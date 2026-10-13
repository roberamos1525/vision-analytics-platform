"""Event sinks: SQLite, in-memory ring buffer, JSONL file and signed webhooks."""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import queue
import socket
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import httpx

from vap.errors import InvalidInputError
from vap.events.model import Event
from vap.observability.logs import get_logger
from vap.observability.metrics import PlatformMetrics

if TYPE_CHECKING:  # imported lazily to avoid a storage <-> events import cycle
    from vap.storage.repositories import EventRepository

log = get_logger("events.sinks")


class SqliteEventSink:
    name = "sqlite"

    def __init__(self, repo: EventRepository) -> None:
        self.repo = repo

    def handle(self, event: Event) -> None:
        self.repo.insert(event)

    def close(self) -> None:
        return None


class RecentEventsSink:
    """Bounded ring buffer of the most recent events for fast API reads."""

    name = "recent"

    def __init__(self, maxlen: int = 500) -> None:
        self._events: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self.seen = 0

    def handle(self, event: Event) -> None:
        with self._cond:
            self._events.append(event.to_dict())
            self.seen += 1
            self._cond.notify_all()

    def recent(self, limit: int = 50, pipeline_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._events)
        if pipeline_id is not None:
            items = [e for e in items if e["pipeline_id"] == pipeline_id]
        return list(reversed(items[-limit:]))

    def wait_for(self, count: int, timeout: float) -> bool:
        """Wait until at least ``count`` events were seen (test/CLI helper)."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while self.seen < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)
            return True

    def close(self) -> None:
        return None


class JsonlEventSink:
    """Append events to a JSON-lines file with simple size-based rotation."""

    name = "jsonl"

    def __init__(self, path: str | Path, max_bytes: int = 50 * 1024 * 1024) -> None:
        if max_bytes < 1024:
            raise InvalidInputError("max_bytes must be >= 1024")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self._lock = threading.Lock()

    def handle(self, event: Event) -> None:
        line = json.dumps(event.to_dict(), separators=(",", ":")) + "\n"
        with self._lock:
            if self.path.exists() and self.path.stat().st_size + len(line) > self.max_bytes:
                self.path.replace(self.path.with_name(self.path.name + ".1"))
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line)

    def close(self) -> None:
        return None


def sign_payload(secret: str, timestamp: str, body: bytes) -> str:
    mac = hmac.new(secret.encode("utf-8"), timestamp.encode("ascii") + b"." + body, hashlib.sha256)
    return "sha256=" + mac.hexdigest()


def verify_signature(secret: str, timestamp: str, body: bytes, signature: str) -> bool:
    return hmac.compare_digest(sign_payload(secret, timestamp, body), signature)


Resolver = Callable[[str], list[str]]


def _resolve(host: str) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise InvalidInputError(f"cannot resolve webhook host {host!r}") from exc
    return sorted({str(info[4][0]) for info in infos})


def validate_webhook_url(url: str, *, allow_private: bool, resolver: Resolver = _resolve) -> str:
    """Reject non-HTTP(S) URLs and, unless allowed, private/loopback targets (SSRF guard)."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise InvalidInputError("webhook URL must be http(s) with a host", details={"url": url})
    if parts.username or parts.password:
        raise InvalidInputError("webhook URL must not embed credentials")
    if allow_private:
        return url
    for addr in resolver(parts.hostname):
        ip = ipaddress.ip_address(addr.split("%", 1)[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise InvalidInputError(
                "webhook URL resolves to a private or loopback address",
                details={"host": parts.hostname, "address": addr},
            )
    return url


class WebhookSink:
    """Deliver events as signed JSON POSTs on a background worker.

    Retries 5xx/429/network failures with exponential backoff; other 4xx
    responses are permanent failures. Headers:
    ``X-VAP-Event``, ``X-VAP-Timestamp`` and ``X-VAP-Signature``
    (``sha256=<hex HMAC of "timestamp.body">``) when a secret is configured.
    """

    name = "webhook"

    def __init__(
        self,
        urls: list[str],
        *,
        secret: str | None = None,
        timeout_s: float = 5.0,
        max_retries: int = 3,
        backoff_s: float = 0.5,
        queue_size: int = 1000,
        event_types: list[str] | None = None,
        allow_private: bool = False,
        client: httpx.Client | None = None,
        metrics: PlatformMetrics | None = None,
        resolver: Resolver = _resolve,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not urls:
            raise InvalidInputError("webhook sink needs at least one URL")
        self.urls = [
            validate_webhook_url(u, allow_private=allow_private, resolver=resolver) for u in urls
        ]
        self.secret = secret
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.event_types = set(event_types or [])
        self._client = client or httpx.Client(timeout=timeout_s, follow_redirects=False)
        self._owns_client = client is None
        self._metrics = metrics
        self._sleep = sleep
        self._queue: queue.Queue[Event | None] = queue.Queue(maxsize=queue_size)
        self.delivered = 0
        self.failed = 0
        self.dropped = 0
        self.last_error: str | None = None
        self._thread = threading.Thread(target=self._run, name="webhook-sink", daemon=True)
        self._thread.start()

    def handle(self, event: Event) -> None:
        if self.event_types and event.type.value not in self.event_types:
            return
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            self.dropped += 1
            self._count("dropped")

    def _count(self, outcome: str) -> None:
        if self._metrics:
            self._metrics.webhook_deliveries.inc(outcome=outcome)

    def _run(self) -> None:
        while True:
            event = self._queue.get()
            try:
                if event is None:
                    return
                body = json.dumps(event.to_dict(), separators=(",", ":")).encode("utf-8")
                for url in self.urls:
                    self._deliver(url, event, body)
            finally:
                self._queue.task_done()

    def _headers(self, event: Event, body: bytes) -> dict[str, str]:
        ts = str(int(time.time()))
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "vision-analytics-platform/webhook",
            "X-VAP-Event": event.type.value,
            "X-VAP-Event-Id": event.event_id,
            "X-VAP-Timestamp": ts,
        }
        if self.secret:
            headers["X-VAP-Signature"] = sign_payload(self.secret, ts, body)
        return headers

    def _deliver(self, url: str, event: Event, body: bytes) -> bool:
        for attempt in range(self.max_retries + 1):
            if attempt:
                self._sleep(self.backoff_s * (2 ** (attempt - 1)))
            try:
                resp = self._client.post(url, content=body, headers=self._headers(event, body))
            except httpx.HTTPError as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                self._count("retry" if attempt < self.max_retries else "failed")
                continue
            if 200 <= resp.status_code < 300:
                self.delivered += 1
                self._count("delivered")
                return True
            self.last_error = f"HTTP {resp.status_code}"
            if resp.status_code != 429 and resp.status_code < 500:
                break
            self._count("retry" if attempt < self.max_retries else "failed")
        self.failed += 1
        self._count("failed")
        log.warning("webhook delivery failed", extra={"url": url, "error": self.last_error})
        return False

    def flush(self, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.01)
        return self._queue.unfinished_tasks == 0

    def stats(self) -> dict[str, Any]:
        return {
            "urls": len(self.urls),
            "delivered": self.delivered,
            "failed": self.failed,
            "dropped": self.dropped,
            "pending": self._queue.qsize(),
            "last_error": self.last_error,
        }

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=5.0)
        if self._owns_client:
            self._client.close()
