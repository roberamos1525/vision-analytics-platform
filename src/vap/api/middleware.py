"""ASGI middleware: request ids, body-size limits, access logs and HTTP metrics."""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from vap.observability.logs import get_logger
from vap.observability.metrics import PlatformMetrics

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

log = get_logger("api.access")
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class _BodyTooLarge(Exception):
    pass


def _error_body(
    code: str, message: str, request_id: str, details: dict[str, Any] | None = None
) -> bytes:
    return json.dumps(
        {
            "error": {"code": code, "message": message, "details": details or {}},
            "request_id": request_id,
        }
    ).encode("utf-8")


class RequestContextMiddleware:
    """Assigns ``X-Request-ID``, enforces body limits and records metrics.

    Body limits are enforced on the actual byte stream (not only the
    ``Content-Length`` header), so chunked uploads cannot bypass them.
    ``upload_prefixes`` select routes that get the larger upload limit.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_body: int,
        max_upload: int,
        upload_prefixes: tuple[str, ...],
        metrics: PlatformMetrics | None = None,
    ) -> None:
        self.app = app
        self.max_body = max_body
        self.max_upload = max_upload
        self.upload_prefixes = upload_prefixes
        self.metrics = metrics

    def _limit_for(self, path: str, method: str) -> int:
        if method == "PUT" and path.startswith(self.upload_prefixes):
            return self.max_upload
        return self.max_body

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {
            k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])
        }
        supplied = headers.get("x-request-id", "")
        request_id = supplied if _REQUEST_ID.match(supplied) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        method = scope.get("method", "GET")
        path = scope.get("path", "")
        limit = self._limit_for(path, method)
        started = time.perf_counter()
        status_holder = {"status": 500, "sent": False}

        declared = headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > limit:
            await self._reject(send, request_id, limit)
            self._record(scope, method, 413, started)
            return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise _BodyTooLarge
            return message

        async def tracking_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                status_holder["status"] = int(message["status"])
                status_holder["sent"] = True
                raw = list(message.get("headers", []))
                raw.append((b"x-request-id", request_id.encode("latin-1")))
                message["headers"] = raw
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _BodyTooLarge:
            if not status_holder["sent"]:
                await self._reject(send, request_id, limit)
            status_holder["status"] = 413
        finally:
            self._record(scope, method, int(status_holder["status"]), started)

    async def _reject(self, send: Send, request_id: str, limit: int) -> None:
        body = _error_body(
            "payload_too_large",
            "request body exceeds the configured limit",
            request_id,
            {"max_bytes": limit},
        )
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"x-request-id", request_id.encode("latin-1")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    def _record(self, scope: Scope, method: str, status: int, started: float) -> None:
        elapsed = time.perf_counter() - started
        route = scope.get("route")
        template = getattr(route, "path", None) or "unmatched"
        if self.metrics:
            self.metrics.http_requests.inc(method=method, route=template, status=str(status))
            self.metrics.http_latency.observe(elapsed, method=method, route=template)
        log.info(
            "request",
            extra={
                "method": method,
                "route": template,
                "status": status,
                "ms": round(elapsed * 1000, 2),
                "request_id": scope.get("state", {}).get("request_id"),
            },
        )
