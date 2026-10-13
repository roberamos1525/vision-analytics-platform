from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from tests.helpers import wait_until
from vap.errors import ConflictError, InvalidInputError
from vap.events.bus import EventBus
from vap.events.model import Event, EventType, Severity
from vap.events.sinks import (
    JsonlEventSink,
    RecentEventsSink,
    WebhookSink,
    sign_payload,
    validate_webhook_url,
    verify_signature,
)
from vap.observability.metrics import PlatformMetrics


def event(etype: EventType = EventType.ZONE_INTRUSION, pipeline: str = "p1") -> Event:
    return Event(etype, pipeline, "cam", 3, 0.3, "something happened", track_ids=[1], zone_id="z")


class Recorder:
    def __init__(self, name: str = "rec", fail: bool = False) -> None:
        self.name = name
        self.fail = fail
        self.events: list[Event] = []
        self.closed = False

    def handle(self, ev: Event) -> None:
        if self.fail:
            raise RuntimeError("sink broke")
        self.events.append(ev)

    def close(self) -> None:
        self.closed = True


class TestBus:
    def test_dispatch_and_error_isolation(self) -> None:
        bus = EventBus()
        good, bad = Recorder("good"), Recorder("bad", fail=True)
        bus.subscribe(bad)
        bus.subscribe(good)
        bus.start()
        for _ in range(5):
            assert bus.publish(event())
        assert bus.flush(5.0)
        assert len(good.events) == 5
        stats = bus.stats()
        assert stats["delivered"] == 5 and stats["sink_errors"] == {"bad": 5}
        bus.stop()
        assert good.closed and not bus.running

    def test_drops_when_full_and_counts(self) -> None:
        metrics = PlatformMetrics()
        bus = EventBus(queue_size=2, metrics=metrics)
        assert bus.publish(event()) and bus.publish(event())
        assert not bus.publish(event())
        assert bus.dropped == 1 and metrics.bus_dropped.value() == 1
        rec = Recorder()
        bus.subscribe(rec)
        assert bus.flush()  # synchronous drain when not started
        assert len(rec.events) == 2

    def test_subscription_management(self) -> None:
        bus = EventBus()
        bus.subscribe(Recorder("a"))
        with pytest.raises(ConflictError):
            bus.subscribe(Recorder("a"))
        assert bus.sink("a") is not None
        assert bus.unsubscribe("a") is not None and bus.sink("a") is None


class TestSinks:
    def test_recent_ring_buffer(self) -> None:
        sink = RecentEventsSink(maxlen=3)
        for p in ("p1", "p2", "p1", "p1"):
            sink.handle(event(pipeline=p))
        assert len(sink.recent(10)) == 3
        assert all(e["pipeline_id"] == "p1" for e in sink.recent(10, "p1"))
        assert sink.wait_for(4, 0.1) and not sink.wait_for(5, 0.05)

    def test_jsonl_rotation(self, tmp_path: Path) -> None:
        sink = JsonlEventSink(tmp_path / "ev" / "events.jsonl", max_bytes=1024)
        for _ in range(10):
            sink.handle(event())
        assert (tmp_path / "ev" / "events.jsonl.1").exists()
        lines = (tmp_path / "ev" / "events.jsonl").read_text().splitlines()
        assert json.loads(lines[0])["type"] == "zone_intrusion"
        with pytest.raises(InvalidInputError):
            JsonlEventSink(tmp_path / "x.jsonl", max_bytes=10)

    def test_event_serialisation(self) -> None:
        d = event().to_dict()
        assert d["severity"] == Severity.WARNING.value and d["track_ids"] == [1]
        assert len(d["event_id"]) == 32


def public_resolver(_: str) -> list[str]:
    return ["93.184.216.34"]


class TestWebhook:
    def test_signature_helpers(self) -> None:
        sig = sign_payload("s3cret", "123", b"{}")
        assert sig.startswith("sha256=") and verify_signature("s3cret", "123", b"{}", sig)
        assert not verify_signature("other", "123", b"{}", sig)

    def test_url_validation(self) -> None:
        with pytest.raises(InvalidInputError):
            validate_webhook_url("ftp://example.com/x", allow_private=True)
        with pytest.raises(InvalidInputError):
            validate_webhook_url("https://user:pw@example.com/x", allow_private=True)
        with pytest.raises(InvalidInputError):
            validate_webhook_url(
                "http://internal/x", allow_private=False, resolver=lambda _: ["10.1.2.3"]
            )
        with pytest.raises(InvalidInputError):
            validate_webhook_url(
                "http://localhost/x", allow_private=False, resolver=lambda _: ["127.0.0.1"]
            )
        assert validate_webhook_url(
            "https://hooks.example.com/x", allow_private=False, resolver=public_resolver
        )
        assert validate_webhook_url("http://127.0.0.1:9/x", allow_private=True)

    def test_delivers_signed_payload_with_retries(self) -> None:
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(503 if len(calls) < 3 else 200)

        sleeps: list[float] = []
        metrics = PlatformMetrics()
        sink = WebhookSink(
            ["https://hooks.example.com/vap"],
            secret="topsecret",
            client=httpx.Client(transport=httpx.MockTransport(handler)),
            resolver=public_resolver,
            backoff_s=0.1,
            sleep=sleeps.append,
            metrics=metrics,
        )
        ev = event()
        sink.handle(ev)
        assert sink.flush(5.0)
        assert len(calls) == 3 and sleeps == [0.1, 0.2]
        req = calls[-1]
        body = req.content
        assert json.loads(body)["event_id"] == ev.event_id
        assert verify_signature(
            "topsecret", req.headers["X-VAP-Timestamp"], body, req.headers["X-VAP-Signature"]
        )
        assert req.headers["X-VAP-Event"] == "zone_intrusion"
        assert sink.stats()["delivered"] == 1
        assert metrics.webhook_deliveries.value(outcome="retry") == 2
        sink.close()

    def test_client_errors_are_not_retried_and_filters_apply(self) -> None:
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(400)

        sink = WebhookSink(
            ["https://hooks.example.com/vap"],
            client=httpx.Client(transport=httpx.MockTransport(handler)),
            resolver=public_resolver,
            event_types=["loitering"],
            sleep=lambda _: None,
        )
        sink.handle(event(EventType.ZONE_INTRUSION))
        sink.handle(event(EventType.LOITERING))
        assert sink.flush(5.0)
        assert len(calls) == 1 and sink.failed == 1 and sink.last_error == "HTTP 400"
        assert "X-VAP-Signature" not in calls[0].headers
        sink.close()

    def test_network_errors_exhaust_retries(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        sink = WebhookSink(
            ["http://127.0.0.1:9/x"],
            allow_private=True,
            max_retries=2,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
            sleep=lambda _: None,
        )
        sink.handle(event())
        assert wait_until(lambda: sink.failed == 1, 5.0)
        assert "ConnectError" in (sink.last_error or "")
        sink.close()
        with pytest.raises(InvalidInputError):
            WebhookSink([])
