"""Runtime wiring (webhooks, retention, health) and PipelineManager behaviour."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from tests.fakes import FakeCapture, ScriptedFactory
from tests.helpers import wait_until
from vap.config import Settings
from vap.errors import ConflictError, InvalidInputError, NotFoundError, SourceTimeoutError
from vap.events.model import Event, EventType
from vap.events.sinks import verify_signature
from vap.extraction import ExtractionSpec
from vap.runtime import Runtime

SCENE = {"kind": "synthetic", "num_frames": 60, "num_objects": 2, "seed": 1}
ZONE_ALL = {"zone_id": "all", "points": [[0, 0], [1, 0], [1, 1], [0, 1]], "normalized": True}
CONFIG = {
    "detector": {"kind": "background", "warmup_frames": 5, "warmup_stride": 2},
    "tracker": {"min_hits": 2},
    "zones": [ZONE_ALL],
    "rules": [{"kind": "zone_intrusion", "rule_id": "in", "zone_id": "all"}],
    "realtime": False,
}


def with_webhook(settings: Settings) -> Settings:
    data = settings.model_dump(mode="json")
    data["webhooks"] = {
        "urls": ["https://hooks.example.com/vap"],
        "secret": "wh-secret",
        "event_types": ["zone_intrusion"],
        "backoff_s": 0,
    }
    return Settings.model_validate(data)


def test_events_reach_signed_webhook(
    client_factory: Callable[..., TestClient], settings: Settings
) -> None:
    received: list[httpx.Request] = []
    lock = threading.Lock()

    def handler(request: httpx.Request) -> httpx.Response:
        with lock:
            received.append(request)
        return httpx.Response(204)

    client = client_factory(
        with_webhook(settings),
        webhook_client=httpx.Client(transport=httpx.MockTransport(handler)),
        webhook_resolver=lambda _: ["93.184.216.34"],
    )
    client.post("/v1/cameras", json={"camera_id": "c", "source": SCENE})
    client.post(
        "/v1/pipelines",
        json={"pipeline_id": "p", "camera_id": "c", "config": CONFIG, "start": True},
    )
    assert wait_until(
        lambda: client.get("/v1/pipelines/p/status").json()["state"] == "completed", 20
    )
    assert wait_until(lambda: len(received) >= 1, 5)
    req = received[0]
    payload = json.loads(req.content)
    assert payload["type"] == "zone_intrusion" and payload["pipeline_id"] == "p"
    assert verify_signature(
        "wh-secret", req.headers["X-VAP-Timestamp"], req.content, req.headers["X-VAP-Signature"]
    )
    health = client.get("/v1/health").json()
    assert health["webhooks"]["delivered"] >= 1 and health["status"] == "ok"
    assert client.get("/v1/system").json()["webhooks_configured"] == 1


def test_private_webhook_target_is_rejected_at_startup(settings: Settings) -> None:
    data = settings.model_dump(mode="json")
    data["webhooks"] = {"urls": ["http://127.0.0.1:9000/hook"]}
    with pytest.raises(InvalidInputError, match="private or loopback"):
        Runtime(Settings.model_validate(data))


def test_retention_purges_old_events(settings: Settings) -> None:
    runtime = Runtime(settings)
    old = Event(EventType.CROWDING, "p", "c", 0, 0.0, "old")
    old.created_at = time.time() - 365 * 86_400
    runtime.repos.events.insert(old)
    runtime.repos.events.insert(Event(EventType.CROWDING, "p", "c", 1, 0.1, "new"))
    runtime.start(restore=False)
    try:
        assert [e["message"] for e in runtime.repos.events.list()] == ["new"]
        runtime.start()  # idempotent
    finally:
        runtime.close()


@pytest.fixture
def runtime(settings: Settings) -> Runtime:
    rt = Runtime(settings)
    rt.start(restore=False)
    yield rt
    rt.close()


class TestManager:
    def test_probe_timeout_and_failure(self, settings: Settings) -> None:
        stall = ScriptedFactory([lambda: FakeCapture(block_after=0)])
        rt = Runtime(settings, capture_factory=stall)
        try:
            rt.manager.create_camera(
                "cam", "", {"kind": "rtsp", "url": "rtsp://cam.local/s", "frame_timeout": 5}
            )
            with pytest.raises(SourceTimeoutError):
                rt.manager.probe_camera("cam", timeout_s=0.3)
            dead = ScriptedFactory([], default=lambda: FakeCapture(opened=False))
            rt.manager.capture_factory = dead
            rt.manager.update_camera(
                "cam",
                "",
                {"kind": "rtsp", "url": "rtsp://cam.local/s", "max_reconnect_attempts": 1},
            )
            with pytest.raises(Exception, match="connection attempts"):
                rt.manager.probe_camera("cam", timeout_s=5)
        finally:
            rt.close()

    def test_ids_and_lookup_errors(self, runtime: Runtime) -> None:
        mgr = runtime.manager
        with pytest.raises(InvalidInputError):
            mgr.create_camera("bad id", "", SCENE)
        with pytest.raises(InvalidInputError):
            mgr.create_camera("c", "", {"kind": "synthetic", "fps": -1})
        mgr.create_camera("c", "Cam", SCENE)
        with pytest.raises(NotFoundError):
            mgr.create_pipeline("p", "missing", {})
        with pytest.raises(InvalidInputError):
            mgr.create_pipeline("p", "c", {"process_every_n": 0})
        with pytest.raises(NotFoundError):
            mgr.start("nope")
        with pytest.raises(NotFoundError):
            mgr.stop("nope")
        with pytest.raises(NotFoundError):
            mgr.require_pipeline("nope")
        assert mgr.get_pipeline("nope") is None

    def test_extraction_job(self, runtime: Runtime, settings: Settings) -> None:
        runtime.manager.create_camera("c", "", SCENE)
        result = runtime.manager.extract("c", ExtractionSpec(mode="interval", interval_s=1.0), "j")
        assert result["count"] == 4  # 60 frames at 15 fps = 4 s
        assert Path(result["directory"]).is_dir()
        with pytest.raises(InvalidInputError):
            runtime.manager.extract("c", ExtractionSpec(), "../escape")

    def test_shutdown_stops_running_pipelines(self, settings: Settings) -> None:
        rt = Runtime(settings)
        rt.start(restore=False)
        rt.manager.create_camera("c", "", {**SCENE, "loop": True})
        rt.manager.create_pipeline("p", "c", {**CONFIG, "realtime": True})
        rt.manager.start("p")
        assert rt.manager.running_ids() == ["p"]
        with pytest.raises(ConflictError):
            rt.manager.delete_pipeline("p")
        pipe = rt.manager.get_pipeline("p")
        rt.close()
        assert pipe is not None and pipe.state.value == "stopped"
        rt2 = Runtime(settings)
        assert rt2.repos.pipelines.get("p").desired_state == "running"
        rt2.start(restore=True)
        try:
            assert wait_until(lambda: rt2.manager.running_ids() == ["p"], 5)
        finally:
            rt2.close()


def test_readiness_reports_degraded_bus(
    client_factory: Callable[..., TestClient], settings: Settings
) -> None:
    client = client_factory(settings)
    runtime: Runtime = client.app.state.runtime  # type: ignore[attr-defined]
    assert client.get("/readyz").json() == {"status": "ok"}
    runtime.bus.stop()
    r = client.get("/readyz")
    assert r.status_code == 503 and r.json() == {"status": "degraded"}
    health = client.get("/v1/health").json()
    assert health["status"] == "degraded" and health["event_bus"]["running"] is False
    runtime.bus.start()
    assert client.get("/readyz").status_code == 200


def test_finished_pipeline_can_be_restarted_with_new_run(
    client_factory: Callable[..., TestClient], settings: Settings
) -> None:
    client = client_factory(settings)
    client.post("/v1/cameras", json={"camera_id": "c", "source": {**SCENE, "num_frames": 20}})
    client.post("/v1/pipelines", json={"pipeline_id": "p", "camera_id": "c", "config": CONFIG})
    runs = set()
    for _ in range(2):
        started = client.post("/v1/pipelines/p/start").json()
        runs.add(started["run_id"])
        assert wait_until(
            lambda: client.get("/v1/pipelines/p/status").json()["state"] == "completed", 10
        )
    history = client.get("/v1/pipelines/p/runs").json()["items"]
    assert len(runs) == 2 and {r["run_id"] for r in history} == runs
    assert all(r["status"] == "completed" and r["frames"] == 20 for r in history)
