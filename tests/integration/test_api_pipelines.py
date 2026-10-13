"""End-to-end API flows: cameras -> pipelines -> events/metrics/exports/preview."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.fakes import FakeCapture, ScriptedFactory
from tests.helpers import ADMIN_KEY, OPERATOR_KEY, VIEWER_KEY, wait_until
from vap.config import Settings

ADMIN = {"X-API-Key": ADMIN_KEY}
OPERATOR = {"X-API-Key": OPERATOR_KEY}
VIEWER = {"Authorization": f"Bearer {VIEWER_KEY}"}

SCENE = {
    "kind": "synthetic",
    "num_frames": 120,
    "fps": 10,
    "objects": [
        {
            "object_id": 1,
            "x": 10,
            "y": 40,
            "width": 30,
            "height": 24,
            "vx": 2.5,
            "color": [230, 60, 60],
        },
        {
            "object_id": 2,
            "x": 270,
            "y": 160,
            "width": 26,
            "height": 30,
            "vx": -2.0,
            "color": [60, 200, 80],
        },
    ],
}
CONFIG = {
    "detector": {"kind": "background", "warmup_frames": 10, "warmup_stride": 2},
    "tracker": {"kind": "sort", "min_hits": 2},
    "zones": [
        {"zone_id": "right", "points": [[0.6, 0], [1, 0], [1, 1], [0.6, 1]], "normalized": True}
    ],
    "lines": [{"line_id": "mid", "points": [[200, 0], [200, 240]]}],
    "rules": [
        {"kind": "zone_intrusion", "rule_id": "intrude", "zone_id": "right"},
        {"kind": "line_crossing", "rule_id": "cross", "line_id": "mid"},
    ],
    "realtime": False,
}


def finish(client: TestClient, pid: str, headers: dict[str, str] = VIEWER) -> dict[str, Any]:
    status: dict[str, Any] = {}

    def done() -> bool:
        nonlocal status
        status = client.get(f"/v1/pipelines/{pid}/status", headers=headers).json()
        return status["state"] in ("completed", "failed", "stopped")

    assert wait_until(done, 30)
    return status


@pytest.fixture
def client(client_factory: Callable[..., TestClient], auth_settings: Settings) -> TestClient:
    return client_factory(auth_settings)


def test_full_pipeline_flow(client: TestClient) -> None:
    r = client.post(
        "/v1/cameras", json={"camera_id": "cam1", "name": "Gate", "source": SCENE}, headers=OPERATOR
    )
    assert r.status_code == 201 and r.json()["source"]["kind"] == "synthetic"
    assert client.get("/v1/cameras", headers=VIEWER).json()["count"] == 1

    probe = client.post("/v1/cameras/cam1/probe", headers=OPERATOR).json()
    assert probe["ok"] and probe["frame"] == {"width": 320, "height": 240}

    r = client.post(
        "/v1/pipelines",
        json={"pipeline_id": "p1", "camera_id": "cam1", "config": CONFIG, "start": True},
        headers=OPERATOR,
    )
    assert r.status_code == 201
    status = finish(client, "p1")
    assert status["state"] == "completed" and status["frames_processed"] == 120

    events = client.get("/v1/events", params={"pipeline_id": "p1"}, headers=VIEWER).json()
    types = {e["type"] for e in events["items"]}
    assert {"zone_intrusion", "line_crossing"} <= types
    one = events["items"][0]
    assert (
        client.get(f"/v1/events/{one['event_id']}", headers=VIEWER).json()["event_id"]
        == one["event_id"]
    )
    snap = client.get(f"/v1/events/{one['event_id']}/snapshot", headers=VIEWER)
    assert snap.status_code == 200 and snap.headers["content-type"] == "image/jpeg"
    stats = client.get("/v1/events/stats", headers=VIEWER).json()
    assert stats["total"] == len(events["items"])
    assert client.get("/v1/events/recent", headers=VIEWER).json()["count"] == stats["total"]

    analytics = client.get("/v1/pipelines/p1/analytics", headers=VIEWER).json()
    assert analytics["zones"][0]["entries"] >= 1 and analytics["lines"][0]["total"] >= 1
    metrics = client.get("/v1/pipelines/p1/metrics", headers=VIEWER).json()
    assert metrics["frames_processed"] == 120
    history = client.get("/v1/pipelines/p1/metrics/history", headers=VIEWER).json()
    assert history["count"] >= 1
    tracks = client.get("/v1/pipelines/p1/tracks", headers=VIEWER).json()
    assert tracks["count"] >= 2
    runs = client.get("/v1/pipelines/p1/runs", headers=VIEWER).json()
    assert runs["items"][0]["status"] == "completed"

    jpg = client.get("/v1/pipelines/p1/snapshot.jpg", headers=VIEWER)
    assert jpg.status_code == 200 and jpg.headers["x-frame-index"] == "119"
    assert client.get("/v1/pipelines/p1/heatmap.png", headers=VIEWER).content[:4] == b"\x89PNG"

    csv_text = client.get("/v1/exports/events", params={"format": "csv"}, headers=VIEWER).text
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    assert len(rows) == stats["total"] and rows[0]["pipeline_id"] == "p1"
    jsonl = client.get("/v1/exports/tracks", params={"format": "jsonl"}, headers=VIEWER).text
    assert all(json.loads(line)["pipeline_id"] == "p1" for line in jsonl.splitlines())
    metrics_csv = client.get("/v1/exports/metrics", params={"pipeline_id": "p1"}, headers=VIEWER)
    assert metrics_csv.text.startswith("ts,")
    assert client.get("/v1/exports/metrics", headers=VIEWER).status_code == 422

    prom = client.get("/metrics", headers=VIEWER).text
    assert 'vap_frames_processed_total{pipeline="p1"} 120' in prom
    assert "vap_http_requests_total" in prom


def test_mjpeg_preview_streams_frames(client: TestClient) -> None:
    looping = {**SCENE, "loop": True, "num_frames": 30}
    client.post("/v1/cameras", json={"camera_id": "c", "source": looping}, headers=OPERATOR)
    cfg = {**CONFIG, "realtime": True}
    client.post(
        "/v1/pipelines",
        json={"pipeline_id": "live", "camera_id": "c", "config": cfg},
        headers=OPERATOR,
    )
    assert client.post("/v1/pipelines/live/start", headers=OPERATOR).json()["state"] in (
        "starting",
        "running",
    )
    assert client.post("/v1/pipelines/live/start", headers=OPERATOR).status_code == 409
    with client.stream(
        "GET",
        "/v1/pipelines/live/preview.mjpg",
        params={"max_frames": 3, "fps": 30},
        headers=VIEWER,
    ) as r:
        assert r.headers["content-type"].startswith("multipart/x-mixed-replace")
        body = b"".join(r.iter_bytes())
    assert body.count(b"--vapframe") == 3 and body.count(b"\xff\xd8") == 3
    stopped = client.post("/v1/pipelines/live/stop", headers=OPERATOR).json()
    assert stopped["state"] == "stopped"
    assert client.post("/v1/pipelines/live/stop", headers=OPERATOR).status_code == 409
    assert client.get("/v1/pipelines/live/preview.mjpg", headers=VIEWER).status_code == 409
    view = client.get("/v1/pipelines/live", headers=VIEWER).json()
    assert view["desired_state"] == "stopped" and view["status"]["state"] == "stopped"


def test_management_conflicts_and_lifecycle(client: TestClient) -> None:
    client.post(
        "/v1/cameras", json={"camera_id": "c", "source": {**SCENE, "loop": True}}, headers=OPERATOR
    )
    client.post(
        "/v1/pipelines",
        json={"pipeline_id": "p", "camera_id": "c", "config": {**CONFIG, "realtime": True}},
        headers=OPERATOR,
    )
    assert (
        client.post(
            "/v1/pipelines", json={"pipeline_id": "p", "camera_id": "c"}, headers=OPERATOR
        ).status_code
        == 409
    )
    assert (
        client.post(
            "/v1/pipelines", json={"pipeline_id": "q", "camera_id": "zzz"}, headers=OPERATOR
        ).status_code
        == 404
    )
    assert client.get("/v1/pipelines/p/status", headers=VIEWER).status_code == 409  # never started
    client.post("/v1/pipelines/p/start", headers=OPERATOR)
    put = client.put("/v1/cameras/c", json={"source": SCENE}, headers=OPERATOR)
    assert put.status_code == 409 and put.json()["error"]["code"] == "conflict"
    assert (
        client.put("/v1/pipelines/p", json={"config": CONFIG}, headers=OPERATOR).status_code == 409
    )
    assert client.delete("/v1/pipelines/p", headers=OPERATOR).status_code == 409
    client.post("/v1/pipelines/p/stop", headers=OPERATOR)
    updated = client.put(
        "/v1/pipelines/p", json={"config": {**CONFIG, "process_every_n": 2}}, headers=OPERATOR
    )
    assert updated.json()["config"]["process_every_n"] == 2
    assert client.delete("/v1/cameras/c", headers=OPERATOR).status_code == 409
    assert client.delete("/v1/pipelines/p", headers=OPERATOR).status_code == 204
    assert client.delete("/v1/cameras/c", headers=OPERATOR).status_code == 204
    assert client.get("/v1/cameras/c", headers=VIEWER).status_code == 404


def test_validation_envelopes(client: TestClient) -> None:
    bad_cam = client.post(
        "/v1/cameras", json={"camera_id": "c", "source": {"kind": "webcam"}}, headers=OPERATOR
    )
    body = bad_cam.json()
    assert bad_cam.status_code == 422 and body["error"]["code"] == "invalid_input"
    assert body["error"]["details"]["errors"] and body["request_id"]
    client.post("/v1/cameras", json={"camera_id": "c", "source": SCENE}, headers=OPERATOR)
    bad_cfg = {
        "rules": [{"kind": "loitering", "rule_id": "r", "zone_id": "ghost", "min_dwell_s": 1}]
    }
    r = client.post(
        "/v1/pipelines",
        json={"pipeline_id": "p", "camera_id": "c", "config": bad_cfg},
        headers=OPERATOR,
    )
    assert r.status_code == 422 and "ghost" in json.dumps(r.json())
    file_cam = client.post(
        "/v1/cameras",
        json={"camera_id": "f", "source": {"kind": "file", "path": "../../etc/passwd"}},
        headers=OPERATOR,
    )
    assert file_cam.status_code == 201  # stored; access is checked when the source is built
    probe = client.post("/v1/cameras/f/probe", headers=OPERATOR)
    assert probe.status_code == 403 and probe.json()["error"]["code"] == "forbidden"
    assert client.get("/v1/events", params={"type": "bogus"}, headers=VIEWER).status_code == 422
    assert client.get("/v1/events", params={"limit": 5000}, headers=VIEWER).status_code == 422
    assert client.get("/v1/nowhere", headers=VIEWER).json()["error"]["code"] == "not_found"


def test_authentication_and_roles(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}
    r = client.get("/v1/cameras")
    assert r.status_code == 401 and r.headers["www-authenticate"] == "ApiKey"
    assert (
        client.get("/v1/cameras", headers={"X-API-Key": "wrong-key-xxxxxxxxxxxx"}).status_code
        == 401
    )
    denied = client.post("/v1/cameras", json={"camera_id": "c", "source": SCENE}, headers=VIEWER)
    assert denied.status_code == 403 and denied.json()["error"]["details"]["role"] == "viewer"
    assert (
        client.post(
            "/v1/cameras", json={"camera_id": "c", "source": SCENE}, headers=ADMIN
        ).status_code
        == 201
    )
    assert (
        client.put("/v1/models/m/versions/1.0.0", content=b"x", headers=OPERATOR).status_code == 403
    )
    assert client.get("/v1/system", headers=VIEWER).json()["auth_enabled"] is True


def test_limits_and_request_ids(client: TestClient) -> None:
    big = {"camera_id": "c", "name": "x" * 3_000_000, "source": SCENE}
    r = client.post("/v1/cameras", json=big, headers=OPERATOR)
    assert r.status_code == 413 and r.json()["error"]["code"] == "payload_too_large"
    ok = client.get("/v1/health", headers={**VIEWER, "X-Request-ID": "trace-123"})
    assert ok.headers["x-request-id"] == "trace-123" and ok.json()["database"]["ok"]
    generated = client.get("/healthz", headers={"X-Request-ID": "bad id with spaces"})
    assert generated.headers["x-request-id"] != "bad id with spaces"
    assert client.get("/readyz").status_code == 200


def test_pipeline_limit(client_factory: Callable[..., TestClient], settings: Settings) -> None:
    data = settings.model_dump(mode="json")
    data["pipelines"]["max_pipelines"] = 1
    client = client_factory(Settings.model_validate(data))
    client.post("/v1/cameras", json={"camera_id": "c", "source": {**SCENE, "loop": True}})
    for pid in ("a", "b"):
        client.post(
            "/v1/pipelines",
            json={"pipeline_id": pid, "camera_id": "c", "config": {**CONFIG, "realtime": True}},
        )
    assert client.post("/v1/pipelines/a/start").status_code == 200
    r = client.post("/v1/pipelines/b/start")
    assert r.status_code == 429 and r.json()["error"]["details"]["max_pipelines"] == 1


def test_rtsp_camera_through_manager(
    client_factory: Callable[..., TestClient], settings: Settings
) -> None:
    factory = ScriptedFactory(
        [
            lambda: FakeCapture(frames=40, size=(64, 48)),
            lambda: FakeCapture(frames=40, size=(64, 48)),
        ],
        default=lambda: FakeCapture(opened=False),
    )
    client = client_factory(settings, capture_factory=factory)
    src = {
        "kind": "rtsp",
        "url": "rtsp://admin:pw@10.0.0.9/live",
        "reconnect_initial_delay": 0.01,
        "reconnect_max_delay": 0.02,
        "max_reconnect_attempts": 3,
        "frame_timeout": 1.0,
    }
    cam = client.post("/v1/cameras", json={"camera_id": "door", "source": src}).json()
    assert cam["source"]["url"] == "rtsp://admin:***@10.0.0.9/live"
    client.post(
        "/v1/pipelines",
        json={
            "pipeline_id": "door-p",
            "camera_id": "door",
            "config": {"realtime": False},
            "start": True,
        },
    )
    status = finish(client, "door-p", {})
    assert status["state"] == "failed"
    assert status["source"]["health"]["reconnects"] == 1
    assert status["frames_processed"] == 80
    assert status["source"]["uri"] == "rtsp://admin:***@10.0.0.9/live"
    assert wait_until(
        lambda: client.get("/v1/events", params={"type": "source_down"}).json()["count"] == 1, 5
    )


def test_restore_running_pipelines_after_restart(
    client_factory: Callable[..., TestClient], settings: Settings
) -> None:
    first = client_factory(settings)
    first.post("/v1/cameras", json={"camera_id": "c", "source": {**SCENE, "loop": True}})
    first.post(
        "/v1/pipelines",
        json={
            "pipeline_id": "p",
            "camera_id": "c",
            "config": {**CONFIG, "realtime": True},
            "start": True,
        },
    )
    first.__exit__(None, None, None)  # shutdown keeps desired_state=running
    second = client_factory(settings)
    assert wait_until(
        lambda: second.get("/v1/pipelines/p/status").json().get("state") == "running", 10
    )
    second.post("/v1/pipelines/p/stop")
