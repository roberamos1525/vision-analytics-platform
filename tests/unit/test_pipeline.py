from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.fakes import FakeCapture, ScriptedFactory
from tests.helpers import two_object_scene, wait_until
from vap.detection.background import BackgroundSubtractionDetector
from vap.detection.base import Detection, Detector
from vap.events.bus import EventBus
from vap.events.model import EventType
from vap.events.sinks import RecentEventsSink, SqliteEventSink
from vap.imaging.frame import Frame
from vap.observability.metrics import PlatformMetrics
from vap.pipeline.config import PipelineConfig
from vap.pipeline.pipeline import CameraPipeline, PipelineOptions, PipelineState
from vap.sources.rtsp import ReconnectPolicy, RtspSource
from vap.sources.synthetic import SyntheticScene, SyntheticSource
from vap.storage.database import Database
from vap.storage.repositories import Repositories
from vap.tracking.trackers import build_tracker


class GroundTruthDetector(Detector):
    """Perfect detector reading boxes from the synthetic scene (isolates analytics)."""

    name = "oracle"

    def __init__(self, scene: SyntheticScene) -> None:
        self.scene = scene

    def detect(self, frame: Frame) -> list[Detection]:
        idx = int(frame.metadata.get("scene_index", frame.index))
        return [Detection(g.bbox, 1.0, "object") for g in self.scene.ground_truth(idx)]


CONFIG: dict[str, Any] = {
    "zones": [
        {"zone_id": "right", "points": [[0.5, 0], [1, 0], [1, 1], [0.5, 1]], "normalized": True}
    ],
    "lines": [{"line_id": "mid", "points": [[160, 0], [160, 240]]}],
    "rules": [
        {"kind": "zone_intrusion", "rule_id": "intrude", "zone_id": "right"},
        {"kind": "line_crossing", "rule_id": "cross", "line_id": "mid"},
        {"kind": "loitering", "rule_id": "loiter", "zone_id": "right", "min_dwell_s": 3.0},
    ],
    "realtime": False,
    "flow_every_n": 5,
    "tracker": {"min_hits": 2},
}


@pytest.fixture
def env(tmp_path: Path) -> dict[str, Any]:
    db = Database(tmp_path / "vap.db")
    repos = Repositories(db)
    metrics = PlatformMetrics()
    bus = EventBus(metrics=metrics)
    recent = RecentEventsSink()
    bus.subscribe(SqliteEventSink(repos.events))
    bus.subscribe(recent)
    bus.start()
    yield {
        "repos": repos,
        "bus": bus,
        "recent": recent,
        "metrics": metrics,
        "snapshots": tmp_path / "snaps",
    }
    bus.stop()
    db.close()


def make_pipeline(
    env: dict[str, Any], source: Any, detector: Detector, config: dict[str, Any]
) -> CameraPipeline:
    cfg = PipelineConfig.model_validate(config)
    return CameraPipeline(
        "p1",
        "cam",
        source,
        detector,
        build_tracker(cfg.tracker),
        cfg,
        bus=env["bus"],
        repos=env["repos"],
        metrics=env["metrics"],
        options=PipelineOptions(
            snapshot_dir=env["snapshots"], metrics_interval_s=0.0, backend="numpy"
        ),
    )


class TestPipelineRun:
    def test_full_run_emits_events_and_persists(self, env: dict[str, Any]) -> None:
        scene = SyntheticScene(two_object_scene(num_frames=100))
        pipe = make_pipeline(
            env, SyntheticSource("cam", scene.spec), GroundTruthDetector(scene), CONFIG
        )
        pipe.start()
        pipe.join(20)
        assert pipe.state == PipelineState.COMPLETED and pipe.frames_processed == 100
        env["bus"].flush()
        events = env["repos"].events.list(limit=100)
        types = {e["type"] for e in events}
        assert {"zone_intrusion", "line_crossing", "loitering"} <= types
        crossing = next(e for e in events if e["type"] == "line_crossing")
        assert crossing["line_id"] == "mid" and crossing["snapshot"]
        assert (env["snapshots"] / crossing["snapshot"]).is_file()
        tracks = env["repos"].tracks.list("p1")
        assert len(tracks) == 2 and {t["direction"] for t in tracks} == {"E", "W"}
        assert env["repos"].metrics.list("p1")
        run = env["repos"].runs.list("p1")[0]
        assert run["status"] == "completed" and run["frames"] == 100
        analytics = pipe.analytics()
        assert analytics["lines"][0]["total"] == 2 and analytics["flow"] is not None
        assert env["metrics"].frames.value(pipeline="p1") == 100
        status = pipe.status()
        assert status["state"] == "completed" and status["source"]["health"]["state"] == "closed"

    def test_preview_frames_and_heatmap(self, env: dict[str, Any]) -> None:
        scene = SyntheticScene(two_object_scene(num_frames=30))
        pipe = make_pipeline(
            env, SyntheticSource("cam", scene.spec), GroundTruthDetector(scene), {"realtime": False}
        )
        assert pipe.latest_jpeg() is None and pipe.heatmap_png() is None
        source = pipe.source
        source.open()
        for frame in source:
            pipe.process_frame(frame)
        index, jpeg = pipe.latest_jpeg()  # type: ignore[misc]
        assert index == 29 and jpeg[:2] == b"\xff\xd8"
        assert pipe.latest_jpeg()[1] is jpeg  # type: ignore[index]  # cached
        assert pipe.heatmap_png()[:4] == b"\x89PNG"  # type: ignore[index]
        assert pipe.wait_for_frame(10, 0.01) and not pipe.wait_for_frame(29, 0.01)

    def test_stop_looping_source(self, env: dict[str, Any]) -> None:
        scene = SyntheticScene(two_object_scene(num_frames=20))
        cfg = {"realtime": True}
        pipe = make_pipeline(
            env, SyntheticSource("cam", scene.spec, loop=True), GroundTruthDetector(scene), cfg
        )
        pipe.start()
        assert wait_until(lambda: pipe.frames_processed > 3, 10)
        assert pipe.state == PipelineState.RUNNING
        assert pipe.stop(5)
        assert pipe.state == PipelineState.STOPPED
        assert env["repos"].runs.list("p1")[0]["status"] == "stopped"
        with pytest.raises(RuntimeError):
            pipe.start()

    def test_frame_skip_resize_and_max_frames(self, env: dict[str, Any]) -> None:
        scene = SyntheticScene(two_object_scene(num_frames=50))
        cfg = {"realtime": False, "process_every_n": 2, "resize_width": 160, "max_frames": 10}
        seen: list[tuple[int, int]] = []

        class Spy(Detector):
            def detect(self, frame: Frame) -> list[Detection]:
                seen.append((frame.index, frame.width))
                return []

        pipe = make_pipeline(env, SyntheticSource("cam", scene.spec), Spy(), cfg)
        pipe.start()
        pipe.join(10)
        assert pipe.frames_processed == 10 and pipe.frames_read == 19
        assert [i for i, _ in seen] == list(range(0, 19, 2))
        assert {w for _, w in seen} == {160}

    def test_source_failure_marks_failed_and_emits_event(self, env: dict[str, Any]) -> None:
        factory = ScriptedFactory(
            [lambda: FakeCapture(frames=3)], default=lambda: FakeCapture(opened=False)
        )
        policy = ReconnectPolicy(initial_delay=0.0, max_delay=0.0, jitter=0.0, max_attempts=2)
        source = RtspSource(
            "cam", "rtsp://cam.local/s", capture_factory=factory, policy=policy, frame_timeout=1.0
        )
        pipe = make_pipeline(
            env, source, BackgroundSubtractionDetector(backend="numpy"), {"realtime": False}
        )
        pipe.start()
        pipe.join(10)
        assert pipe.state == PipelineState.FAILED and "connection attempts" in (pipe.error or "")
        assert pipe.frames_processed == 3
        env["bus"].flush()
        events = env["repos"].events.list(event_type="source_down")
        assert len(events) == 1 and events[0]["severity"] == "critical"

    def test_detector_crash_is_contained(self, env: dict[str, Any]) -> None:
        class Boom(Detector):
            def detect(self, frame: Frame) -> list[Detection]:
                raise ValueError("kaboom")

        scene = SyntheticScene(two_object_scene(num_frames=5))
        pipe = make_pipeline(env, SyntheticSource("cam", scene.spec), Boom(), {"realtime": False})
        pipe.start()
        pipe.join(10)
        assert pipe.state == PipelineState.FAILED and pipe.error == "ValueError: kaboom"
        assert env["metrics"].pipeline_errors.value(pipeline="p1") == 1


class TestPipelineConfig:
    @pytest.mark.parametrize(
        ("patch", "message"),
        [
            (
                {"rules": [{"kind": "zone_intrusion", "rule_id": "r", "zone_id": "nope"}]},
                "unknown zone",
            ),
            (
                {"rules": [{"kind": "line_crossing", "rule_id": "r", "line_id": "nope"}]},
                "unknown line",
            ),
            (
                {
                    "rules": [
                        {"kind": "crowding", "rule_id": "r", "zone_id": "nope", "min_count": 2}
                    ]
                },
                "unknown zone",
            ),
            ({"zones": [CONFIG["zones"][0], CONFIG["zones"][0]]}, "duplicate zone"),
            ({"process_every_n": 0}, "greater than or equal"),
            ({"unknown": 1}, "Extra inputs"),
        ],
    )
    def test_validation(self, patch: dict[str, Any], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            PipelineConfig.model_validate(patch)

    def test_defaults(self) -> None:
        cfg = PipelineConfig()
        assert cfg.detector.kind == "background" and cfg.tracker.kind == "sort"
        assert EventType.CROWDING.value == "crowding"
