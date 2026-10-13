"""Per-camera processing pipeline running on its own thread.

Each frame flows through: source -> (frame skip / resize) -> detector ->
tracker -> analytics engine -> rules -> events (with annotated snapshots).
The pipeline keeps the latest annotated frame for MJPEG previews, publishes
periodic metric samples, persists finished track summaries, and records the
run in SQLite.
"""

from __future__ import annotations

import enum
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vap.analytics.engine import AnalyticsEngine
from vap.analytics.rules import RuleContext, RuleEngine
from vap.detection.base import Detector
from vap.errors import SourceError
from vap.events.bus import EventBus
from vap.events.model import Event, EventType, Severity
from vap.imaging.backend import get_backend
from vap.imaging.codec import encode_image, save_image
from vap.imaging.frame import Frame
from vap.motion.flow import block_matching_flow
from vap.motion.heatmap import MotionHeatmap
from vap.motion.kinematics import compass_direction, estimate_kinematics, path_length
from vap.observability.logs import get_logger
from vap.observability.metrics import PlatformMetrics
from vap.pipeline.annotate import annotate
from vap.pipeline.config import PipelineConfig
from vap.sources.base import FrameSource
from vap.sources.rtsp import RtspSource
from vap.storage.repositories import Repositories
from vap.tracking.base import Track, Tracker

log = get_logger("pipeline")


class PipelineState(enum.StrEnum):
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"


TERMINAL_STATES = {PipelineState.STOPPED, PipelineState.COMPLETED, PipelineState.FAILED}


@dataclass(slots=True)
class PipelineOptions:
    snapshot_dir: Path | None = None
    snapshot_quality: int = 85
    preview_quality: int = 70
    metrics_interval_s: float = 5.0
    backend: str = "auto"


class _FpsMeter:
    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self.value = 0.0
        self._last: float | None = None

    def tick(self, now: float) -> None:
        if self._last is not None:
            dt = now - self._last
            if dt > 0:
                inst = 1.0 / dt
                self.value = (
                    inst if self.value == 0 else self.value + self.alpha * (inst - self.value)
                )
        self._last = now


class CameraPipeline:
    def __init__(
        self,
        pipeline_id: str,
        camera_id: str,
        source: FrameSource,
        detector: Detector,
        tracker: Tracker,
        config: PipelineConfig,
        *,
        bus: EventBus | None = None,
        repos: Repositories | None = None,
        metrics: PlatformMetrics | None = None,
        options: PipelineOptions | None = None,
    ) -> None:
        self.pipeline_id = pipeline_id
        self.camera_id = camera_id
        self.source = source
        self.detector = detector
        self.tracker = tracker
        self.config = config
        self.bus = bus
        self.repos = repos
        self.metrics = metrics
        self.options = options or PipelineOptions()
        self.ops = get_backend(self.options.backend)
        self.engine = AnalyticsEngine(
            config.zones,
            config.lines,
            window_s=config.counter_window_s,
            pixels_per_meter=config.pixels_per_meter,
        )
        self.rules = RuleEngine(list(config.rules))
        self.heatmap: MotionHeatmap | None = None
        self.run_id = uuid.uuid4().hex
        self.state = PipelineState.CREATED
        self.error: str | None = None
        self.frames_read = 0
        self.frames_processed = 0
        self.detections_total = 0
        self.events_total = 0
        self.started_at: float | None = None
        self.stopped_at: float | None = None
        self.last_flow: dict[str, Any] | None = None
        self._fps = _FpsMeter()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame_cond = threading.Condition()
        self._latest_frame: Frame | None = None
        self._latest_tracks: list[Track] = []
        self._latest_jpeg: tuple[int, bytes] | None = None
        self._prev_gray: np.ndarray | None = None
        self._last_metrics_at = 0.0
        self._lock = threading.Lock()

    # -- lifecycle ------------------------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self.state not in (PipelineState.CREATED,):
                raise RuntimeError(
                    f"pipeline {self.pipeline_id} cannot start from state {self.state}"
                )
            self.state = PipelineState.STARTING
        self.started_at = time.time()
        if self.repos:
            self.repos.runs.start(self.run_id, self.pipeline_id)
        self._thread = threading.Thread(
            target=self._run, name=f"pipeline-{self.pipeline_id}", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> bool:
        """Request stop and wait; returns True when the worker exited in time."""
        self._stop.set()
        with self._lock:
            if self.state in (PipelineState.STARTING, PipelineState.RUNNING):
                self.state = PipelineState.STOPPING
        if isinstance(self.source, RtspSource):
            self.source.stop()
        with self._frame_cond:
            self._frame_cond.notify_all()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout)
            return not self._thread.is_alive()
        return True

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- worker ---------------------------------------------------------------------------

    def _run(self) -> None:
        final_state = PipelineState.COMPLETED
        try:
            self.source.open()
            with self._lock:
                if self.state == PipelineState.STARTING:
                    self.state = PipelineState.RUNNING
            self._loop()
            if self._stop.is_set():
                final_state = PipelineState.STOPPED
        except SourceError as exc:
            final_state = PipelineState.STOPPED if self._stop.is_set() else PipelineState.FAILED
            if final_state == PipelineState.FAILED:
                self.error = exc.message
                self._emit_source_down(exc.message)
                log.error(
                    "pipeline source failed",
                    extra={"pipeline": self.pipeline_id, "error": exc.message},
                )
        except Exception as exc:
            final_state = PipelineState.FAILED
            self.error = f"{type(exc).__name__}: {exc}"
            if self.metrics:
                self.metrics.pipeline_errors.inc(pipeline=self.pipeline_id)
            log.exception("pipeline crashed", extra={"pipeline": self.pipeline_id})
        finally:
            self._finalize(final_state)

    def _loop(self) -> None:
        cfg = self.config
        position = 0
        wall_start: float | None = None
        stream_start: float | None = None
        live = self.source.info().live
        while not self._stop.is_set():
            frame = self.source.read()
            if frame is None:
                return
            self.frames_read += 1
            position += 1
            if (position - 1) % cfg.process_every_n == 0:
                self.process_frame(frame)
            if cfg.max_frames is not None and self.frames_processed >= cfg.max_frames:
                return
            if cfg.realtime and not live:
                now = time.monotonic()
                if wall_start is None or stream_start is None:
                    wall_start, stream_start = now, frame.timestamp
                delay = (frame.timestamp - stream_start) - (now - wall_start)
                if delay > 0 and self._stop.wait(min(delay, 1.0)):
                    return

    def _prepare(self, frame: Frame) -> Frame:
        width = self.config.resize_width
        if width and frame.width != width:
            height = max(1, round(frame.height * width / frame.width))
            frame = frame.with_image(self.ops.resize(frame.image, (width, height), "area"))
        return frame

    def process_frame(self, frame: Frame) -> list[Event]:
        """Run one frame through the analytics chain (also used directly by tests/CLI)."""
        t0 = time.perf_counter()
        frame = self._prepare(frame)
        self.engine.configure(frame.width, frame.height)
        if self.config.heatmap and self.heatmap is None:
            self.heatmap = MotionHeatmap(frame.height, frame.width, decay=self.config.heatmap_decay)
        detections = self.detector.detect(frame)
        tracks = self.tracker.update(detections, frame.index, frame.timestamp)
        finished = self.tracker.pop_finished()
        observations = self.engine.update(tracks, frame.index, frame.timestamp, finished)
        ctx = RuleContext(
            self.pipeline_id,
            self.camera_id,
            frame.index,
            frame.timestamp,
            observations,
            self.engine,
        )
        events = self.rules.evaluate(ctx)
        if self.heatmap is not None:
            self.heatmap.add_points([t.anchor for t in tracks])
        if self.config.flow_every_n and self.frames_processed % self.config.flow_every_n == 0:
            self._update_flow(frame)
        if events:
            self._publish(frame, tracks, events)
        if finished and self.config.persist_tracks:
            self._persist_tracks(finished)

        self.frames_processed += 1
        self.detections_total += len(detections)
        self._fps.tick(time.monotonic())
        with self._frame_cond:
            self._latest_frame = frame
            self._latest_tracks = list(tracks)
            self._frame_cond.notify_all()
        if self.metrics:
            self.metrics.frames.inc(pipeline=self.pipeline_id)
            self.metrics.detections.inc(len(detections), pipeline=self.pipeline_id)
            self.metrics.active_tracks.set(len(tracks), pipeline=self.pipeline_id)
            self.metrics.pipeline_fps.set(round(self._fps.value, 3), pipeline=self.pipeline_id)
            self.metrics.frame_latency.observe(time.perf_counter() - t0, pipeline=self.pipeline_id)
        self._maybe_sample_metrics()
        return events

    def _update_flow(self, frame: Frame) -> None:
        gray = self.ops.to_gray(frame.image)
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            self.last_flow = (
                block_matching_flow(self._prev_gray, gray, block=16, search=4).summary().to_dict()
            )
        self._prev_gray = gray

    def _publish(self, frame: Frame, tracks: list[Track], events: list[Event]) -> None:
        snapshot_name: str | None = None
        if self.config.snapshots and self.options.snapshot_dir is not None:
            highlight = {tid for ev in events for tid in ev.track_ids}
            image = annotate(
                frame.image,
                tracks,
                list(self.engine.zones.values()),
                list(self.engine.lines.values()),
                highlight=highlight,
            )
            snapshot_name = f"{self.pipeline_id}_{frame.index:08d}_{events[0].event_id[:8]}.jpg"
            try:
                save_image(
                    self.options.snapshot_dir / snapshot_name, image, self.options.snapshot_quality
                )
            except OSError:
                log.exception("snapshot write failed", extra={"pipeline": self.pipeline_id})
                snapshot_name = None
        for ev in events:
            ev.snapshot = snapshot_name
            self.events_total += 1
            if self.metrics:
                self.metrics.events.inc(pipeline=self.pipeline_id, type=ev.type.value)
            if self.bus:
                self.bus.publish(ev)

    def _emit_source_down(self, message: str) -> None:
        event = Event(
            type=EventType.SOURCE_DOWN,
            pipeline_id=self.pipeline_id,
            camera_id=self.camera_id,
            frame_index=self.frames_read,
            stream_time=self.engine.last_ts,
            message=f"source failed: {message}",
            severity=Severity.CRITICAL,
            rule_id="system",
        )
        self.events_total += 1
        if self.metrics:
            self.metrics.events.inc(pipeline=self.pipeline_id, type=event.type.value)
        if self.bus:
            self.bus.publish(event)

    def _persist_tracks(self, finished: list[Track]) -> None:
        if not self.repos:
            return
        rows = []
        for t in finished:
            if not t.ever_confirmed:
                continue
            kin = estimate_kinematics(t.history, pixels_per_meter=self.config.pixels_per_meter)
            dur = max(t.duration, 1e-9)
            rows.append(
                {
                    "run_id": self.run_id,
                    "pipeline_id": self.pipeline_id,
                    "track_id": t.track_id,
                    "label": t.label,
                    "first_frame": t.first_frame,
                    "last_frame": t.last_frame,
                    "first_ts": t.first_timestamp,
                    "last_ts": t.last_timestamp,
                    "hits": t.hits,
                    "path_length": round(path_length(t.history), 3),
                    "mean_speed": round(path_length(t.history) / dur, 3) if t.duration > 0 else 0.0,
                    "direction": compass_direction(kin.heading_deg)
                    if kin.heading_deg is not None
                    else None,
                    "last_bbox": [round(v, 2) for v in t.bbox.to_list()],
                }
            )
        try:
            self.repos.tracks.insert_many(rows)
        except Exception:
            log.exception("persisting tracks failed", extra={"pipeline": self.pipeline_id})

    def _maybe_sample_metrics(self, force: bool = False) -> None:
        if not self.repos:
            return
        now = time.time()
        if not force and now - self._last_metrics_at < self.options.metrics_interval_s:
            return
        self._last_metrics_at = now
        try:
            self.repos.metrics.insert(self.pipeline_id, now, self.metrics_sample())
        except Exception:
            log.exception("metric sample failed", extra={"pipeline": self.pipeline_id})

    def _finalize(self, final_state: PipelineState) -> None:
        if final_state != PipelineState.FAILED and self.config.persist_tracks:
            finished = self.tracker.flush()
            if self.engine.frame_size is not None:
                self.engine.update([], self.frames_read, self.engine.last_ts, finished)
            self._persist_tracks(finished)
        try:
            self.source.close()
        except Exception:
            log.exception("closing source failed", extra={"pipeline": self.pipeline_id})
        self._maybe_sample_metrics(force=True)
        with self._lock:
            self.state = final_state
        self.stopped_at = time.time()
        if self.repos:
            try:
                self.repos.runs.finish(
                    self.run_id, self.frames_processed, final_state.value, self.error
                )
            except Exception:
                log.exception("recording run failed", extra={"pipeline": self.pipeline_id})
        with self._frame_cond:
            self._frame_cond.notify_all()
        log.info(
            "pipeline finished",
            extra={
                "pipeline": self.pipeline_id,
                "state": final_state.value,
                "frames": self.frames_processed,
            },
        )

    # -- read side (API) ----------------------------------------------------------------------

    def metrics_sample(self) -> dict[str, Any]:
        snap = self.engine.snapshot()
        return {
            "frames_processed": self.frames_processed,
            "detections": self.detections_total,
            "events": self.events_total,
            "fps": round(self._fps.value, 3),
            "live_tracks": snap["live_tracks"],
            "unique_tracks": snap["unique_tracks"],
            "zones": {z["zone_id"]: z["occupancy"] for z in snap["zones"]},
            "lines": {ln["line_id"]: ln["total"] for ln in snap["lines"]},
        }

    def status(self) -> dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "camera_id": self.camera_id,
            "run_id": self.run_id,
            "state": self.state.value,
            "error": self.error,
            "frames_read": self.frames_read,
            "frames_processed": self.frames_processed,
            "detections": self.detections_total,
            "events": self.events_total,
            "fps": round(self._fps.value, 3),
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "source": {**self.source.info().to_dict(), "health": self.source.health().to_dict()},
            "detector": self.detector.describe(),
            "tracker": self.tracker.name,
        }

    def analytics(self) -> dict[str, Any]:
        with self._frame_cond:
            tracks = list(self._latest_tracks)
        out = self.engine.snapshot()
        out["tracks"] = []
        for t in tracks:
            kin = estimate_kinematics(t.history, pixels_per_meter=self.config.pixels_per_meter)
            out["tracks"].append({**t.to_dict(), "kinematics": kin.to_dict()})
        out["flow"] = self.last_flow
        out["heatmap_hotspots"] = self.heatmap.hotspots(4)[:4] if self.heatmap is not None else []
        return out

    def latest_jpeg(self) -> tuple[int, bytes] | None:
        with self._frame_cond:
            frame, tracks = self._latest_frame, list(self._latest_tracks)
        if frame is None:
            return None
        cached = self._latest_jpeg
        if cached is not None and cached[0] == frame.index:
            return cached
        image = annotate(
            frame.image,
            tracks,
            list(self.engine.zones.values()),
            list(self.engine.lines.values()),
            heatmap=self.heatmap,
        )
        result = (frame.index, encode_image(image, "jpeg", self.options.preview_quality))
        self._latest_jpeg = result
        return result

    def heatmap_png(self) -> bytes | None:
        if self.heatmap is None:
            return None
        return encode_image(self.heatmap.render(), "png")

    def wait_for_frame(self, after_index: int, timeout: float) -> bool:
        """Block until a frame newer than ``after_index`` exists or the pipeline ends."""
        deadline = time.monotonic() + timeout
        with self._frame_cond:
            while True:
                if self._latest_frame is not None and self._latest_frame.index > after_index:
                    return True
                if self.state in TERMINAL_STATES:
                    return False
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._frame_cond.wait(remaining)
