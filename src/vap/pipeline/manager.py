"""Camera and pipeline management: CRUD, start/stop and restore on restart."""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from vap.config import Settings
from vap.detection.specs import build_detector
from vap.errors import (
    ConflictError,
    InvalidInputError,
    LimitExceededError,
    SourceError,
    SourceTimeoutError,
    validation_details,
)
from vap.events.bus import EventBus
from vap.extraction import ExtractionSpec, extract_frames
from vap.observability.logs import get_logger
from vap.observability.metrics import PlatformMetrics
from vap.pipeline.config import PipelineConfig
from vap.pipeline.pipeline import TERMINAL_STATES, CameraPipeline, PipelineOptions
from vap.registry.models import ModelRegistry
from vap.sources.base import FrameSource
from vap.sources.specs import SourceSpec, build_source, public_spec
from vap.sources.video_file import CaptureFactory
from vap.storage.repositories import CameraRecord, PipelineRecord, Repositories
from vap.tracking.trackers import build_tracker

log = get_logger("pipeline.manager")

_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}$")
_SOURCE_ADAPTER: TypeAdapter[Any] = TypeAdapter(SourceSpec)


def validate_id(value: str, kind: str) -> str:
    if not _ID_RE.match(value):
        raise InvalidInputError(f"{kind} id must match {_ID_RE.pattern}", details={"id": value})
    return value


def parse_source_spec(data: dict[str, Any]) -> Any:
    try:
        return _SOURCE_ADAPTER.validate_python(data)
    except ValidationError as exc:
        raise InvalidInputError(
            "invalid source spec", details={"errors": validation_details(exc.errors())}
        ) from exc


def parse_pipeline_config(data: dict[str, Any]) -> PipelineConfig:
    try:
        return PipelineConfig.model_validate(data)
    except ValidationError as exc:
        raise InvalidInputError(
            "invalid pipeline config", details={"errors": validation_details(exc.errors())}
        ) from exc


class PipelineManager:
    def __init__(
        self,
        settings: Settings,
        repos: Repositories,
        bus: EventBus,
        metrics: PlatformMetrics,
        registry: ModelRegistry,
        *,
        capture_factory: CaptureFactory | None = None,
    ) -> None:
        self.settings = settings
        self.repos = repos
        self.bus = bus
        self.metrics = metrics
        self.registry = registry
        self.capture_factory = capture_factory
        self._running: dict[str, CameraPipeline] = {}
        self._last: dict[str, CameraPipeline] = {}
        self._lock = threading.RLock()

    # -- cameras -------------------------------------------------------------------------

    def create_camera(self, camera_id: str, name: str, spec: dict[str, Any]) -> CameraRecord:
        validate_id(camera_id, "camera")
        parsed = parse_source_spec(spec)
        return self.repos.cameras.create(
            camera_id, name or camera_id, parsed.model_dump(mode="json")
        )

    def update_camera(self, camera_id: str, name: str, spec: dict[str, Any]) -> CameraRecord:
        parsed = parse_source_spec(spec)
        with self._lock:
            if any(p.camera_id == camera_id for p in self._running.values()):
                raise ConflictError(f"camera {camera_id!r} is in use by a running pipeline")
            return self.repos.cameras.update(
                camera_id, name or camera_id, parsed.model_dump(mode="json")
            )

    def delete_camera(self, camera_id: str) -> None:
        self.repos.cameras.delete(camera_id)

    def camera_view(self, record: CameraRecord) -> dict[str, Any]:
        spec = parse_source_spec(record.spec)
        return {
            "camera_id": record.camera_id,
            "name": record.name,
            "source": public_spec(spec),
            "created_at": record.created_at,
            "updated_at": record.updated_at,
        }

    def _media_roots(self) -> list[Path]:
        return list(self.settings.media.roots)

    def build_source(self, camera_id: str) -> FrameSource:
        record = self.repos.cameras.get(camera_id)
        spec = parse_source_spec(record.spec)
        return build_source(
            camera_id, spec, media_roots=self._media_roots(), capture_factory=self.capture_factory
        )

    def probe_camera(self, camera_id: str, timeout_s: float = 10.0) -> dict[str, Any]:
        """Open the source, read one frame and report its properties and health."""
        source = self.build_source(camera_id)
        result: dict[str, Any] = {}
        failure: list[BaseException] = []

        def worker() -> None:
            try:
                source.open()
                frame = source.read()
                result["frame"] = (
                    None if frame is None else {"width": frame.width, "height": frame.height}
                )
            except BaseException as exc:
                failure.append(exc)

        started = time.monotonic()
        t = threading.Thread(target=worker, name=f"probe-{camera_id}", daemon=True)
        t.start()
        t.join(timeout_s)
        try:
            if t.is_alive():
                stop = getattr(source, "stop", None)
                if callable(stop):
                    stop()
                raise SourceTimeoutError(
                    f"probe of camera {camera_id!r} timed out after {timeout_s:.1f}s"
                )
            if failure:
                exc = failure[0]
                if isinstance(exc, SourceError | InvalidInputError):
                    raise exc
                raise SourceError(f"probe failed: {type(exc).__name__}: {exc}") from exc
            return {
                "camera_id": camera_id,
                "ok": result.get("frame") is not None,
                "latency_ms": round((time.monotonic() - started) * 1000, 1),
                "frame": result.get("frame"),
                "info": source.info().to_dict(),
                "health": source.health().to_dict(),
            }
        finally:
            source.close()

    def extract(self, camera_id: str, spec: ExtractionSpec, job_name: str) -> dict[str, Any]:
        validate_id(job_name, "extraction")
        out_dir = self.settings.storage.exports_dir / "extractions" / camera_id / job_name
        source = self.build_source(camera_id)
        with source:
            return extract_frames(
                source, spec, out_dir, backend=self.settings.pipelines.image_backend
            )

    # -- pipelines -----------------------------------------------------------------------

    def create_pipeline(
        self, pipeline_id: str, camera_id: str, config: dict[str, Any]
    ) -> PipelineRecord:
        validate_id(pipeline_id, "pipeline")
        parsed = parse_pipeline_config(config)
        self.repos.cameras.get(camera_id)
        return self.repos.pipelines.create(pipeline_id, camera_id, parsed.model_dump(mode="json"))

    def update_pipeline(self, pipeline_id: str, config: dict[str, Any]) -> PipelineRecord:
        parsed = parse_pipeline_config(config)
        with self._lock:
            if pipeline_id in self._running:
                raise ConflictError(
                    f"pipeline {pipeline_id!r} is running; stop it before reconfiguring"
                )
            return self.repos.pipelines.update_config(pipeline_id, parsed.model_dump(mode="json"))

    def delete_pipeline(self, pipeline_id: str) -> None:
        with self._lock:
            if pipeline_id in self._running:
                raise ConflictError(f"pipeline {pipeline_id!r} is running; stop it first")
            self.repos.pipelines.delete(pipeline_id)
            self._last.pop(pipeline_id, None)

    def _reap(self) -> None:
        """Move pipelines whose worker finished out of the running set."""
        with self._lock:
            for pid, pipe in list(self._running.items()):
                if pipe.state in TERMINAL_STATES and not pipe.alive:
                    self._running.pop(pid)
                    self._last[pid] = pipe
            self.metrics.pipelines_running.set(len(self._running))

    def start(self, pipeline_id: str) -> dict[str, Any]:
        self._reap()
        with self._lock:
            if pipeline_id in self._running:
                raise ConflictError(f"pipeline {pipeline_id!r} is already running")
            if len(self._running) >= self.settings.pipelines.max_pipelines:
                raise LimitExceededError(
                    "maximum number of concurrent pipelines reached",
                    details={"max_pipelines": self.settings.pipelines.max_pipelines},
                )
            record = self.repos.pipelines.get(pipeline_id)
            config = parse_pipeline_config(record.config)
            source = self.build_source(record.camera_id)
            backend = self.settings.pipelines.image_backend
            detector = build_detector(
                config.detector,
                backend=backend,
                media_roots=self._media_roots(),
                model_resolver=self.registry.resolve,
            )
            pipe = CameraPipeline(
                pipeline_id,
                record.camera_id,
                source,
                detector,
                build_tracker(config.tracker),
                config,
                bus=self.bus,
                repos=self.repos,
                metrics=self.metrics,
                options=PipelineOptions(
                    snapshot_dir=self.settings.storage.snapshots_dir,
                    snapshot_quality=self.settings.pipelines.snapshot_quality,
                    preview_quality=self.settings.pipelines.preview_quality,
                    metrics_interval_s=self.settings.pipelines.metrics_interval_s,
                    backend=backend,
                ),
            )
            pipe.start()
            self._running[pipeline_id] = pipe
            self.repos.pipelines.set_desired_state(pipeline_id, "running")
            self.metrics.pipelines_running.set(len(self._running))
        log.info("pipeline started", extra={"pipeline": pipeline_id, "run_id": pipe.run_id})
        return pipe.status()

    def stop(self, pipeline_id: str, *, persist_state: bool = True) -> dict[str, Any]:
        with self._lock:
            pipe = self._running.get(pipeline_id)
            if pipe is None:
                self.repos.pipelines.get(pipeline_id)
                raise ConflictError(f"pipeline {pipeline_id!r} is not running")
        stopped = pipe.stop(self.settings.pipelines.stop_timeout_s)
        if not stopped:
            log.warning("pipeline did not stop in time", extra={"pipeline": pipeline_id})
        if persist_state:
            self.repos.pipelines.set_desired_state(pipeline_id, "stopped")
        self._reap()
        return pipe.status()

    def get_pipeline(self, pipeline_id: str) -> CameraPipeline | None:
        """Running pipeline or the most recent finished run, if any."""
        self._reap()
        with self._lock:
            return self._running.get(pipeline_id) or self._last.get(pipeline_id)

    def require_pipeline(self, pipeline_id: str) -> CameraPipeline:
        pipe = self.get_pipeline(pipeline_id)
        if pipe is None:
            self.repos.pipelines.get(pipeline_id)
            raise ConflictError(f"pipeline {pipeline_id!r} has not been started")
        return pipe

    def pipeline_view(self, record: PipelineRecord) -> dict[str, Any]:
        pipe = self.get_pipeline(record.pipeline_id)
        return {
            "pipeline_id": record.pipeline_id,
            "camera_id": record.camera_id,
            "desired_state": record.desired_state,
            "config": record.config,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "status": pipe.status() if pipe else {"state": "stopped"},
        }

    def running_ids(self) -> list[str]:
        self._reap()
        with self._lock:
            return sorted(self._running)

    def restore(self) -> list[str]:
        """Start pipelines whose desired state is ``running`` (after a restart)."""
        started: list[str] = []
        for record in self.repos.pipelines.list():
            if record.desired_state != "running":
                continue
            try:
                self.start(record.pipeline_id)
                started.append(record.pipeline_id)
            except Exception as exc:
                log.error(
                    "failed to restore pipeline",
                    extra={"pipeline": record.pipeline_id, "error": str(exc)},
                )
        return started

    def shutdown(self) -> None:
        with self._lock:
            pipes = list(self._running.items())
        for pid, pipe in pipes:
            pipe.stop(self.settings.pipelines.stop_timeout_s)
            log.info("pipeline stopped for shutdown", extra={"pipeline": pid})
        self._reap()
