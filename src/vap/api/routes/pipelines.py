"""Pipeline management, live analytics, MJPEG preview and snapshots."""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, Query, Response
from fastapi.responses import StreamingResponse

from vap.api.deps import Operator, RuntimeDep, Viewer
from vap.api.schemas import PipelineCreate, PipelineUpdate
from vap.errors import ConflictError
from vap.pipeline.pipeline import TERMINAL_STATES, CameraPipeline

router = APIRouter(prefix="/v1/pipelines", tags=["pipelines"])

BOUNDARY = "vapframe"


@router.post("", status_code=201)
def create_pipeline(body: PipelineCreate, runtime: RuntimeDep, _: Operator) -> dict[str, Any]:
    record = runtime.manager.create_pipeline(body.pipeline_id, body.camera_id, body.config)
    if body.start:
        runtime.manager.start(body.pipeline_id)
        record = runtime.repos.pipelines.get(body.pipeline_id)
    return runtime.manager.pipeline_view(record)


@router.get("")
def list_pipelines(runtime: RuntimeDep, _: Viewer) -> dict[str, Any]:
    items = [runtime.manager.pipeline_view(r) for r in runtime.repos.pipelines.list()]
    return {"items": items, "count": len(items)}


@router.get("/{pipeline_id}")
def get_pipeline(pipeline_id: str, runtime: RuntimeDep, _: Viewer) -> dict[str, Any]:
    return runtime.manager.pipeline_view(runtime.repos.pipelines.get(pipeline_id))


@router.put("/{pipeline_id}")
def update_pipeline(
    pipeline_id: str, body: PipelineUpdate, runtime: RuntimeDep, _: Operator
) -> dict[str, Any]:
    return runtime.manager.pipeline_view(runtime.manager.update_pipeline(pipeline_id, body.config))


@router.delete("/{pipeline_id}", status_code=204)
def delete_pipeline(pipeline_id: str, runtime: RuntimeDep, _: Operator) -> Response:
    runtime.manager.delete_pipeline(pipeline_id)
    return Response(status_code=204)


@router.post("/{pipeline_id}/start")
def start_pipeline(pipeline_id: str, runtime: RuntimeDep, _: Operator) -> dict[str, Any]:
    return runtime.manager.start(pipeline_id)


@router.post("/{pipeline_id}/stop")
def stop_pipeline(pipeline_id: str, runtime: RuntimeDep, _: Operator) -> dict[str, Any]:
    return runtime.manager.stop(pipeline_id)


@router.get("/{pipeline_id}/status")
def pipeline_status(pipeline_id: str, runtime: RuntimeDep, _: Viewer) -> dict[str, Any]:
    return runtime.manager.require_pipeline(pipeline_id).status()


@router.get("/{pipeline_id}/analytics")
def pipeline_analytics(pipeline_id: str, runtime: RuntimeDep, _: Viewer) -> dict[str, Any]:
    pipe = runtime.manager.require_pipeline(pipeline_id)
    return {"pipeline_id": pipeline_id, "state": pipe.state.value, **pipe.analytics()}


@router.get("/{pipeline_id}/metrics")
def pipeline_metrics(pipeline_id: str, runtime: RuntimeDep, _: Viewer) -> dict[str, Any]:
    pipe = runtime.manager.require_pipeline(pipeline_id)
    return {"pipeline_id": pipeline_id, "state": pipe.state.value, **pipe.metrics_sample()}


@router.get("/{pipeline_id}/metrics/history")
def pipeline_metrics_history(
    pipeline_id: str,
    runtime: RuntimeDep,
    _: Viewer,
    since: float | None = Query(None, ge=0),
    limit: int = Query(100, ge=1, le=1000),
) -> dict[str, Any]:
    runtime.repos.pipelines.get(pipeline_id)
    items = runtime.repos.metrics.list(pipeline_id, since, limit)
    return {"pipeline_id": pipeline_id, "items": items, "count": len(items)}


@router.get("/{pipeline_id}/runs")
def pipeline_runs(
    pipeline_id: str, runtime: RuntimeDep, _: Viewer, limit: int = Query(20, ge=1, le=200)
) -> dict[str, Any]:
    runtime.repos.pipelines.get(pipeline_id)
    items = runtime.repos.runs.list(pipeline_id, limit)
    return {"items": items, "count": len(items)}


@router.get("/{pipeline_id}/tracks")
def pipeline_tracks(
    pipeline_id: str,
    runtime: RuntimeDep,
    _: Viewer,
    run_id: str | None = Query(None, max_length=64),
    limit: int = Query(100, ge=1, le=1000),
) -> dict[str, Any]:
    runtime.repos.pipelines.get(pipeline_id)
    items = runtime.repos.tracks.list(pipeline_id, run_id, limit)
    return {"items": items, "count": len(items)}


def _frame_or_conflict(pipe: CameraPipeline) -> tuple[int, bytes]:
    latest = pipe.latest_jpeg()
    if latest is None:
        raise ConflictError("no frame has been processed yet")
    return latest


@router.get("/{pipeline_id}/snapshot.jpg")
def pipeline_snapshot(pipeline_id: str, runtime: RuntimeDep, _: Viewer) -> Response:
    pipe = runtime.manager.require_pipeline(pipeline_id)
    index, data = _frame_or_conflict(pipe)
    return Response(data, media_type="image/jpeg", headers={"X-Frame-Index": str(index)})


@router.get("/{pipeline_id}/heatmap.png")
def pipeline_heatmap(pipeline_id: str, runtime: RuntimeDep, _: Viewer) -> Response:
    pipe = runtime.manager.require_pipeline(pipeline_id)
    data = pipe.heatmap_png()
    if data is None:
        raise ConflictError("heatmap is disabled or no frame has been processed yet")
    return Response(data, media_type="image/png")


def mjpeg_stream(
    pipe: CameraPipeline, max_fps: float, max_frames: int | None, idle_timeout: float
) -> Iterator[bytes]:
    """Multipart JPEG stream; ends when the pipeline stops or limits are reached."""
    last_index = -1
    sent = 0
    min_interval = 1.0 / max_fps
    last_sent = 0.0
    while max_frames is None or sent < max_frames:
        if not pipe.wait_for_frame(last_index, idle_timeout):
            if pipe.state in TERMINAL_STATES:
                return
            # Yield control so a disconnected client cancels the stream promptly.
            yield b""
            continue
        wait = min_interval - (time.monotonic() - last_sent)
        if wait > 0:
            time.sleep(wait)
        latest = pipe.latest_jpeg()
        if latest is None:
            continue
        last_index, data = latest
        last_sent = time.monotonic()
        sent += 1
        yield (
            (
                f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(data)}\r\n"
                f"X-Frame-Index: {last_index}\r\n\r\n"
            ).encode("ascii")
            + data
            + b"\r\n"
        )


@router.get("/{pipeline_id}/preview.mjpg")
def pipeline_preview(
    pipeline_id: str,
    runtime: RuntimeDep,
    _: Viewer,
    fps: float = Query(5.0, gt=0, le=30),
    max_frames: int | None = Query(None, ge=1, le=100_000),
) -> StreamingResponse:
    pipe = runtime.manager.require_pipeline(pipeline_id)
    if pipe.state in TERMINAL_STATES:
        raise ConflictError(f"pipeline {pipeline_id!r} is not running")
    return StreamingResponse(
        mjpeg_stream(pipe, fps, max_frames, idle_timeout=5.0),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
        headers={"Cache-Control": "no-store"},
    )
