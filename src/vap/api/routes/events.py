"""Event queries, snapshots and data exports."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Query, Response
from fastapi.responses import StreamingResponse

from vap.api.deps import RuntimeDep, Viewer
from vap.errors import InvalidInputError, NotFoundError
from vap.events.model import EventType
from vap.exporting import render

router = APIRouter(tags=["events"])

EVENT_TYPES = [t.value for t in EventType]


def _check_type(event_type: str | None) -> None:
    if event_type is not None and event_type not in EVENT_TYPES:
        raise InvalidInputError("unknown event type", details={"allowed": EVENT_TYPES})


@router.get("/v1/events")
def list_events(
    runtime: RuntimeDep,
    _: Viewer,
    pipeline_id: str | None = Query(None, max_length=64),
    camera_id: str | None = Query(None, max_length=64),
    type: str | None = Query(None, max_length=32),
    since: float | None = Query(None, ge=0),
    until: float | None = Query(None, ge=0),
    before: float | None = Query(None, ge=0, description="pagination cursor (created_at)"),
    limit: int = Query(100, ge=1, le=1000),
) -> dict[str, Any]:
    _check_type(type)
    items = runtime.repos.events.list(
        pipeline_id=pipeline_id,
        camera_id=camera_id,
        event_type=type,
        since=since,
        until=until,
        before=before,
        limit=limit,
    )
    next_cursor = items[-1]["created_at"] if len(items) == limit else None
    return {"items": items, "count": len(items), "next_before": next_cursor}


@router.get("/v1/events/recent")
def recent_events(
    runtime: RuntimeDep,
    _: Viewer,
    pipeline_id: str | None = Query(None, max_length=64),
    limit: int = Query(50, ge=1, le=500),
) -> dict[str, Any]:
    items = runtime.recent.recent(limit, pipeline_id)
    return {"items": items, "count": len(items)}


@router.get("/v1/events/stats")
def event_stats(
    runtime: RuntimeDep, _: Viewer, pipeline_id: str | None = Query(None, max_length=64)
) -> dict[str, Any]:
    counts = runtime.repos.events.counts_by_type(pipeline_id)
    return {"by_type": counts, "total": sum(counts.values()), "bus": runtime.bus.stats()}


@router.get("/v1/events/{event_id}")
def get_event(event_id: str, runtime: RuntimeDep, _: Viewer) -> dict[str, Any]:
    return runtime.repos.events.get(event_id)


@router.get("/v1/events/{event_id}/snapshot")
def event_snapshot(event_id: str, runtime: RuntimeDep, _: Viewer) -> Response:
    event = runtime.repos.events.get(event_id)
    name = event.get("snapshot")
    if not name:
        raise NotFoundError("event has no snapshot")
    root = runtime.settings.storage.snapshots_dir.resolve()
    path = (root / name).resolve()
    if root not in path.parents or not path.is_file():
        raise NotFoundError("snapshot file not found")
    return Response(path.read_bytes(), media_type="image/jpeg")


@router.get("/v1/exports/{kind}")
def export_data(
    kind: Literal["events", "tracks", "metrics"],
    runtime: RuntimeDep,
    _: Viewer,
    format: Literal["csv", "jsonl"] = Query("csv"),
    pipeline_id: str | None = Query(None, max_length=64),
    since: float | None = Query(None, ge=0),
    limit: int = Query(1000, ge=1, le=1000),
) -> StreamingResponse:
    repos = runtime.repos
    rows: list[dict[str, Any]]
    if kind == "events":
        rows = repos.events.list(pipeline_id=pipeline_id, since=since, limit=limit, ascending=True)
    elif kind == "tracks":
        rows = repos.tracks.list(pipeline_id, None, limit)
    else:
        if pipeline_id is None:
            raise InvalidInputError("pipeline_id is required for metrics exports")
        rows = repos.metrics.list(pipeline_id, since, limit)
    media = "text/csv" if format == "csv" else "application/x-ndjson"
    filename = f"{kind}.{format}"
    return StreamingResponse(
        render(rows, kind, format),
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
