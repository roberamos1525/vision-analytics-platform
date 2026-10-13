"""Camera (video source) management endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Response

from vap.api.deps import Operator, RuntimeDep, Viewer
from vap.api.schemas import CameraCreate, CameraUpdate, ExtractionRequest

router = APIRouter(prefix="/v1/cameras", tags=["cameras"])


@router.post("", status_code=201)
def create_camera(body: CameraCreate, runtime: RuntimeDep, _: Operator) -> dict[str, Any]:
    record = runtime.manager.create_camera(body.camera_id, body.name, body.source)
    return runtime.manager.camera_view(record)


@router.get("")
def list_cameras(runtime: RuntimeDep, _: Viewer) -> dict[str, Any]:
    items = [runtime.manager.camera_view(r) for r in runtime.repos.cameras.list()]
    return {"items": items, "count": len(items)}


@router.get("/{camera_id}")
def get_camera(camera_id: str, runtime: RuntimeDep, _: Viewer) -> dict[str, Any]:
    return runtime.manager.camera_view(runtime.repos.cameras.get(camera_id))


@router.put("/{camera_id}")
def update_camera(
    camera_id: str, body: CameraUpdate, runtime: RuntimeDep, _: Operator
) -> dict[str, Any]:
    record = runtime.manager.update_camera(camera_id, body.name, body.source)
    return runtime.manager.camera_view(record)


@router.delete("/{camera_id}", status_code=204)
def delete_camera(camera_id: str, runtime: RuntimeDep, _: Operator) -> Response:
    runtime.manager.delete_camera(camera_id)
    return Response(status_code=204)


@router.post("/{camera_id}/probe")
def probe_camera(
    camera_id: str,
    runtime: RuntimeDep,
    _: Operator,
    timeout_s: float = Query(10.0, gt=0, le=60),
) -> dict[str, Any]:
    return runtime.manager.probe_camera(camera_id, timeout_s)


@router.post("/{camera_id}/extract", status_code=201)
def extract_frames(
    camera_id: str, body: ExtractionRequest, runtime: RuntimeDep, _: Operator
) -> dict[str, Any]:
    result = runtime.manager.extract(camera_id, body.spec, body.job)
    # Do not leak absolute server paths to clients.
    result["directory"] = f"extractions/{camera_id}/{body.job}"
    return result
