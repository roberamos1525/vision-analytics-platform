"""Model registry endpoints (artefacts are uploaded as raw request bodies)."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator
from typing import IO, Any

from fastapi import APIRouter, Header, Query, Request, Response
from starlette.concurrency import run_in_threadpool

from vap.api.deps import Admin, RuntimeDep, Viewer
from vap.api.schemas import ModelActivate
from vap.errors import InvalidInputError, NotFoundError, PayloadTooLargeError
from vap.registry.models import ModelRegistry

router = APIRouter(prefix="/v1/models", tags=["models"])


@router.get("")
def list_models(runtime: RuntimeDep, _: Viewer) -> dict[str, Any]:
    items = runtime.registry.names()
    return {"items": items, "count": len(items)}


@router.get("/{name}")
def get_model(name: str, runtime: RuntimeDep, _: Viewer) -> dict[str, Any]:
    ModelRegistry.validate_name(name)
    versions = [m.to_dict() for m in runtime.registry.list(name)]
    if not versions:
        raise NotFoundError(f"model {name!r} not found")
    active = next((v["version"] for v in versions if v["active"]), None)
    return {"name": name, "active": active, "versions": versions}


def _chunks(fh: IO[bytes]) -> Iterator[bytes]:
    fh.seek(0)
    while block := fh.read(1 << 20):
        yield block


@router.put("/{name}/versions/{version}", status_code=201)
async def upload_model(
    name: str,
    version: str,
    request: Request,
    runtime: RuntimeDep,
    _: Admin,
    format: str = Query("onnx", max_length=16),
    sha256: str | None = Query(None, min_length=64, max_length=64),
    activate: bool = Query(False),
    x_model_metadata: str | None = Header(None, max_length=16_384),
) -> dict[str, Any]:
    try:
        metadata = json.loads(x_model_metadata) if x_model_metadata else {}
    except json.JSONDecodeError as exc:
        raise InvalidInputError("X-Model-Metadata must be a JSON object") from exc
    if not isinstance(metadata, dict):
        raise InvalidInputError("X-Model-Metadata must be a JSON object")
    limit = runtime.settings.server.max_upload_bytes
    with tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024) as spool:
        size = 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > limit:
                raise PayloadTooLargeError("model artefact exceeds the upload limit")
            spool.write(chunk)
        model = await run_in_threadpool(
            runtime.registry.register,
            name,
            version,
            _chunks(spool),
            fmt=format,
            metadata=metadata,
            expected_sha256=sha256,
            activate=activate,
        )
    return model.to_dict()


@router.get("/{name}/versions/{version}")
def get_version(name: str, version: str, runtime: RuntimeDep, _: Viewer) -> dict[str, Any]:
    return runtime.registry.get(name, version).to_dict()


@router.post("/{name}/activate")
def activate(name: str, body: ModelActivate, runtime: RuntimeDep, _: Admin) -> dict[str, Any]:
    return runtime.registry.activate(name, body.version).to_dict()


@router.post("/{name}/deactivate", status_code=204)
def deactivate(name: str, runtime: RuntimeDep, _: Admin) -> Response:
    ModelRegistry.validate_name(name)
    runtime.registry.deactivate(name)
    return Response(status_code=204)


@router.post("/{name}/versions/{version}/verify")
def verify(name: str, version: str, runtime: RuntimeDep, _: Viewer) -> dict[str, Any]:
    model = runtime.registry.verify(name, version)
    return {"name": name, "version": version, "sha256": model.sha256, "valid": True}


@router.delete("/{name}/versions/{version}", status_code=204)
def delete_version(
    name: str, version: str, runtime: RuntimeDep, _: Admin, force: bool = Query(False)
) -> Response:
    runtime.registry.delete(name, version, force=force)
    return Response(status_code=204)
