"""Health, readiness, system info and Prometheus metrics."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse, PlainTextResponse

from vap import __version__
from vap.api.deps import RuntimeDep, Viewer
from vap.imaging.backend import available_backends

router = APIRouter()


@router.get("/healthz", tags=["system"])
def healthz() -> dict[str, str]:
    """Liveness probe (no authentication, no dependencies)."""
    return {"status": "ok"}


@router.get("/readyz", tags=["system"])
def readyz(runtime: RuntimeDep) -> JSONResponse:
    health = runtime.health()
    code = 200 if health["status"] == "ok" else 503
    return JSONResponse({"status": health["status"]}, status_code=code)


@router.get("/v1/health", tags=["system"])
def health(runtime: RuntimeDep, _: Viewer) -> dict[str, Any]:
    return runtime.health()


@router.get("/v1/system", tags=["system"])
def system_info(runtime: RuntimeDep, _: Viewer) -> dict[str, Any]:
    s = runtime.settings
    return {
        "version": __version__,
        "image_backends": available_backends(),
        "image_backend": s.pipelines.image_backend,
        "limits": {
            "max_pipelines": s.pipelines.max_pipelines,
            "max_request_bytes": s.server.max_request_bytes,
            "max_upload_bytes": s.server.max_upload_bytes,
            "max_image_pixels": s.server.max_image_pixels,
        },
        "auth_enabled": s.auth.enabled,
        "webhooks_configured": len(s.webhooks.urls),
        "media_roots": len(s.media.roots),
    }


@router.get("/metrics", tags=["system"], response_class=PlainTextResponse)
def prometheus(runtime: RuntimeDep, _: Viewer) -> PlainTextResponse:
    return PlainTextResponse(runtime.metrics.render(), media_type="text/plain; version=0.0.4")
