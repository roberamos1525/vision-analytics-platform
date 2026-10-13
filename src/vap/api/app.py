"""FastAPI application factory with consistent error envelopes."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from vap import __version__
from vap.api.middleware import RequestContextMiddleware
from vap.api.routes import analysis, cameras, events, models, pipelines, system
from vap.config import Settings
from vap.errors import VapError, validation_details
from vap.observability.logs import get_logger
from vap.runtime import Runtime

log = get_logger("api")

_HTTP_CODES = {
    400: "bad_request",
    401: "unauthenticated",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    415: "unsupported_media_type",
    422: "invalid_input",
    429: "limit_exceeded",
}


def _envelope(
    request: Request, status: int, code: str, message: str, details: Any = None
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None) or request.scope.get("state", {}).get(
        "request_id"
    )
    headers = {"WWW-Authenticate": "ApiKey"} if status == 401 else None
    return JSONResponse(
        {
            "error": {"code": code, "message": message, "details": details or {}},
            "request_id": request_id,
        },
        status_code=status,
        headers=headers,
    )


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(VapError)
    async def vap_error(request: Request, exc: VapError) -> JSONResponse:
        if exc.status_code >= 500:
            log.error("request failed", extra={"code": exc.code, "error": exc.message})
        return _envelope(request, exc.status_code, exc.code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _envelope(
            request,
            422,
            "invalid_input",
            "request validation failed",
            {"errors": validation_details(list(exc.errors()))},
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _HTTP_CODES.get(exc.status_code, "http_error")
        return _envelope(request, exc.status_code, code, str(exc.detail))

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error", extra={"path": request.url.path})
        return _envelope(request, 500, "internal_error", "an unexpected error occurred")


def create_app(
    settings: Settings | None = None, *, runtime: Runtime | None = None, restore: bool = True
) -> FastAPI:
    """Build the API. Pass ``runtime`` to inject fakes (tests) or share state."""
    if runtime is None:
        runtime = Runtime(settings or Settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        runtime.start(restore=restore)
        try:
            yield
        finally:
            runtime.close()

    app = FastAPI(
        title="Vision Analytics Platform",
        version=__version__,
        description="Local computer-vision analytics: cameras, pipelines, events, models and evaluation.",
        lifespan=lifespan,
    )
    app.state.runtime = runtime
    install_error_handlers(app)
    for module in (system, cameras, pipelines, events, models, analysis):
        app.include_router(module.router)
    app.add_middleware(
        RequestContextMiddleware,
        max_body=runtime.settings.server.max_request_bytes,
        max_upload=runtime.settings.server.max_upload_bytes,
        upload_prefixes=("/v1/models/",),
        metrics=runtime.metrics,
    )
    return app


def app_from_env() -> FastAPI:
    """Entry point for ``uvicorn --factory vap.api.app:app_from_env``."""
    from vap.config import load_settings
    from vap.observability.logs import configure_logging

    settings = load_settings()
    configure_logging(settings.logging.level, settings.logging.format)
    return create_app(settings)
