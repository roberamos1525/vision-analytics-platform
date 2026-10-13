"""Security, middleware and helper units of the API layer (without a full runtime)."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import MutableMapping
from types import SimpleNamespace
from typing import Any

import pytest

from vap.api.deps import decode_b64
from vap.api.middleware import RequestContextMiddleware
from vap.api.routes.pipelines import mjpeg_stream
from vap.api.security import ANONYMOUS_ADMIN, authenticate, key_fingerprint, require
from vap.config import Role, Settings
from vap.errors import AuthenticationError, InvalidInputError, PermissionDeniedError
from vap.observability.metrics import PlatformMetrics
from vap.pipeline.pipeline import PipelineState

KEYS = {"a" * 20: "admin", "v" * 20: "viewer"}


def fake_request(headers: dict[str, str], settings: Settings) -> Any:
    lowered = {k.lower(): v for k, v in headers.items()}
    app = SimpleNamespace(state=SimpleNamespace(runtime=SimpleNamespace(settings=settings)))
    return SimpleNamespace(headers=lowered, app=app, state=SimpleNamespace())


class TestSecurity:
    settings = Settings.model_validate({"auth": {"enabled": True, "api_keys": KEYS}})

    def test_header_and_bearer(self) -> None:
        p = authenticate(self.settings, fake_request({"X-API-Key": "a" * 20}, self.settings))
        assert p.role == Role.ADMIN and p.key_id == key_fingerprint("a" * 20)
        p = authenticate(
            self.settings, fake_request({"Authorization": "Bearer " + "v" * 20}, self.settings)
        )
        assert p.role == Role.VIEWER

    @pytest.mark.parametrize("headers", [{}, {"X-API-Key": "nope"}, {"Authorization": "Basic abc"}])
    def test_rejections(self, headers: dict[str, str]) -> None:
        with pytest.raises(AuthenticationError):
            authenticate(self.settings, fake_request(headers, self.settings))

    def test_role_dependency(self) -> None:
        dep = require(Role.OPERATOR)
        with pytest.raises(PermissionDeniedError):
            dep(fake_request({"X-API-Key": "v" * 20}, self.settings))
        req = fake_request({"X-API-Key": "a" * 20}, self.settings)
        assert dep(req).role == Role.ADMIN and req.state.principal.role == Role.ADMIN

    def test_auth_disabled_grants_admin(self) -> None:
        settings = Settings()
        assert authenticate(settings, fake_request({}, settings)) is ANONYMOUS_ADMIN


def test_decode_b64_accepts_data_urls() -> None:
    raw = b"\x89PNGdata"
    encoded = base64.b64encode(raw).decode()
    assert decode_b64(encoded) == raw
    assert decode_b64("data:image/png;base64," + encoded) == raw
    with pytest.raises(InvalidInputError):
        decode_b64("***")


class TestMiddleware:
    def run(
        self, app: Any, scope: dict[str, Any], chunks: list[bytes]
    ) -> list[MutableMapping[str, Any]]:
        sent: list[MutableMapping[str, Any]] = []
        queue = [
            {"type": "http.request", "body": c, "more_body": i < len(chunks) - 1}
            for i, c in enumerate(chunks)
        ]

        async def receive() -> MutableMapping[str, Any]:
            return queue.pop(0) if queue else {"type": "http.disconnect"}

        async def send(message: MutableMapping[str, Any]) -> None:
            sent.append(message)

        asyncio.run(app(scope, receive, send))
        return sent

    @staticmethod
    async def echo_app(scope: Any, receive: Any, send: Any) -> None:
        body = b""
        while True:
            msg = await receive()
            body += msg.get("body", b"")
            if not msg.get("more_body"):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": body})

    def scope(
        self,
        path: str = "/v1/x",
        method: str = "POST",
        headers: list[tuple[bytes, bytes]] | None = None,
    ) -> dict[str, Any]:
        return {"type": "http", "method": method, "path": path, "headers": headers or []}

    def test_streamed_body_over_limit_is_rejected(self) -> None:
        metrics = PlatformMetrics()
        mw = RequestContextMiddleware(
            self.echo_app, max_body=10, max_upload=100, upload_prefixes=("/up/",), metrics=metrics
        )
        sent = self.run(mw, self.scope(), [b"12345", b"67890", b"x"])
        assert sent[0]["status"] == 413
        assert json.loads(sent[1]["body"])["error"]["details"]["max_bytes"] == 10
        assert metrics.http_requests.value(method="POST", route="unmatched", status="413") == 1

    def test_upload_prefix_gets_larger_limit_and_request_id(self) -> None:
        mw = RequestContextMiddleware(
            self.echo_app, max_body=10, max_upload=100, upload_prefixes=("/up/",)
        )
        headers = [(b"x-request-id", b"abc-1")]
        sent = self.run(mw, self.scope("/up/model", "PUT", headers), [b"x" * 50])
        assert sent[0]["status"] == 200 and sent[1]["body"] == b"x" * 50
        assert (b"x-request-id", b"abc-1") in sent[0]["headers"]

    def test_declared_length_short_circuits(self) -> None:
        mw = RequestContextMiddleware(
            self.echo_app, max_body=10, max_upload=100, upload_prefixes=()
        )
        sent = self.run(mw, self.scope(headers=[(b"content-length", b"999")]), [b""])
        assert sent[0]["status"] == 413

    def test_non_http_scopes_pass_through(self) -> None:
        called: list[str] = []

        async def app(scope: Any, receive: Any, send: Any) -> None:
            called.append(scope["type"])

        mw = RequestContextMiddleware(app, max_body=1, max_upload=1, upload_prefixes=())
        asyncio.run(mw({"type": "lifespan"}, None, None))  # type: ignore[arg-type]
        assert called == ["lifespan"]


class FakePipe:
    def __init__(self, frames: int) -> None:
        self.remaining = frames
        self.index = -1
        self.state = PipelineState.RUNNING

    def wait_for_frame(self, after: int, timeout: float) -> bool:
        if self.remaining == 0:
            self.state = PipelineState.COMPLETED
            return False
        self.remaining -= 1
        self.index += 1
        return True

    def latest_jpeg(self) -> tuple[int, bytes]:
        return self.index, b"\xff\xd8jpeg"


def test_mjpeg_stream_ends_with_pipeline() -> None:
    parts = list(mjpeg_stream(FakePipe(2), max_fps=1000, max_frames=None, idle_timeout=0.01))  # type: ignore[arg-type]
    assert len(parts) == 2 and b"X-Frame-Index: 1" in parts[1]
    limited = list(mjpeg_stream(FakePipe(10), max_fps=1000, max_frames=3, idle_timeout=0.01))  # type: ignore[arg-type]
    assert len(limited) == 3
