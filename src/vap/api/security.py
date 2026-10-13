"""API-key authentication and role-based authorisation."""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Callable
from dataclasses import dataclass

from fastapi import Request

from vap.config import Role, Settings
from vap.errors import AuthenticationError, PermissionDeniedError


@dataclass(frozen=True, slots=True)
class Principal:
    key_id: str
    role: Role


ANONYMOUS_ADMIN = Principal("auth-disabled", Role.ADMIN)


def key_fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def _extract_key(request: Request) -> str | None:
    key = request.headers.get("x-api-key")
    if key:
        return key.strip()
    auth = request.headers.get("authorization", "")
    scheme, _, token = auth.partition(" ")
    if scheme.lower() == "bearer" and token:
        return token.strip()
    return None


def authenticate(settings: Settings, request: Request) -> Principal:
    if not settings.auth.enabled:
        return ANONYMOUS_ADMIN
    supplied = _extract_key(request)
    if not supplied:
        raise AuthenticationError("missing API key (X-API-Key header or Bearer token)")
    match: Principal | None = None
    # Compare against every key so timing does not reveal which prefix matched.
    for key, role in settings.auth.api_keys.items():
        if hmac.compare_digest(key.encode("utf-8"), supplied.encode("utf-8")):
            match = Principal(key_fingerprint(key), role)
    if match is None:
        raise AuthenticationError("invalid API key")
    return match


def require(role: Role) -> Callable[[Request], Principal]:
    """FastAPI dependency factory enforcing a minimum role."""

    def dependency(request: Request) -> Principal:
        settings: Settings = request.app.state.runtime.settings
        principal = authenticate(settings, request)
        if not principal.role.allows(role):
            raise PermissionDeniedError(
                f"this operation requires the {role.value} role",
                details={"role": principal.role.value},
            )
        request.state.principal = principal
        return principal

    return dependency
