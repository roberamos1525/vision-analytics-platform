"""Typed error hierarchy shared by the library, the CLI and the HTTP API.

Every error carries a stable machine-readable ``code`` and the HTTP status the
API layer should map it to, so the error envelope stays consistent everywhere.
"""

from __future__ import annotations

from typing import Any


class VapError(Exception):
    """Base class for all platform errors."""

    code: str = "internal_error"
    status_code: int = 500

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


class InvalidInputError(VapError):
    code = "invalid_input"
    status_code = 422


class NotFoundError(VapError):
    code = "not_found"
    status_code = 404


class ConflictError(VapError):
    code = "conflict"
    status_code = 409


class AuthenticationError(VapError):
    code = "unauthenticated"
    status_code = 401


class PermissionDeniedError(VapError):
    code = "forbidden"
    status_code = 403


class PayloadTooLargeError(VapError):
    code = "payload_too_large"
    status_code = 413


class LimitExceededError(VapError):
    code = "limit_exceeded"
    status_code = 429


class BackendUnavailableError(VapError):
    """An optional dependency (OpenCV, onnxruntime, ...) is required but missing."""

    code = "backend_unavailable"
    status_code = 501


class SourceError(VapError):
    """A video source could not be opened or read."""

    code = "source_error"
    status_code = 502


class SourceTimeoutError(SourceError):
    code = "source_timeout"
    status_code = 504


class ModelError(VapError):
    code = "model_error"
    status_code = 422


class ChecksumMismatchError(ModelError):
    code = "checksum_mismatch"


class StorageError(VapError):
    code = "storage_error"
    status_code = 500


class ConfigError(VapError):
    code = "config_error"
    status_code = 500


def validation_details(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reduce pydantic error dicts to JSON-safe ``{loc, msg, type}`` records."""
    out: list[dict[str, Any]] = []
    for err in errors:
        loc = ".".join(str(p) for p in err.get("loc", ()))
        out.append({"loc": loc, "msg": str(err.get("msg", "")), "type": str(err.get("type", ""))})
    return out
