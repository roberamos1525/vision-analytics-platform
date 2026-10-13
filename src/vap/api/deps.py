"""Shared FastAPI dependencies and small response helpers."""

from __future__ import annotations

import base64
import binascii
from typing import Annotated

from fastapi import Depends, Request

from vap.api.security import Principal, require
from vap.config import Role
from vap.errors import InvalidInputError
from vap.runtime import Runtime


def get_runtime(request: Request) -> Runtime:
    runtime: Runtime = request.app.state.runtime
    return runtime


RuntimeDep = Annotated[Runtime, Depends(get_runtime)]
Viewer = Annotated[Principal, Depends(require(Role.VIEWER))]
Operator = Annotated[Principal, Depends(require(Role.OPERATOR))]
Admin = Annotated[Principal, Depends(require(Role.ADMIN))]


def decode_b64(data: str, field: str = "image_b64") -> bytes:
    payload = data.split(",", 1)[1] if data.startswith("data:") and "," in data else data
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidInputError(f"{field} is not valid base64") from exc
