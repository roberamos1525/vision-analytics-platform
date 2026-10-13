"""Backend selection for image-processing operations."""

from __future__ import annotations

import threading
from typing import Literal

from vap.errors import InvalidInputError
from vap.imaging.numpy_backend import NumpyBackend
from vap.imaging.opencv_backend import OpenCVBackend, opencv_available
from vap.imaging.ops import ImageBackend

BackendName = Literal["auto", "numpy", "opencv"]

_cache: dict[str, ImageBackend] = {}
_lock = threading.Lock()


def available_backends() -> list[str]:
    names = ["numpy"]
    if opencv_available():
        names.append("opencv")
    return names


def get_backend(name: str = "auto") -> ImageBackend:
    """Return a shared backend instance.

    ``auto`` prefers OpenCV when importable and falls back to numpy.
    Requesting ``opencv`` explicitly raises ``BackendUnavailableError`` when it
    is not installed instead of silently degrading.
    """
    if name not in ("auto", "numpy", "opencv"):
        raise InvalidInputError(f"unknown image backend {name!r}")
    resolved = name
    if name == "auto":
        resolved = "opencv" if opencv_available() else "numpy"
    with _lock:
        backend = _cache.get(resolved)
        if backend is None:
            backend = OpenCVBackend() if resolved == "opencv" else NumpyBackend()
            _cache[resolved] = backend
        return backend
