"""Test helpers shared by unit and integration tests."""

from __future__ import annotations

import time
from collections.abc import Callable

import pytest

from vap.imaging.backend import available_backends
from vap.sources.synthetic import SceneSpec, SyntheticObject

HAS_OPENCV = "opencv" in available_backends()
try:
    import scipy  # noqa: F401

    HAS_SCIPY = True
except ImportError:  # pragma: no cover - depends on environment
    HAS_SCIPY = False

requires_opencv = pytest.mark.skipif(not HAS_OPENCV, reason="opencv-python-headless not installed")
requires_scipy = pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")

ADMIN_KEY = "admin-key-0123456789abcdef"
OPERATOR_KEY = "operator-key-0123456789abcd"
VIEWER_KEY = "viewer-key-0123456789abcdef"


def two_object_scene(num_frames: int = 120, noise: float = 0.0) -> SceneSpec:
    """Two rectangles moving horizontally in opposite directions on separate rows."""
    return SceneSpec(
        width=320,
        height=240,
        fps=10.0,
        num_frames=num_frames,
        noise_std=noise,
        objects=[
            SyntheticObject(1, x=10, y=40, width=30, height=24, vx=2.5, color=(230, 60, 60)),
            SyntheticObject(2, x=270, y=160, width=26, height=30, vx=-2.0, color=(60, 200, 80)),
        ],
    )


def wait_until(
    predicate: Callable[[], bool], timeout: float = 10.0, interval: float = 0.02
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()
