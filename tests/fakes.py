"""Fake capture backends standing in for ``cv2.VideoCapture`` (files and RTSP)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import numpy as np

from vap.sources.video_file import (
    CAP_PROP_FPS,
    CAP_PROP_FRAME_COUNT,
    CAP_PROP_FRAME_HEIGHT,
    CAP_PROP_FRAME_WIDTH,
)


def bgr_frame(index: int, width: int = 32, height: int = 24) -> np.ndarray:
    """A BGR frame whose blue channel encodes the index (to check BGR->RGB)."""
    img = np.zeros((height, width, 3), np.uint8)
    img[..., 0] = index % 256
    img[..., 2] = 7
    return img


class FakeCapture:
    """Scriptable capture.

    ``frames``: number of frames before reads fail (``None`` = endless).
    ``block_after``: reads after this many frames block until released (a stall).
    ``raise_after``: reads after this many frames raise an exception.
    """

    def __init__(
        self,
        *,
        opened: bool = True,
        frames: int | None = None,
        block_after: int | None = None,
        raise_after: int | None = None,
        fps: float = 25.0,
        size: tuple[int, int] = (32, 24),
        start_index: int = 0,
        read_delay: float = 0.005,
    ) -> None:
        self.read_delay = read_delay
        self.opened = opened
        self.frames = frames
        self.block_after = block_after
        self.raise_after = raise_after
        self.fps = fps
        self.size = size
        self.reads = 0
        self.index = start_index
        self.released = threading.Event()

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.read_delay:
            time.sleep(self.read_delay)
        if self.raise_after is not None and self.reads >= self.raise_after:
            raise RuntimeError("decoder exploded")
        if self.block_after is not None and self.reads >= self.block_after:
            self.released.wait(5.0)
            return False, None
        if self.frames is not None and self.reads >= self.frames:
            return False, None
        self.reads += 1
        frame = bgr_frame(self.index, *self.size)
        self.index += 1
        return True, frame

    def get(self, prop_id: int) -> float:
        return {
            CAP_PROP_FPS: self.fps,
            CAP_PROP_FRAME_WIDTH: float(self.size[0]),
            CAP_PROP_FRAME_HEIGHT: float(self.size[1]),
            CAP_PROP_FRAME_COUNT: float(self.frames or 0),
        }.get(prop_id, 0.0)

    def release(self) -> None:
        self.released.set()


class ScriptedFactory:
    """Capture factory returning captures from a script, one per connection attempt."""

    def __init__(
        self,
        script: list[Callable[[], FakeCapture]],
        default: Callable[[], FakeCapture] | None = None,
    ) -> None:
        self.script = list(script)
        self.default = default
        self.urls: list[str] = []
        self.created: list[FakeCapture] = []
        self._lock = threading.Lock()

    def __call__(self, url: str) -> FakeCapture:
        with self._lock:
            self.urls.append(url)
            if self.script:
                cap = self.script.pop(0)()
            elif self.default is not None:
                cap = self.default()
            else:
                cap = FakeCapture(opened=False)
            self.created.append(cap)
            return cap
