"""OpenCV-backed video file source and writer.

OpenCV is optional: without it, opening the default capture raises
:class:`vap.errors.BackendUnavailableError`. A ``capture_factory`` may be
injected, which is how the tests exercise this source without real codecs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from vap.errors import InvalidInputError, SourceError
from vap.imaging.frame import Frame, ensure_image
from vap.imaging.opencv_backend import load_cv2
from vap.sources.base import FrameSource, SourceInfo


class Capture(Protocol):
    """Subset of ``cv2.VideoCapture`` used by the file and RTSP sources."""

    def isOpened(self) -> bool: ...

    def read(self) -> tuple[bool, np.ndarray | None]: ...

    def get(self, prop_id: int) -> float: ...

    def release(self) -> None: ...


CaptureFactory = Callable[[str], Capture]

# OpenCV property ids (stable across versions); kept here so fakes do not need cv2.
CAP_PROP_POS_MSEC = 0
CAP_PROP_FRAME_WIDTH = 3
CAP_PROP_FRAME_HEIGHT = 4
CAP_PROP_FPS = 5
CAP_PROP_FRAME_COUNT = 7


def opencv_file_capture(path: str) -> Capture:
    cv2: Any = load_cv2()
    return cv2.VideoCapture(path)


def bgr_to_frame_image(image: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR (or grey) frame into the internal RGB layout."""
    if image.ndim == 2:
        return np.repeat(ensure_image(image)[:, :, None], 3, axis=2)
    return ensure_image(np.ascontiguousarray(image[:, :, 2::-1]), allow_gray=False)


class VideoFileSource(FrameSource):
    kind = "file"

    def __init__(
        self,
        source_id: str,
        path: str | Path,
        *,
        loop: bool = False,
        capture_factory: CaptureFactory | None = None,
        fallback_fps: float = 25.0,
    ) -> None:
        super().__init__(source_id)
        self.path = Path(path)
        self.loop = loop
        self._factory = capture_factory or opencv_file_capture
        self._fallback_fps = fallback_fps
        self._cap: Capture | None = None
        self._fps = fallback_fps
        self._size: tuple[int, int] | None = None
        self._count: int | None = None
        self._emitted = 0

    def _open(self) -> None:
        if not self.path.is_file():
            raise SourceError(f"video file not found: {self.path}")
        self._cap = self._factory(str(self.path))
        if not self._cap.isOpened():
            self._cap.release()
            self._cap = None
            raise SourceError(f"could not open video file {self.path.name}")
        fps = float(self._cap.get(CAP_PROP_FPS) or 0.0)
        self._fps = fps if 0 < fps <= 1000 else self._fallback_fps
        w = int(self._cap.get(CAP_PROP_FRAME_WIDTH) or 0)
        h = int(self._cap.get(CAP_PROP_FRAME_HEIGHT) or 0)
        self._size = (w, h) if w > 0 and h > 0 else None
        count = int(self._cap.get(CAP_PROP_FRAME_COUNT) or 0)
        self._count = count if count > 0 else None
        self._emitted = 0

    def _read(self) -> Frame | None:
        if self._cap is None:
            return None
        ok, image = self._cap.read()
        if (not ok or image is None) and self.loop and self._emitted > 0:
            self._cap.release()
            self._cap = self._factory(str(self.path))
            ok, image = self._cap.read()
        if not ok or image is None:
            return None
        frame = Frame(
            bgr_to_frame_image(image), self._emitted, self._emitted / self._fps, self.source_id
        )
        self._emitted += 1
        return frame

    def _close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def info(self) -> SourceInfo:
        width, height = self._size if self._size else (None, None)
        return SourceInfo(
            kind=self.kind,
            uri=str(self.path),
            width=width,
            height=height,
            fps=self._fps,
            frame_count=None if self.loop else self._count,
        )


def write_video(
    path: str | Path, frames: Iterable[np.ndarray], fps: float, codec: str = "MJPG"
) -> int:
    """Write RGB frames to a video file with OpenCV; returns the number written."""
    cv2: Any = load_cv2()
    if len(codec) != 4:
        raise InvalidInputError("codec must be a four-character code")
    if fps <= 0:
        raise InvalidInputError("fps must be positive")
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    written = 0
    try:
        for image in frames:
            image = ensure_image(image, allow_gray=False)
            if writer is None:
                h, w = image.shape[:2]
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*codec), fps, (w, h))
                if not writer.isOpened():
                    raise SourceError(f"could not open video writer for {out_path.name}")
            writer.write(np.ascontiguousarray(image[:, :, ::-1]))
            written += 1
    finally:
        if writer is not None:
            writer.release()
    return written
