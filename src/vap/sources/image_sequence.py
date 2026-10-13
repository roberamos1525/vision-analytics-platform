"""Frame source reading a directory of still images in natural sort order."""

from __future__ import annotations

import re
from pathlib import Path

from vap.errors import InvalidInputError, SourceError
from vap.imaging.codec import DEFAULT_MAX_PIXELS, SUPPORTED_SUFFIXES, load_image
from vap.imaging.frame import Frame
from vap.sources.base import FrameSource, SourceInfo

_NUM = re.compile(r"(\d+)")


def natural_key(name: str) -> tuple[object, ...]:
    """Sort key so ``frame_2.png`` precedes ``frame_10.png``."""
    return tuple(int(part) if part.isdigit() else part.lower() for part in _NUM.split(name))


def list_images(directory: Path, pattern: str = "*") -> list[Path]:
    files = [
        p for p in directory.glob(pattern) if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return sorted(files, key=lambda p: natural_key(p.name))


class ImageSequenceSource(FrameSource):
    kind = "images"

    def __init__(
        self,
        source_id: str,
        directory: str | Path,
        *,
        pattern: str = "*",
        fps: float = 10.0,
        loop: bool = False,
        max_pixels: int = DEFAULT_MAX_PIXELS,
    ) -> None:
        super().__init__(source_id)
        if fps <= 0:
            raise InvalidInputError("fps must be positive")
        if "/" in pattern or "\\" in pattern or ".." in pattern:
            raise InvalidInputError("pattern must not contain path separators")
        self.directory = Path(directory)
        self.pattern = pattern
        self.fps = fps
        self.loop = loop
        self.max_pixels = max_pixels
        self._files: list[Path] = []
        self._pos = 0
        self._emitted = 0
        self._size: tuple[int, int] | None = None

    def _open(self) -> None:
        if not self.directory.is_dir():
            raise SourceError(f"image directory not found: {self.directory}")
        self._files = list_images(self.directory, self.pattern)
        if not self._files:
            raise SourceError(f"no images matching {self.pattern!r} in {self.directory}")
        self._pos = 0
        self._emitted = 0

    def _read(self) -> Frame | None:
        if self._pos >= len(self._files):
            if not self.loop:
                return None
            self._pos = 0
        path = self._files[self._pos]
        try:
            image = load_image(path, max_pixels=self.max_pixels)
        except InvalidInputError as exc:
            raise SourceError(f"failed to decode {path.name}: {exc.message}") from exc
        if self._size is None:
            self._size = (image.shape[1], image.shape[0])
        frame = Frame(
            image, self._emitted, self._emitted / self.fps, self.source_id, {"path": path.name}
        )
        self._pos += 1
        self._emitted += 1
        return frame

    def _close(self) -> None:
        self._files = []

    def info(self) -> SourceInfo:
        width, height = self._size if self._size else (None, None)
        return SourceInfo(
            kind=self.kind,
            uri=str(self.directory / self.pattern),
            width=width,
            height=height,
            fps=self.fps,
            frame_count=None if self.loop else (len(self._files) or None),
        )
