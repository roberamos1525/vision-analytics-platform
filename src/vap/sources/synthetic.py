"""Deterministic synthetic scenes with exact ground truth.

A scene is a static background plus a set of moving rectangles and circles.
Object positions are a closed-form function of time, so any frame can be
rendered independently (random access) and the ground-truth boxes are exact.
Used by the demo "synthetic camera", the evaluation harness and the tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from vap.errors import InvalidInputError
from vap.geometry import BBox
from vap.imaging.frame import Frame
from vap.sources.base import FrameSource, SourceInfo

Shape = Literal["rect", "circle"]


@dataclass(frozen=True, slots=True)
class SyntheticObject:
    object_id: int
    x: float
    y: float
    width: float
    height: float
    vx: float = 0.0
    vy: float = 0.0
    shape: Shape = "rect"
    color: tuple[int, int, int] = (255, 255, 255)
    label: str = "object"
    bounce: bool = True
    start_frame: int = 0
    end_frame: int | None = None

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise InvalidInputError("synthetic object size must be positive")
        if self.shape not in ("rect", "circle"):
            raise InvalidInputError(f"unknown synthetic shape {self.shape!r}")
        if self.end_frame is not None and self.end_frame < self.start_frame:
            raise InvalidInputError("end_frame must be >= start_frame")

    def active(self, index: int) -> bool:
        return index >= self.start_frame and (self.end_frame is None or index <= self.end_frame)


@dataclass(frozen=True, slots=True)
class GroundTruth:
    frame_index: int
    object_id: int
    bbox: BBox
    label: str


def _reflect(value: float, span: float) -> float:
    """Triangle-wave reflection of ``value`` into ``[0, span]``."""
    if span <= 0:
        return 0.0
    period = 2.0 * span
    m = math.fmod(value, period)
    if m < 0:
        m += period
    return m if m <= span else period - m


@dataclass(slots=True)
class SceneSpec:
    width: int = 320
    height: int = 240
    fps: float = 15.0
    num_frames: int | None = 150
    background: tuple[int, int, int] = (32, 32, 32)
    gradient: bool = False
    noise_std: float = 0.0
    seed: int = 0
    objects: list[SyntheticObject] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not (8 <= self.width <= 4096 and 8 <= self.height <= 4096):
            raise InvalidInputError("scene dimensions must be within [8, 4096]")
        if not (0 < self.fps <= 240):
            raise InvalidInputError("fps must be in (0, 240]")
        if self.num_frames is not None and self.num_frames < 1:
            raise InvalidInputError("num_frames must be >= 1")
        if self.noise_std < 0:
            raise InvalidInputError("noise_std must be >= 0")
        ids = [o.object_id for o in self.objects]
        if len(ids) != len(set(ids)):
            raise InvalidInputError("synthetic object ids must be unique")


class SyntheticScene:
    def __init__(self, spec: SceneSpec) -> None:
        self.spec = spec
        self._background = self._make_background()
        h, w = spec.height, spec.width
        self._yy, self._xx = np.mgrid[0:h, 0:w]

    def _make_background(self) -> np.ndarray:
        s = self.spec
        bg = np.empty((s.height, s.width, 3), dtype=np.float32)
        bg[:] = np.asarray(s.background, dtype=np.float32)
        if s.gradient:
            ramp = np.linspace(-20.0, 20.0, s.width, dtype=np.float32)
            bg += ramp[None, :, None]
        return np.clip(bg, 0, 255).astype(np.uint8)

    @property
    def background(self) -> np.ndarray:
        return self._background.copy()

    def object_box(self, obj: SyntheticObject, index: int) -> BBox:
        t = index - obj.start_frame
        x = obj.x + obj.vx * t
        y = obj.y + obj.vy * t
        if obj.bounce:
            x = _reflect(x, self.spec.width - obj.width)
            y = _reflect(y, self.spec.height - obj.height)
        return BBox(x, y, x + obj.width, y + obj.height)

    def ground_truth(self, index: int) -> list[GroundTruth]:
        out: list[GroundTruth] = []
        w, h = self.spec.width, self.spec.height
        for obj in self.spec.objects:
            if not obj.active(index):
                continue
            box = self.object_box(obj, index)
            clipped = BBox(
                min(max(box.x1, 0.0), w),
                min(max(box.y1, 0.0), h),
                min(max(box.x2, 0.0), w),
                min(max(box.y2, 0.0), h),
            )
            if clipped.area < 1.0:
                continue
            out.append(GroundTruth(index, obj.object_id, clipped, obj.label))
        return out

    def render(self, index: int) -> np.ndarray:
        if index < 0:
            raise InvalidInputError("frame index must be >= 0")
        img = self._background.copy()
        for obj in self.spec.objects:
            if not obj.active(index):
                continue
            box = self.object_box(obj, index)
            if obj.shape == "rect":
                x1, y1 = max(round(box.x1), 0), max(round(box.y1), 0)
                x2, y2 = min(round(box.x2), self.spec.width), min(round(box.y2), self.spec.height)
                if x2 > x1 and y2 > y1:
                    img[y1:y2, x1:x2] = obj.color
            else:
                cx, cy = box.center
                rx, ry = obj.width / 2.0, obj.height / 2.0
                mask = ((self._xx + 0.5 - cx) / rx) ** 2 + ((self._yy + 0.5 - cy) / ry) ** 2 <= 1.0
                img[mask] = obj.color
        if self.spec.noise_std > 0:
            rng = np.random.default_rng((self.spec.seed, index))
            noise = rng.normal(0.0, self.spec.noise_std, img.shape)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return img

    def frame(self, index: int, source_id: str = "synthetic") -> Frame:
        return Frame(self.render(index), index, index / self.spec.fps, source_id)

    def frames(self, source_id: str = "synthetic") -> list[Frame]:
        if self.spec.num_frames is None:
            raise InvalidInputError("scene is infinite; cannot materialise all frames")
        return [self.frame(i, source_id) for i in range(self.spec.num_frames)]


_DISTINCT_COLORS: tuple[tuple[int, int, int], ...] = (
    (230, 60, 60),
    (60, 200, 80),
    (70, 110, 240),
    (240, 220, 60),
    (220, 80, 220),
    (60, 220, 220),
    (250, 150, 40),
    (200, 200, 200),
)


def random_scene(
    *,
    seed: int = 0,
    num_objects: int = 3,
    width: int = 320,
    height: int = 240,
    fps: float = 15.0,
    num_frames: int | None = 150,
    min_size: int = 14,
    max_size: int = 36,
    max_speed: float = 4.0,
    noise_std: float = 2.0,
    shapes: tuple[Shape, ...] = ("rect", "circle"),
) -> SceneSpec:
    """Generate a reproducible scene of bouncing objects with distinct colours."""
    if num_objects < 0 or num_objects > 64:
        raise InvalidInputError("num_objects must be in [0, 64]")
    if not (1 <= min_size <= max_size) or max_size >= min(width, height):
        raise InvalidInputError("object sizes must satisfy 1 <= min <= max < frame size")
    rng = np.random.default_rng(seed)
    objects: list[SyntheticObject] = []
    for i in range(num_objects):
        w = float(rng.integers(min_size, max_size + 1))
        h = float(rng.integers(min_size, max_size + 1))
        speed = float(rng.uniform(max_speed * 0.4, max_speed))
        angle = float(rng.uniform(0, 2 * math.pi))
        objects.append(
            SyntheticObject(
                object_id=i + 1,
                x=float(rng.uniform(0, width - w)),
                y=float(rng.uniform(0, height - h)),
                width=w,
                height=h,
                vx=speed * math.cos(angle),
                vy=speed * math.sin(angle),
                shape=shapes[i % len(shapes)],
                color=_DISTINCT_COLORS[i % len(_DISTINCT_COLORS)],
            )
        )
    return SceneSpec(
        width=width,
        height=height,
        fps=fps,
        num_frames=num_frames,
        noise_std=noise_std,
        seed=seed,
        objects=objects,
    )


class SyntheticSource(FrameSource):
    """Frame source backed by a :class:`SyntheticScene`."""

    kind = "synthetic"

    def __init__(self, source_id: str, spec: SceneSpec, *, loop: bool = False) -> None:
        super().__init__(source_id)
        self.scene = SyntheticScene(spec)
        self.loop = loop
        self._index = 0
        self._emitted = 0

    def _open(self) -> None:
        self._index = 0
        self._emitted = 0

    def _read(self) -> Frame | None:
        n = self.scene.spec.num_frames
        if n is not None and self._index >= n:
            if not self.loop:
                return None
            self._index = 0
        frame = self.scene.frame(self._index, self.source_id)
        # Index and timestamp keep increasing across loops so downstream analytics stay monotonic.
        frame.timestamp = self._emitted / self.scene.spec.fps
        frame.index = self._emitted
        frame.metadata["scene_index"] = self._index
        self._index += 1
        self._emitted += 1
        return frame

    def _close(self) -> None:
        return None

    def info(self) -> SourceInfo:
        s = self.scene.spec
        return SourceInfo(
            kind=self.kind,
            uri=f"synthetic://{len(s.objects)}-objects?seed={s.seed}",
            width=s.width,
            height=s.height,
            fps=s.fps,
            frame_count=None if self.loop else s.num_frames,
        )
