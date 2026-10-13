"""Geometric primitives: bounding boxes, polygons and directed line segments.

Coordinates are in pixels with the origin at the top-left corner; ``x`` grows to
the right and ``y`` grows downwards. Boxes use ``(x1, y1, x2, y2)`` corners with
``x2 >= x1`` and ``y2 >= y1`` (continuous coordinates, so a box covering pixel
columns 0..9 is ``x1=0, x2=10``).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from vap.errors import InvalidInputError

Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        values = (self.x1, self.y1, self.x2, self.y2)
        if not all(math.isfinite(v) for v in values):
            raise InvalidInputError(
                "bounding box coordinates must be finite", details={"box": values}
            )
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise InvalidInputError("bounding box has negative extent", details={"box": values})

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> BBox:
        return cls(float(x), float(y), float(x + w), float(y + h))

    @classmethod
    def from_cxcywh(cls, cx: float, cy: float, w: float, h: float) -> BBox:
        return cls(float(cx - w / 2), float(cy - h / 2), float(cx + w / 2), float(cy + h / 2))

    @classmethod
    def from_array(cls, values: Sequence[float] | np.ndarray) -> BBox:
        if len(values) != 4:
            raise InvalidInputError("a bounding box needs exactly four values")
        x1, y1, x2, y2 = (float(v) for v in values)
        return cls(x1, y1, x2, y2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Point:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def bottom_center(self) -> Point:
        """Ground-contact point, the usual anchor for zone and line analytics."""
        return ((self.x1 + self.x2) / 2.0, self.y2)

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    def to_xywh(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.width, self.height)

    def clip(self, width: float, height: float) -> BBox:
        x1 = min(max(self.x1, 0.0), width)
        y1 = min(max(self.y1, 0.0), height)
        x2 = min(max(self.x2, x1), width)
        y2 = min(max(self.y2, y1), height)
        return BBox(x1, y1, x2, y2)

    def scale(self, sx: float, sy: float | None = None) -> BBox:
        sy = sx if sy is None else sy
        return BBox(self.x1 * sx, self.y1 * sy, self.x2 * sx, self.y2 * sy)

    def translate(self, dx: float, dy: float) -> BBox:
        return BBox(self.x1 + dx, self.y1 + dy, self.x2 + dx, self.y2 + dy)

    def intersection(self, other: BBox) -> float:
        w = min(self.x2, other.x2) - max(self.x1, other.x1)
        h = min(self.y2, other.y2) - max(self.y1, other.y1)
        return max(w, 0.0) * max(h, 0.0)

    def iou(self, other: BBox) -> float:
        inter = self.intersection(other)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def contains_point(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


def boxes_to_array(boxes: Iterable[BBox]) -> np.ndarray:
    """Stack boxes into an ``(N, 4)`` float64 array."""
    rows = [b.to_list() for b in boxes]
    if not rows:
        return np.zeros((0, 4), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between ``(N, 4)`` and ``(M, 4)`` corner-format box arrays."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(union > 0, inter / union, 0.0)
    return out


def _validate_points(
    points: Sequence[Sequence[float]], minimum: int, kind: str
) -> tuple[Point, ...]:
    if len(points) < minimum:
        raise InvalidInputError(f"a {kind} needs at least {minimum} points")
    out: list[Point] = []
    for p in points:
        if len(p) != 2:
            raise InvalidInputError(f"{kind} points must be (x, y) pairs")
        x, y = float(p[0]), float(p[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            raise InvalidInputError(f"{kind} coordinates must be finite")
        out.append((x, y))
    return tuple(out)


@dataclass(frozen=True, slots=True)
class Polygon:
    points: tuple[Point, ...]

    @classmethod
    def from_points(cls, points: Sequence[Sequence[float]]) -> Polygon:
        poly = cls(_validate_points(points, 3, "polygon"))
        if poly.area <= 0:
            raise InvalidInputError("polygon is degenerate (zero area)")
        return poly

    @property
    def area(self) -> float:
        pts = np.asarray(self.points)
        x, y = pts[:, 0], pts[:, 1]
        return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)

    @property
    def bbox(self) -> BBox:
        pts = np.asarray(self.points)
        return BBox(
            float(pts[:, 0].min()),
            float(pts[:, 1].min()),
            float(pts[:, 0].max()),
            float(pts[:, 1].max()),
        )

    def contains(self, x: float, y: float) -> bool:
        return bool(self.contains_many(np.array([[x, y]], dtype=np.float64))[0])

    def contains_many(self, pts: np.ndarray) -> np.ndarray:
        """Even-odd ray casting for an ``(N, 2)`` array of points."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        px, py = pts[:, 0][:, None], pts[:, 1][:, None]
        verts = np.asarray(self.points, dtype=np.float64)
        x1, y1 = verts[:, 0][None, :], verts[:, 1][None, :]
        x2, y2 = np.roll(verts[:, 0], -1)[None, :], np.roll(verts[:, 1], -1)[None, :]
        straddles = (y1 > py) != (y2 > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
        crossings = straddles & (px < x_cross)
        return (np.count_nonzero(crossings, axis=1) % 2) == 1

    def mask(self, height: int, width: int) -> np.ndarray:
        """Rasterise the polygon into a boolean mask using pixel centres."""
        ys, xs = np.mgrid[0:height, 0:width]
        centres = np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], axis=1)
        return self.contains_many(centres).reshape(height, width)

    def to_list(self) -> list[list[float]]:
        return [[x, y] for x, y in self.points]


def _cross(o: Point, a: Point, b: Point) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


@dataclass(frozen=True, slots=True)
class LineSegment:
    """A directed tripwire from ``start`` to ``end``.

    The *positive* side is the side where the cross product
    ``(end - start) x (point - start)`` is positive. With image coordinates
    (``y`` pointing down) that is the right-hand side when walking from start
    to end as seen on screen: for a line drawn left-to-right, "below" it.
    """

    start: Point
    end: Point

    @classmethod
    def from_points(cls, points: Sequence[Sequence[float]]) -> LineSegment:
        pts = _validate_points(points, 2, "line")
        if len(pts) != 2:
            raise InvalidInputError("a line needs exactly two points")
        if pts[0] == pts[1]:
            raise InvalidInputError("line endpoints must differ")
        return cls(pts[0], pts[1])

    @property
    def length(self) -> float:
        return math.dist(self.start, self.end)

    def side(self, point: Point) -> int:
        value = _cross(self.start, self.end, point)
        if value > 0:
            return 1
        if value < 0:
            return -1
        return 0

    def crossing_direction(self, prev: Point, cur: Point) -> int:
        """Return +1/-1 when the movement ``prev -> cur`` crosses the segment, else 0.

        +1 means the point moved from the negative to the positive side.
        Touching the line without passing through it does not count.
        """
        s_prev, s_cur = self.side(prev), self.side(cur)
        if s_prev == 0 or s_cur == 0 or s_prev == s_cur:
            return 0
        d1 = _cross(prev, cur, self.start)
        d2 = _cross(prev, cur, self.end)
        if (d1 > 0 and d2 > 0) or (d1 < 0 and d2 < 0):
            return 0
        return 1 if s_cur > 0 else -1

    def to_list(self) -> list[list[float]]:
        return [list(self.start), list(self.end)]
