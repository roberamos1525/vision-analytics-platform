"""Block-matching optical flow with an optional OpenCV Farneback path.

For each ``block x block`` tile of the previous frame the displacement within
``[-search, search]^2`` that minimises the sum of absolute differences (SAD)
in the current frame is selected. The search is vectorised over all blocks:
one shifted copy of the current frame is compared per candidate displacement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from vap.errors import InvalidInputError
from vap.imaging.frame import ensure_gray
from vap.imaging.opencv_backend import load_cv2


@dataclass(slots=True)
class FlowField:
    """Per-block displacement (pixels/frame) with block geometry."""

    dx: np.ndarray
    dy: np.ndarray
    block: int
    valid: np.ndarray

    @property
    def magnitude(self) -> np.ndarray:
        return np.hypot(self.dx, self.dy)

    def summary(self, min_magnitude: float = 0.5) -> FlowSummary:
        mag = self.magnitude
        moving = self.valid & (mag >= min_magnitude)
        if not moving.any():
            return FlowSummary(0.0, 0.0, None, 0.0, int(self.valid.sum()))
        mdx = float(self.dx[moving].mean())
        mdy = float(self.dy[moving].mean())
        return FlowSummary(
            mean_magnitude=float(mag[moving].mean()),
            max_magnitude=float(mag[moving].max()),
            direction_deg=heading_degrees(mdx, mdy),
            moving_fraction=float(moving.sum()) / max(int(self.valid.sum()), 1),
            valid_blocks=int(self.valid.sum()),
        )


@dataclass(frozen=True, slots=True)
class FlowSummary:
    mean_magnitude: float
    max_magnitude: float
    direction_deg: float | None
    moving_fraction: float
    valid_blocks: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_magnitude": round(self.mean_magnitude, 4),
            "max_magnitude": round(self.max_magnitude, 4),
            "direction_deg": None if self.direction_deg is None else round(self.direction_deg, 2),
            "moving_fraction": round(self.moving_fraction, 4),
            "valid_blocks": self.valid_blocks,
        }


def heading_degrees(dx: float, dy: float) -> float:
    """Heading in degrees, 0 = image right, 90 = image up (counter-clockwise)."""
    return math.degrees(math.atan2(-dy, dx)) % 360.0


def block_matching_flow(
    prev: np.ndarray,
    cur: np.ndarray,
    *,
    block: int = 8,
    search: int = 6,
    min_texture: float = 4.0,
) -> FlowField:
    """Estimate per-block motion from ``prev`` to ``cur``.

    Blocks whose intensity standard deviation is below ``min_texture`` are
    marked invalid (flat regions have no reliable match). Ties prefer the
    smallest displacement, so static scenes yield zero flow.
    """
    a = ensure_gray(prev).astype(np.int16)
    b = ensure_gray(cur).astype(np.int16)
    if a.shape != b.shape:
        raise InvalidInputError("frames must have the same shape")
    if block < 2 or search < 1 or search > 32:
        raise InvalidInputError("block must be >= 2 and search in [1, 32]")
    h, w = a.shape
    rows, cols = h // block, w // block
    if rows == 0 or cols == 0:
        raise InvalidInputError("frame is smaller than one block")
    hh, ww = rows * block, cols * block
    a_crop = a[:hh, :ww]
    padded = np.pad(b, search, mode="edge")
    best = np.full((rows, cols), np.iinfo(np.int64).max, dtype=np.int64)
    best_dx = np.zeros((rows, cols), dtype=np.float32)
    best_dy = np.zeros((rows, cols), dtype=np.float32)
    best_r2 = np.full((rows, cols), np.inf, dtype=np.float32)
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            shifted = padded[search + dy : search + dy + hh, search + dx : search + dx + ww]
            sad = np.abs(a_crop - shifted).reshape(rows, block, cols, block).sum(axis=(1, 3))
            r2 = float(dx * dx + dy * dy)
            better = (sad < best) | ((sad == best) & (r2 < best_r2))
            best = np.where(better, sad, best)
            best_dx = np.where(better, dx, best_dx)
            best_dy = np.where(better, dy, best_dy)
            best_r2 = np.where(better, r2, best_r2)
    texture = a_crop.reshape(rows, block, cols, block).std(axis=(1, 3))
    valid = texture >= min_texture
    best_dx = np.where(valid, best_dx, 0.0).astype(np.float32)
    best_dy = np.where(valid, best_dy, 0.0).astype(np.float32)
    return FlowField(best_dx, best_dy, block, valid)


def farneback_flow(prev: np.ndarray, cur: np.ndarray, *, block: int = 8) -> FlowField:
    """Dense Farneback flow (OpenCV) averaged onto the same block grid."""
    cv2: Any = load_cv2()
    a, b = ensure_gray(prev), ensure_gray(cur)
    if a.shape != b.shape:
        raise InvalidInputError("frames must have the same shape")
    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = a.shape
    rows, cols = h // block, w // block
    if rows == 0 or cols == 0:
        raise InvalidInputError("frame is smaller than one block")
    crop = (
        flow[: rows * block, : cols * block].reshape(rows, block, cols, block, 2).mean(axis=(1, 3))
    )
    texture = a[: rows * block, : cols * block].reshape(rows, block, cols, block).std(axis=(1, 3))
    return FlowField(
        crop[..., 0].astype(np.float32), crop[..., 1].astype(np.float32), block, texture >= 1.0
    )
