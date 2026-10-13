"""Speed and direction estimation from track trajectories."""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from vap.errors import InvalidInputError
from vap.motion.flow import heading_degrees


class PositionSample(Protocol):
    """Anything with a timestamp and a position (e.g. ``vap.tracking.TrackPoint``)."""

    @property
    def timestamp(self) -> float: ...

    @property
    def x(self) -> float: ...

    @property
    def y(self) -> float: ...


COMPASS = ("E", "NE", "N", "NW", "W", "SW", "S", "SE")


def compass_direction(heading: float) -> str:
    """8-way compass label where N is image-up and E is image-right."""
    return COMPASS[int(((heading % 360.0) + 22.5) // 45.0) % 8]


@dataclass(frozen=True, slots=True)
class Kinematics:
    speed_px_s: float
    vx_px_s: float
    vy_px_s: float
    heading_deg: float | None
    direction: str | None
    speed_m_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "speed_px_s": round(self.speed_px_s, 3),
            "velocity_px_s": [round(self.vx_px_s, 3), round(self.vy_px_s, 3)],
            "heading_deg": None if self.heading_deg is None else round(self.heading_deg, 2),
            "direction": self.direction,
            "speed_m_s": None if self.speed_m_s is None else round(self.speed_m_s, 3),
            "speed_km_h": None if self.speed_m_s is None else round(self.speed_m_s * 3.6, 3),
        }


STATIONARY = Kinematics(0.0, 0.0, 0.0, None, None)


def estimate_kinematics(
    points: Sequence[PositionSample],
    *,
    window: int = 10,
    pixels_per_meter: float | None = None,
    min_speed: float = 1.0,
) -> Kinematics:
    """Least-squares velocity over the last ``window`` points.

    A linear fit is far less sensitive to per-frame box jitter than a
    two-point difference. Speeds below ``min_speed`` px/s report no heading.
    """
    if window < 2:
        raise InvalidInputError("window must be >= 2")
    if pixels_per_meter is not None and pixels_per_meter <= 0:
        raise InvalidInputError("pixels_per_meter must be positive")
    pts = list(points)[-window:]
    if len(pts) < 2:
        return STATIONARY
    t = np.asarray([p.timestamp for p in pts], dtype=np.float64)
    if float(t.max() - t.min()) <= 0:
        return STATIONARY
    x = np.asarray([p.x for p in pts], dtype=np.float64)
    y = np.asarray([p.y for p in pts], dtype=np.float64)
    tc = t - t.mean()
    denom = float((tc * tc).sum())
    vx = float((tc * (x - x.mean())).sum() / denom)
    vy = float((tc * (y - y.mean())).sum() / denom)
    speed = math.hypot(vx, vy)
    heading = heading_degrees(vx, vy) if speed >= min_speed else None
    return Kinematics(
        speed_px_s=speed,
        vx_px_s=vx,
        vy_px_s=vy,
        heading_deg=heading,
        direction=compass_direction(heading) if heading is not None else None,
        speed_m_s=speed / pixels_per_meter if pixels_per_meter else None,
    )


def path_length(points: Sequence[PositionSample]) -> float:
    pts = list(points)
    return float(sum(math.hypot(b.x - a.x, b.y - a.y) for a, b in itertools.pairwise(pts)))
