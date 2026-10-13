"""Zone (polygon) and tripwire (line) definitions with validation.

Coordinates may be given in pixels or normalised to ``[0, 1]`` of the frame
size (``normalized=true``); normalised geometry is resolved against the actual
frame size when the pipeline sees its first frame, so the same configuration
works across camera resolutions.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vap.errors import InvalidInputError
from vap.geometry import LineSegment, Polygon

_ID_PATTERN = r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}$"

PointList = list[tuple[float, float]]


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _check_normalized(points: PointList) -> None:
    for x, y in points:
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError("normalized coordinates must be within [0, 1]")


class ZoneSpec(_Spec):
    zone_id: str = Field(pattern=_ID_PATTERN)
    name: str = Field("", max_length=128)
    points: PointList = Field(min_length=3, max_length=64)
    normalized: bool = False

    @model_validator(mode="after")
    def _validate(self) -> ZoneSpec:
        if self.normalized:
            _check_normalized(self.points)
        try:
            Polygon.from_points(self.points)
        except InvalidInputError as exc:
            raise ValueError(exc.message) from exc
        return self


class LineSpec(_Spec):
    line_id: str = Field(pattern=_ID_PATTERN)
    name: str = Field("", max_length=128)
    points: PointList = Field(min_length=2, max_length=2)
    normalized: bool = False

    @field_validator("points")
    @classmethod
    def _distinct(cls, value: PointList) -> PointList:
        if value[0] == value[1]:
            raise ValueError("line endpoints must differ")
        return value

    @model_validator(mode="after")
    def _validate(self) -> LineSpec:
        if self.normalized:
            _check_normalized(self.points)
        return self


@dataclass(frozen=True, slots=True)
class Zone:
    zone_id: str
    name: str
    polygon: Polygon

    def contains(self, point: tuple[float, float]) -> bool:
        return self.polygon.contains(point[0], point[1])


@dataclass(frozen=True, slots=True)
class Tripwire:
    line_id: str
    name: str
    segment: LineSegment


def _scale(points: PointList, normalized: bool, width: int, height: int) -> PointList:
    if not normalized:
        return [(float(x), float(y)) for x, y in points]
    return [(x * width, y * height) for x, y in points]


def resolve_zone(spec: ZoneSpec, width: int, height: int) -> Zone:
    poly = Polygon.from_points(_scale(spec.points, spec.normalized, width, height))
    return Zone(spec.zone_id, spec.name or spec.zone_id, poly)


def resolve_line(spec: LineSpec, width: int, height: int) -> Tripwire:
    seg = LineSegment.from_points(_scale(spec.points, spec.normalized, width, height))
    return Tripwire(spec.line_id, spec.name or spec.line_id, seg)
