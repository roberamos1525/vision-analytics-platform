"""Concrete trackers: plain IoU association and SORT (Kalman + Hungarian)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from vap.detection.base import Detection
from vap.errors import InvalidInputError
from vap.geometry import BBox
from vap.tracking.base import Track, Tracker, TrackerConfig
from vap.tracking.kalman import KalmanBoxModel


class IoUTracker(Tracker):
    """Associates detections to the last observed box of each track."""

    name = "iou"


class SortTracker(Tracker):
    """SORT: constant-velocity Kalman prediction + Hungarian IoU assignment.

    The reported box after a match is the Kalman posterior, which smooths
    detector jitter; while a track is lost its box follows the prediction so it
    can be re-associated after short occlusions.
    """

    name = "sort"

    def __init__(
        self,
        config: TrackerConfig | None = None,
        *,
        position_noise: float = 1.0,
        velocity_noise: float = 0.01,
    ) -> None:
        super().__init__(config)
        self.position_noise = position_noise
        self.velocity_noise = velocity_noise

    def _init_model(self, track: Track) -> None:
        track.model = KalmanBoxModel(
            track.bbox, position_noise=self.position_noise, velocity_noise=self.velocity_noise
        )

    def _predict(self, track: Track) -> BBox:
        model: KalmanBoxModel = track.model
        return model.predict()

    def _on_match(self, track: Track, det: Detection) -> BBox:
        model: KalmanBoxModel = track.model
        return model.update(det.bbox)


class TrackerSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["iou", "sort"] = "sort"
    iou_threshold: float = Field(0.3, gt=0, le=1)
    min_hits: int = Field(3, ge=1, le=100)
    max_age: int = Field(10, ge=0, le=1000)
    solver: Literal["hungarian", "scipy", "greedy"] = "hungarian"
    class_aware: bool = False
    position_noise: float = Field(1.0, gt=0, le=1000)
    velocity_noise: float = Field(0.01, gt=0, le=1000)

    def config(self) -> TrackerConfig:
        return TrackerConfig(
            iou_threshold=self.iou_threshold,
            min_hits=self.min_hits,
            max_age=self.max_age,
            solver=self.solver,
            class_aware=self.class_aware,
        )


def build_tracker(spec: TrackerSpec | dict[str, Any] | None = None) -> Tracker:
    if spec is None:
        spec = TrackerSpec()
    elif isinstance(spec, dict):
        spec = TrackerSpec.model_validate(spec)
    if spec.kind == "iou":
        return IoUTracker(spec.config())
    if spec.kind == "sort":
        return SortTracker(
            spec.config(), position_noise=spec.position_noise, velocity_noise=spec.velocity_noise
        )
    raise InvalidInputError(f"unknown tracker kind {spec.kind!r}")
