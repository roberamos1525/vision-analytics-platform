"""Track lifecycle shared by the IoU and SORT trackers.

Lifecycle: a new detection starts a ``TENTATIVE`` track; after ``min_hits``
consecutive matches it becomes ``CONFIRMED``. A track missing for at least one
frame is ``LOST``; after ``max_age`` frames without a match it is ``DELETED``
and handed to :meth:`Tracker.pop_finished`. Tentative tracks that miss a frame
are deleted immediately (they never reached confirmation).
"""

from __future__ import annotations

import enum
import itertools
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vap.detection.base import Detection
from vap.errors import InvalidInputError
from vap.geometry import BBox, boxes_to_array, iou_matrix
from vap.tracking.assignment import Solver, match


class TrackState(enum.StrEnum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    DELETED = "deleted"


@dataclass(frozen=True, slots=True)
class TrackPoint:
    frame_index: int
    timestamp: float
    x: float
    y: float


@dataclass(slots=True)
class Track:
    track_id: int
    bbox: BBox
    label: str
    score: float
    first_frame: int
    first_timestamp: float
    last_frame: int
    last_timestamp: float
    state: TrackState = TrackState.TENTATIVE
    hits: int = 1
    consecutive_hits: int = 1
    misses: int = 0
    history: deque[TrackPoint] = field(default_factory=lambda: deque(maxlen=256))
    ever_confirmed: bool = False
    model: Any = None

    @property
    def anchor(self) -> tuple[float, float]:
        return self.bbox.bottom_center

    @property
    def duration(self) -> float:
        return self.last_timestamp - self.first_timestamp

    @property
    def is_active(self) -> bool:
        return self.state in (TrackState.TENTATIVE, TrackState.CONFIRMED)

    def velocity(self, window: int = 5) -> tuple[float, float]:
        """Centre velocity in pixels/second over the last ``window`` points."""
        pts = list(self.history)[-max(window, 2) :]
        if len(pts) < 2:
            return (0.0, 0.0)
        dt = pts[-1].timestamp - pts[0].timestamp
        if dt <= 0:
            return (0.0, 0.0)
        return ((pts[-1].x - pts[0].x) / dt, (pts[-1].y - pts[0].y) / dt)

    def to_dict(self) -> dict[str, Any]:
        vx, vy = self.velocity()
        return {
            "track_id": self.track_id,
            "bbox": [round(v, 2) for v in self.bbox.to_list()],
            "label": self.label,
            "score": round(float(self.score), 4),
            "state": self.state.value,
            "hits": self.hits,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "first_timestamp": round(self.first_timestamp, 4),
            "last_timestamp": round(self.last_timestamp, 4),
            "velocity": [round(vx, 3), round(vy, 3)],
        }


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    iou_threshold: float = 0.3
    min_hits: int = 3
    max_age: int = 10
    solver: Solver = "hungarian"
    class_aware: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.iou_threshold <= 1.0:
            raise InvalidInputError("iou_threshold must be in (0, 1]")
        if self.min_hits < 1 or self.max_age < 0:
            raise InvalidInputError("min_hits must be >= 1 and max_age >= 0")


class Tracker:
    """Associates per-frame detections into persistent tracks."""

    name: str = "abstract"

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self._ids = itertools.count(1)
        self._tracks: list[Track] = []
        self._finished: list[Track] = []
        self.frame_count = 0

    # -- hooks ----------------------------------------------------------------------

    def _predict(self, track: Track) -> BBox:
        """Expected box for ``track`` in the current frame."""
        return track.bbox

    def _on_match(self, track: Track, det: Detection) -> BBox:
        """Return the box the track should report after matching ``det``."""
        return det.bbox

    def _init_model(self, track: Track) -> None:
        return None

    # -- core -----------------------------------------------------------------------

    def reset(self) -> None:
        self._ids = itertools.count(1)
        self._tracks = []
        self._finished = []
        self.frame_count = 0

    @property
    def tracks(self) -> list[Track]:
        return list(self._tracks)

    def confirmed_tracks(self) -> list[Track]:
        return [t for t in self._tracks if t.state == TrackState.CONFIRMED]

    def pop_finished(self) -> list[Track]:
        out, self._finished = self._finished, []
        return out

    def flush(self) -> list[Track]:
        """Terminate all live tracks (end of stream) and return everything finished."""
        for t in self._tracks:
            t.state = TrackState.DELETED
            self._finished.append(t)
        self._tracks = []
        return self.pop_finished()

    def _cost_matrix(self, predicted: list[BBox], dets: list[Detection]) -> np.ndarray:
        ious = iou_matrix(boxes_to_array(predicted), boxes_to_array(d.bbox for d in dets))
        cost = 1.0 - ious
        if self.config.class_aware and cost.size:
            labels_t = np.asarray([t.label for t in self._tracks], dtype=object)
            labels_d = np.asarray([d.label for d in dets], dtype=object)
            cost = np.where(labels_t[:, None] == labels_d[None, :], cost, np.inf)
        return cost

    def update(
        self, detections: list[Detection], frame_index: int, timestamp: float
    ) -> list[Track]:
        """Advance one frame; returns the confirmed tracks updated in this frame."""
        self.frame_count += 1
        predicted = [self._predict(t) for t in self._tracks]
        cost = self._cost_matrix(predicted, detections)
        matches, lost_idx, new_idx = match(
            cost.reshape(len(self._tracks), len(detections)),
            1.0 - self.config.iou_threshold,
            self.config.solver,
        )
        for ti, di in matches:
            self._apply_match(self._tracks[ti], detections[di], frame_index, timestamp)
        for ti in lost_idx:
            self._apply_miss(self._tracks[ti], predicted[ti])
        for di in new_idx:
            self._spawn(detections[di], frame_index, timestamp)
        alive: list[Track] = []
        for t in self._tracks:
            if t.state == TrackState.DELETED:
                self._finished.append(t)
            else:
                alive.append(t)
        self._tracks = alive
        return [t for t in self._tracks if t.state == TrackState.CONFIRMED and t.misses == 0]

    def _apply_match(
        self, track: Track, det: Detection, frame_index: int, timestamp: float
    ) -> None:
        track.bbox = self._on_match(track, det).clip(1e9, 1e9)
        track.score = det.score
        track.hits += 1
        track.consecutive_hits += 1
        track.misses = 0
        track.last_frame = frame_index
        track.last_timestamp = timestamp
        cx, cy = track.bbox.center
        track.history.append(TrackPoint(frame_index, timestamp, cx, cy))
        if track.state != TrackState.CONFIRMED and (
            track.ever_confirmed or track.consecutive_hits >= self.config.min_hits
        ):
            track.state = TrackState.CONFIRMED
            track.ever_confirmed = True

    def _apply_miss(self, track: Track, predicted: BBox) -> None:
        track.misses += 1
        track.consecutive_hits = 0
        if not track.ever_confirmed or track.misses > self.config.max_age:
            track.state = TrackState.DELETED
            return
        track.state = TrackState.LOST
        track.bbox = predicted

    def _spawn(self, det: Detection, frame_index: int, timestamp: float) -> None:
        track = Track(
            track_id=next(self._ids),
            bbox=det.bbox,
            label=det.label,
            score=det.score,
            first_frame=frame_index,
            first_timestamp=timestamp,
            last_frame=frame_index,
            last_timestamp=timestamp,
        )
        cx, cy = det.bbox.center
        track.history.append(TrackPoint(frame_index, timestamp, cx, cy))
        if self.config.min_hits <= 1:
            track.state = TrackState.CONFIRMED
            track.ever_confirmed = True
        self._init_model(track)
        self._tracks.append(track)
