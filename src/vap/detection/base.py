"""Detector interface and detection result type."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any

import numpy as np

from vap.geometry import BBox, boxes_to_array
from vap.imaging.frame import Frame


@dataclass(frozen=True, slots=True)
class Detection:
    bbox: BBox
    score: float = 1.0
    label: str = "object"
    class_id: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": [round(v, 2) for v in self.bbox.to_list()],
            "score": round(float(self.score), 4),
            "label": self.label,
            "class_id": self.class_id,
        }


def detections_to_arrays(dets: list[Detection]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = boxes_to_array(d.bbox for d in dets)
    scores = np.asarray([d.score for d in dets], dtype=np.float64)
    classes = np.asarray([d.class_id for d in dets], dtype=np.int64)
    return boxes, scores, classes


class Detector(abc.ABC):
    """Produces detections for one frame at a time.

    Detectors may be stateful (background models); :meth:`reset` clears that
    state, e.g. when a pipeline restarts or the camera changes.
    """

    name: str = "abstract"

    @abc.abstractmethod
    def detect(self, frame: Frame) -> list[Detection]: ...

    def reset(self) -> None:
        return None

    def describe(self) -> dict[str, Any]:
        return {"name": self.name}
