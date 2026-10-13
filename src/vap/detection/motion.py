"""Frame-differencing detector: moving regions without a background model."""

from __future__ import annotations

from typing import Any

from vap.detection.background import blobs_to_detections
from vap.detection.base import Detection, Detector
from vap.errors import InvalidInputError
from vap.imaging.frame import Frame
from vap.motion.differencing import MotionAnalyzer


class FrameDifferenceDetector(Detector):
    """Detects moving regions with frame differencing (no background model)."""

    name = "motion"

    def __init__(
        self,
        *,
        threshold: float = 20.0,
        mode: str = "three_frame",
        min_area: int = 30,
        max_area_ratio: float = 0.5,
        dilate_ksize: int = 5,
        label: str = "motion",
        backend: str = "auto",
    ) -> None:
        if min_area < 1 or not 0 < max_area_ratio <= 1:
            raise InvalidInputError("min_area must be >= 1 and max_area_ratio in (0, 1]")
        self.analyzer = MotionAnalyzer(threshold=threshold, mode=mode, backend=backend)
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self.dilate_ksize = dilate_ksize
        self.label = label
        self.last_energy = 0.0

    def reset(self) -> None:
        self.analyzer.reset()
        self.last_energy = 0.0

    def detect(self, frame: Frame) -> list[Detection]:
        mask = self.analyzer.update(frame.image)
        self.last_energy = MotionAnalyzer.energy(mask)
        if not mask.any():
            return []
        backend = self.analyzer.backend
        if self.dilate_ksize > 1:
            mask = backend.morphology(mask, "dilate", self.dilate_ksize, "ellipse")
        blobs = backend.connected_components(mask, 8, self.min_area)
        max_area = self.max_area_ratio * frame.width * frame.height
        return blobs_to_detections(
            blobs, min_area=self.min_area, max_area=max_area, label=self.label
        )

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.analyzer.mode,
            "threshold": self.analyzer.threshold,
            "backend": self.analyzer.backend.name,
        }
