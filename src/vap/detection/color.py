"""Colour-range detector working in HSV space (OpenCV ranges: H 0-179)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from vap.detection.background import blobs_to_detections
from vap.detection.base import Detection, Detector
from vap.detection.nms import suppress
from vap.errors import InvalidInputError
from vap.imaging.backend import get_backend
from vap.imaging.frame import ColorSpace, Frame


@dataclass(frozen=True, slots=True)
class ColorRange:
    label: str
    hue_min: int
    hue_max: int
    sat_min: int = 80
    sat_max: int = 255
    val_min: int = 60
    val_max: int = 255
    class_id: int = 0

    def __post_init__(self) -> None:
        if not (0 <= self.hue_min <= 179 and 0 <= self.hue_max <= 179):
            raise InvalidInputError("hue bounds must be within [0, 179]")
        for lo, hi, name in (
            (self.sat_min, self.sat_max, "saturation"),
            (self.val_min, self.val_max, "value"),
        ):
            if not (0 <= lo <= hi <= 255):
                raise InvalidInputError(f"{name} bounds must satisfy 0 <= min <= max <= 255")

    def mask(self, hsv: np.ndarray) -> np.ndarray:
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        if self.hue_min <= self.hue_max:
            hue_ok = (h >= self.hue_min) & (h <= self.hue_max)
        else:  # wraps around red (e.g. 170..10)
            hue_ok = (h >= self.hue_min) | (h <= self.hue_max)
        return (
            hue_ok
            & (s >= self.sat_min)
            & (s <= self.sat_max)
            & (v >= self.val_min)
            & (v <= self.val_max)
        )


def range_for_rgb(
    label: str, rgb: tuple[int, int, int], hue_tolerance: int = 10, class_id: int = 0
) -> ColorRange:
    """Build a hue band centred on a reference RGB colour."""
    hsv = get_backend("numpy").convert_color(
        np.asarray(rgb, dtype=np.uint8).reshape(1, 1, 3), ColorSpace.RGB, ColorSpace.HSV
    )[0, 0]
    hue = int(hsv[0])
    sat_min = max(int(hsv[1]) - 80, 40)
    val_min = max(int(hsv[2]) - 80, 40)
    return ColorRange(
        label=label,
        hue_min=(hue - hue_tolerance) % 180,
        hue_max=(hue + hue_tolerance) % 180,
        sat_min=sat_min,
        val_min=val_min,
        class_id=class_id,
    )


class ColorDetector(Detector):
    name = "color"

    def __init__(
        self,
        ranges: list[ColorRange],
        *,
        min_area: int = 30,
        max_area_ratio: float = 0.5,
        morph_ksize: int = 3,
        nms_iou: float = 0.5,
        backend: str = "auto",
    ) -> None:
        if not ranges:
            raise InvalidInputError("colour detector needs at least one range")
        if min_area < 1 or not 0 < max_area_ratio <= 1:
            raise InvalidInputError("min_area must be >= 1 and max_area_ratio in (0, 1]")
        self.ranges = list(ranges)
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self.morph_ksize = morph_ksize
        self.nms_iou = nms_iou
        self.backend = get_backend(backend)

    def detect(self, frame: Frame) -> list[Detection]:
        hsv = self.backend.convert_color(frame.image, ColorSpace.RGB, ColorSpace.HSV)
        max_area = self.max_area_ratio * frame.width * frame.height
        dets: list[Detection] = []
        for rng in self.ranges:
            mask = rng.mask(hsv)
            if self.morph_ksize > 1 and mask.any():
                mask = self.backend.morphology(mask, "open", self.morph_ksize, "rect")
            if not mask.any():
                continue
            blobs = self.backend.connected_components(mask, 8, self.min_area)
            dets.extend(
                blobs_to_detections(
                    blobs,
                    min_area=self.min_area,
                    max_area=max_area,
                    label=rng.label,
                    class_id=rng.class_id,
                    score_fn=lambda b: b.fill_ratio,
                )
            )
        return suppress(dets, self.nms_iou, class_aware=True)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "backend": self.backend.name,
            "ranges": [r.label for r in self.ranges],
            "min_area": self.min_area,
        }
