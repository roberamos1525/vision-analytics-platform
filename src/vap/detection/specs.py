"""Validated detector specifications and the detector factory."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vap.detection.background import BackgroundSubtractionDetector
from vap.detection.base import Detector
from vap.detection.color import ColorDetector, ColorRange
from vap.detection.motion import FrameDifferenceDetector
from vap.detection.onnx import YoloOnnxDetector
from vap.detection.template import TemplateDetector
from vap.errors import InvalidInputError
from vap.imaging.codec import decode_image, load_image
from vap.sources.specs import resolve_media_path

MAX_TEMPLATE_B64 = 2_000_000

# (name, version or None) -> (model file path, metadata dict)
ModelResolver = Callable[[str, str | None], tuple[Path, dict[str, Any]]]


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BackgroundDetectorSpec(_Spec):
    kind: Literal["background"] = "background"
    engine: Literal["gaussian", "opencv_mog2"] = "gaussian"
    learning_rate: float = Field(0.02, gt=0, le=1)
    k_sigma: float = Field(2.5, gt=0, le=20)
    min_diff: float = Field(18.0, ge=0, le=255)
    blur_sigma: float = Field(1.0, ge=0, le=10)
    morph_ksize: int = Field(3, ge=0, le=31)
    min_area: int = Field(30, ge=1, le=10_000_000)
    max_area_ratio: float = Field(0.5, gt=0, le=1)
    warmup_frames: int = Field(20, ge=0, le=500)
    warmup_stride: int = Field(2, ge=1, le=30)
    absorb_after: int = Field(150, ge=0, le=100_000)
    label: str = Field("object", min_length=1, max_length=64)


class ColorRangeSpec(_Spec):
    label: str = Field(min_length=1, max_length=64)
    hue_min: int = Field(ge=0, le=179)
    hue_max: int = Field(ge=0, le=179)
    sat_min: int = Field(80, ge=0, le=255)
    sat_max: int = Field(255, ge=0, le=255)
    val_min: int = Field(60, ge=0, le=255)
    val_max: int = Field(255, ge=0, le=255)
    class_id: int = Field(0, ge=0, le=10_000)


class ColorDetectorSpec(_Spec):
    kind: Literal["color"] = "color"
    ranges: list[ColorRangeSpec] = Field(min_length=1, max_length=32)
    min_area: int = Field(30, ge=1)
    max_area_ratio: float = Field(0.5, gt=0, le=1)
    morph_ksize: int = Field(3, ge=0, le=31)
    nms_iou: float = Field(0.5, ge=0, le=1)


class TemplateDetectorSpec(_Spec):
    kind: Literal["template"] = "template"
    template_path: str | None = Field(None, max_length=1024)
    template_png_b64: str | None = Field(None, max_length=MAX_TEMPLATE_B64)
    threshold: float = Field(0.8, ge=-1, le=1)
    scales: list[float] = Field(default_factory=lambda: [1.0], min_length=1, max_length=8)
    max_matches: int = Field(20, ge=1, le=500)
    nms_iou: float = Field(0.3, ge=0, le=1)
    label: str = Field("template", min_length=1, max_length=64)

    @model_validator(mode="after")
    def _one_template(self) -> TemplateDetectorSpec:
        if (self.template_path is None) == (self.template_png_b64 is None):
            raise ValueError("exactly one of template_path or template_png_b64 is required")
        return self


class MotionDetectorSpec(_Spec):
    kind: Literal["motion"] = "motion"
    threshold: float = Field(20.0, ge=0, le=255)
    mode: Literal["two_frame", "three_frame"] = "three_frame"
    min_area: int = Field(30, ge=1)
    max_area_ratio: float = Field(0.5, gt=0, le=1)
    dilate_ksize: int = Field(5, ge=0, le=31)
    label: str = Field("motion", min_length=1, max_length=64)


class OnnxDetectorSpec(_Spec):
    kind: Literal["onnx"] = "onnx"
    model: str = Field(min_length=1, max_length=128)
    version: str | None = Field(None, max_length=64)
    input_width: int = Field(640, ge=32, le=4096)
    input_height: int = Field(640, ge=32, le=4096)
    conf_threshold: float = Field(0.25, ge=0, le=1)
    iou_threshold: float = Field(0.45, ge=0, le=1)
    max_detections: int = Field(100, ge=1, le=1000)
    output_format: Literal["auto", "yolov5", "yolov8"] = "auto"


AnyDetectorSpec = (
    BackgroundDetectorSpec
    | ColorDetectorSpec
    | TemplateDetectorSpec
    | MotionDetectorSpec
    | OnnxDetectorSpec
)
DetectorSpec = Annotated[AnyDetectorSpec, Field(discriminator="kind")]


def _load_template(spec: TemplateDetectorSpec, media_roots: list[Path]) -> Any:
    if spec.template_path is not None:
        return load_image(resolve_media_path(spec.template_path, media_roots), max_pixels=4_000_000)
    assert spec.template_png_b64 is not None
    try:
        raw = base64.b64decode(spec.template_png_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidInputError("template_png_b64 is not valid base64") from exc
    return decode_image(raw, max_pixels=4_000_000)


def build_detector(
    spec: AnyDetectorSpec,
    *,
    backend: str = "auto",
    media_roots: list[Path] | None = None,
    model_resolver: ModelResolver | None = None,
) -> Detector:
    if isinstance(spec, BackgroundDetectorSpec):
        params = spec.model_dump(exclude={"kind"})
        return BackgroundSubtractionDetector(**params, backend=backend)
    if isinstance(spec, ColorDetectorSpec):
        ranges = [ColorRange(**r.model_dump()) for r in spec.ranges]
        return ColorDetector(
            ranges,
            min_area=spec.min_area,
            max_area_ratio=spec.max_area_ratio,
            morph_ksize=spec.morph_ksize,
            nms_iou=spec.nms_iou,
            backend=backend,
        )
    if isinstance(spec, TemplateDetectorSpec):
        return TemplateDetector(
            _load_template(spec, media_roots or []),
            threshold=spec.threshold,
            scales=tuple(spec.scales),
            max_matches=spec.max_matches,
            nms_iou=spec.nms_iou,
            label=spec.label,
            backend=backend,
        )
    if isinstance(spec, MotionDetectorSpec):
        return FrameDifferenceDetector(**spec.model_dump(exclude={"kind"}), backend=backend)
    if isinstance(spec, OnnxDetectorSpec):
        if model_resolver is None:
            raise InvalidInputError("ONNX detectors require a model registry")
        path, metadata = model_resolver(spec.model, spec.version)
        names = metadata.get("class_names")
        return YoloOnnxDetector(
            model_path=path,
            input_size=(spec.input_width, spec.input_height),
            class_names=list(names) if isinstance(names, list) else None,
            conf_threshold=spec.conf_threshold,
            iou_threshold=spec.iou_threshold,
            max_detections=spec.max_detections,
            output_format=spec.output_format,
            backend=backend,
        )
    raise InvalidInputError(f"unsupported detector spec {type(spec).__name__}")
