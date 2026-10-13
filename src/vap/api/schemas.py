"""Request bodies for the REST API (responses are plain JSON dicts)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from vap.detection.specs import DetectorSpec
from vap.extraction import ExtractionSpec
from vap.imaging.chain import ImageOp

ID = r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}$"
MAX_IMAGE_B64 = 1_500_000


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CameraCreate(_Body):
    camera_id: str = Field(pattern=ID)
    name: str = Field("", max_length=128)
    source: dict[str, Any]


class CameraUpdate(_Body):
    name: str = Field("", max_length=128)
    source: dict[str, Any]


class PipelineCreate(_Body):
    pipeline_id: str = Field(pattern=ID)
    camera_id: str = Field(pattern=ID)
    config: dict[str, Any] = Field(default_factory=dict)
    start: bool = False


class PipelineUpdate(_Body):
    config: dict[str, Any]


class ExtractionRequest(_Body):
    job: str = Field(pattern=ID)
    spec: ExtractionSpec = Field(default_factory=ExtractionSpec)


class AnalyzeImageRequest(_Body):
    image_b64: str = Field(min_length=4, max_length=MAX_IMAGE_B64)
    detector: DetectorSpec


class ProcessImageRequest(_Body):
    image_b64: str = Field(min_length=4, max_length=MAX_IMAGE_B64)
    ops: list[ImageOp] = Field(min_length=1, max_length=32)
    output_format: str = Field("png", pattern="^(png|jpeg)$")


class ModelActivate(_Body):
    version: str = Field(min_length=1, max_length=64)
