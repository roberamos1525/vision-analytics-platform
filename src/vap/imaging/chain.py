"""Declarative chains of image operations (used by the API and the CLI).

Example: ``[{"op": "resize", "width": 320}, {"op": "gray"}, {"op": "canny", "low": 50, "high": 120}]``.
Boolean masks produced by an op (edges, threshold) are converted to 0/255
images when they become the chain's final output.
"""

from __future__ import annotations

import time
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from vap.errors import InvalidInputError
from vap.imaging.frame import ColorSpace
from vap.imaging.ops import ImageBackend


class _Op(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ResizeOp(_Op):
    op: Literal["resize"] = "resize"
    width: int = Field(ge=1, le=8192)
    height: int | None = Field(None, ge=1, le=8192)
    interpolation: Literal["nearest", "bilinear", "area"] = "bilinear"


class GrayOp(_Op):
    op: Literal["gray"] = "gray"


class ConvertOp(_Op):
    op: Literal["convert"] = "convert"
    to: Literal["bgr", "hsv"]


class BlurOp(_Op):
    op: Literal["blur"] = "blur"
    kind: Literal["gaussian", "box"] = "gaussian"
    sigma: float = Field(1.0, gt=0, le=50)
    ksize: int = Field(3, ge=1, le=99)


class CannyOp(_Op):
    op: Literal["canny"] = "canny"
    low: float = Field(50.0, ge=0, le=2000)
    high: float = Field(120.0, ge=0, le=2000)
    sigma: float = Field(1.0, ge=0, le=10)


class SobelOp(_Op):
    op: Literal["sobel"] = "sobel"


class ThresholdOp(_Op):
    op: Literal["threshold"] = "threshold"
    value: float | None = Field(None, ge=0, le=255)


class MorphologyOp(_Op):
    op: Literal["morphology"] = "morphology"
    operation: Literal["erode", "dilate", "open", "close", "gradient"]
    ksize: int = Field(3, ge=1, le=31)
    shape: Literal["rect", "ellipse", "cross"] = "rect"
    iterations: int = Field(1, ge=1, le=20)


ImageOp = Annotated[
    ResizeOp | GrayOp | ConvertOp | BlurOp | CannyOp | SobelOp | ThresholdOp | MorphologyOp,
    Field(discriminator="op"),
]


class OpChain(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ops: list[ImageOp] = Field(min_length=1, max_length=32)


def _as_image(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == bool:
        return arr.astype(np.uint8) * 255
    return arr


def apply_chain(
    image: np.ndarray, chain: OpChain, backend: ImageBackend
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Apply ``chain`` and return the result plus per-step shape/timing info."""
    current = image
    space = ColorSpace.RGB
    steps: list[dict[str, Any]] = []
    for op in chain.ops:
        t0 = time.perf_counter()
        if isinstance(op, ResizeOp):
            h, w = current.shape[:2]
            height = op.height or max(1, round(h * op.width / w))
            current = backend.resize(_as_image(current), (op.width, height), op.interpolation)
        elif isinstance(op, GrayOp):
            if current.ndim == 3:
                current = backend.convert_color(current, space, ColorSpace.GRAY)
            space = ColorSpace.GRAY
        elif isinstance(op, ConvertOp):
            if space != ColorSpace.RGB:
                raise InvalidInputError("convert requires an RGB input")
            target = ColorSpace(op.to)
            current = backend.convert_color(current, ColorSpace.RGB, target)
            space = target
        elif isinstance(op, BlurOp):
            current = _as_image(current)
            if op.kind == "gaussian":
                current = backend.gaussian_blur(current, op.sigma)
            else:
                current = backend.box_blur(current, op.ksize)
        elif isinstance(op, CannyOp | SobelOp | ThresholdOp | MorphologyOp):
            gray = _as_image(current)
            if gray.ndim == 3:
                gray = backend.convert_color(gray, space, ColorSpace.GRAY)
            space = ColorSpace.GRAY
            if isinstance(op, CannyOp):
                if op.high < op.low:
                    raise InvalidInputError("canny high threshold must be >= low")
                current = backend.canny(gray, op.low, op.high, op.sigma)
            elif isinstance(op, SobelOp):
                gx, gy = backend.sobel(gray)
                mag = np.hypot(gx, gy)
                peak = float(mag.max()) or 1.0
                current = np.clip(mag * (255.0 / peak), 0, 255).astype(np.uint8)
            elif isinstance(op, ThresholdOp):
                value = op.value if op.value is not None else backend.otsu_threshold(gray)
                current = backend.threshold(gray, value)
            else:
                src = current if current.dtype == bool else gray
                current = backend.morphology(src, op.operation, op.ksize, op.shape, op.iterations)
        steps.append(
            {
                "op": op.op,
                "shape": list(current.shape),
                "ms": round((time.perf_counter() - t0) * 1000, 3),
            }
        )
    return _as_image(current), steps
