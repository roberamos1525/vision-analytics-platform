"""OpenCV implementation of :class:`vap.imaging.ops.ImageBackend`.

Imported lazily: constructing :class:`OpenCVBackend` without OpenCV installed
raises :class:`vap.errors.BackendUnavailableError`.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np

from vap.errors import BackendUnavailableError, InvalidInputError
from vap.geometry import BBox
from vap.imaging.components import Blob
from vap.imaging.frame import ColorSpace, ensure_image
from vap.imaging.ops import (
    INTERPOLATIONS,
    ImageBackend,
    Interpolation,
    KernelShape,
    MorphOp,
    gaussian_kernel_size,
    structuring_element,
    validate_size,
)


def load_cv2() -> ModuleType:
    """Import OpenCV or raise a typed error explaining how to install it."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - exercised only without OpenCV
        raise BackendUnavailableError(
            "OpenCV is not installed; install the 'opencv' extra "
            "(pip install opencv-python-headless)",
            details={"backend": "opencv"},
        ) from exc
    return cv2


def opencv_available() -> bool:
    try:
        load_cv2()
    except BackendUnavailableError:
        return False
    return True


class OpenCVBackend(ImageBackend):
    name = "opencv"

    def __init__(self) -> None:
        self.cv2: Any = load_cv2()
        cv2 = self.cv2
        self._interp = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "area": cv2.INTER_AREA,
        }
        self._conversions = {
            (ColorSpace.RGB, ColorSpace.BGR): cv2.COLOR_RGB2BGR,
            (ColorSpace.BGR, ColorSpace.RGB): cv2.COLOR_BGR2RGB,
            (ColorSpace.RGB, ColorSpace.GRAY): cv2.COLOR_RGB2GRAY,
            (ColorSpace.BGR, ColorSpace.GRAY): cv2.COLOR_BGR2GRAY,
            (ColorSpace.GRAY, ColorSpace.RGB): cv2.COLOR_GRAY2RGB,
            (ColorSpace.GRAY, ColorSpace.BGR): cv2.COLOR_GRAY2BGR,
            (ColorSpace.RGB, ColorSpace.HSV): cv2.COLOR_RGB2HSV,
            (ColorSpace.BGR, ColorSpace.HSV): cv2.COLOR_BGR2HSV,
            (ColorSpace.HSV, ColorSpace.RGB): cv2.COLOR_HSV2RGB,
            (ColorSpace.HSV, ColorSpace.BGR): cv2.COLOR_HSV2BGR,
        }
        self._morph = {
            "erode": cv2.MORPH_ERODE,
            "dilate": cv2.MORPH_DILATE,
            "open": cv2.MORPH_OPEN,
            "close": cv2.MORPH_CLOSE,
            "gradient": cv2.MORPH_GRADIENT,
        }

    def resize(
        self, image: np.ndarray, size: tuple[int, int], interpolation: Interpolation = "bilinear"
    ) -> np.ndarray:
        image = ensure_image(image)
        width, height = validate_size(size)
        if interpolation not in INTERPOLATIONS:
            raise InvalidInputError(f"unknown interpolation {interpolation!r}")
        return self.cv2.resize(image, (width, height), interpolation=self._interp[interpolation])

    def convert_color(self, image: np.ndarray, src: ColorSpace, dst: ColorSpace) -> np.ndarray:
        src, dst = ColorSpace(src), ColorSpace(dst)
        image = ensure_image(image)
        if src == dst:
            return image.copy()
        if (image.ndim == 2) != (src == ColorSpace.GRAY):
            raise InvalidInputError(f"image shape does not match colour space {src.value}")
        code = self._conversions.get((src, dst))
        if code is not None:
            return self.cv2.cvtColor(image, code)
        rgb = self.convert_color(image, src, ColorSpace.RGB)
        return self.convert_color(rgb, ColorSpace.RGB, dst)

    def gaussian_blur(
        self, image: np.ndarray, sigma: float, ksize: int | None = None
    ) -> np.ndarray:
        image = ensure_image(image)
        if sigma <= 0:
            raise InvalidInputError("sigma must be positive")
        k = ksize or gaussian_kernel_size(sigma)
        if k < 1 or k % 2 == 0:
            raise InvalidInputError("kernel size must be a positive odd integer")
        return self.cv2.GaussianBlur(image, (k, k), sigmaX=sigma, sigmaY=sigma)

    def box_blur(self, image: np.ndarray, ksize: int) -> np.ndarray:
        image = ensure_image(image)
        if ksize < 1:
            raise InvalidInputError("kernel size must be >= 1")
        return self.cv2.blur(image, (ksize, ksize))

    def sobel(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gray = ensure_image(gray)
        if gray.ndim != 2:
            raise InvalidInputError("sobel expects a single-channel image")
        cv2 = self.cv2
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return gx, gy

    def canny(self, gray: np.ndarray, low: float, high: float, sigma: float = 1.0) -> np.ndarray:
        gray = ensure_image(gray)
        if gray.ndim != 2:
            raise InvalidInputError("canny expects a single-channel image")
        if low < 0 or high < low:
            raise InvalidInputError("canny thresholds must satisfy 0 <= low <= high")
        smoothed = self.gaussian_blur(gray, sigma) if sigma > 0 else gray
        return self.cv2.Canny(smoothed, low, high, L2gradient=True) > 0

    def morphology(
        self,
        mask: np.ndarray,
        op: MorphOp,
        ksize: int = 3,
        shape: KernelShape = "rect",
        iterations: int = 1,
    ) -> np.ndarray:
        self.check_morph(op, ksize, shape, iterations)
        if mask.ndim != 2:
            raise InvalidInputError("morphology expects a 2-D image")
        is_bool = mask.dtype == bool
        arr = mask.astype(np.uint8) if is_bool else ensure_image(mask)
        kernel = structuring_element(shape, ksize).astype(np.uint8)
        out = self.cv2.morphologyEx(arr, self._morph[op], kernel, iterations=iterations)
        return out.astype(bool) if is_bool else out

    def connected_components(
        self, mask: np.ndarray, connectivity: int = 8, min_area: int = 1
    ) -> list[Blob]:
        if mask.ndim != 2:
            raise InvalidInputError("connected components expects a 2-D mask")
        if connectivity not in (4, 8):
            raise InvalidInputError("connectivity must be 4 or 8")
        count, _, stats, centroids = self.cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=connectivity
        )
        cv2 = self.cv2
        blobs: list[Blob] = []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x = float(stats[label, cv2.CC_STAT_LEFT])
            y = float(stats[label, cv2.CC_STAT_TOP])
            w = float(stats[label, cv2.CC_STAT_WIDTH])
            h = float(stats[label, cv2.CC_STAT_HEIGHT])
            cx, cy = (float(c) for c in centroids[label])
            blobs.append(
                Blob(label=label, area=area, bbox=BBox(x, y, x + w, y + h), centroid=(cx, cy))
            )
        return blobs
