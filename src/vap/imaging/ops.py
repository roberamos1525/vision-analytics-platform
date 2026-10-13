"""Image-processing backend interface.

Two implementations exist: :mod:`vap.imaging.numpy_backend` (always available)
and :mod:`vap.imaging.opencv_backend` (when ``opencv-python-headless`` is
installed). Both accept and return arrays in the conventions described in
:mod:`vap.imaging.frame` so callers can switch backends without code changes.
"""

from __future__ import annotations

import abc
from typing import Literal

import numpy as np

from vap.errors import InvalidInputError
from vap.imaging.components import Blob
from vap.imaging.frame import ColorSpace

Interpolation = Literal["nearest", "bilinear", "area"]
MorphOp = Literal["erode", "dilate", "open", "close", "gradient"]
KernelShape = Literal["rect", "ellipse", "cross"]

MORPH_OPS: tuple[str, ...] = ("erode", "dilate", "open", "close", "gradient")
KERNEL_SHAPES: tuple[str, ...] = ("rect", "ellipse", "cross")
INTERPOLATIONS: tuple[str, ...] = ("nearest", "bilinear", "area")


def structuring_element(shape: KernelShape, ksize: int) -> np.ndarray:
    """Boolean structuring element compatible with ``cv2.getStructuringElement``."""
    if ksize < 1:
        raise InvalidInputError("kernel size must be >= 1")
    if shape == "rect":
        return np.ones((ksize, ksize), dtype=bool)
    if shape == "cross":
        el = np.zeros((ksize, ksize), dtype=bool)
        el[ksize // 2, :] = True
        el[:, ksize // 2] = True
        return el
    if shape == "ellipse":
        r = (ksize - 1) / 2.0
        if r == 0:
            return np.ones((1, 1), dtype=bool)
        yy, xx = np.mgrid[0:ksize, 0:ksize]
        return ((xx - r) ** 2 + (yy - r) ** 2) <= r * r + 1e-9
    raise InvalidInputError(f"unknown kernel shape {shape!r}")


def gaussian_kernel_size(sigma: float) -> int:
    """Kernel width covering +-3 sigma, always odd and at least 3."""
    k = int(2 * round(3.0 * sigma) + 1)
    return max(k, 3)


def validate_size(size: tuple[int, int]) -> tuple[int, int]:
    width, height = int(size[0]), int(size[1])
    if width < 1 or height < 1:
        raise InvalidInputError("target size must be positive", details={"size": [width, height]})
    if width * height > 100_000_000:
        raise InvalidInputError("target size is unreasonably large")
    return width, height


class ImageBackend(abc.ABC):
    """Abstract image-processing operations."""

    name: str = "abstract"

    @abc.abstractmethod
    def resize(
        self, image: np.ndarray, size: tuple[int, int], interpolation: Interpolation = "bilinear"
    ) -> np.ndarray:
        """Resize to ``size=(width, height)``."""

    @abc.abstractmethod
    def convert_color(self, image: np.ndarray, src: ColorSpace, dst: ColorSpace) -> np.ndarray:
        """Convert between RGB, BGR, GRAY and HSV (OpenCV ranges: H 0-179)."""

    @abc.abstractmethod
    def gaussian_blur(
        self, image: np.ndarray, sigma: float, ksize: int | None = None
    ) -> np.ndarray:
        """Separable Gaussian blur with reflect-101 borders."""

    @abc.abstractmethod
    def box_blur(self, image: np.ndarray, ksize: int) -> np.ndarray:
        """Mean filter with a ``ksize x ksize`` window."""

    @abc.abstractmethod
    def sobel(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """3x3 Sobel derivatives ``(gx, gy)`` as float32."""

    @abc.abstractmethod
    def canny(self, gray: np.ndarray, low: float, high: float, sigma: float = 1.0) -> np.ndarray:
        """Canny edge map as a boolean array."""

    @abc.abstractmethod
    def morphology(
        self,
        mask: np.ndarray,
        op: MorphOp,
        ksize: int = 3,
        shape: KernelShape = "rect",
        iterations: int = 1,
    ) -> np.ndarray:
        """Binary or grey-level morphology; returns the input dtype."""

    @abc.abstractmethod
    def connected_components(
        self, mask: np.ndarray, connectivity: int = 8, min_area: int = 1
    ) -> list[Blob]:
        """Foreground blobs of a binary mask."""

    def threshold(self, gray: np.ndarray, value: float) -> np.ndarray:
        return np.asarray(gray) > value

    def otsu_threshold(self, gray: np.ndarray) -> int:
        """Return the Otsu threshold of a ``uint8`` image."""
        hist = np.bincount(np.asarray(gray, dtype=np.uint8).ravel(), minlength=256).astype(
            np.float64
        )
        total = hist.sum()
        if total == 0:
            return 0
        prob = hist / total
        omega = np.cumsum(prob)
        mu = np.cumsum(prob * np.arange(256))
        mu_t = mu[-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            between = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega))
        between = np.nan_to_num(between, nan=0.0, posinf=0.0)
        # Between-class variance is flat across empty histogram gaps; take the middle
        # of the optimal plateau rather than its first bin.
        best = np.flatnonzero(between >= between.max() * (1 - 1e-9))
        return int((best[0] + best[-1]) // 2)

    def to_gray(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        return self.convert_color(image, ColorSpace.RGB, ColorSpace.GRAY)

    @staticmethod
    def check_morph(op: str, ksize: int, shape: str, iterations: int) -> None:
        if op not in MORPH_OPS:
            raise InvalidInputError(f"unknown morphology op {op!r}")
        if shape not in KERNEL_SHAPES:
            raise InvalidInputError(f"unknown kernel shape {shape!r}")
        if ksize < 1 or ksize > 99:
            raise InvalidInputError("kernel size must be in [1, 99]")
        if iterations < 1 or iterations > 50:
            raise InvalidInputError("iterations must be in [1, 50]")
