"""Frame container and image validation helpers.

Internally every colour image is an ``(H, W, 3)`` ``uint8`` array in RGB order
and every single-channel image is ``(H, W)``. Sources that produce BGR (OpenCV)
convert at the boundary so the rest of the system never has to guess.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np

from vap.errors import InvalidInputError


class ColorSpace(enum.StrEnum):
    RGB = "rgb"
    BGR = "bgr"
    GRAY = "gray"
    HSV = "hsv"


def ensure_image(image: np.ndarray, *, allow_gray: bool = True, name: str = "image") -> np.ndarray:
    """Validate an image array and return it as a contiguous ``uint8`` array."""
    if not isinstance(image, np.ndarray):
        raise InvalidInputError(f"{name} must be a numpy array")
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    if image.ndim == 2:
        if not allow_gray:
            raise InvalidInputError(f"{name} must have 3 channels")
    elif image.ndim != 3 or image.shape[2] != 3:
        raise InvalidInputError(
            f"{name} must be HxW or HxWx3", details={"shape": list(image.shape)}
        )
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise InvalidInputError(f"{name} is empty")
    if image.dtype != np.uint8:
        if (
            np.issubdtype(image.dtype, np.floating)
            and image.size
            and float(np.nanmax(image)) <= 1.0
        ):
            image = np.clip(image * 255.0 + 0.5, 0, 255)
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def ensure_gray(image: np.ndarray) -> np.ndarray:
    """Return a 2-D view of an image, converting RGB with ITU-R BT.601 weights."""
    image = ensure_image(image)
    if image.ndim == 2:
        return image
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gray = image.astype(np.float32) @ weights
    return np.clip(gray + 0.5, 0, 255).astype(np.uint8)


def ensure_mask(mask: np.ndarray) -> np.ndarray:
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise InvalidInputError("mask must be a 2-D array")
    return mask.astype(bool, copy=False)


@dataclass(slots=True)
class Frame:
    """A decoded video frame plus its position in the stream."""

    image: np.ndarray
    index: int
    timestamp: float
    source_id: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def with_image(self, image: np.ndarray) -> Frame:
        return Frame(image, self.index, self.timestamp, self.source_id, dict(self.metadata))
