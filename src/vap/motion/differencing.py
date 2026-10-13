"""Frame differencing: two-frame and three-frame motion masks."""

from __future__ import annotations

from collections import deque

import numpy as np

from vap.errors import InvalidInputError
from vap.imaging.backend import get_backend
from vap.imaging.frame import ensure_gray


def frame_difference(prev: np.ndarray, cur: np.ndarray, threshold: float = 20.0) -> np.ndarray:
    """Boolean mask of pixels whose grey level changed by more than ``threshold``."""
    a, b = ensure_gray(prev), ensure_gray(cur)
    if a.shape != b.shape:
        raise InvalidInputError(
            "frames must have the same shape", details={"a": a.shape, "b": b.shape}
        )
    diff = np.abs(b.astype(np.int16) - a.astype(np.int16))
    return diff > threshold


def three_frame_difference(
    f1: np.ndarray, f2: np.ndarray, f3: np.ndarray, threshold: float = 20.0
) -> np.ndarray:
    """Moving pixels of the *middle* frame: AND of consecutive differences.

    Unlike a two-frame difference this suppresses the "ghost" left behind at
    the object's previous position.
    """
    return frame_difference(f1, f2, threshold) & frame_difference(f2, f3, threshold)


class MotionAnalyzer:
    """Stateful differencer producing a motion mask and an energy score per frame."""

    def __init__(
        self,
        *,
        threshold: float = 20.0,
        mode: str = "two_frame",
        blur_sigma: float = 1.0,
        morph_ksize: int = 3,
        backend: str = "auto",
    ) -> None:
        if mode not in ("two_frame", "three_frame"):
            raise InvalidInputError("mode must be 'two_frame' or 'three_frame'")
        if threshold < 0:
            raise InvalidInputError("threshold must be >= 0")
        self.threshold = threshold
        self.mode = mode
        self.blur_sigma = blur_sigma
        self.morph_ksize = morph_ksize
        self.backend = get_backend(backend)
        self._history: deque[np.ndarray] = deque(maxlen=3 if mode == "three_frame" else 2)

    def reset(self) -> None:
        self._history.clear()

    def update(self, image: np.ndarray) -> np.ndarray:
        gray = ensure_gray(image)
        if self.blur_sigma > 0:
            gray = self.backend.gaussian_blur(gray, self.blur_sigma)
        if self._history and self._history[-1].shape != gray.shape:
            self._history.clear()
        self._history.append(gray)
        if len(self._history) < self._history.maxlen:  # type: ignore[operator]
            return np.zeros(gray.shape, dtype=bool)
        if self.mode == "three_frame":
            mask = three_frame_difference(*self._history, threshold=self.threshold)
        else:
            mask = frame_difference(self._history[0], self._history[1], self.threshold)
        if self.morph_ksize > 1 and mask.any():
            mask = self.backend.morphology(mask, "close", self.morph_ksize, "ellipse")
        return mask

    @staticmethod
    def energy(mask: np.ndarray) -> float:
        """Fraction of pixels in motion."""
        return float(np.count_nonzero(mask)) / mask.size if mask.size else 0.0
