"""Background-subtraction detector (running Gaussian model + connected components).

The pure-numpy model keeps a per-pixel running mean and variance of the
grey-level image. A pixel is foreground when it deviates from the mean by
more than ``k`` standard deviations *and* by an absolute minimum. Background
pixels adapt at ``learning_rate``; foreground pixels adapt at a much slower
rate so objects that stop moving are eventually absorbed rather than
corrupting the model immediately. With ``engine="opencv_mog2"`` OpenCV's
MOG2 subtractor is used instead (shadows disabled).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from vap.detection.base import Detection, Detector
from vap.errors import InvalidInputError
from vap.imaging.backend import get_backend
from vap.imaging.components import Blob
from vap.imaging.frame import Frame, ensure_gray
from vap.imaging.opencv_backend import load_cv2

Engine = Literal["gaussian", "opencv_mog2"]


class RunningGaussianModel:
    def __init__(
        self,
        learning_rate: float = 0.02,
        k_sigma: float = 2.5,
        min_diff: float = 18.0,
        min_std: float = 4.0,
        foreground_rate_factor: float = 0.05,
        absorb_after: int = 150,
        static_threshold: float = 6.0,
    ) -> None:
        if not 0.0 < learning_rate <= 1.0:
            raise InvalidInputError("learning_rate must be in (0, 1]")
        if k_sigma <= 0 or min_diff < 0 or min_std <= 0:
            raise InvalidInputError("k_sigma and min_std must be positive, min_diff >= 0")
        if not 0.0 <= foreground_rate_factor <= 1.0:
            raise InvalidInputError("foreground_rate_factor must be in [0, 1]")
        if absorb_after < 0 or static_threshold < 0:
            raise InvalidInputError("absorb_after and static_threshold must be >= 0")
        self.absorb_after = absorb_after
        self.static_threshold = static_threshold
        self._prev: np.ndarray | None = None
        self._static_fg: np.ndarray | None = None
        self.learning_rate = learning_rate
        self.k_sigma = k_sigma
        self.min_diff = min_diff
        self.min_std = min_std
        self.fg_factor = foreground_rate_factor
        self.mean: np.ndarray | None = None
        self.var: np.ndarray | None = None
        self.frames_seen = 0

    def reset(self) -> None:
        self._prev = None
        self._static_fg = None
        self.mean = None
        self.var = None
        self.frames_seen = 0

    def initialize(self, frames: list[np.ndarray]) -> None:
        """Seed the model from a warm-up stack using the per-pixel median.

        The median rejects objects that cover a pixel for less than half of the
        warm-up frames, which avoids "ghost" detections where objects started.
        """
        if not frames:
            raise InvalidInputError("cannot initialise a background model from zero frames")
        stack = np.stack([f.astype(np.float32) for f in frames], axis=0)
        self.mean = np.median(stack, axis=0).astype(np.float32)
        spread = np.median(np.abs(stack - self.mean[None]), axis=0) * 1.4826
        self.var = np.maximum(spread, self.min_std).astype(np.float32) ** 2
        self.frames_seen = len(frames)

    def apply(self, gray: np.ndarray) -> np.ndarray:
        """Classify pixels and update the model; returns a boolean foreground mask."""
        x = gray.astype(np.float32)
        if self.mean is None or self.var is None or self.mean.shape != x.shape:
            self.initialize([x])
            return np.zeros(x.shape, dtype=bool)
        assert self.var is not None
        diff = x - self.mean
        std = np.sqrt(np.maximum(self.var, self.min_std**2))
        fg = (np.abs(diff) > self.k_sigma * std) & (np.abs(diff) > self.min_diff)
        alpha = np.where(fg, self.learning_rate * self.fg_factor, self.learning_rate)
        alpha = alpha.astype(np.float32)
        self.mean += alpha * diff
        self.var += alpha * (diff * diff - self.var)
        self.frames_seen += 1
        if self.absorb_after:
            fg = self._absorb_static(x, fg)
        self._prev = x
        return fg

    def _absorb_static(self, x: np.ndarray, fg: np.ndarray) -> np.ndarray:
        """Absorb foreground pixels that stayed unchanged for ``absorb_after`` frames.

        This removes "ghosts" (background revealed where an object sat during
        initialisation) and eventually absorbs parked objects, the classic
        trade-off of adaptive background models.
        """
        assert self.mean is not None and self.var is not None
        if self._static_fg is None or self._static_fg.shape != x.shape or self._prev is None:
            self._static_fg = np.zeros(x.shape, dtype=np.int32)
            return fg
        static = np.abs(x - self._prev) <= self.static_threshold
        self._static_fg = np.where(fg & static, self._static_fg + 1, 0).astype(np.int32)
        absorb = self._static_fg >= self.absorb_after
        if absorb.any():
            self.mean[absorb] = x[absorb]
            self.var[absorb] = np.float32(self.min_std**2 * 4)
            self._static_fg[absorb] = 0
            fg = fg & ~absorb
        return fg


def blobs_to_detections(
    blobs: list[Blob],
    *,
    min_area: int,
    max_area: float,
    label: str,
    class_id: int = 0,
    score_fn: Any = None,
) -> list[Detection]:
    out: list[Detection] = []
    for blob in blobs:
        if blob.area < min_area or blob.area > max_area:
            continue
        score = float(score_fn(blob)) if score_fn else min(1.0, 0.5 + 0.5 * blob.fill_ratio)
        out.append(Detection(blob.bbox, score, label, class_id))
    return out


class BackgroundSubtractionDetector(Detector):
    name = "background"

    def __init__(
        self,
        *,
        engine: Engine = "gaussian",
        learning_rate: float = 0.02,
        k_sigma: float = 2.5,
        min_diff: float = 18.0,
        blur_sigma: float = 1.0,
        morph_ksize: int = 3,
        min_area: int = 30,
        max_area_ratio: float = 0.5,
        warmup_frames: int = 20,
        warmup_stride: int = 2,
        absorb_after: int = 150,
        label: str = "object",
        backend: str = "auto",
    ) -> None:
        if engine not in ("gaussian", "opencv_mog2"):
            raise InvalidInputError(f"unknown background engine {engine!r}")
        if min_area < 1 or not 0 < max_area_ratio <= 1:
            raise InvalidInputError("min_area must be >= 1 and max_area_ratio in (0, 1]")
        if warmup_frames < 0 or warmup_frames > 500 or morph_ksize < 0 or blur_sigma < 0:
            raise InvalidInputError(
                "warmup_frames must be in [0, 500]; morph_ksize, blur_sigma >= 0"
            )
        self.engine = engine
        self.blur_sigma = blur_sigma
        self.morph_ksize = morph_ksize
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        if not 1 <= warmup_stride <= 30:
            raise InvalidInputError("warmup_stride must be in [1, 30]")
        self.warmup_frames = warmup_frames
        self.warmup_stride = warmup_stride
        self.label = label
        self.backend = get_backend(backend)
        self.model = RunningGaussianModel(
            learning_rate=learning_rate,
            k_sigma=k_sigma,
            min_diff=min_diff,
            absorb_after=absorb_after,
        )
        self._learning_rate = learning_rate
        self._mog2: Any = None
        self._frames = 0
        self._warmup: list[np.ndarray] = []
        self.last_mask: np.ndarray | None = None
        if engine == "opencv_mog2":
            self._mog2 = self._make_mog2()

    def _make_mog2(self) -> Any:
        cv2 = load_cv2()
        return cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=16, detectShadows=False)

    def reset(self) -> None:
        self.model.reset()
        self._frames = 0
        self._warmup = []
        self.last_mask = None
        if self.engine == "opencv_mog2":
            self._mog2 = self._make_mog2()

    def foreground_mask(self, frame: Frame) -> np.ndarray:
        gray = ensure_gray(frame.image)
        if self.blur_sigma > 0:
            gray = self.backend.gaussian_blur(gray, self.blur_sigma)
        warmup = self._frames < self.warmup_frames * self.warmup_stride
        position = self._frames
        self._frames += 1
        if self.engine == "opencv_mog2":
            raw = self._mog2.apply(gray, learningRate=-1) > 127
            mask = np.zeros_like(raw) if warmup else raw
        elif warmup:
            # Sampling every ``warmup_stride`` frames widens the temporal window of the
            # median so slow objects are still rejected from the initial background.
            if position % self.warmup_stride == 0:
                self._warmup.append(gray)
            if len(self._warmup) == self.warmup_frames:
                self.model.initialize(self._warmup)
                self._warmup = []
            mask = np.zeros(gray.shape, dtype=bool)
        else:
            mask = self.model.apply(gray)
        if self.morph_ksize > 1 and mask.any():
            mask = self.backend.morphology(mask, "open", self.morph_ksize, "ellipse")
            mask = self.backend.morphology(mask, "close", self.morph_ksize + 2, "ellipse")
        self.last_mask = mask
        return mask

    def detect(self, frame: Frame) -> list[Detection]:
        mask = self.foreground_mask(frame)
        if not mask.any():
            return []
        blobs = self.backend.connected_components(mask, 8, self.min_area)
        max_area = self.max_area_ratio * frame.width * frame.height
        return blobs_to_detections(
            blobs, min_area=self.min_area, max_area=max_area, label=self.label
        )

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "engine": self.engine,
            "backend": self.backend.name,
            "learning_rate": self._learning_rate,
            "min_area": self.min_area,
            "warmup_frames": self.warmup_frames,
            "warmup_stride": self.warmup_stride,
        }
