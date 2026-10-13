"""Template-matching detector using zero-mean normalised cross-correlation.

The numpy path computes the correlation with real FFTs and the local image
energy with integral images, which is equivalent to OpenCV's
``TM_CCOEFF_NORMED``. When OpenCV is selected, ``cv2.matchTemplate`` is used.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vap.detection.base import Detection, Detector
from vap.detection.nms import nms
from vap.errors import InvalidInputError
from vap.geometry import BBox
from vap.imaging.backend import get_backend
from vap.imaging.frame import Frame, ensure_gray


def _integral(img: np.ndarray) -> np.ndarray:
    out = np.zeros((img.shape[0] + 1, img.shape[1] + 1), dtype=np.float64)
    out[1:, 1:] = img.cumsum(axis=0).cumsum(axis=1)
    return out


def _window_sums(integral: np.ndarray, h: int, w: int) -> np.ndarray:
    return integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]


def match_template_ncc(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Return the ``(H-h+1, W-w+1)`` ZNCC response map in ``[-1, 1]``."""
    img = image.astype(np.float64)
    tpl = template.astype(np.float64)
    H, W = img.shape
    h, w = tpl.shape
    if h > H or w > W:
        raise InvalidInputError("template is larger than the image")
    n = float(h * w)
    t0 = tpl - tpl.mean()
    t_norm = float(np.sqrt((t0 * t0).sum()))
    if t_norm == 0:
        raise InvalidInputError("template has no contrast (constant image)")
    shape = (H, W)
    spectrum = np.fft.rfft2(img, s=shape) * np.fft.rfft2(t0[::-1, ::-1], s=shape)
    corr = np.fft.irfft2(spectrum, s=shape)[h - 1 : H, w - 1 : W]
    sums = _window_sums(_integral(img), h, w)
    sq_sums = _window_sums(_integral(img * img), h, w)
    variance = np.maximum(sq_sums - sums * sums / n, 0.0)
    denom = np.sqrt(variance) * t_norm
    with np.errstate(divide="ignore", invalid="ignore"):
        resp = np.where(denom > 1e-6 * t_norm, corr / denom, 0.0)
    return np.clip(resp, -1.0, 1.0)


class TemplateDetector(Detector):
    name = "template"

    def __init__(
        self,
        template: np.ndarray,
        *,
        threshold: float = 0.8,
        scales: tuple[float, ...] = (1.0,),
        max_matches: int = 20,
        nms_iou: float = 0.3,
        label: str = "template",
        backend: str = "auto",
    ) -> None:
        tpl = ensure_gray(template)
        if tpl.shape[0] < 3 or tpl.shape[1] < 3:
            raise InvalidInputError("template must be at least 3x3 pixels")
        if not -1.0 <= threshold <= 1.0:
            raise InvalidInputError("threshold must be within [-1, 1]")
        if not scales or any(s <= 0 or s > 8 for s in scales):
            raise InvalidInputError("scales must be positive and <= 8")
        if max_matches < 1:
            raise InvalidInputError("max_matches must be >= 1")
        self.template = tpl
        self.threshold = threshold
        self.scales = tuple(scales)
        self.max_matches = max_matches
        self.nms_iou = nms_iou
        self.label = label
        self.backend = get_backend(backend)

    def _response(self, gray: np.ndarray, tpl: np.ndarray) -> np.ndarray:
        if self.backend.name == "opencv":
            cv2 = self.backend.cv2  # type: ignore[attr-defined]
            return cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED).astype(np.float64)
        return match_template_ncc(gray, tpl)

    def detect(self, frame: Frame) -> list[Detection]:
        gray = ensure_gray(frame.image)
        boxes: list[list[float]] = []
        scores: list[float] = []
        for scale in self.scales:
            th = max(3, round(self.template.shape[0] * scale))
            tw = max(3, round(self.template.shape[1] * scale))
            if th > gray.shape[0] or tw > gray.shape[1]:
                continue
            tpl = self.template if scale == 1.0 else self.backend.resize(self.template, (tw, th))
            if float(tpl.std()) == 0.0:
                continue
            resp = self._response(gray, tpl)
            ys, xs = np.nonzero(resp >= self.threshold)
            if ys.size == 0:
                continue
            vals = resp[ys, xs]
            # Keep only the strongest candidates before NMS to bound the work.
            if vals.size > self.max_matches * 50:
                top = np.argpartition(-vals, self.max_matches * 50)[: self.max_matches * 50]
                ys, xs, vals = ys[top], xs[top], vals[top]
            for y, x, v in zip(ys.tolist(), xs.tolist(), vals.tolist(), strict=True):
                boxes.append([x, y, x + tw, y + th])
                scores.append(v)
        if not boxes:
            return []
        keep = nms(np.asarray(boxes), np.asarray(scores), self.nms_iou)[: self.max_matches]
        return [Detection(BBox.from_array(boxes[i]), float(scores[i]), self.label) for i in keep]

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "backend": self.backend.name,
            "template_size": list(self.template.shape[::-1]),
            "threshold": self.threshold,
            "scales": list(self.scales),
        }
