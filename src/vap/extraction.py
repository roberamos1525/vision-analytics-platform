"""Frame extraction: every N frames, fixed time interval or scene-change keyframes.

Scene changes are scored with the total-variation distance between
normalised colour histograms of consecutive (downscaled) frames, combined
with the mean absolute pixel difference so that both global colour shifts and
large structural changes trigger a keyframe.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from vap.errors import InvalidInputError
from vap.imaging.backend import get_backend
from vap.imaging.codec import save_image
from vap.imaging.frame import Frame

ExtractionMode = Literal["every_n", "interval", "scene_change"]


class ExtractionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: ExtractionMode = "every_n"
    every_n: int = Field(10, ge=1, le=1_000_000)
    interval_s: float = Field(1.0, gt=0, le=86_400)
    scene_threshold: float = Field(0.3, gt=0, le=1)
    min_gap_frames: int = Field(5, ge=0, le=100_000)
    max_frames: int = Field(100, ge=1, le=10_000)
    max_scan_frames: int = Field(10_000, ge=1, le=10_000_000)
    image_format: Literal["png", "jpeg"] = "jpeg"
    resize_width: int | None = Field(None, ge=16, le=4096)


@dataclass(frozen=True, slots=True)
class Selection:
    frame: Frame
    reason: str
    score: float = 0.0


def histogram_signature(image: np.ndarray, bins: int = 16) -> np.ndarray:
    """Concatenated per-channel histograms, normalised to sum to 1 per channel."""
    small = image[::4, ::4] if image.shape[0] >= 64 and image.shape[1] >= 64 else image
    if small.ndim == 2:
        small = small[:, :, None]
    q = (small.astype(np.uint16) * bins) // 256
    hists = [
        np.bincount(q[..., c].ravel(), minlength=bins).astype(np.float64) for c in range(q.shape[2])
    ]
    return np.concatenate([h / max(h.sum(), 1.0) for h in hists])


def scene_distance(a: np.ndarray, b: np.ndarray, sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Score in [0, 1]: max of histogram TV distance and normalised mean abs diff."""
    channels = max(len(sig_a) // 16, 1)
    tv = float(np.abs(sig_a - sig_b).sum()) / (2.0 * channels)
    sa = a[::4, ::4].astype(np.float32)
    sb = b[::4, ::4].astype(np.float32)
    mad = float(np.abs(sa - sb).mean()) / 255.0 if sa.shape == sb.shape else 1.0
    return max(tv, min(mad * 2.0, 1.0))


class FrameSelector:
    def __init__(self, spec: ExtractionSpec) -> None:
        self.spec = spec
        self._next_ts: float | None = None
        self._prev: tuple[np.ndarray, np.ndarray] | None = None
        self._last_key = -(10**12)

    def consider(self, frame: Frame, position: int) -> Selection | None:
        s = self.spec
        if s.mode == "every_n":
            return Selection(frame, "every_n") if position % s.every_n == 0 else None
        if s.mode == "interval":
            if self._next_ts is None or frame.timestamp >= self._next_ts - 1e-9:
                base = frame.timestamp if self._next_ts is None else self._next_ts
                self._next_ts = base + s.interval_s
                while self._next_ts <= frame.timestamp:
                    self._next_ts += s.interval_s
                return Selection(frame, "interval")
            return None
        sig = histogram_signature(frame.image)
        if self._prev is None:
            self._prev = (frame.image, sig)
            self._last_key = position
            return Selection(frame, "first", 1.0)
        score = scene_distance(self._prev[0], frame.image, self._prev[1], sig)
        self._prev = (frame.image, sig)
        if score >= s.scene_threshold and position - self._last_key >= s.min_gap_frames:
            self._last_key = position
            return Selection(frame, "scene_change", score)
        return None

    def select(self, frames: Iterable[Frame]) -> Iterator[Selection]:
        emitted = 0
        for position, frame in enumerate(frames):
            if position >= self.spec.max_scan_frames or emitted >= self.spec.max_frames:
                return
            sel = self.consider(frame, position)
            if sel is not None:
                emitted += 1
                yield sel


def extract_frames(
    frames: Iterable[Frame], spec: ExtractionSpec, out_dir: str | Path, *, backend: str = "auto"
) -> dict[str, Any]:
    """Select frames and write them plus a ``manifest.jsonl`` into ``out_dir``."""
    target = Path(out_dir)
    if target.exists() and any(target.iterdir()):
        raise InvalidInputError(
            "extraction output directory must be empty", details={"dir": str(target)}
        )
    target.mkdir(parents=True, exist_ok=True)
    ops = get_backend(backend)
    ext = "png" if spec.image_format == "png" else "jpg"
    manifest: list[dict[str, Any]] = []
    started = time.perf_counter()
    for sel in FrameSelector(spec).select(frames):
        image = sel.frame.image
        if spec.resize_width and image.shape[1] != spec.resize_width:
            h = max(1, round(image.shape[0] * spec.resize_width / image.shape[1]))
            image = ops.resize(image, (spec.resize_width, h), "area")
        name = f"frame_{sel.frame.index:06d}.{ext}"
        save_image(target / name, image)
        manifest.append(
            {
                "file": name,
                "frame_index": sel.frame.index,
                "timestamp": round(sel.frame.timestamp, 4),
                "reason": sel.reason,
                "score": round(sel.score, 4),
            }
        )
    with (target / "manifest.jsonl").open("w", encoding="utf-8") as fh:
        for item in manifest:
            fh.write(json.dumps(item) + "\n")
    return {
        "directory": str(target),
        "mode": spec.mode,
        "count": len(manifest),
        "elapsed_s": round(time.perf_counter() - started, 3),
        "frames": manifest,
    }
