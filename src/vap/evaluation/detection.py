"""Detection metrics: precision, recall, F1 and average precision at IoU thresholds."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from vap.errors import InvalidInputError
from vap.geometry import BBox, iou_matrix

APMethod = Literal["all_points", "coco101"]


@dataclass(frozen=True, slots=True)
class GTBox:
    frame_index: int
    bbox: BBox
    label: str = "object"
    object_id: int | None = None


@dataclass(frozen=True, slots=True)
class PredBox:
    frame_index: int
    bbox: BBox
    score: float
    label: str = "object"


@dataclass(slots=True)
class ThresholdResult:
    iou_threshold: float
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    ap: float
    pr_curve: list[tuple[float, float]] = field(default_factory=list)

    def to_dict(self, include_curve: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "iou_threshold": self.iou_threshold,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
            "ap": round(self.ap, 6),
        }
        if include_curve:
            out["pr_curve"] = [[round(r, 4), round(p, 4)] for r, p in self.pr_curve]
        return out


def average_precision(
    recall: np.ndarray, precision: np.ndarray, method: APMethod = "all_points"
) -> float:
    """Area under the interpolated precision/recall curve."""
    if recall.size == 0:
        return 0.0
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    # Precision envelope: make precision monotonically decreasing.
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    if method == "coco101":
        points = np.linspace(0.0, 1.0, 101)
        idx = np.searchsorted(mrec, points, side="left")
        vals = np.where(idx < mpre.size, mpre[np.minimum(idx, mpre.size - 1)], 0.0)
        # Recall levels beyond the achieved recall contribute zero precision.
        vals = np.where(points <= recall[-1], vals, 0.0)
        return float(vals.mean())
    if method != "all_points":
        raise InvalidInputError(f"unknown AP method {method!r}")
    change = np.nonzero(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[change + 1] - mrec[change]) * mpre[change + 1]))


def evaluate_threshold(
    gts: Sequence[GTBox],
    preds: Sequence[PredBox],
    iou_threshold: float,
    *,
    class_aware: bool = False,
    method: APMethod = "all_points",
) -> ThresholdResult:
    if not 0.0 < iou_threshold <= 1.0:
        raise InvalidInputError("IoU threshold must be in (0, 1]")

    def key(frame: int, label: str) -> tuple[int, str]:
        return (frame, label if class_aware else "*")

    gt_by_key: dict[tuple[int, str], list[GTBox]] = defaultdict(list)
    for g in gts:
        gt_by_key[key(g.frame_index, g.label)].append(g)
    gt_arrays = {
        k: np.asarray([g.bbox.to_list() for g in v], dtype=np.float64) for k, v in gt_by_key.items()
    }
    taken = {k: np.zeros(len(v), dtype=bool) for k, v in gt_by_key.items()}
    order = sorted(range(len(preds)), key=lambda i: -preds[i].score)
    is_tp = np.zeros(len(order), dtype=bool)
    for rank, i in enumerate(order):
        p = preds[i]
        k = key(p.frame_index, p.label)
        boxes = gt_arrays.get(k)
        if boxes is None:
            continue
        ious = iou_matrix(np.asarray([p.bbox.to_list()]), boxes)[0]
        ious = np.where(taken[k], -1.0, ious)
        best = int(np.argmax(ious))
        if ious[best] >= iou_threshold:
            taken[k][best] = True
            is_tp[rank] = True
    n_gt = len(gts)
    tp_cum = np.cumsum(is_tp)
    fp_cum = np.cumsum(~is_tp)
    recall = tp_cum / n_gt if n_gt else np.zeros_like(tp_cum, dtype=np.float64)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    tp = int(tp_cum[-1]) if tp_cum.size else 0
    fp = int(fp_cum[-1]) if fp_cum.size else 0
    fn = n_gt - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / n_gt if n_gt else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    ap = (
        average_precision(recall.astype(np.float64), precision.astype(np.float64), method)
        if n_gt
        else 0.0
    )
    step = max(1, len(order) // 50)
    curve = [(float(recall[i]), float(precision[i])) for i in range(0, len(order), step)]
    return ThresholdResult(iou_threshold, tp, fp, fn, prec, rec, f1, ap, curve)


@dataclass(slots=True)
class DetectionReport:
    thresholds: list[ThresholdResult]
    num_gt: int
    num_predictions: int
    num_frames: int

    @property
    def map(self) -> float:
        return float(np.mean([t.ap for t in self.thresholds])) if self.thresholds else 0.0

    def at(self, iou_threshold: float) -> ThresholdResult:
        for t in self.thresholds:
            if abs(t.iou_threshold - iou_threshold) < 1e-9:
                return t
        raise InvalidInputError(f"threshold {iou_threshold} was not evaluated")

    def to_dict(self, include_curve: bool = False) -> dict[str, Any]:
        return {
            "num_gt": self.num_gt,
            "num_predictions": self.num_predictions,
            "num_frames": self.num_frames,
            "map": round(self.map, 6),
            "thresholds": [t.to_dict(include_curve) for t in self.thresholds],
        }


COCO_THRESHOLDS: tuple[float, ...] = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))


def evaluate_detections(
    gts: Iterable[GTBox],
    preds: Iterable[PredBox],
    iou_thresholds: Sequence[float] = (0.5,),
    *,
    class_aware: bool = False,
    method: APMethod = "all_points",
) -> DetectionReport:
    gt_list, pred_list = list(gts), list(preds)
    if not iou_thresholds:
        raise InvalidInputError("at least one IoU threshold is required")
    results = [
        evaluate_threshold(gt_list, pred_list, t, class_aware=class_aware, method=method)
        for t in iou_thresholds
    ]
    frames = {g.frame_index for g in gt_list} | {p.frame_index for p in pred_list}
    return DetectionReport(results, len(gt_list), len(pred_list), len(frames))
