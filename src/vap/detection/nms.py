"""Non-maximum suppression in numpy."""

from __future__ import annotations

import numpy as np

from vap.detection.base import Detection, detections_to_arrays
from vap.errors import InvalidInputError


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy NMS; returns kept indices sorted by descending score."""
    if not 0.0 <= iou_threshold <= 1.0:
        raise InvalidInputError("iou_threshold must be within [0, 1]")
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if boxes.shape[0] != scores.shape[0]:
        raise InvalidInputError("boxes and scores must have the same length")
    if boxes.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    x1, y1, x2, y2 = boxes.T
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)
        order = rest[iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def batched_nms(
    boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, iou_threshold: float
) -> np.ndarray:
    """Class-aware NMS using the coordinate-offset trick (one NMS pass)."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if boxes.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    span = float(boxes.max() - min(boxes.min(), 0.0)) + 1.0
    offsets = np.asarray(class_ids, dtype=np.float64).reshape(-1, 1) * span
    return nms(boxes + offsets, scores, iou_threshold)


def suppress(
    dets: list[Detection], iou_threshold: float, class_aware: bool = True
) -> list[Detection]:
    if not dets:
        return []
    boxes, scores, classes = detections_to_arrays(dets)
    keep = (
        batched_nms(boxes, scores, classes, iou_threshold)
        if class_aware
        else nms(boxes, scores, iou_threshold)
    )
    return [dets[int(i)] for i in keep]
