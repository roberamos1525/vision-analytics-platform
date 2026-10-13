"""Evaluation: detection AP/precision/recall and MOT metrics on synthetic ground truth."""

from vap.evaluation.detection import (
    COCO_THRESHOLDS,
    DetectionReport,
    GTBox,
    PredBox,
    average_precision,
    evaluate_detections,
)
from vap.evaluation.runner import EvaluationRequest, evaluate_files, run_evaluation
from vap.evaluation.tracking import MOTReport, evaluate_mot

__all__ = [
    "COCO_THRESHOLDS",
    "DetectionReport",
    "EvaluationRequest",
    "GTBox",
    "MOTReport",
    "PredBox",
    "average_precision",
    "evaluate_detections",
    "evaluate_files",
    "evaluate_mot",
    "run_evaluation",
]
