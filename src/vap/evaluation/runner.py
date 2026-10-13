"""Evaluation harness: run detector + tracker over a synthetic scene and score it.

Ground truth comes from the synthetic scene generator, so the numbers reflect
how well the classical pipeline recovers the rendered objects. Frames inside
the detector warm-up window can be excluded with ``skip_frames``.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from vap.detection.base import Detector
from vap.detection.specs import (
    AnyDetectorSpec,
    BackgroundDetectorSpec,
    DetectorSpec,
    build_detector,
)
from vap.errors import InvalidInputError
from vap.evaluation.detection import (
    COCO_THRESHOLDS,
    GTBox,
    PredBox,
    evaluate_detections,
)
from vap.evaluation.tracking import evaluate_mot
from vap.geometry import BBox
from vap.sources.specs import SyntheticSourceSpec
from vap.sources.synthetic import SyntheticScene
from vap.tracking.trackers import TrackerSpec, build_tracker


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene: SyntheticSourceSpec = Field(default_factory=lambda: SyntheticSourceSpec(num_frames=150))
    detector: DetectorSpec = Field(default_factory=BackgroundDetectorSpec)
    tracker: TrackerSpec = Field(default_factory=TrackerSpec)
    iou_thresholds: list[float] = Field(default_factory=lambda: [0.5], min_length=1, max_length=20)
    coco_map: bool = False
    tracking_iou: float = Field(0.5, gt=0, le=1)
    skip_frames: int | None = Field(None, ge=0)
    backend: str = Field("auto", pattern="^(auto|numpy|opencv)$")

    @field_validator("iou_thresholds")
    @classmethod
    def _check_thresholds(cls, values: list[float]) -> list[float]:
        for v in values:
            if not 0.0 < v <= 1.0:
                raise ValueError("IoU thresholds must be in (0, 1]")
        return sorted(set(values))

    @field_validator("scene")
    @classmethod
    def _finite_scene(cls, scene: SyntheticSourceSpec) -> SyntheticSourceSpec:
        if scene.num_frames is None or scene.loop:
            raise ValueError("evaluation scenes must be finite (num_frames set, loop false)")
        if scene.num_frames > 5000:
            raise ValueError("evaluation scenes are limited to 5000 frames")
        return scene


def default_skip(spec: AnyDetectorSpec) -> int:
    if isinstance(spec, BackgroundDetectorSpec):
        return spec.warmup_frames * spec.warmup_stride
    return 0


def run_evaluation(
    request: EvaluationRequest,
    *,
    detector: Detector | None = None,
    media_roots: list[Path] | None = None,
) -> dict[str, Any]:
    """Execute an evaluation and return a JSON-serialisable report."""
    started = time.perf_counter()
    scene = SyntheticScene(request.scene.scene_spec())
    det = detector or build_detector(
        request.detector, backend=request.backend, media_roots=media_roots
    )
    det.reset()
    tracker = build_tracker(request.tracker)
    skip = (
        request.skip_frames if request.skip_frames is not None else default_skip(request.detector)
    )
    n = scene.spec.num_frames or 0
    if skip >= n:
        raise InvalidInputError(
            "skip_frames leaves no frames to evaluate", details={"skip": skip, "frames": n}
        )

    gts: list[GTBox] = []
    preds: list[PredBox] = []
    gt_frames: dict[int, list[tuple[int, BBox]]] = defaultdict(list)
    hyp_frames: dict[int, list[tuple[int, BBox]]] = defaultdict(list)
    detect_time = 0.0
    for i in range(n):
        frame = scene.frame(i, "evaluation")
        t0 = time.perf_counter()
        detections = det.detect(frame)
        detect_time += time.perf_counter() - t0
        tracks = tracker.update(detections, i, frame.timestamp)
        if i < skip:
            continue
        for g in scene.ground_truth(i):
            gts.append(GTBox(i, g.bbox, g.label, g.object_id))
            gt_frames[i].append((g.object_id, g.bbox))
        for d in detections:
            preds.append(PredBox(i, d.bbox, d.score, d.label))
        for t in tracks:
            hyp_frames[i].append((t.track_id, t.bbox))

    thresholds = list(COCO_THRESHOLDS) if request.coco_map else request.iou_thresholds
    det_report = evaluate_detections(gts, preds, thresholds)
    mot = evaluate_mot(gt_frames, hyp_frames, request.tracking_iou)
    elapsed = time.perf_counter() - started
    return {
        "frames": n,
        "evaluated_frames": n - skip,
        "skip_frames": skip,
        "detector": det.describe(),
        "tracker": request.tracker.model_dump(),
        "detection": det_report.to_dict(),
        "tracking": mot.to_dict(),
        "timing": {
            "total_s": round(elapsed, 4),
            "detector_ms_per_frame": round(detect_time / max(n, 1) * 1000, 3),
        },
    }


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise InvalidInputError(f"{path.name}:{lineno}: invalid JSON") from exc


def evaluate_files(
    gt_path: Path, pred_path: Path, iou_thresholds: list[float], tracking_iou: float = 0.5
) -> dict[str, Any]:
    """Score JSONL files of ``{"frame", "bbox", "id"?, "score"?, "label"?}`` records.

    Detection metrics are always computed; tracking metrics are computed when
    every prediction carries an ``id``.
    """
    gts: list[GTBox] = []
    preds: list[PredBox] = []
    gt_frames: dict[int, list[tuple[int, BBox]]] = defaultdict(list)
    hyp_frames: dict[int, list[tuple[int, BBox]]] = defaultdict(list)
    has_ids = True
    for rec in _read_jsonl(gt_path):
        try:
            frame, box = int(rec["frame"]), BBox.from_array(rec["bbox"])
            oid = int(rec.get("id", len(gts)))
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidInputError(f"bad ground-truth record: {rec!r}") from exc
        gts.append(GTBox(frame, box, str(rec.get("label", "object")), oid))
        gt_frames[frame].append((oid, box))
    for rec in _read_jsonl(pred_path):
        try:
            frame, box = int(rec["frame"]), BBox.from_array(rec["bbox"])
            score = float(rec.get("score", 1.0))
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidInputError(f"bad prediction record: {rec!r}") from exc
        preds.append(PredBox(frame, box, score, str(rec.get("label", "object"))))
        if "id" in rec:
            hyp_frames[frame].append((int(rec["id"]), box))
        else:
            has_ids = False
    report: dict[str, Any] = {
        "detection": evaluate_detections(gts, preds, iou_thresholds).to_dict()
    }
    if has_ids and hyp_frames:
        report["tracking"] = evaluate_mot(gt_frames, hyp_frames, tracking_iou).to_dict()
    return report
