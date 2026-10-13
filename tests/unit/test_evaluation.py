from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vap.errors import InvalidInputError
from vap.evaluation.detection import (
    COCO_THRESHOLDS,
    GTBox,
    PredBox,
    average_precision,
    evaluate_detections,
)
from vap.evaluation.runner import EvaluationRequest, evaluate_files, run_evaluation
from vap.evaluation.tracking import evaluate_mot
from vap.geometry import BBox

A = BBox(0, 0, 10, 10)
B = BBox(50, 50, 60, 60)


class TestDetectionMetrics:
    def test_perfect(self) -> None:
        gts = [GTBox(0, A), GTBox(1, B)]
        preds = [PredBox(0, A, 0.9), PredBox(1, B, 0.8)]
        r = evaluate_detections(gts, preds, [0.5]).at(0.5)
        assert (r.tp, r.fp, r.fn) == (2, 0, 0)
        assert r.precision == r.recall == r.ap == pytest.approx(1.0)

    def test_known_pr_curve(self) -> None:
        # Ranked: TP, FP, TP -> precision 1, .5, .667 at recall .5, .5, 1.
        gts = [GTBox(0, A), GTBox(0, B)]
        preds = [PredBox(0, A, 0.9), PredBox(0, BBox(100, 100, 110, 110), 0.8), PredBox(0, B, 0.7)]
        r = evaluate_detections(gts, preds, [0.5]).at(0.5)
        assert r.ap == pytest.approx(0.5 * 1.0 + 0.5 * (2 / 3))
        assert r.precision == pytest.approx(2 / 3) and r.recall == 1.0
        assert r.f1 == pytest.approx(0.8)

    def test_duplicates_are_false_positives(self) -> None:
        r = evaluate_detections([GTBox(0, A)], [PredBox(0, A, 0.9), PredBox(0, A, 0.8)]).at(0.5)
        assert (r.tp, r.fp) == (1, 1)

    def test_iou_threshold_sensitivity_and_map(self) -> None:
        shifted = BBox(2, 0, 12, 10)  # IoU = 80/120 = 0.667
        rep = evaluate_detections([GTBox(0, A)], [PredBox(0, shifted, 1.0)], COCO_THRESHOLDS)
        assert rep.at(0.5).tp == 1 and rep.at(0.7).tp == 0
        assert rep.map == pytest.approx(4 / 10)
        with pytest.raises(InvalidInputError):
            rep.at(0.99)

    def test_class_aware_matching(self) -> None:
        gts = [GTBox(0, A, "car")]
        preds = [PredBox(0, A, 0.9, "person")]
        assert evaluate_detections(gts, preds, class_aware=False).at(0.5).tp == 1
        assert evaluate_detections(gts, preds, class_aware=True).at(0.5).tp == 0

    def test_edge_cases(self) -> None:
        empty = evaluate_detections([], [PredBox(0, A, 0.5)]).at(0.5)
        assert empty.ap == 0 and empty.fp == 1
        none = evaluate_detections([GTBox(0, A)], []).at(0.5)
        assert none.recall == 0 and none.fn == 1
        with pytest.raises(InvalidInputError):
            evaluate_detections([], [], [])
        with pytest.raises(InvalidInputError):
            evaluate_detections([], [], [1.5])
        assert evaluate_detections([GTBox(0, A)], [PredBox(0, A, 1)]).to_dict(include_curve=True)[
            "thresholds"
        ][0]["pr_curve"]

    def test_average_precision_methods(self) -> None:
        recall = np.array([0.5, 1.0])
        precision = np.array([1.0, 0.5])
        assert average_precision(recall, precision) == pytest.approx(0.75)
        assert average_precision(recall, precision, "coco101") == pytest.approx(
            (51 * 1.0 + 50 * 0.5) / 101
        )
        assert average_precision(np.array([]), np.array([])) == 0.0
        with pytest.raises(InvalidInputError):
            average_precision(recall, precision, "bogus")  # type: ignore[arg-type]


def frames_of(*tracks: list[tuple[int, BBox]]) -> dict[int, list[tuple[int, BBox]]]:
    return {i: list(t) for i, t in enumerate(tracks)}


class TestMOT:
    def test_id_switch_and_idf1(self) -> None:
        gt = frames_of(*[[(1, A.translate(i, 0))] for i in range(4)])
        hyp = frames_of(
            [(10, A)],
            [(10, A.translate(1, 0))],
            [(20, A.translate(2, 0))],
            [(20, A.translate(3, 0))],
        )
        r = evaluate_mot(gt, hyp)
        assert r.id_switches == 1 and r.matches == 4
        assert r.mota == pytest.approx(1 - 1 / 4)
        assert r.idtp == 2 and r.idf1 == pytest.approx(0.5)
        assert r.motp == pytest.approx(1.0)

    def test_fp_fn_fragmentation_and_mostly_tracked(self) -> None:
        gt = frames_of(*[[(1, A), (2, B)] for _ in range(5)])
        hyp = frames_of(
            [(1, A), (9, BBox(200, 200, 210, 210))],
            [(1, A)],
            [],
            [(1, A)],
            [(1, A)],
        )
        r = evaluate_mot(gt, hyp)
        assert r.false_positives == 1 and r.misses == 6
        assert r.fragmentations == 1
        assert (
            r.mostly_tracked == 1 and r.partially_tracked == 0 and r.mostly_lost == 1
        )  # 4/5 = 80%
        assert r.to_dict()["gt_tracks"] == 2

    def test_previous_correspondence_is_kept(self) -> None:
        # Two hypotheses overlap the GT; the established one must keep the match.
        near = BBox(1, 0, 11, 10)
        gt = frames_of([(1, A)], [(1, A)])
        hyp = frames_of([(5, near)], [(6, A), (5, near)])
        r = evaluate_mot(gt, hyp)
        assert r.id_switches == 0 and r.false_positives == 1

    def test_empty_and_validation(self) -> None:
        r = evaluate_mot({}, {})
        assert r.mota == 0 and r.idf1 == 0
        with pytest.raises(InvalidInputError):
            evaluate_mot({}, {}, 0)


class TestRunner:
    def test_synthetic_benchmark(self) -> None:
        req = EvaluationRequest.model_validate(
            {
                "scene": {"num_frames": 100, "seed": 2, "num_objects": 3},
                "backend": "numpy",
                "iou_thresholds": [0.5, 0.3],
            }
        )
        report = run_evaluation(req)
        assert report["skip_frames"] == 40 and report["evaluated_frames"] == 60
        det = report["detection"]["thresholds"]
        assert [t["iou_threshold"] for t in det] == [0.3, 0.5]
        assert det[1]["recall"] > 0.6 and det[1]["precision"] > 0.8
        assert report["tracking"]["mota"] > 0.5
        assert report["timing"]["detector_ms_per_frame"] > 0

    def test_request_validation(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            EvaluationRequest.model_validate({"scene": {"loop": True}})
        with pytest.raises(ValueError, match="IoU"):
            EvaluationRequest.model_validate({"iou_thresholds": [0]})
        with pytest.raises(InvalidInputError):
            run_evaluation(
                EvaluationRequest.model_validate({"scene": {"num_frames": 10}, "skip_frames": 10})
            )

    def test_coco_map_and_color_detector(self) -> None:
        req = EvaluationRequest.model_validate(
            {
                "scene": {
                    "num_frames": 20,
                    "num_objects": 0,
                    "noise_std": 0,
                    "objects": [
                        {
                            "object_id": 1,
                            "x": 10,
                            "y": 10,
                            "width": 20,
                            "height": 20,
                            "vx": 3,
                            "color": [230, 40, 40],
                        }
                    ],
                },
                "detector": {
                    "kind": "color",
                    "ranges": [{"label": "object", "hue_min": 170, "hue_max": 10}],
                },
                "tracker": {"kind": "iou", "min_hits": 1},
                "coco_map": True,
                "backend": "numpy",
            }
        )
        report = run_evaluation(req)
        assert report["skip_frames"] == 0
        assert len(report["detection"]["thresholds"]) == 10
        assert report["detection"]["map"] > 0.9
        assert report["tracking"]["idf1"] == pytest.approx(1.0)

    def test_evaluate_files(self, tmp_path: Path) -> None:
        gt = tmp_path / "gt.jsonl"
        pred = tmp_path / "pred.jsonl"
        gt.write_text(
            "\n".join(
                json.dumps({"frame": i, "id": 1, "bbox": [i, 0, i + 10, 10]}) for i in range(3)
            )
        )
        pred.write_text(
            "\n".join(
                json.dumps({"frame": i, "id": 4, "score": 0.9, "bbox": [i, 0, i + 10, 10]})
                for i in range(3)
            )
            + "\n\n"
        )
        report = evaluate_files(gt, pred, [0.5])
        assert report["detection"]["thresholds"][0]["ap"] == pytest.approx(1.0)
        assert report["tracking"]["mota"] == pytest.approx(1.0)
        pred.write_text(json.dumps({"frame": 0, "bbox": [0, 0, 10, 10]}))
        assert "tracking" not in evaluate_files(gt, pred, [0.5])
        pred.write_text("{broken")
        with pytest.raises(InvalidInputError):
            evaluate_files(gt, pred, [0.5])
        pred.write_text(json.dumps({"frame": 0}))
        with pytest.raises(InvalidInputError):
            evaluate_files(gt, pred, [0.5])
