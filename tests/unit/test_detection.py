from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tests.helpers import requires_opencv, two_object_scene
from vap.detection.background import BackgroundSubtractionDetector, RunningGaussianModel
from vap.detection.base import Detection
from vap.detection.color import ColorDetector, ColorRange, range_for_rgb
from vap.detection.motion import FrameDifferenceDetector
from vap.detection.nms import batched_nms, nms, suppress
from vap.detection.onnx import YoloOnnxDetector, decode_predictions, letterbox, to_nchw, unletterbox
from vap.detection.specs import (
    BackgroundDetectorSpec,
    ColorDetectorSpec,
    MotionDetectorSpec,
    OnnxDetectorSpec,
    TemplateDetectorSpec,
    build_detector,
)
from vap.detection.template import TemplateDetector, match_template_ncc
from vap.errors import BackendUnavailableError, InvalidInputError, ModelError
from vap.geometry import BBox, boxes_to_array, iou_matrix
from vap.imaging.codec import encode_image, save_image
from vap.imaging.frame import Frame
from vap.sources.synthetic import SyntheticScene


def recall_on_scene(
    detector: Any, scene: SyntheticScene, start: int, iou: float = 0.5
) -> tuple[float, float]:
    tp = fp = n_gt = 0
    for i in range(scene.spec.num_frames or 0):
        dets = detector.detect(scene.frame(i))
        if i < start:
            continue
        gts = scene.ground_truth(i)
        m = iou_matrix(boxes_to_array(d.bbox for d in dets), boxes_to_array(g.bbox for g in gts))
        hit = int((m.max(axis=0) >= iou).sum()) if m.size else 0
        tp += hit
        n_gt += len(gts)
        fp += len(dets) - hit
    return tp / n_gt, tp / max(tp + fp, 1)


class TestBackgroundSubtraction:
    @pytest.mark.parametrize(
        "backend", ["numpy", pytest.param("opencv", marks=[pytest.mark.opencv, requires_opencv])]
    )
    def test_finds_moving_objects(self, backend: str) -> None:
        scene = SyntheticScene(two_object_scene(num_frames=90, noise=2.0))
        det = BackgroundSubtractionDetector(backend=backend)
        recall, precision = recall_on_scene(det, scene, start=40)
        assert recall > 0.9 and precision > 0.9

    def test_ghost_from_short_warmup_is_absorbed(self) -> None:
        # With a short warm-up the median still contains the objects, which leaves
        # "ghosts" at their start positions until static absorption removes them.
        scene = SyntheticScene(two_object_scene(num_frames=120, noise=2.0))
        det = BackgroundSubtractionDetector(
            backend="numpy", warmup_frames=10, warmup_stride=1, absorb_after=20
        )
        early_dets = []
        for i in range(120):
            dets = det.detect(scene.frame(i))
            if i == 25:
                early_dets = dets
        late = det.detect(scene.frame(119))
        assert len(early_dets) > len(scene.ground_truth(25))
        assert len(late) == len(scene.ground_truth(119))

    @pytest.mark.opencv
    @requires_opencv
    def test_mog2_engine(self) -> None:
        scene = SyntheticScene(two_object_scene(num_frames=80, noise=2.0))
        det = BackgroundSubtractionDetector(engine="opencv_mog2", warmup_frames=20, warmup_stride=1)
        recall, _ = recall_on_scene(det, scene, start=30)
        assert recall > 0.8

    def test_warmup_and_reset(self, scene: SyntheticScene) -> None:
        det = BackgroundSubtractionDetector(backend="numpy", warmup_frames=3, warmup_stride=1)
        assert [det.detect(scene.frame(i)) for i in range(3)] == [[], [], []]
        assert det.detect(scene.frame(10))
        assert det.last_mask is not None and det.last_mask.any()
        det.reset()
        assert det.detect(scene.frame(11)) == []
        assert det.describe()["warmup_frames"] == 3

    def test_model_median_init_rejects_transients(self) -> None:
        model = RunningGaussianModel()
        frames = [np.full((4, 4), 50, np.uint8) for _ in range(5)]
        frames[2] = np.full((4, 4), 250, np.uint8)
        model.initialize(frames)
        assert model.mean is not None and np.allclose(model.mean, 50)
        fg = model.apply(np.full((4, 4), 200, np.uint8))
        assert fg.all()
        with pytest.raises(InvalidInputError):
            model.initialize([])

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"engine": "mog9"},
            {"min_area": 0},
            {"max_area_ratio": 2},
            {"warmup_frames": 1000},
            {"warmup_stride": 0},
        ],
    )
    def test_validation(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(InvalidInputError):
            BackgroundSubtractionDetector(**kwargs)

    def test_model_validation(self) -> None:
        with pytest.raises(InvalidInputError):
            RunningGaussianModel(learning_rate=0)
        with pytest.raises(InvalidInputError):
            RunningGaussianModel(foreground_rate_factor=2)
        with pytest.raises(InvalidInputError):
            RunningGaussianModel(absorb_after=-1)


class TestColorDetector:
    def test_detects_each_colour_with_label(self, backend: Any, scene: SyntheticScene) -> None:
        ranges = [
            range_for_rgb("red", (230, 60, 60)),
            range_for_rgb("green", (60, 200, 80), class_id=1),
        ]
        det = ColorDetector(ranges, backend=backend.name)
        dets = det.detect(scene.frame(5))
        by_label = {d.label: d for d in dets}
        assert set(by_label) == {"red", "green"}
        gt = {g.object_id: g.bbox for g in scene.ground_truth(5)}
        assert by_label["red"].bbox.iou(gt[1]) > 0.9
        assert by_label["green"].class_id == 1

    def test_hue_wraparound(self) -> None:
        rng = ColorRange("red", hue_min=170, hue_max=10)
        hsv = np.array([[[175, 200, 200], [5, 200, 200], [90, 200, 200]]], np.uint8)
        assert rng.mask(hsv).tolist() == [[True, True, False]]

    def test_validation(self) -> None:
        with pytest.raises(InvalidInputError):
            ColorRange("x", hue_min=0, hue_max=200)
        with pytest.raises(InvalidInputError):
            ColorRange("x", hue_min=0, hue_max=10, sat_min=200, sat_max=100)
        with pytest.raises(InvalidInputError):
            ColorDetector([])


class TestTemplate:
    def test_ncc_matches_known_location(self, rng: np.random.Generator) -> None:
        img = rng.integers(0, 256, (60, 80)).astype(np.uint8)
        tpl = img[20:32, 30:45].copy()
        resp = match_template_ncc(img, tpl)
        assert resp.shape == (49, 66)
        y, x = np.unravel_index(np.argmax(resp), resp.shape)
        assert (y, x) == (20, 30) and resp[y, x] == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.opencv
    @requires_opencv
    def test_ncc_matches_opencv(self, rng: np.random.Generator) -> None:
        import cv2

        img = rng.integers(0, 256, (50, 70)).astype(np.uint8)
        tpl = img[5:20, 10:30].copy()
        ours = match_template_ncc(img, tpl)
        theirs = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
        assert np.abs(ours - theirs).max() < 1e-3

    def test_detector_finds_all_instances(self, backend: Any) -> None:
        img = np.full((80, 120, 3), 30, np.uint8)
        tpl = np.zeros((10, 10, 3), np.uint8)
        tpl[2:8, 2:8] = 220
        tpl[4:6, :] = 120
        for x, y in ((10, 10), (70, 50)):
            img[y : y + 10, x : x + 10] = tpl
        det = TemplateDetector(tpl, threshold=0.9, backend=backend.name, label="logo")
        dets = det.detect(Frame(img, 0, 0.0))
        assert sorted((round(d.bbox.x1), round(d.bbox.y1)) for d in dets) == [(10, 10), (70, 50)]
        assert all(d.label == "logo" for d in dets)

    def test_validation(self) -> None:
        with pytest.raises(InvalidInputError):
            TemplateDetector(np.zeros((2, 2), np.uint8))
        with pytest.raises(InvalidInputError):
            match_template_ncc(np.zeros((5, 5)), np.ones((3, 3)))
        with pytest.raises(InvalidInputError):
            match_template_ncc(np.zeros((5, 5)), np.ones((6, 6)))
        with pytest.raises(InvalidInputError):
            TemplateDetector(np.eye(5, dtype=np.uint8) * 200, scales=(0,))


class TestNms:
    def test_suppresses_overlaps_keeps_best(self) -> None:
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]])
        keep = nms(boxes, np.array([0.5, 0.9, 0.8]), 0.5)
        assert keep.tolist() == [1, 2]

    def test_class_aware(self) -> None:
        boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]])
        assert sorted(batched_nms(boxes, np.array([0.9, 0.8]), np.array([0, 1]), 0.5).tolist()) == [
            0,
            1,
        ]
        dets = [Detection(BBox(0, 0, 10, 10), 0.9, "a"), Detection(BBox(0, 0, 10, 10), 0.8, "a", 0)]
        assert len(suppress(dets, 0.5)) == 1
        assert suppress([], 0.5) == []

    def test_validation(self) -> None:
        with pytest.raises(InvalidInputError):
            nms(np.zeros((1, 4)), np.zeros(2), 0.5)
        with pytest.raises(InvalidInputError):
            nms(np.zeros((1, 4)), np.zeros(1), 1.5)
        assert nms(np.zeros((0, 4)), np.zeros(0), 0.5).size == 0


class FakeSession:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output
        self.feeds: list[dict[str, np.ndarray]] = []

    def get_inputs(self) -> list[Any]:
        return [SimpleNamespace(name="images", shape=[1, 3, 64, 64])]

    def run(self, output_names: Any, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.feeds.append(feed)
        return [self.output]


class TestOnnxAdapter:
    def test_letterbox_and_inverse(self) -> None:
        img = np.full((50, 100, 3), 255, np.uint8)
        canvas, ratio, pad = letterbox(img, (64, 64), backend="numpy")
        assert canvas.shape == (64, 64, 3) and ratio == pytest.approx(0.64)
        assert pad == (0.0, 16.0)
        assert canvas[0, 0, 0] == 114 and canvas[32, 32, 0] == 255
        back = unletterbox(np.array([[0.0, 16.0, 64.0, 48.0]]), ratio, pad, (100, 50))
        assert np.allclose(back, [[0, 0, 100, 50]])
        assert to_nchw(canvas).shape == (1, 3, 64, 64)

    def test_decode_yolov5_and_v8_layouts(self) -> None:
        v5 = np.array([[[32, 32, 10, 10, 0.9, 0.1, 0.8], [5, 5, 2, 2, 0.1, 0.9, 0.1]]], np.float32)
        boxes, scores, classes = decode_predictions(v5, 0.5, "yolov5")
        assert boxes.tolist() == [[27, 27, 37, 37]] and classes.tolist() == [1]
        assert scores[0] == pytest.approx(0.72)
        v8 = np.array([[[32, 5], [32, 5], [10, 2], [10, 2], [0.2, 0.1], [0.7, 0.2]]], np.float32)
        boxes, scores, classes = decode_predictions(v8, 0.5, "yolov8")
        assert classes.tolist() == [1] and scores[0] == pytest.approx(0.7)
        # "auto" infers the layout from the realistic case of many more anchors than channels.
        wide = np.zeros((1, 6, 40), np.float32)
        wide[0, :, 3] = [20, 20, 8, 8, 0.1, 0.95]
        _, scores, classes = decode_predictions(wide, 0.5)
        assert classes.tolist() == [1] and scores.tolist() == pytest.approx([0.95])
        with pytest.raises(ModelError):
            decode_predictions(np.zeros((2, 3, 7)), 0.5)
        with pytest.raises(ModelError):
            decode_predictions(np.zeros(5), 0.5)

    def test_detector_end_to_end_with_fake_session(self) -> None:
        out = np.array(
            [
                [
                    [32, 32, 20, 20, 0.95, 0.9, 0.1],
                    [33, 33, 20, 20, 0.9, 0.9, 0.1],
                    [10, 10, 4, 4, 0.2, 0.5, 0.5],
                ]
            ],
            np.float32,
        )
        session = FakeSession(out)
        det = YoloOnnxDetector(
            session=session,
            input_size=(64, 64),
            class_names=["person", "car"],
            output_format="yolov5",
            backend="numpy",
        )
        dets = det.detect(Frame(np.zeros((32, 64, 3), np.uint8), 0, 0.0))
        assert len(dets) == 1 and dets[0].label == "person"
        assert (
            dets[0].bbox.iou(BBox(22, 6, 42, 26)) > 0.9
        )  # mapped back from the letterboxed canvas
        assert session.feeds[0]["images"].shape == (1, 3, 64, 64)
        assert det.describe()["classes"] == 2

    def test_errors(self, tmp_path: Path) -> None:
        with pytest.raises(InvalidInputError):
            YoloOnnxDetector()
        with pytest.raises((BackendUnavailableError, ModelError)):
            YoloOnnxDetector(model_path=tmp_path / "missing.onnx")

        class Broken(FakeSession):
            def run(self, output_names: Any, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
                raise RuntimeError("bad shape")

        det = YoloOnnxDetector(session=Broken(np.zeros(1)), input_size=(32, 32), backend="numpy")
        with pytest.raises(ModelError, match="inference failed"):
            det.detect(Frame(np.zeros((8, 8, 3), np.uint8), 0, 0.0))


class TestMotionDetector:
    def test_detects_moving_object(self, backend: Any, scene: SyntheticScene) -> None:
        det = FrameDifferenceDetector(backend=backend.name)
        assert det.detect(scene.frame(0)) == [] and det.detect(scene.frame(1)) == []
        dets = det.detect(scene.frame(2))
        assert len(dets) >= 2 and det.last_energy > 0
        det.reset()
        assert det.last_energy == 0
        assert det.describe()["mode"] == "three_frame"


class TestSpecs:
    def test_build_all_kinds(self, tmp_path: Path) -> None:
        assert isinstance(
            build_detector(BackgroundDetectorSpec(), backend="numpy"), BackgroundSubtractionDetector
        )
        color = ColorDetectorSpec.model_validate(
            {"ranges": [{"label": "r", "hue_min": 0, "hue_max": 10}]}
        )
        assert isinstance(build_detector(color, backend="numpy"), ColorDetector)
        assert isinstance(
            build_detector(MotionDetectorSpec(), backend="numpy"), FrameDifferenceDetector
        )
        tpl = np.eye(8, dtype=np.uint8) * 200
        save_image(tmp_path / "tpl.png", tpl)
        by_path = TemplateDetectorSpec(template_path="tpl.png")
        assert isinstance(build_detector(by_path, media_roots=[tmp_path]), TemplateDetector)
        b64 = base64.b64encode(encode_image(tpl, "png")).decode()
        assert isinstance(
            build_detector(TemplateDetectorSpec(template_png_b64=b64)), TemplateDetector
        )

    def test_template_and_onnx_validation(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            TemplateDetectorSpec()
        with pytest.raises(InvalidInputError):
            build_detector(TemplateDetectorSpec(template_png_b64="!!notbase64"))
        with pytest.raises(InvalidInputError):
            build_detector(OnnxDetectorSpec(model="yolo"))

    def test_onnx_spec_uses_registry_resolver(self, tmp_path: Path) -> None:
        calls: list[tuple[str, str | None]] = []

        def resolver(name: str, version: str | None) -> tuple[Path, dict[str, Any]]:
            calls.append((name, version))
            return tmp_path / "missing.onnx", {"class_names": ["a"]}

        with pytest.raises((BackendUnavailableError, ModelError)):
            build_detector(OnnxDetectorSpec(model="yolo", version="1.0.0"), model_resolver=resolver)
        assert calls == [("yolo", "1.0.0")]


def test_template_multi_scale_finds_resized_instance() -> None:
    tpl = np.zeros((16, 16, 3), np.uint8)
    tpl[3:13, 3:13] = 200
    tpl[6:10, :] = 90
    img = np.full((80, 80, 3), 30, np.uint8)
    big = np.kron(tpl, np.ones((2, 2, 1), np.uint8))  # 32x32 instance
    img[20:52, 30:62] = big
    single = TemplateDetector(tpl, threshold=0.85, backend="numpy")
    multi = TemplateDetector(tpl, threshold=0.85, scales=(1.0, 2.0), backend="numpy")
    assert single.detect(Frame(img, 0, 0.0)) == []
    found = multi.detect(Frame(img, 0, 0.0))
    assert len(found) == 1 and found[0].bbox.iou(BBox(30, 20, 62, 52)) > 0.9
