from __future__ import annotations

import itertools

import numpy as np
import pytest

from tests.helpers import requires_scipy, two_object_scene
from vap.detection.base import Detection
from vap.errors import InvalidInputError
from vap.evaluation.tracking import evaluate_mot
from vap.geometry import BBox
from vap.sources.synthetic import SyntheticScene
from vap.tracking.assignment import greedy, hungarian, match, solve
from vap.tracking.base import TrackerConfig, TrackState
from vap.tracking.kalman import KalmanBoxModel, KalmanFilter, bbox_to_z, x_to_bbox
from vap.tracking.trackers import IoUTracker, SortTracker, TrackerSpec, build_tracker


def brute_force(cost: np.ndarray) -> float:
    n, m = cost.shape
    if n <= m:
        return min(
            sum(cost[i, p[i]] for i in range(n)) for p in itertools.permutations(range(m), n)
        )
    return brute_force(cost.T)


class TestAssignment:
    @pytest.mark.parametrize("shape", [(1, 1), (3, 3), (2, 5), (5, 2), (4, 4)])
    def test_hungarian_is_optimal(self, rng: np.random.Generator, shape: tuple[int, int]) -> None:
        for _ in range(10):
            cost = rng.random(shape) * 10
            rows, cols = hungarian(cost)
            assert len(rows) == min(shape)
            assert len(set(cols.tolist())) == len(cols)
            assert cost[rows, cols].sum() == pytest.approx(brute_force(cost))

    def test_hungarian_handles_ties_and_negative(self) -> None:
        cost = np.array([[-1.0, -1.0], [-1.0, -1.0]])
        _, cols = hungarian(cost)
        assert sorted(cols.tolist()) == [0, 1]
        assert hungarian(np.zeros((0, 3)))[0].size == 0
        with pytest.raises(InvalidInputError):
            hungarian(np.array([[np.inf]]))
        with pytest.raises(InvalidInputError):
            hungarian(np.zeros(3))

    @requires_scipy
    def test_matches_scipy(self, rng: np.random.Generator) -> None:
        for _ in range(20):
            cost = rng.random((6, 8))
            r1, c1 = solve(cost, "hungarian")
            r2, c2 = solve(cost, "scipy")
            assert cost[r1, c1].sum() == pytest.approx(cost[r2, c2].sum())

    def test_greedy_is_valid_but_not_always_optimal(self) -> None:
        cost = np.array([[1.0, 2.0], [2.0, 100.0]])
        rows, cols = greedy(cost)
        assert cost[rows, cols].sum() == 101.0
        assert cost[hungarian(cost)].sum() == 4.0
        with pytest.raises(InvalidInputError):
            solve(cost, "magic")  # type: ignore[arg-type]

    def test_match_gates_expensive_pairs(self) -> None:
        cost = np.array([[0.1, 0.9], [0.95, 0.99]])
        matches, rows, cols = match(cost, 0.5)
        assert matches == [(0, 0)] and rows == [1] and cols == [1]
        assert match(np.zeros((0, 2)), 0.5) == ([], [], [0, 1])
        inf_cost = np.array([[np.inf, 0.2]])
        assert match(inf_cost, 0.5)[0] == [(0, 1)]


class TestKalman:
    def test_constant_velocity_converges(self) -> None:
        model = KalmanBoxModel(BBox(0, 0, 10, 10))
        for t in range(1, 30):
            model.predict()
            model.update(BBox(3 * t, 0, 3 * t + 10, 10))
        vx, vy = model.velocity
        assert vx == pytest.approx(3.0, abs=0.2) and abs(vy) < 0.2
        predicted = model.predict()
        assert predicted.center[0] == pytest.approx(3 * 30 + 5, abs=1.0)

    def test_state_conversion_roundtrip(self) -> None:
        box = BBox(10, 20, 40, 80)
        back = x_to_bbox(np.concatenate([bbox_to_z(box), np.zeros(3)]))
        assert np.allclose(back.to_list(), box.to_list())

    def test_filter_shape_validation_and_mahalanobis(self) -> None:
        eye = np.eye(2)
        kf = KalmanFilter(eye, eye, eye * 0.1, eye, np.zeros(2), eye)
        assert kf.mahalanobis(np.array([0.0, 0.0])) == 0.0
        assert kf.mahalanobis(np.array([3.0, 0.0])) > kf.mahalanobis(np.array([1.0, 0.0]))
        with pytest.raises(InvalidInputError):
            KalmanFilter(eye, np.eye(3), eye, eye, np.zeros(2), eye)


def det(x: float, y: float, w: float = 10, h: float = 10, label: str = "object") -> Detection:
    return Detection(BBox(x, y, x + w, y + h), 0.9, label)


class TestLifecycle:
    @pytest.mark.parametrize("cls", [IoUTracker, SortTracker])
    def test_confirmation_loss_and_deletion(self, cls: type) -> None:
        tracker = cls(TrackerConfig(min_hits=3, max_age=2))
        out = [tracker.update([det(2 * i, 0)], i, i * 0.1) for i in range(3)]
        assert out[0] == [] and out[1] == [] and len(out[2]) == 1
        track = out[2][0]
        assert track.state == TrackState.CONFIRMED and track.track_id == 1
        tracker.update([], 3, 0.3)
        assert track.state == TrackState.LOST and track.misses == 1
        tracker.update([], 4, 0.4)
        tracker.update([], 5, 0.5)
        finished = tracker.pop_finished()
        assert [t.track_id for t in finished] == [1] and finished[0].state == TrackState.DELETED
        assert tracker.tracks == []

    def test_tentative_tracks_die_on_first_miss(self) -> None:
        tracker = IoUTracker(TrackerConfig(min_hits=3))
        tracker.update([det(0, 0)], 0, 0.0)
        tracker.update([], 1, 0.1)
        assert tracker.tracks == [] and len(tracker.pop_finished()) == 1

    def test_sort_bridges_short_occlusion(self) -> None:
        tracker = SortTracker(TrackerConfig(min_hits=2, max_age=5, iou_threshold=0.2))
        for i in range(10):
            tracker.update([det(4 * i, 0, 20, 20)], i, i * 0.1)
        for i in range(10, 13):  # occluded
            tracker.update([], i, i * 0.1)
        out = tracker.update([det(4 * 13, 0, 20, 20)], 13, 1.3)
        assert [t.track_id for t in out] == [1]

    def test_class_aware_does_not_mix_labels(self) -> None:
        tracker = IoUTracker(TrackerConfig(min_hits=1, class_aware=True))
        tracker.update([det(0, 0, label="car")], 0, 0.0)
        out = tracker.update([det(1, 0, label="person")], 1, 0.1)
        assert [t.label for t in out] == ["person"] and out[0].track_id == 2

    def test_velocity_and_flush(self) -> None:
        tracker = IoUTracker(TrackerConfig(min_hits=1))
        for i in range(6):
            tracker.update([det(5 * i, 0)], i, i * 0.5)
        track = tracker.confirmed_tracks()[0]
        assert track.velocity()[0] == pytest.approx(10.0)
        assert track.to_dict()["velocity"] == [10.0, 0.0]
        assert len(tracker.flush()) == 1 and tracker.tracks == []
        tracker.reset()
        assert tracker.frame_count == 0

    def test_config_and_spec_validation(self) -> None:
        with pytest.raises(InvalidInputError):
            TrackerConfig(iou_threshold=0)
        with pytest.raises(InvalidInputError):
            TrackerConfig(min_hits=0)
        assert isinstance(build_tracker({"kind": "iou"}), IoUTracker)
        assert isinstance(build_tracker(None), SortTracker)
        with pytest.raises(ValueError, match="kind"):
            TrackerSpec.model_validate({"kind": "deepsort"})


@pytest.mark.parametrize("kind", ["iou", "sort"])
def test_perfect_detections_give_perfect_tracking(kind: str) -> None:
    scene = SyntheticScene(two_object_scene(num_frames=60))
    tracker = build_tracker({"kind": kind, "min_hits": 1})
    gt_frames: dict[int, list[tuple[int, BBox]]] = {}
    hyp_frames: dict[int, list[tuple[int, BBox]]] = {}
    for i in range(60):
        gts = scene.ground_truth(i)
        dets = [Detection(g.bbox, 1.0) for g in gts]
        tracks = tracker.update(dets, i, i / 10)
        gt_frames[i] = [(g.object_id, g.bbox) for g in gts]
        hyp_frames[i] = [(t.track_id, t.bbox) for t in tracks]
    report = evaluate_mot(gt_frames, hyp_frames)
    assert report.mota == pytest.approx(1.0)
    assert report.idf1 == pytest.approx(1.0)
    assert report.id_switches == 0
