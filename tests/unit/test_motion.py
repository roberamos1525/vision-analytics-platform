from __future__ import annotations

import numpy as np
import pytest

from tests.helpers import requires_opencv
from vap.errors import InvalidInputError
from vap.motion.differencing import MotionAnalyzer, frame_difference, three_frame_difference
from vap.motion.flow import block_matching_flow, farneback_flow, heading_degrees
from vap.motion.heatmap import MotionHeatmap, apply_colormap
from vap.motion.kinematics import compass_direction, estimate_kinematics, path_length
from vap.tracking.base import TrackPoint


def textured(seed: int = 0, shape: tuple[int, int] = (64, 80)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (shape[0] // 4, shape[1] // 4), dtype=np.uint8)
    return np.kron(base, np.ones((4, 4), np.uint8))


class TestDifferencing:
    def test_two_and_three_frame(self) -> None:
        f = [np.zeros((20, 30), np.uint8) for _ in range(3)]
        for i, img in enumerate(f):
            img[5:10, 5 + 6 * i : 11 + 6 * i] = 200
        two = frame_difference(f[0], f[1])
        three = three_frame_difference(*f)
        assert two.sum() > three.sum() > 0
        # Three-frame differencing leaves no ghost at the first position.
        assert not three[5:10, 5:11].any()
        with pytest.raises(InvalidInputError):
            frame_difference(np.zeros((2, 2), np.uint8), np.zeros((3, 3), np.uint8))

    def test_analyzer_energy_and_reset(self, backend: object) -> None:
        analyzer = MotionAnalyzer(
            mode="two_frame", blur_sigma=0, morph_ksize=0, backend=backend.name
        )  # type: ignore[attr-defined]
        a = np.zeros((10, 10), np.uint8)
        b = a.copy()
        b[0:5, 0:2] = 255
        assert not analyzer.update(a).any()
        mask = analyzer.update(b)
        assert MotionAnalyzer.energy(mask) == pytest.approx(0.1)
        analyzer.reset()
        assert not analyzer.update(b).any()
        with pytest.raises(InvalidInputError):
            MotionAnalyzer(mode="optical")


class TestHeatmap:
    def test_points_mask_decay_and_render(self) -> None:
        hm = MotionHeatmap(40, 60, decay=0.5, splat_sigma=2.0)
        hm.add_points([(30, 20), (-100, 5)])
        peak = np.unravel_index(np.argmax(hm.values), hm.values.shape)
        assert peak == (20, 30)
        before = float(hm.values.max())
        hm.add_mask(np.zeros((40, 60), bool))
        assert float(hm.values.max()) == pytest.approx(before * 0.5)
        rgb = hm.render()
        assert rgb.shape == (40, 60, 3) and rgb.dtype == np.uint8
        hot = hm.hotspots(2)
        assert hot[0]["row"] == 1 and hot[0]["col"] == 1
        hm.reset()
        assert hm.updates == 0 and not hm.values.any()

    def test_overlay_leaves_cold_pixels(self) -> None:
        hm = MotionHeatmap(20, 20)
        img = np.full((20, 20, 3), 50, np.uint8)
        hm.add_points([(10, 10)])
        out = hm.overlay(img, 0.5)
        assert np.array_equal(out[0, 0], img[0, 0]) and not np.array_equal(out[10, 10], img[10, 10])
        with pytest.raises(InvalidInputError):
            hm.add_mask(np.zeros((5, 5), bool))
        with pytest.raises(InvalidInputError):
            hm.overlay(img, 2.0)

    def test_colormap_endpoints(self) -> None:
        cm = apply_colormap(np.array([0.0, 1.0]))
        assert cm[0, 2] > cm[0, 0] and cm[1, 0] > cm[1, 2]

    @pytest.mark.parametrize(
        "kwargs", [{"height": 0, "width": 5}, {"height": 5, "width": 5, "decay": 0}]
    )
    def test_validation(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(InvalidInputError):
            MotionHeatmap(**kwargs)  # type: ignore[arg-type]


class TestFlow:
    @pytest.mark.parametrize(("dx", "dy"), [(3, 0), (0, -2), (2, 2), (-4, 1)])
    def test_block_matching_recovers_global_shift(self, dx: int, dy: int) -> None:
        a = textured()
        b = np.roll(a, shift=(dy, dx), axis=(0, 1))
        field = block_matching_flow(a, b, block=8, search=6)
        inner = (slice(1, -1), slice(1, -1))
        assert np.median(field.dx[inner]) == dx and np.median(field.dy[inner]) == dy
        summary = field.summary()
        assert summary.direction_deg == pytest.approx(heading_degrees(dx, dy), abs=10)

    def test_static_scene_has_zero_flow(self) -> None:
        a = textured(3)
        field = block_matching_flow(a, a)
        assert not field.dx.any() and not field.dy.any()
        s = field.summary()
        assert s.direction_deg is None and s.moving_fraction == 0
        assert s.to_dict()["valid_blocks"] == field.valid.sum()

    def test_flat_blocks_are_invalid(self) -> None:
        field = block_matching_flow(np.zeros((32, 32), np.uint8), np.zeros((32, 32), np.uint8))
        assert not field.valid.any()
        with pytest.raises(InvalidInputError):
            block_matching_flow(np.zeros((4, 4), np.uint8), np.zeros((4, 4), np.uint8))
        with pytest.raises(InvalidInputError):
            block_matching_flow(
                np.zeros((32, 32), np.uint8), np.zeros((32, 32), np.uint8), search=40
            )

    @pytest.mark.opencv
    @requires_opencv
    def test_farneback_direction(self) -> None:
        a = textured(5, (96, 96))
        b = np.roll(a, shift=(0, 2), axis=(0, 1))
        s = farneback_flow(a, b).summary()
        assert s.direction_deg is not None
        assert min(s.direction_deg, 360 - s.direction_deg) < 20


def points(xs: list[float], ys: list[float], dt: float = 0.1) -> list[TrackPoint]:
    return [TrackPoint(i, i * dt, x, y) for i, (x, y) in enumerate(zip(xs, ys, strict=True))]


class TestKinematics:
    def test_speed_heading_and_calibration(self) -> None:
        pts = points([0, 1, 2, 3, 4], [0, -1, -2, -3, -4])
        k = estimate_kinematics(pts, pixels_per_meter=10)
        assert k.speed_px_s == pytest.approx(np.hypot(10, 10))
        assert k.heading_deg == pytest.approx(45)
        assert k.direction == "NE"
        assert k.speed_m_s == pytest.approx(np.hypot(10, 10) / 10)
        assert k.to_dict()["speed_km_h"] == pytest.approx(round(k.speed_m_s * 3.6, 3))

    def test_jitter_is_smoothed_by_least_squares(self) -> None:
        xs = [i * 2 + (0.8 if i % 2 else -0.8) for i in range(10)]
        k = estimate_kinematics(points(xs, [0] * 10, dt=1.0))
        assert k.vx_px_s == pytest.approx(2.0, abs=0.3)

    def test_stationary_and_validation(self) -> None:
        assert estimate_kinematics(points([5, 5, 5], [5, 5, 5])).heading_deg is None
        assert estimate_kinematics(points([1], [1])).speed_px_s == 0
        assert (
            estimate_kinematics([TrackPoint(0, 1.0, 0, 0), TrackPoint(1, 1.0, 5, 0)]).speed_px_s
            == 0
        )
        with pytest.raises(InvalidInputError):
            estimate_kinematics(points([0, 1], [0, 0]), window=1)
        with pytest.raises(InvalidInputError):
            estimate_kinematics(points([0, 1], [0, 0]), pixels_per_meter=0)

    @pytest.mark.parametrize(
        ("heading", "label"),
        [(0, "E"), (90, "N"), (180, "W"), (270, "S"), (350, "E"), (135, "NW"), (300, "SE")],
    )
    def test_compass(self, heading: float, label: str) -> None:
        assert compass_direction(heading) == label

    def test_path_length(self) -> None:
        assert path_length(points([0, 3, 3], [0, 4, 0])) == pytest.approx(9)
        assert path_length([]) == 0
