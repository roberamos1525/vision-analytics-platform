from __future__ import annotations

import math

import numpy as np
import pytest

from vap.errors import InvalidInputError
from vap.geometry import BBox, LineSegment, Polygon, boxes_to_array, iou_matrix


class TestBBox:
    def test_basic_properties(self) -> None:
        b = BBox(10, 20, 40, 60)
        assert b.width == 30
        assert b.height == 40
        assert b.area == 1200
        assert b.center == (25, 40)
        assert b.bottom_center == (25, 60)
        assert b.to_xywh() == (10, 20, 30, 40)

    def test_constructors_round_trip(self) -> None:
        assert BBox.from_xywh(1, 2, 3, 4) == BBox(1, 2, 4, 6)
        assert BBox.from_cxcywh(5, 5, 4, 2) == BBox(3, 4, 7, 6)
        assert BBox.from_array(np.array([1, 2, 3, 4])) == BBox(1, 2, 3, 4)

    @pytest.mark.parametrize(
        "coords",
        [(5, 0, 4, 1), (0, 5, 1, 4), (0, 0, math.inf, 1), (math.nan, 0, 1, 1)],
    )
    def test_rejects_invalid(self, coords: tuple[float, ...]) -> None:
        with pytest.raises(InvalidInputError):
            BBox(*coords)

    def test_from_array_requires_four_values(self) -> None:
        with pytest.raises(InvalidInputError):
            BBox.from_array([1, 2, 3])

    def test_iou(self) -> None:
        a = BBox(0, 0, 10, 10)
        assert a.iou(a) == pytest.approx(1.0)
        assert a.iou(BBox(5, 0, 15, 10)) == pytest.approx(50 / 150)
        assert a.iou(BBox(20, 20, 30, 30)) == 0.0
        assert BBox(0, 0, 0, 0).iou(BBox(0, 0, 0, 0)) == 0.0

    def test_clip_scale_translate(self) -> None:
        b = BBox(-5, -5, 50, 50).clip(20, 30)
        assert b == BBox(0, 0, 20, 30)
        assert BBox(1, 2, 3, 4).scale(2) == BBox(2, 4, 6, 8)
        assert BBox(1, 2, 3, 4).scale(2, 3) == BBox(2, 6, 6, 12)
        assert BBox(1, 2, 3, 4).translate(1, -1) == BBox(2, 1, 4, 3)
        assert BBox(0, 0, 10, 10).contains_point(10, 10)
        assert not BBox(0, 0, 10, 10).contains_point(10.1, 5)


def test_iou_matrix_matches_pairwise(rng: np.random.Generator) -> None:
    xy = rng.uniform(0, 100, (6, 2))
    wh = rng.uniform(1, 40, (6, 2))
    boxes = [BBox(x, y, x + w, y + h) for (x, y), (w, h) in zip(xy, wh, strict=True)]
    mat = iou_matrix(boxes_to_array(boxes[:3]), boxes_to_array(boxes[3:]))
    for i in range(3):
        for j in range(3):
            assert mat[i, j] == pytest.approx(boxes[i].iou(boxes[3 + j]))


def test_iou_matrix_empty_shapes() -> None:
    assert iou_matrix(np.zeros((0, 4)), np.zeros((3, 4))).shape == (0, 3)
    assert boxes_to_array([]).shape == (0, 4)


class TestPolygon:
    square = Polygon.from_points([[0, 0], [10, 0], [10, 10], [0, 10]])

    def test_area_and_bbox(self) -> None:
        assert self.square.area == pytest.approx(100)
        assert self.square.bbox == BBox(0, 0, 10, 10)

    def test_contains(self) -> None:
        assert self.square.contains(5, 5)
        assert not self.square.contains(15, 5)
        tri = Polygon.from_points([[0, 0], [10, 0], [0, 10]])
        flags = tri.contains_many(np.array([[1, 1], [9, 9], [2, 7]]))
        assert flags.tolist() == [True, False, True]

    def test_concave_polygon(self) -> None:
        # "U" shape: the notch at the top middle is outside.
        u = Polygon.from_points(
            [[0, 0], [3, 0], [3, 8], [7, 8], [7, 0], [10, 0], [10, 10], [0, 10]]
        )
        assert u.contains(1, 5)
        assert not u.contains(5, 4)
        assert u.contains(5, 9)

    def test_mask_rasterises_pixel_centres(self) -> None:
        mask = Polygon.from_points([[2, 2], [6, 2], [6, 5], [2, 5]]).mask(8, 8)
        assert mask.sum() == 4 * 3
        assert mask[2:5, 2:6].all()

    @pytest.mark.parametrize(
        "points",
        [
            [[0, 0], [1, 1]],
            [[0, 0], [1, 1], [2, 2]],
            [[0, 0], [1], [2, 2]],
            [[0, 0], [1, math.inf], [2, 0]],
        ],
    )
    def test_invalid(self, points: list[list[float]]) -> None:
        with pytest.raises(InvalidInputError):
            Polygon.from_points(points)


class TestLineSegment:
    line = LineSegment.from_points([[50, 0], [50, 100]])

    def test_side(self) -> None:
        # Walking down the screen along x=50, the right-hand side is x < 50.
        assert self.line.side((40, 10)) == 1
        assert self.line.side((60, 10)) == -1
        horizontal = LineSegment.from_points([[0, 50], [100, 50]])
        assert horizontal.side((50, 60)) == 1
        assert self.line.side((50, 10)) == 0

    def test_crossing_directions(self) -> None:
        assert self.line.crossing_direction((60, 50), (40, 50)) == 1
        assert self.line.crossing_direction((40, 50), (60, 50)) == -1

    def test_no_crossing_outside_segment_or_touching(self) -> None:
        assert self.line.crossing_direction((40, 150), (60, 150)) == 0
        assert self.line.crossing_direction((40, 50), (50, 50)) == 0
        assert self.line.crossing_direction((40, 50), (45, 60)) == 0

    def test_validation(self) -> None:
        with pytest.raises(InvalidInputError):
            LineSegment.from_points([[0, 0], [0, 0]])
        with pytest.raises(InvalidInputError):
            LineSegment.from_points([[0, 0], [1, 1], [2, 2]])
        assert self.line.length == pytest.approx(100)
        assert self.line.to_list() == [[50, 0], [50, 100]]
