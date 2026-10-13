"""The pure-numpy backend must agree with OpenCV (skipped when OpenCV is absent)."""

from __future__ import annotations

import numpy as np
import pytest

from tests.helpers import requires_opencv
from vap.imaging.backend import get_backend
from vap.imaging.frame import ColorSpace
from vap.sources.synthetic import SyntheticScene, random_scene

pytestmark = [pytest.mark.opencv, requires_opencv]


@pytest.fixture(scope="module")
def image() -> np.ndarray:
    return SyntheticScene(random_scene(seed=11, num_objects=5, noise_std=3.0)).render(7)


def max_diff(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.abs(a.astype(np.int32) - b.astype(np.int32)).max())


@pytest.mark.parametrize("interp", ["nearest", "bilinear", "area"])
@pytest.mark.parametrize("size", [(160, 120), (97, 61), (400, 300)])
def test_resize(image: np.ndarray, interp: str, size: tuple[int, int]) -> None:
    np_out = get_backend("numpy").resize(image, size, interp)  # type: ignore[arg-type]
    cv_out = get_backend("opencv").resize(image, size, interp)  # type: ignore[arg-type]
    assert np_out.shape == cv_out.shape
    if interp == "nearest":
        # Nearest-neighbour sampling conventions differ by at most one source pixel.
        assert np.mean(np_out == cv_out) > 0.9
    elif interp == "area" and size[0] > image.shape[1]:
        # Upscaling with "area" is bilinear in numpy; OpenCV uses its own variant.
        assert np.abs(np_out.astype(int) - cv_out).mean() < 2.0
    else:
        assert max_diff(np_out, cv_out) <= 2


@pytest.mark.parametrize(
    ("src", "dst"),
    [
        (ColorSpace.RGB, ColorSpace.GRAY),
        (ColorSpace.RGB, ColorSpace.HSV),
        (ColorSpace.BGR, ColorSpace.GRAY),
    ],
)
def test_color(image: np.ndarray, src: ColorSpace, dst: ColorSpace) -> None:
    a = get_backend("numpy").convert_color(image, src, dst)
    b = get_backend("opencv").convert_color(image, src, dst)
    assert max_diff(a, b) <= 1


def test_blur_and_sobel(image: np.ndarray) -> None:
    np_b, cv_b = get_backend("numpy"), get_backend("opencv")
    assert max_diff(np_b.gaussian_blur(image, 1.5), cv_b.gaussian_blur(image, 1.5)) <= 1
    assert max_diff(np_b.box_blur(image, 5), cv_b.box_blur(image, 5)) <= 1
    gray = cv_b.to_gray(image)
    for a, b in zip(np_b.sobel(gray), cv_b.sobel(gray), strict=True):
        assert np.allclose(a, b, atol=1e-3)


def test_canny_overlap(image: np.ndarray) -> None:
    gray = get_backend("opencv").to_gray(image)
    a = get_backend("numpy").canny(gray, 40, 100)
    b = get_backend("opencv").canny(gray, 40, 100)
    overlap = (a & b).sum() / max(a.sum(), b.sum())
    assert overlap > 0.9


@pytest.mark.parametrize("op", ["erode", "dilate", "open", "close", "gradient"])
@pytest.mark.parametrize("shape", ["rect", "ellipse", "cross"])
def test_morphology_identical(image: np.ndarray, op: str, shape: str) -> None:
    mask = get_backend("numpy").to_gray(image) > 60
    a = get_backend("numpy").morphology(mask, op, 5, shape)  # type: ignore[arg-type]
    b = get_backend("opencv").morphology(mask, op, 5, shape)  # type: ignore[arg-type]
    assert np.array_equal(a, b)


def test_components_identical(image: np.ndarray) -> None:
    mask = get_backend("numpy").to_gray(image) > 60
    a = get_backend("numpy").connected_components(mask, 8, 1)
    b = get_backend("opencv").connected_components(mask, 8, 1)
    assert sorted((x.area, x.bbox.to_list()) for x in a) == sorted(
        (x.area, x.bbox.to_list()) for x in b
    )
