"""Image-processing behaviour, run against every available backend."""

from __future__ import annotations

import numpy as np
import pytest

from vap.errors import BackendUnavailableError, InvalidInputError
from vap.imaging.backend import available_backends, get_backend
from vap.imaging.components import component_stats, find_blobs, label_components
from vap.imaging.frame import ColorSpace, Frame, ensure_gray, ensure_image, ensure_mask
from vap.imaging.ops import ImageBackend, gaussian_kernel_size, structuring_element


def square_mask(size: int = 20, lo: int = 5, hi: int = 15) -> np.ndarray:
    m = np.zeros((size, size), dtype=bool)
    m[lo:hi, lo:hi] = True
    return m


class TestFrameHelpers:
    def test_ensure_image_converts_float_and_squeezes(self) -> None:
        img = ensure_image(np.full((4, 4, 1), 0.5, dtype=np.float32))
        assert img.dtype == np.uint8 and img.shape == (4, 4)
        assert int(img[0, 0]) == 128

    @pytest.mark.parametrize("bad", [np.zeros((4, 4, 4)), np.zeros((0, 3, 3)), np.zeros(5)])
    def test_ensure_image_rejects(self, bad: np.ndarray) -> None:
        with pytest.raises(InvalidInputError):
            ensure_image(bad)

    def test_ensure_image_requires_colour_when_asked(self) -> None:
        with pytest.raises(InvalidInputError):
            ensure_image(np.zeros((4, 4), np.uint8), allow_gray=False)

    def test_ensure_gray_weights(self) -> None:
        img = np.zeros((1, 3, 3), np.uint8)
        img[0, 0] = (255, 0, 0)
        img[0, 1] = (0, 255, 0)
        img[0, 2] = (0, 0, 255)
        assert ensure_gray(img)[0].tolist() == [76, 150, 29]

    def test_frame_properties(self) -> None:
        f = Frame(np.zeros((10, 20, 3), np.uint8), 3, 0.3, "cam")
        assert f.shape == (10, 20) and f.width == 20 and f.height == 10
        g = f.with_image(np.ones((5, 5, 3), np.uint8))
        assert g.index == 3 and g.shape == (5, 5)
        with pytest.raises(InvalidInputError):
            ensure_mask(np.zeros((2, 2, 2)))


class TestBackendSelection:
    def test_numpy_always_available(self) -> None:
        assert "numpy" in available_backends()
        assert get_backend("numpy").name == "numpy"

    def test_auto_prefers_opencv_when_installed(self) -> None:
        expected = "opencv" if "opencv" in available_backends() else "numpy"
        assert get_backend("auto").name == expected

    def test_unknown_backend(self) -> None:
        with pytest.raises(InvalidInputError):
            get_backend("cuda")

    def test_missing_opencv_raises_typed_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins

        from vap.imaging import opencv_backend

        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "cv2":
                raise ImportError("no cv2")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(BackendUnavailableError) as exc:
            opencv_backend.load_cv2()
        assert exc.value.status_code == 501
        assert not opencv_backend.opencv_available()


class TestResize:
    def test_nearest_upscale_replicates(self, backend: ImageBackend) -> None:
        img = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        out = backend.resize(img, (4, 4), "nearest")
        assert out.shape == (4, 4)
        assert out[:2, :2].tolist() == [[0, 0], [0, 0]]
        assert out[2:, :2].min() == 255

    def test_area_integer_downscale_is_block_mean(
        self, backend: ImageBackend, rng: np.random.Generator
    ) -> None:
        img = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
        out = backend.resize(img, (4, 4), "area")
        expected = img.reshape(4, 2, 4, 2, 3).astype(np.float64).mean(axis=(1, 3))
        assert np.abs(out.astype(np.float64) - expected).max() <= 1.0

    def test_bilinear_preserves_constant(self, backend: ImageBackend) -> None:
        img = np.full((13, 17, 3), 77, np.uint8)
        out = backend.resize(img, (31, 7), "bilinear")
        assert out.shape == (7, 31, 3)
        assert np.all(out == 77)

    def test_invalid_arguments(self, backend: ImageBackend) -> None:
        img = np.zeros((4, 4), np.uint8)
        with pytest.raises(InvalidInputError):
            backend.resize(img, (0, 4))
        with pytest.raises(InvalidInputError):
            backend.resize(img, (4, 4), "cubic")  # type: ignore[arg-type]


class TestColor:
    def test_rgb_bgr_roundtrip(self, backend: ImageBackend, rng: np.random.Generator) -> None:
        img = rng.integers(0, 256, (5, 6, 3), dtype=np.uint8)
        bgr = backend.convert_color(img, ColorSpace.RGB, ColorSpace.BGR)
        assert np.array_equal(bgr[..., 0], img[..., 2])
        assert np.array_equal(backend.convert_color(bgr, ColorSpace.BGR, ColorSpace.RGB), img)

    @pytest.mark.parametrize(
        ("rgb", "hsv"),
        [
            ((255, 0, 0), (0, 255, 255)),
            ((0, 255, 0), (60, 255, 255)),
            ((0, 0, 255), (120, 255, 255)),
            ((255, 255, 255), (0, 0, 255)),
            ((0, 0, 0), (0, 0, 0)),
            ((128, 128, 0), (30, 255, 128)),
        ],
    )
    def test_hsv_known_values(
        self, backend: ImageBackend, rgb: tuple[int, int, int], hsv: tuple[int, int, int]
    ) -> None:
        px = np.array(rgb, np.uint8).reshape(1, 1, 3)
        out = backend.convert_color(px, ColorSpace.RGB, ColorSpace.HSV)[0, 0]
        assert np.abs(out.astype(int) - np.array(hsv)).max() <= 1

    def test_hsv_roundtrip(self, backend: ImageBackend, rng: np.random.Generator) -> None:
        img = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
        hsv = backend.convert_color(img, ColorSpace.RGB, ColorSpace.HSV)
        back = backend.convert_color(hsv, ColorSpace.HSV, ColorSpace.RGB)
        # Hue is quantised to 2 degrees, so allow a small error.
        assert np.abs(back.astype(int) - img).max() <= 8

    def test_gray_to_rgb_and_shape_validation(self, backend: ImageBackend) -> None:
        gray = np.full((3, 3), 9, np.uint8)
        rgb = backend.convert_color(gray, ColorSpace.GRAY, ColorSpace.RGB)
        assert rgb.shape == (3, 3, 3) and np.all(rgb == 9)
        with pytest.raises(InvalidInputError):
            backend.convert_color(gray, ColorSpace.RGB, ColorSpace.GRAY)
        assert backend.to_gray(gray) is gray


class TestFilters:
    def test_gaussian_blur_constant_and_smoothing(self, backend: ImageBackend) -> None:
        flat = np.full((20, 20), 100, np.uint8)
        assert np.all(backend.gaussian_blur(flat, 2.0) == 100)
        spike = np.zeros((21, 21), np.uint8)
        spike[10, 10] = 255
        blurred = backend.gaussian_blur(spike, 1.5)
        assert blurred[10, 10] < 255
        assert blurred[10, 10] == blurred.max()
        assert blurred[10, 9] == blurred[10, 11] == blurred[9, 10]

    def test_gaussian_validation(self, backend: ImageBackend) -> None:
        img = np.zeros((5, 5), np.uint8)
        with pytest.raises(InvalidInputError):
            backend.gaussian_blur(img, 0)
        with pytest.raises(InvalidInputError):
            backend.gaussian_blur(img, 1.0, ksize=4)
        assert gaussian_kernel_size(1.0) == 7
        assert gaussian_kernel_size(0.1) == 3

    def test_box_blur(self, backend: ImageBackend) -> None:
        img = np.zeros((9, 9), np.uint8)
        img[4, 4] = 90
        out = backend.box_blur(img, 3)
        assert out[4, 4] == 10 and out[3, 3] == 10 and out[0, 0] == 0

    def test_sobel_on_ramp(self, backend: ImageBackend) -> None:
        ramp = np.tile(np.arange(0, 100, 5, dtype=np.uint8), (10, 1))
        gx, gy = backend.sobel(ramp)
        assert gx.dtype == np.float32
        assert gx[5, 5] == pytest.approx(5 * 2 * 4)
        assert abs(float(gy[5, 5])) < 1e-3

    def test_canny_outlines_square(self, backend: ImageBackend) -> None:
        img = np.zeros((40, 40), np.uint8)
        img[10:30, 10:30] = 200
        edges = backend.canny(img, 50, 150)
        assert edges.dtype == bool
        assert edges[10:30, 8:12].any() and edges[10:30, 28:32].any()
        assert not edges[15:25, 15:25].any()
        assert not edges[:5, :5].any()

    def test_canny_validation(self, backend: ImageBackend) -> None:
        with pytest.raises(InvalidInputError):
            backend.canny(np.zeros((4, 4), np.uint8), 100, 50)
        with pytest.raises(InvalidInputError):
            backend.canny(np.zeros((4, 4, 3), np.uint8), 10, 50)

    def test_otsu_separates_bimodal(self, backend: ImageBackend, rng: np.random.Generator) -> None:
        img = np.concatenate([rng.normal(50, 5, 500), rng.normal(200, 5, 500)])
        img = np.clip(img, 0, 255).astype(np.uint8).reshape(20, 50)
        t = backend.otsu_threshold(img)
        assert 70 < t < 180
        assert backend.threshold(img, t).sum() == 500


class TestMorphology:
    def test_erode_and_dilate(self, backend: ImageBackend) -> None:
        m = square_mask()
        assert backend.morphology(m, "erode", 3).sum() == 8 * 8
        assert backend.morphology(m, "dilate", 3).sum() == 12 * 12
        assert backend.morphology(m, "erode", 3, iterations=2).sum() == 6 * 6

    def test_open_removes_specks_close_fills_holes(self, backend: ImageBackend) -> None:
        m = square_mask()
        noisy = m.copy()
        noisy[1, 1] = True
        assert not backend.morphology(noisy, "open", 3)[1, 1]
        holed = m.copy()
        holed[10, 10] = False
        assert backend.morphology(holed, "close", 3)[10, 10]

    def test_gradient_is_outline(self, backend: ImageBackend) -> None:
        grad = backend.morphology(square_mask(), "gradient", 3)
        assert grad[5, 10] and not grad[10, 10] and not grad[0, 0]

    def test_grayscale_input_keeps_dtype(self, backend: ImageBackend) -> None:
        img = np.zeros((10, 10), np.uint8)
        img[5, 5] = 200
        out = backend.morphology(img, "dilate", 3, "cross")
        assert out.dtype == np.uint8 and out[4, 5] == 200 and out[4, 4] == 0

    @pytest.mark.parametrize(
        ("op", "ksize", "shape", "iterations"),
        [
            ("melt", 3, "rect", 1),
            ("erode", 0, "rect", 1),
            ("erode", 3, "star", 1),
            ("erode", 3, "rect", 0),
        ],
    )
    def test_validation(
        self, backend: ImageBackend, op: str, ksize: int, shape: str, iterations: int
    ) -> None:
        with pytest.raises(InvalidInputError):
            backend.morphology(square_mask(), op, ksize, shape, iterations)  # type: ignore[arg-type]

    def test_structuring_elements(self) -> None:
        assert structuring_element("rect", 3).sum() == 9
        assert structuring_element("cross", 5).sum() == 9
        ellipse = structuring_element("ellipse", 5)
        assert ellipse[2, 2] and not ellipse[0, 0] and ellipse.sum() == 13
        assert np.array_equal(ellipse, ellipse.T) and np.array_equal(ellipse, ellipse[::-1])
        with pytest.raises(InvalidInputError):
            structuring_element("rect", 0)


class TestComponents:
    def test_labels_and_stats(self) -> None:
        m = np.zeros((10, 12), bool)
        m[1:3, 1:4] = True
        m[5:9, 6:11] = True
        labels, count = label_components(m)
        assert count == 2
        assert labels[1, 1] == 1 and labels[5, 6] == 2
        blobs = component_stats(labels, count)
        assert [b.area for b in blobs] == [6, 20]
        assert blobs[1].bbox.to_list() == [6, 5, 11, 9]
        assert blobs[0].centroid == pytest.approx((2.0, 1.5))
        assert blobs[1].fill_ratio == pytest.approx(1.0)

    def test_connectivity_on_diagonal(self) -> None:
        m = np.eye(5, dtype=bool)
        assert label_components(m, 8)[1] == 1
        assert label_components(m, 4)[1] == 5

    def test_u_shape_merges_branches(self) -> None:
        m = np.zeros((6, 7), bool)
        m[0:5, 1] = True
        m[0:5, 5] = True
        m[4, 1:6] = True
        labels, count = label_components(m, 4)
        assert count == 1 and labels[0, 1] == labels[0, 5]

    def test_empty_and_validation(self) -> None:
        labels, count = label_components(np.zeros((3, 3), bool))
        assert count == 0 and not labels.any()
        assert component_stats(labels, 0) == []
        with pytest.raises(InvalidInputError):
            label_components(np.zeros((3, 3), bool), 6)
        with pytest.raises(InvalidInputError):
            label_components(np.zeros((3, 3, 3), bool))

    def test_find_blobs_min_area_and_backend_agreement(
        self, backend: ImageBackend, rng: np.random.Generator
    ) -> None:
        m = rng.random((60, 80)) > 0.7
        ours = find_blobs(m, 8, min_area=3)
        theirs = backend.connected_components(m, 8, 3)
        assert sorted(b.area for b in ours) == sorted(b.area for b in theirs)
        assert all(b.area >= 3 for b in ours)
