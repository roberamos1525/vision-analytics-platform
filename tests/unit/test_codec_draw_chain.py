from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vap.errors import InvalidInputError, PayloadTooLargeError
from vap.geometry import BBox, LineSegment, Polygon
from vap.imaging import draw
from vap.imaging.chain import OpChain, apply_chain
from vap.imaging.codec import decode_image, encode_image, format_for_path, load_image, save_image
from vap.imaging.ops import ImageBackend


class TestCodec:
    def test_png_roundtrip_is_lossless(self, rng: np.random.Generator) -> None:
        img = rng.integers(0, 256, (12, 9, 3), dtype=np.uint8)
        assert np.array_equal(decode_image(encode_image(img, "png")), img)

    def test_jpeg_roundtrip_is_close(self) -> None:
        img = np.full((16, 16, 3), (200, 100, 50), np.uint8)
        out = decode_image(encode_image(img, "jpeg", quality=95))
        assert np.abs(out.astype(int) - img).max() <= 4

    def test_gray_and_palette_decode_to_rgb(self) -> None:
        buf = io.BytesIO()
        Image.new("P", (4, 3), 5).save(buf, format="PNG")
        assert decode_image(buf.getvalue()).shape == (3, 4, 3)
        gray = encode_image(np.zeros((5, 5), np.uint8), "bmp")
        assert decode_image(gray).shape == (5, 5, 3)

    def test_rejects_garbage_and_empty(self) -> None:
        with pytest.raises(InvalidInputError):
            decode_image(b"not an image at all")
        with pytest.raises(InvalidInputError):
            decode_image(b"")

    def test_pixel_limit_checked_before_decoding(self) -> None:
        data = encode_image(np.zeros((100, 100, 3), np.uint8), "png")
        with pytest.raises(PayloadTooLargeError) as exc:
            decode_image(data, max_pixels=1000)
        assert exc.value.details["width"] == 100

    def test_encode_validation(self) -> None:
        img = np.zeros((2, 2, 3), np.uint8)
        with pytest.raises(InvalidInputError):
            encode_image(img, "gif")  # type: ignore[arg-type]
        with pytest.raises(InvalidInputError):
            encode_image(img, "jpeg", quality=0)

    def test_save_and_load(self, tmp_path: Path) -> None:
        img = np.full((6, 7, 3), 99, np.uint8)
        path = save_image(tmp_path / "sub" / "a.png", img)
        assert np.array_equal(load_image(path), img)
        assert not list(tmp_path.glob("**/*.tmp"))
        with pytest.raises(InvalidInputError):
            format_for_path(tmp_path / "x.tiff")
        with pytest.raises(InvalidInputError):
            load_image(tmp_path / "missing.png")


class TestDraw:
    def test_rect_outline_clips(self) -> None:
        img = np.zeros((20, 20, 3), np.uint8)
        draw.draw_rect(img, BBox(-5, 5, 10, 30), (255, 0, 0), thickness=1)
        assert img[5, 0:10, 0].all()
        assert img[5:20, 9, 0].all()
        assert img[10, 5, 0] == 0

    def test_fill_rect_alpha(self) -> None:
        img = np.full((10, 10, 3), 100, np.uint8)
        draw.fill_rect(img, BBox(0, 0, 5, 5), (200, 200, 200), alpha=0.5)
        assert img[2, 2, 0] == 150 and img[7, 7, 0] == 100

    def test_line_endpoints_and_thickness(self) -> None:
        img = np.zeros((10, 10, 3), np.uint8)
        draw.draw_line(img, (0, 0), (9, 9), (0, 255, 0), thickness=1)
        assert all(img[i, i, 1] == 255 for i in range(10))
        draw.draw_line(img, (0, 5), (100, 5), (0, 0, 255), thickness=3)
        assert img[4:7, 9, 2].all()

    def test_polygon_fill_and_tripwire(self) -> None:
        img = np.zeros((20, 20, 3), np.uint8)
        poly = Polygon.from_points([[2, 2], [12, 2], [12, 12], [2, 12]])
        draw.draw_polygon(img, poly, (0, 0, 200), fill_alpha=1.0)
        assert img[7, 7, 2] == 200
        draw.draw_tripwire(img, LineSegment.from_points([[0, 15], [19, 15]]), (255, 255, 255))
        assert img[15, 3].tolist() == [255, 255, 255]
        assert img[17:20, 8:12].any()  # tick points below a left-to-right line

    def test_text_and_label(self) -> None:
        img = np.zeros((30, 60, 3), np.uint8)
        draw.draw_text(img, "#12", (1, 1), (255, 255, 255), scale=2)
        assert img.any()
        assert draw.text_size("12", 2) == (14, 10)
        draw.draw_label(img, BBox(30, 20, 50, 28), "7", (10, 200, 10))
        assert img[18, 31, 1] == 200
        assert draw.color_for(9) == draw.PALETTE[1]


class TestChain:
    def test_edges_pipeline(self, backend: ImageBackend) -> None:
        img = np.zeros((40, 60, 3), np.uint8)
        img[10:30, 20:40] = (200, 50, 50)
        chain = OpChain.model_validate(
            {
                "ops": [
                    {"op": "resize", "width": 30},
                    {"op": "gray"},
                    {"op": "canny", "low": 30, "high": 90},
                ]
            }
        )
        out, steps = apply_chain(img, chain, backend)
        assert out.shape == (20, 30) and out.dtype == np.uint8
        assert set(np.unique(out)) <= {0, 255} and out.max() == 255
        assert [s["op"] for s in steps] == ["resize", "gray", "canny"]

    def test_threshold_morphology_and_sobel(self, backend: ImageBackend) -> None:
        img = np.zeros((20, 20, 3), np.uint8)
        img[5:15, 5:15] = 255
        chain = OpChain.model_validate(
            {
                "ops": [
                    {"op": "blur", "kind": "box", "ksize": 3},
                    {"op": "threshold"},
                    {"op": "morphology", "operation": "erode", "ksize": 3},
                ]
            }
        )
        out, _ = apply_chain(img, chain, backend)
        assert out[10, 10] == 255 and out[0, 0] == 0
        sob, _ = apply_chain(img, OpChain.model_validate({"ops": [{"op": "sobel"}]}), backend)
        assert sob.max() == 255 and sob[10, 10] == 0

    def test_convert_and_errors(self, backend: ImageBackend) -> None:
        img = np.zeros((4, 4, 3), np.uint8)
        hsv, _ = apply_chain(
            img, OpChain.model_validate({"ops": [{"op": "convert", "to": "hsv"}]}), backend
        )
        assert hsv.shape == (4, 4, 3)
        with pytest.raises(InvalidInputError):
            apply_chain(
                img,
                OpChain.model_validate({"ops": [{"op": "gray"}, {"op": "convert", "to": "bgr"}]}),
                backend,
            )
        with pytest.raises(InvalidInputError):
            apply_chain(
                img,
                OpChain.model_validate({"ops": [{"op": "canny", "low": 100, "high": 10}]}),
                backend,
            )
        with pytest.raises(ValueError, match="ops"):
            OpChain.model_validate({"ops": []})
