"""Pure-numpy implementation of :class:`vap.imaging.ops.ImageBackend`."""

from __future__ import annotations

import numpy as np

from vap.errors import InvalidInputError
from vap.imaging.components import Blob, find_blobs, label_components
from vap.imaging.frame import ColorSpace, ensure_image
from vap.imaging.ops import (
    INTERPOLATIONS,
    ImageBackend,
    Interpolation,
    KernelShape,
    MorphOp,
    gaussian_kernel_size,
    structuring_element,
    validate_size,
)

_GRAY_WEIGHTS_RGB = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(arr), 0, 255).astype(np.uint8)


def _axis_slice(ndim: int, axis: int, start: int, stop: int) -> tuple[slice, ...]:
    sl = [slice(None)] * ndim
    sl[axis] = slice(start, stop)
    return tuple(sl)


def correlate_axis(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """1-D correlation along ``axis`` with reflect-101 borders (float32 output)."""
    k = len(kernel)
    r = k // 2
    pad = [(0, 0)] * arr.ndim
    pad[axis] = (r, k - 1 - r)
    padded = np.pad(arr.astype(np.float32, copy=False), pad, mode="reflect")
    n = arr.shape[axis]
    out = np.zeros(arr.shape, dtype=np.float32)
    for i, weight in enumerate(kernel):
        if weight != 0:
            out += np.float32(weight) * padded[_axis_slice(arr.ndim, axis, i, i + n)]
    return out


def gaussian_kernel(sigma: float, ksize: int) -> np.ndarray:
    x = np.arange(ksize, dtype=np.float64) - (ksize - 1) / 2.0
    k = np.exp(-(x**2) / (2.0 * sigma * sigma))
    return (k / k.sum()).astype(np.float32)


def shift(arr: np.ndarray, dy: int, dx: int, fill: float | int | bool = 0) -> np.ndarray:
    """Return ``out`` with ``out[y, x] = arr[y + dy, x + dx]`` (``fill`` outside)."""
    h, w = arr.shape[:2]
    out = np.full_like(arr, fill)
    ys, yd = (slice(dy, h), slice(0, h - dy)) if dy >= 0 else (slice(0, h + dy), slice(-dy, h))
    xs, xd = (slice(dx, w), slice(0, w - dx)) if dx >= 0 else (slice(0, w + dx), slice(-dx, w))
    out[yd, xd] = arr[ys, xs]
    return out


class NumpyBackend(ImageBackend):
    name = "numpy"

    # -- geometry -----------------------------------------------------------------

    def resize(
        self, image: np.ndarray, size: tuple[int, int], interpolation: Interpolation = "bilinear"
    ) -> np.ndarray:
        image = ensure_image(image)
        width, height = validate_size(size)
        if interpolation not in INTERPOLATIONS:
            raise InvalidInputError(f"unknown interpolation {interpolation!r}")
        h, w = image.shape[:2]
        if (h, w) == (height, width):
            return image.copy()
        if interpolation == "nearest":
            ys = np.minimum((np.arange(height) * (h / height)).astype(np.int64), h - 1)
            xs = np.minimum((np.arange(width) * (w / width)).astype(np.int64), w - 1)
            return image[ys[:, None], xs[None, :]]
        if interpolation == "area" and height <= h and width <= w:
            return self._resize_area(image, width, height)
        return self._resize_bilinear(image, width, height)

    @staticmethod
    def _resize_bilinear(image: np.ndarray, width: int, height: int) -> np.ndarray:
        h, w = image.shape[:2]
        src = image.astype(np.float32)
        y = np.clip((np.arange(height) + 0.5) * (h / height) - 0.5, 0, h - 1)
        x = np.clip((np.arange(width) + 0.5) * (w / width) - 0.5, 0, w - 1)
        y0 = np.floor(y).astype(np.int64)
        x0 = np.floor(x).astype(np.int64)
        y1 = np.minimum(y0 + 1, h - 1)
        x1 = np.minimum(x0 + 1, w - 1)
        wy = (y - y0).astype(np.float32)
        wx = (x - x0).astype(np.float32)
        if src.ndim == 3:
            wy = wy[:, None, None]
            wx = wx[None, :, None]
        else:
            wy = wy[:, None]
            wx = wx[None, :]
        top = src[y0][:, x0] * (1 - wx) + src[y0][:, x1] * wx
        bottom = src[y1][:, x0] * (1 - wx) + src[y1][:, x1] * wx
        return _to_uint8(top * (1 - wy) + bottom * wy)

    @staticmethod
    def _resize_area(image: np.ndarray, width: int, height: int) -> np.ndarray:
        """Exact box-average downscaling via bilinear sampling of the integral image."""
        h, w = image.shape[:2]
        src = image.astype(np.float64)
        if src.ndim == 2:
            src = src[:, :, None]
        integral = np.zeros((h + 1, w + 1, src.shape[2]), dtype=np.float64)
        integral[1:, 1:] = src.cumsum(axis=0).cumsum(axis=1)

        def sample(ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
            y0 = np.minimum(np.floor(ys).astype(np.int64), h - 1)
            x0 = np.minimum(np.floor(xs).astype(np.int64), w - 1)
            fy = (ys - y0)[:, None, None]
            fx = (xs - x0)[None, :, None]
            a = integral[y0][:, x0]
            b = integral[y0][:, x0 + 1]
            c = integral[y0 + 1][:, x0]
            d = integral[y0 + 1][:, x0 + 1]
            return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy

        ey = np.arange(height + 1) * (h / height)
        ex = np.arange(width + 1) * (w / width)
        s = sample(ey, ex)
        sums = s[1:, 1:] - s[:-1, 1:] - s[1:, :-1] + s[:-1, :-1]
        cell = (h / height) * (w / width)
        out = _to_uint8(sums / cell)
        return out[:, :, 0] if image.ndim == 2 else out

    # -- colour -------------------------------------------------------------------

    def convert_color(self, image: np.ndarray, src: ColorSpace, dst: ColorSpace) -> np.ndarray:
        src, dst = ColorSpace(src), ColorSpace(dst)
        image = ensure_image(image)
        if src == dst:
            return image.copy()
        if (image.ndim == 2) != (src == ColorSpace.GRAY):
            raise InvalidInputError(f"image shape does not match colour space {src.value}")
        rgb = self._to_rgb(image, src)
        if dst == ColorSpace.RGB:
            return rgb
        if dst == ColorSpace.BGR:
            return np.ascontiguousarray(rgb[:, :, ::-1])
        if dst == ColorSpace.GRAY:
            return _to_uint8(rgb.astype(np.float32) @ _GRAY_WEIGHTS_RGB)
        return self._rgb_to_hsv(rgb)

    def _to_rgb(self, image: np.ndarray, src: ColorSpace) -> np.ndarray:
        if src == ColorSpace.RGB:
            return image
        if src == ColorSpace.BGR:
            return np.ascontiguousarray(image[:, :, ::-1])
        if src == ColorSpace.GRAY:
            return np.repeat(image[:, :, None], 3, axis=2)
        return self._hsv_to_rgb(image)

    @staticmethod
    def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
        f = rgb.astype(np.float32)
        r, g, b = f[..., 0], f[..., 1], f[..., 2]
        v = f.max(axis=2)
        mn = f.min(axis=2)
        diff = v - mn
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(v > 0, diff / v * 255.0, 0.0)
            safe = np.where(diff > 0, diff, 1.0)
            h = np.where(
                v == r,
                60.0 * (g - b) / safe,
                np.where(v == g, 120.0 + 60.0 * (b - r) / safe, 240.0 + 60.0 * (r - g) / safe),
            )
        h = np.where(diff > 0, h, 0.0)
        h = np.where(h < 0, h + 360.0, h)
        hsv = np.stack([np.rint(h / 2.0) % 180, s, v], axis=2)
        return _to_uint8(hsv)

    @staticmethod
    def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
        f = hsv.astype(np.float32)
        h = (f[..., 0] * 2.0) % 360.0
        s = f[..., 1] / 255.0
        v = f[..., 2]
        c = v * s
        hp = h / 60.0
        x = c * (1 - np.abs(hp % 2 - 1))
        sector = np.floor(hp).astype(np.int64) % 6
        zeros = np.zeros_like(c)
        table = [
            (c, x, zeros),
            (x, c, zeros),
            (zeros, c, x),
            (zeros, x, c),
            (x, zeros, c),
            (c, zeros, x),
        ]
        r = np.select([sector == i for i in range(6)], [t[0] for t in table])
        g = np.select([sector == i for i in range(6)], [t[1] for t in table])
        b = np.select([sector == i for i in range(6)], [t[2] for t in table])
        m = v - c
        return _to_uint8(np.stack([r + m, g + m, b + m], axis=2))

    # -- filtering ----------------------------------------------------------------

    def gaussian_blur(
        self, image: np.ndarray, sigma: float, ksize: int | None = None
    ) -> np.ndarray:
        image = ensure_image(image)
        if sigma <= 0:
            raise InvalidInputError("sigma must be positive")
        k = ksize or gaussian_kernel_size(sigma)
        if k < 1 or k % 2 == 0:
            raise InvalidInputError("kernel size must be a positive odd integer")
        kernel = gaussian_kernel(sigma, k)
        out = correlate_axis(correlate_axis(image, kernel, 0), kernel, 1)
        return _to_uint8(out)

    def box_blur(self, image: np.ndarray, ksize: int) -> np.ndarray:
        image = ensure_image(image)
        if ksize < 1:
            raise InvalidInputError("kernel size must be >= 1")
        out = image.astype(np.float64)
        for axis in (0, 1):
            r = ksize // 2
            pad = [(0, 0)] * out.ndim
            pad[axis] = (r, ksize - 1 - r)
            padded = np.pad(out, pad, mode="reflect")
            csum = np.cumsum(padded, axis=axis)
            zero_shape = list(csum.shape)
            zero_shape[axis] = 1
            csum = np.concatenate([np.zeros(zero_shape), csum], axis=axis)
            n = out.shape[axis]
            hi = csum[_axis_slice(out.ndim, axis, ksize, ksize + n)]
            lo = csum[_axis_slice(out.ndim, axis, 0, n)]
            out = (hi - lo) / ksize
        return _to_uint8(out)

    def sobel(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gray = ensure_image(gray)
        if gray.ndim != 2:
            raise InvalidInputError("sobel expects a single-channel image")
        deriv = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        smooth = np.array([1.0, 2.0, 1.0], dtype=np.float32)
        gx = correlate_axis(correlate_axis(gray, smooth, 0), deriv, 1)
        gy = correlate_axis(correlate_axis(gray, deriv, 0), smooth, 1)
        return gx, gy

    def canny(self, gray: np.ndarray, low: float, high: float, sigma: float = 1.0) -> np.ndarray:
        gray = ensure_image(gray)
        if gray.ndim != 2:
            raise InvalidInputError("canny expects a single-channel image")
        if low < 0 or high < low:
            raise InvalidInputError("canny thresholds must satisfy 0 <= low <= high")
        smoothed = self.gaussian_blur(gray, sigma) if sigma > 0 else gray
        gx, gy = self.sobel(smoothed)
        mag = np.hypot(gx, gy)
        angle = (np.degrees(np.arctan2(gy, gx)) + 180.0) % 180.0
        horizontal = (angle < 22.5) | (angle >= 157.5)
        diag_down = (angle >= 22.5) & (angle < 67.5)
        vertical = (angle >= 67.5) & (angle < 112.5)
        diag_up = (angle >= 112.5) & (angle < 157.5)
        keep = np.zeros(mag.shape, dtype=bool)
        for sel, (dy, dx) in (
            (horizontal, (0, 1)),
            (diag_down, (1, 1)),
            (vertical, (1, 0)),
            (diag_up, (1, -1)),
        ):
            fwd = shift(mag, dy, dx)
            bwd = shift(mag, -dy, -dx)
            keep |= sel & (mag >= fwd) & (mag > bwd)
        weak = keep & (mag >= low)
        strong = keep & (mag >= high)
        labels, count = label_components(weak, connectivity=8)
        if count == 0:
            return strong
        has_strong = np.zeros(count + 1, dtype=bool)
        has_strong[np.unique(labels[strong])] = True
        has_strong[0] = False
        return has_strong[labels]

    # -- morphology ---------------------------------------------------------------

    def morphology(
        self,
        mask: np.ndarray,
        op: MorphOp,
        ksize: int = 3,
        shape: KernelShape = "rect",
        iterations: int = 1,
    ) -> np.ndarray:
        self.check_morph(op, ksize, shape, iterations)
        if mask.ndim != 2:
            raise InvalidInputError("morphology expects a 2-D image")
        is_bool = mask.dtype == bool
        arr = mask.astype(np.uint8) if is_bool else ensure_image(mask)
        el = structuring_element(shape, ksize)

        def repeat(fn: str, a: np.ndarray) -> np.ndarray:
            for _ in range(iterations):
                a = self._erode(a, el) if fn == "erode" else self._dilate(a, el)
            return a

        if op == "erode":
            out = repeat("erode", arr)
        elif op == "dilate":
            out = repeat("dilate", arr)
        elif op == "open":
            out = repeat("dilate", repeat("erode", arr))
        elif op == "close":
            out = repeat("erode", repeat("dilate", arr))
        else:
            out = repeat("dilate", arr).astype(np.int16) - repeat("erode", arr).astype(np.int16)
            out = np.clip(out, 0, 255).astype(np.uint8)
        return out.astype(bool) if is_bool else out

    @staticmethod
    def _rank_filter(arr: np.ndarray, el: np.ndarray, fill: int, reducer: np.ufunc) -> np.ndarray:
        h, w = arr.shape
        kh, kw = el.shape
        ry, rx = kh // 2, kw // 2
        if el.all() and kh > 1 and kw > 1:
            row = NumpyBackend._rank_filter(arr, np.ones((1, kw), dtype=bool), fill, reducer)
            return NumpyBackend._rank_filter(row, np.ones((kh, 1), dtype=bool), fill, reducer)
        padded = np.pad(arr, ((ry, kh - 1 - ry), (rx, kw - 1 - rx)), constant_values=fill)
        out = np.full_like(arr, fill)
        for dy, dx in zip(*np.nonzero(el), strict=True):
            reducer(out, padded[dy : dy + h, dx : dx + w], out=out)
        return out

    def _erode(self, arr: np.ndarray, el: np.ndarray) -> np.ndarray:
        return self._rank_filter(arr, el, 255, np.minimum)

    def _dilate(self, arr: np.ndarray, el: np.ndarray) -> np.ndarray:
        return self._rank_filter(arr, el, 0, np.maximum)

    # -- components ---------------------------------------------------------------

    def connected_components(
        self, mask: np.ndarray, connectivity: int = 8, min_area: int = 1
    ) -> list[Blob]:
        if mask.ndim != 2:
            raise InvalidInputError("connected components expects a 2-D mask")
        return find_blobs(mask.astype(bool, copy=False), connectivity, min_area)
