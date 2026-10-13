"""Motion heatmaps: exponentially decayed accumulation of motion masks and track points."""

from __future__ import annotations

import numpy as np

from vap.errors import InvalidInputError


def _build_colormap() -> np.ndarray:
    """256-entry RGB lookup table approximating the 'turbo'-like jet ramp."""
    x = np.linspace(0.0, 1.0, 256)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0, 1)
    return (np.stack([r, g, b], axis=1) * 255).astype(np.uint8)


COLORMAP = _build_colormap()


def apply_colormap(values: np.ndarray) -> np.ndarray:
    """Map a ``[0, 1]`` float array to RGB colours."""
    idx = np.clip(np.rint(np.nan_to_num(values) * 255), 0, 255).astype(np.uint8)
    return COLORMAP[idx]


class MotionHeatmap:
    """Accumulates where motion happens.

    ``decay`` is applied once per update (``1.0`` = never forget). The map can
    be fed motion masks (dense) or object anchor points (sparse, splatted with
    a Gaussian kernel) - typically track bottom-centres.
    """

    def __init__(
        self, height: int, width: int, *, decay: float = 0.995, splat_sigma: float = 4.0
    ) -> None:
        if height < 1 or width < 1:
            raise InvalidInputError("heatmap dimensions must be positive")
        if not 0.0 < decay <= 1.0:
            raise InvalidInputError("decay must be in (0, 1]")
        if splat_sigma <= 0:
            raise InvalidInputError("splat_sigma must be positive")
        self.height = height
        self.width = width
        self.decay = decay
        self.splat_sigma = splat_sigma
        self.values = np.zeros((height, width), dtype=np.float32)
        self.updates = 0
        r = int(np.ceil(3 * splat_sigma))
        ax = np.arange(-r, r + 1, dtype=np.float32)
        self._kernel = np.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2 * splat_sigma**2))
        self._radius = r

    def reset(self) -> None:
        self.values[:] = 0.0
        self.updates = 0

    def _check_shape(self, shape: tuple[int, ...]) -> None:
        if shape[:2] != (self.height, self.width):
            raise InvalidInputError(
                "mask shape does not match heatmap", details={"expected": [self.height, self.width]}
            )

    def add_mask(self, mask: np.ndarray, weight: float = 1.0) -> None:
        self._check_shape(mask.shape)
        self.values *= self.decay
        self.values += mask.astype(np.float32) * weight
        self.updates += 1

    def add_points(self, points: list[tuple[float, float]], weight: float = 1.0) -> None:
        self.values *= self.decay
        r = self._radius
        for x, y in points:
            cx, cy = round(x), round(y)
            x0, x1 = max(cx - r, 0), min(cx + r + 1, self.width)
            y0, y1 = max(cy - r, 0), min(cy + r + 1, self.height)
            if x0 >= x1 or y0 >= y1:
                continue
            k = self._kernel[y0 - (cy - r) : y1 - (cy - r), x0 - (cx - r) : x1 - (cx - r)]
            self.values[y0:y1, x0:x1] += k * weight
        self.updates += 1

    def normalized(self) -> np.ndarray:
        peak = float(self.values.max())
        if peak <= 0:
            return np.zeros_like(self.values)
        return self.values / peak

    def render(self) -> np.ndarray:
        return apply_colormap(self.normalized())

    def overlay(self, image: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Blend the heatmap over an RGB image; cold areas stay untouched."""
        self._check_shape(image.shape)
        if not 0.0 <= alpha <= 1.0:
            raise InvalidInputError("alpha must be within [0, 1]")
        norm = self.normalized()
        heat = apply_colormap(norm).astype(np.float32)
        a = (alpha * np.clip(norm * 3.0, 0, 1))[:, :, None]
        out = image.astype(np.float32) * (1 - a) + heat * a
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)

    def hotspots(self, grid: int = 4) -> list[dict[str, float | int]]:
        """Summarise the heatmap on a ``grid x grid`` lattice, hottest first."""
        if grid < 1:
            raise InvalidInputError("grid must be >= 1")
        norm = self.normalized()
        ys = np.linspace(0, self.height, grid + 1).astype(int)
        xs = np.linspace(0, self.width, grid + 1).astype(int)
        cells: list[dict[str, float | int]] = []
        for i in range(grid):
            for j in range(grid):
                cell = norm[ys[i] : ys[i + 1], xs[j] : xs[j + 1]]
                cells.append({"row": i, "col": j, "mean": float(cell.mean()) if cell.size else 0.0})
        return sorted(cells, key=lambda c: -float(c["mean"]))
