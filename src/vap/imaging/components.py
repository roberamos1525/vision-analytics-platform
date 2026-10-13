"""Connected-component labelling in pure numpy.

The algorithm works on horizontal runs rather than pixels: each row is reduced
to ``[start, end)`` runs with a vectorised diff, overlapping runs in adjacent
rows are merged with a small union-find, and labels are painted back with
``np.repeat``. The Python-level work is proportional to the number of runs,
which for analytics masks is orders of magnitude smaller than the pixel count.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vap.errors import InvalidInputError
from vap.geometry import BBox


@dataclass(frozen=True, slots=True)
class Blob:
    label: int
    area: int
    bbox: BBox
    centroid: tuple[float, float]

    @property
    def fill_ratio(self) -> float:
        return self.area / self.bbox.area if self.bbox.area > 0 else 0.0


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        parent = self.parent
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = mask.shape
    padded = np.zeros((h, w + 2), dtype=np.int8)
    padded[:, 1:-1] = mask
    d = np.diff(padded, axis=1)
    rows, starts = np.nonzero(d == 1)
    _, ends = np.nonzero(d == -1)
    return rows.astype(np.int64), starts.astype(np.int64), ends.astype(np.int64)


def label_components(mask: np.ndarray, connectivity: int = 8) -> tuple[np.ndarray, int]:
    """Label connected foreground regions.

    Returns ``(labels, count)`` where ``labels`` is an ``int32`` array with 0 for
    background and ``1..count`` for components ordered by first appearance in
    raster order.
    """
    if connectivity not in (4, 8):
        raise InvalidInputError("connectivity must be 4 or 8")
    if mask.ndim != 2:
        raise InvalidInputError("mask must be 2-D")
    fg = mask.astype(bool, copy=False)
    h, w = fg.shape
    labels = np.zeros((h, w), dtype=np.int32)
    rows, starts, ends = _runs(fg)
    n = rows.shape[0]
    if n == 0:
        return labels, 0

    slack = 1 if connectivity == 8 else 0
    uf = _UnionFind(n)
    row_first = np.searchsorted(rows, np.arange(h + 1))
    for r in range(1, h):
        cur_lo, cur_hi = int(row_first[r]), int(row_first[r + 1])
        prev_lo, prev_hi = int(row_first[r - 1]), int(row_first[r])
        if cur_lo == cur_hi or prev_lo == prev_hi:
            continue
        prev_starts = starts[prev_lo:prev_hi]
        prev_ends = ends[prev_lo:prev_hi]
        # For each current run, previous runs overlap when
        # prev_end + slack > cur_start and prev_start < cur_end + slack.
        first = np.searchsorted(prev_ends + slack, starts[cur_lo:cur_hi], side="right")
        last = np.searchsorted(prev_starts, ends[cur_lo:cur_hi] + slack, side="left")
        for offset in range(cur_hi - cur_lo):
            for j in range(int(first[offset]), int(last[offset])):
                uf.union(cur_lo + offset, prev_lo + j)

    roots = np.fromiter((uf.find(i) for i in range(n)), dtype=np.int64, count=n)
    _, first_idx, inverse = np.unique(roots, return_index=True, return_inverse=True)
    order = np.argsort(first_idx)
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    run_labels = (rank[inverse] + 1).astype(np.int32)

    lengths = ends - starts
    total = int(lengths.sum())
    run_offsets = np.repeat(np.cumsum(lengths) - lengths, lengths)
    flat_pos = np.repeat(rows * w + starts, lengths) + (np.arange(total) - run_offsets)
    labels.ravel()[flat_pos] = np.repeat(run_labels, lengths)
    return labels, int(order.size)


def component_stats(labels: np.ndarray, count: int) -> list[Blob]:
    """Compute area, bounding box and centroid for each label ``1..count``."""
    if count == 0:
        return []
    ys, xs = np.nonzero(labels)
    lab = labels[ys, xs].astype(np.int64)
    area = np.bincount(lab, minlength=count + 1)
    sum_x = np.bincount(lab, weights=xs, minlength=count + 1)
    sum_y = np.bincount(lab, weights=ys, minlength=count + 1)
    big = np.iinfo(np.int64).max
    min_x = np.full(count + 1, big, dtype=np.int64)
    min_y = np.full(count + 1, big, dtype=np.int64)
    max_x = np.full(count + 1, -1, dtype=np.int64)
    max_y = np.full(count + 1, -1, dtype=np.int64)
    np.minimum.at(min_x, lab, xs)
    np.minimum.at(min_y, lab, ys)
    np.maximum.at(max_x, lab, xs)
    np.maximum.at(max_y, lab, ys)
    blobs: list[Blob] = []
    for i in range(1, count + 1):
        if area[i] == 0:
            continue
        bbox = BBox(float(min_x[i]), float(min_y[i]), float(max_x[i] + 1), float(max_y[i] + 1))
        centroid = (float(sum_x[i] / area[i]), float(sum_y[i] / area[i]))
        blobs.append(Blob(label=i, area=int(area[i]), bbox=bbox, centroid=centroid))
    return blobs


def find_blobs(mask: np.ndarray, connectivity: int = 8, min_area: int = 1) -> list[Blob]:
    labels, count = label_components(mask, connectivity)
    return [b for b in component_stats(labels, count) if b.area >= min_area]
