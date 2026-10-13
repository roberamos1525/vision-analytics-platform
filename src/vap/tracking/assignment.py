"""Linear assignment: Hungarian (Kuhn-Munkres) in numpy, plus greedy matching.

The Hungarian solver is the shortest-augmenting-path formulation with row and
column potentials, O(n^2 m) for an ``n x m`` matrix with ``n <= m``. The inner
loop over columns is vectorised. ``scipy.optimize.linear_sum_assignment`` can
be used instead when SciPy is installed (``solver="scipy"``).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from vap.errors import BackendUnavailableError, InvalidInputError

Solver = Literal["hungarian", "scipy", "greedy"]


def hungarian(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Minimum-cost assignment; returns ``(rows, cols)`` like SciPy."""
    c = np.asarray(cost, dtype=np.float64)
    if c.ndim != 2:
        raise InvalidInputError("cost matrix must be 2-D")
    if c.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    if not np.all(np.isfinite(c)):
        raise InvalidInputError("cost matrix must be finite")
    transposed = c.shape[0] > c.shape[1]
    if transposed:
        c = c.T
    n, m = c.shape
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=np.int64)  # p[j] = row (1-based) assigned to column j
    way = np.zeros(m + 1, dtype=np.int64)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            free = ~used[1:]
            cur = c[i0 - 1] - u[i0] - v[1:]
            better = free & (cur < minv[1:])
            minv[1:][better] = cur[better]
            way[1:][better] = j0
            masked = np.where(free, minv[1:], np.inf)
            j1 = int(np.argmin(masked)) + 1
            delta = masked[j1 - 1]
            used_idx = np.nonzero(used)[0]
            u[p[used_idx]] += delta
            v[used_idx] -= delta
            minv[1:][free] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1
    cols = np.nonzero(p[1:])[0]
    rows = p[1:][cols] - 1
    if transposed:
        rows, cols = cols, rows
    order = np.argsort(rows)
    return rows[order].astype(np.int64), cols[order].astype(np.int64)


def greedy(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Repeatedly take the globally cheapest remaining pair."""
    c = np.asarray(cost, dtype=np.float64)
    if c.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    order = np.argsort(c, axis=None, kind="stable")
    used_r: set[int] = set()
    used_c: set[int] = set()
    rows: list[int] = []
    cols: list[int] = []
    limit = min(c.shape)
    for flat in order.tolist():
        r, col = divmod(flat, c.shape[1])
        if r in used_r or col in used_c:
            continue
        used_r.add(r)
        used_c.add(col)
        rows.append(r)
        cols.append(col)
        if len(rows) == limit:
            break
    idx = np.argsort(rows)
    return np.asarray(rows, dtype=np.int64)[idx], np.asarray(cols, dtype=np.int64)[idx]


def solve(cost: np.ndarray, solver: Solver = "hungarian") -> tuple[np.ndarray, np.ndarray]:
    if solver == "hungarian":
        return hungarian(cost)
    if solver == "greedy":
        return greedy(cost)
    if solver == "scipy":
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise BackendUnavailableError("scipy is not installed") from exc
        r, c = linear_sum_assignment(np.asarray(cost, dtype=np.float64))
        return r.astype(np.int64), c.astype(np.int64)
    raise InvalidInputError(f"unknown assignment solver {solver!r}")


def match(
    cost: np.ndarray, max_cost: float, solver: Solver = "hungarian"
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Assign rows to columns, rejecting pairs costlier than ``max_cost``.

    Returns ``(matches, unmatched_rows, unmatched_cols)``. Pairs above the gate
    are replaced by a large finite cost before solving so the solver never
    trades a valid match for a forbidden one.
    """
    c = np.asarray(cost, dtype=np.float64)
    n, m = c.shape
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))
    gated = np.where(np.isfinite(c) & (c <= max_cost), c, np.nan)
    big = (np.nanmax(np.abs(gated)) if np.any(~np.isnan(gated)) else 1.0) * 2 + 1e6
    rows, cols = solve(np.where(np.isnan(gated), big, gated), solver)
    matches: list[tuple[int, int]] = []
    for r, col in zip(rows.tolist(), cols.tolist(), strict=True):
        if not np.isnan(gated[r, col]):
            matches.append((r, col))
    matched_r = {r for r, _ in matches}
    matched_c = {col for _, col in matches}
    return (
        matches,
        [r for r in range(n) if r not in matched_r],
        [col for col in range(m) if col not in matched_c],
    )
