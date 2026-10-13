"""Rolling counters and per-zone / per-line statistics."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vap.errors import InvalidInputError


class RollingCounter:
    """Counts events within a sliding time window using fixed-width buckets.

    Memory is bounded by ``window / bucket`` regardless of event volume.
    Timestamps must be non-decreasing (stream time).
    """

    def __init__(self, window_s: float = 60.0, bucket_s: float = 1.0) -> None:
        if window_s <= 0 or bucket_s <= 0 or bucket_s > window_s:
            raise InvalidInputError("require 0 < bucket_s <= window_s")
        self.window_s = window_s
        self.bucket_s = bucket_s
        self._buckets: deque[tuple[int, float]] = deque()
        self.total = 0.0

    def _bucket(self, ts: float) -> int:
        return math.floor(ts / self.bucket_s)

    def _expire(self, now: float) -> None:
        oldest = self._bucket(now - self.window_s)
        while self._buckets and self._buckets[0][0] <= oldest:
            self._buckets.popleft()

    def add(self, ts: float, amount: float = 1.0) -> None:
        b = self._bucket(ts)
        if self._buckets and self._buckets[-1][0] == b:
            self._buckets[-1] = (b, self._buckets[-1][1] + amount)
        elif self._buckets and b < self._buckets[-1][0]:
            # Late event: fold into the newest bucket rather than reorder history.
            last_b, last_v = self._buckets[-1]
            self._buckets[-1] = (last_b, last_v + amount)
        else:
            self._buckets.append((b, amount))
        self.total += amount
        self._expire(ts)

    def count(self, now: float) -> float:
        self._expire(now)
        return float(sum(v for _, v in self._buckets))

    def rate_per_minute(self, now: float) -> float:
        return self.count(now) * 60.0 / self.window_s


@dataclass(slots=True)
class DwellStats:
    """Streaming dwell-time statistics with a bounded reservoir for quantiles."""

    count: int = 0
    total_s: float = 0.0
    max_s: float = 0.0
    recent: deque[float] = field(default_factory=lambda: deque(maxlen=1000))

    def add(self, duration: float) -> None:
        d = max(0.0, float(duration))
        self.count += 1
        self.total_s += d
        self.max_s = max(self.max_s, d)
        self.recent.append(d)

    @property
    def mean_s(self) -> float:
        return self.total_s / self.count if self.count else 0.0

    def quantile(self, q: float) -> float:
        if not self.recent:
            return 0.0
        return float(np.quantile(np.asarray(self.recent), q))

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean_s": round(self.mean_s, 3),
            "max_s": round(self.max_s, 3),
            "p50_s": round(self.quantile(0.5), 3),
            "p90_s": round(self.quantile(0.9), 3),
        }


@dataclass(slots=True)
class ZoneCounters:
    zone_id: str
    window_s: float = 60.0
    occupancy: int = 0
    peak_occupancy: int = 0
    entries: int = 0
    exits: int = 0
    dwell: DwellStats = field(default_factory=DwellStats)
    entries_window: RollingCounter = field(init=False)

    def __post_init__(self) -> None:
        self.entries_window = RollingCounter(self.window_s)

    def snapshot(self, now: float) -> dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "occupancy": self.occupancy,
            "peak_occupancy": self.peak_occupancy,
            "entries": self.entries,
            "exits": self.exits,
            "entries_per_min": round(self.entries_window.rate_per_minute(now), 3),
            "dwell": self.dwell.to_dict(),
        }


@dataclass(slots=True)
class LineCounters:
    line_id: str
    window_s: float = 60.0
    positive: int = 0
    negative: int = 0
    window: RollingCounter = field(init=False)

    def __post_init__(self) -> None:
        self.window = RollingCounter(self.window_s)

    def record(self, direction: int, ts: float) -> None:
        if direction > 0:
            self.positive += 1
        elif direction < 0:
            self.negative += 1
        else:
            return
        self.window.add(ts)

    def snapshot(self, now: float) -> dict[str, Any]:
        return {
            "line_id": self.line_id,
            "positive": self.positive,
            "negative": self.negative,
            "total": self.positive + self.negative,
            "net": self.positive - self.negative,
            "crossings_per_min": round(self.window.rate_per_minute(now), 3),
        }
