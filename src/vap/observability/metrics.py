"""Minimal thread-safe metrics registry with Prometheus text exposition.

Supports counters, gauges and histograms with label sets. Kept dependency-free
on purpose; the exposition format follows the Prometheus text format 0.0.4.
"""

from __future__ import annotations

import math
import re
import threading
from collections.abc import Sequence
from typing import Any

from vap.errors import InvalidInputError

_NAME = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")
_LABEL = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
DEFAULT_BUCKETS: tuple[float, ...] = (
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
)

LabelKey = tuple[str, ...]


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _fmt(value: float) -> str:
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(float(value))


class _Metric:
    kind = "untyped"

    def __init__(self, name: str, help_text: str, labelnames: Sequence[str]) -> None:
        if not _NAME.match(name):
            raise InvalidInputError(f"invalid metric name {name!r}")
        for label in labelnames:
            if not _LABEL.match(label) or label.startswith("__"):
                raise InvalidInputError(f"invalid label name {label!r}")
        self.name = name
        self.help = help_text
        self.labelnames = tuple(labelnames)
        self._lock = threading.Lock()

    def _key(self, labels: dict[str, str]) -> LabelKey:
        if set(labels) != set(self.labelnames):
            raise InvalidInputError(
                f"metric {self.name} expects labels {list(self.labelnames)}",
                details={"got": list(labels)},
            )
        return tuple(str(labels[n]) for n in self.labelnames)

    def _label_str(self, key: LabelKey, extra: dict[str, str] | None = None) -> str:
        pairs = [f'{n}="{_escape(v)}"' for n, v in zip(self.labelnames, key, strict=True)]
        if extra:
            pairs.extend(f'{n}="{_escape(v)}"' for n, v in extra.items())
        return "{" + ",".join(pairs) + "}" if pairs else ""

    def header(self) -> list[str]:
        return [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} {self.kind}"]

    def expose(self) -> list[str]:
        raise NotImplementedError


class Counter(_Metric):
    kind = "counter"

    def __init__(self, name: str, help_text: str, labelnames: Sequence[str] = ()) -> None:
        super().__init__(name, help_text, labelnames)
        self._values: dict[LabelKey, float] = {}

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        if amount < 0:
            raise InvalidInputError("counters can only increase")
        key = self._key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def value(self, **labels: str) -> float:
        with self._lock:
            return self._values.get(self._key(labels), 0.0)

    def remove(self, **labels: str) -> None:
        with self._lock:
            self._values.pop(self._key(labels), None)

    def expose(self) -> list[str]:
        with self._lock:
            items = sorted(self._values.items())
        return self.header() + [f"{self.name}{self._label_str(k)} {_fmt(v)}" for k, v in items]


class Gauge(Counter):
    kind = "gauge"

    def set(self, value: float, **labels: str) -> None:
        key = self._key(labels)
        with self._lock:
            self._values[key] = float(value)

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        key = self._key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def dec(self, amount: float = 1.0, **labels: str) -> None:
        self.inc(-amount, **labels)


class Histogram(_Metric):
    kind = "histogram"

    def __init__(
        self,
        name: str,
        help_text: str,
        labelnames: Sequence[str] = (),
        buckets: Sequence[float] = DEFAULT_BUCKETS,
    ) -> None:
        super().__init__(name, help_text, labelnames)
        bounds = sorted(float(b) for b in buckets)
        if not bounds or any(b <= 0 for b in bounds):
            raise InvalidInputError("histogram buckets must be positive")
        self.buckets = tuple(bounds)
        self._counts: dict[LabelKey, list[int]] = {}
        self._sums: dict[LabelKey, float] = {}

    def observe(self, value: float, **labels: str) -> None:
        key = self._key(labels)
        with self._lock:
            counts = self._counts.setdefault(key, [0] * (len(self.buckets) + 1))
            for i, bound in enumerate(self.buckets):
                if value <= bound:
                    counts[i] += 1
            counts[-1] += 1
            self._sums[key] = self._sums.get(key, 0.0) + value

    def snapshot(self, **labels: str) -> dict[str, Any]:
        key = self._key(labels)
        with self._lock:
            counts = list(self._counts.get(key, [0] * (len(self.buckets) + 1)))
            total = self._sums.get(key, 0.0)
        return {
            "count": counts[-1],
            "sum": total,
            "buckets": dict(zip(self.buckets, counts[:-1], strict=True)),
        }

    def expose(self) -> list[str]:
        lines = self.header()
        with self._lock:
            items = sorted((k, list(v), self._sums.get(k, 0.0)) for k, v in self._counts.items())
        for key, counts, total in items:
            for bound, count in zip(self.buckets, counts[:-1], strict=True):
                lines.append(
                    f"{self.name}_bucket{self._label_str(key, {'le': _fmt(bound)})} {count}"
                )
            lines.append(f"{self.name}_bucket{self._label_str(key, {'le': '+Inf'})} {counts[-1]}")
            lines.append(f"{self.name}_sum{self._label_str(key)} {_fmt(total)}")
            lines.append(f"{self.name}_count{self._label_str(key)} {counts[-1]}")
        return lines


class MetricsRegistry:
    def __init__(self, namespace: str = "vap") -> None:
        if not _LABEL.match(namespace):
            raise InvalidInputError("invalid metrics namespace")
        self.namespace = namespace
        self._metrics: dict[str, _Metric] = {}
        self._lock = threading.Lock()

    def _register(self, metric: _Metric) -> Any:
        with self._lock:
            existing = self._metrics.get(metric.name)
            if existing is not None:
                if type(existing) is not type(metric) or existing.labelnames != metric.labelnames:
                    raise InvalidInputError(f"metric {metric.name} already registered differently")
                return existing
            self._metrics[metric.name] = metric
            return metric

    def counter(self, name: str, help_text: str, labelnames: Sequence[str] = ()) -> Counter:
        return self._register(Counter(f"{self.namespace}_{name}", help_text, labelnames))

    def gauge(self, name: str, help_text: str, labelnames: Sequence[str] = ()) -> Gauge:
        return self._register(Gauge(f"{self.namespace}_{name}", help_text, labelnames))

    def histogram(
        self,
        name: str,
        help_text: str,
        labelnames: Sequence[str] = (),
        buckets: Sequence[float] = DEFAULT_BUCKETS,
    ) -> Histogram:
        return self._register(Histogram(f"{self.namespace}_{name}", help_text, labelnames, buckets))

    def render(self) -> str:
        with self._lock:
            metrics = sorted(self._metrics.values(), key=lambda m: m.name)
        lines: list[str] = []
        for metric in metrics:
            lines.extend(metric.expose())
        return "\n".join(lines) + "\n"


class PlatformMetrics:
    """The concrete metric families used across the platform."""

    def __init__(self, registry: MetricsRegistry | None = None) -> None:
        self.registry = registry or MetricsRegistry()
        r = self.registry
        self.frames = r.counter("frames_processed_total", "Frames processed", ["pipeline"])
        self.detections = r.counter("detections_total", "Detections produced", ["pipeline"])
        self.events = r.counter("events_total", "Events emitted", ["pipeline", "type"])
        self.pipeline_errors = r.counter("pipeline_errors_total", "Pipeline errors", ["pipeline"])
        self.frame_latency = r.histogram(
            "frame_processing_seconds", "Per-frame processing time", ["pipeline"]
        )
        self.active_tracks = r.gauge(
            "active_tracks", "Confirmed tracks in the last frame", ["pipeline"]
        )
        self.pipeline_fps = r.gauge("pipeline_fps", "Measured processing rate", ["pipeline"])
        self.pipelines_running = r.gauge("pipelines_running", "Running pipelines")
        self.http_requests = r.counter(
            "http_requests_total", "HTTP requests", ["method", "route", "status"]
        )
        self.http_latency = r.histogram("http_request_seconds", "HTTP latency", ["method", "route"])
        self.webhook_deliveries = r.counter(
            "webhook_deliveries_total", "Webhook delivery attempts", ["outcome"]
        )
        self.bus_dropped = r.counter("event_bus_dropped_total", "Events dropped by the bus")

    def render(self) -> str:
        return self.registry.render()
