"""In-process event bus with a bounded queue and a dispatcher thread.

Publishers (pipeline threads) never block: when the queue is full the event
is dropped and counted. A single dispatcher thread fans events out to sinks;
a failing sink is logged and counted but never affects other sinks.
"""

from __future__ import annotations

import queue
import threading
from collections import defaultdict
from typing import Any, Protocol

from vap.errors import ConflictError
from vap.events.model import Event
from vap.observability.logs import get_logger
from vap.observability.metrics import PlatformMetrics

log = get_logger("events.bus")

_STOP = object()


class EventSink(Protocol):
    name: str

    def handle(self, event: Event) -> None: ...

    def close(self) -> None: ...


class EventBus:
    def __init__(self, queue_size: int = 10_000, metrics: PlatformMetrics | None = None) -> None:
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=queue_size)
        self._sinks: dict[str, EventSink] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._metrics = metrics
        self.published = 0
        self.delivered = 0
        self.dropped = 0
        self.sink_errors: dict[str, int] = defaultdict(int)

    def subscribe(self, sink: EventSink) -> None:
        with self._lock:
            if sink.name in self._sinks:
                raise ConflictError(f"sink {sink.name!r} already subscribed")
            self._sinks[sink.name] = sink

    def unsubscribe(self, name: str) -> EventSink | None:
        with self._lock:
            return self._sinks.pop(name, None)

    def sink(self, name: str) -> EventSink | None:
        with self._lock:
            return self._sinks.get(name)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._thread = threading.Thread(target=self._run, name="event-bus", daemon=True)
        self._thread.start()

    def publish(self, event: Event) -> bool:
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            self.dropped += 1
            if self._metrics:
                self._metrics.bus_dropped.inc()
            log.warning("event dropped: bus queue full", extra={"event_type": event.type.value})
            return False
        self.published += 1
        return True

    def _dispatch(self, event: Event) -> None:
        with self._lock:
            sinks = list(self._sinks.values())
        for sink in sinks:
            try:
                sink.handle(event)
            except Exception:
                self.sink_errors[sink.name] += 1
                log.exception(
                    "event sink failed", extra={"sink": sink.name, "event_id": event.event_id}
                )
        self.delivered += 1

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _STOP:
                    return
                self._dispatch(item)
            finally:
                self._queue.task_done()

    def flush(self, timeout: float = 5.0) -> bool:
        """Block until all queued events were dispatched (or timeout)."""
        if not self.running:
            while True:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    return True
                if item is not _STOP:
                    self._dispatch(item)
                self._queue.task_done()
        done = threading.Event()

        def waiter() -> None:
            self._queue.join()
            done.set()

        threading.Thread(target=waiter, daemon=True).start()
        return done.wait(timeout)

    def stop(self, timeout: float = 5.0) -> None:
        if self.running:
            self._queue.put(_STOP)
            assert self._thread is not None
            self._thread.join(timeout)
        self._thread = None
        with self._lock:
            sinks = list(self._sinks.values())
        for sink in sinks:
            try:
                sink.close()
            except Exception:
                log.exception("closing sink failed", extra={"sink": sink.name})

    def stats(self) -> dict[str, Any]:
        with self._lock:
            names = sorted(self._sinks)
        return {
            "running": self.running,
            "queued": self._queue.qsize(),
            "published": self.published,
            "delivered": self.delivered,
            "dropped": self.dropped,
            "sinks": names,
            "sink_errors": dict(self.sink_errors),
        }
