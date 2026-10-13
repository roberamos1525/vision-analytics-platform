"""RTSP / network camera adapter with reconnect, backoff and frame timeouts.

Design
------
* The capture object (``cv2.VideoCapture`` with the FFmpeg backend by default)
  is created through an injectable factory so the adapter can be tested with
  fakes and swapped for other backends.
* Blocking ``read()`` calls run on a per-connection reader thread that keeps
  only the most recent frame (live cameras should never build up latency).
  :meth:`RtspSource.read` waits for that slot with ``frame_timeout``; a stall
  or read failure tears the connection down and reconnects.
* Reconnects use exponential backoff with jitter and an optional attempt cap.
  When the cap is reached the source enters ``FAILED`` and raises.
* :meth:`close` interrupts any pending backoff wait immediately.

This adapter has been exercised only against fake capture backends in the
test-suite; it has not been tested against a physical camera.
"""

from __future__ import annotations

import contextlib
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import numpy as np

from vap.errors import InvalidInputError, SourceError
from vap.imaging.frame import Frame
from vap.imaging.opencv_backend import load_cv2
from vap.sources.base import FrameSource, SourceInfo, SourceState
from vap.sources.video_file import (
    CAP_PROP_FPS,
    CAP_PROP_FRAME_HEIGHT,
    CAP_PROP_FRAME_WIDTH,
    Capture,
    CaptureFactory,
    bgr_to_frame_image,
)

ALLOWED_SCHEMES = ("rtsp", "rtsps", "rtmp", "http", "https")


def validate_stream_url(url: str) -> str:
    if not url or len(url) > 2048:
        raise InvalidInputError("stream URL must be 1..2048 characters")
    parts = urlsplit(url)
    if parts.scheme.lower() not in ALLOWED_SCHEMES:
        raise InvalidInputError(
            "unsupported stream scheme", details={"allowed": list(ALLOWED_SCHEMES)}
        )
    if not parts.hostname:
        raise InvalidInputError("stream URL must include a host")
    if any(c in url for c in "\r\n\t "):
        raise InvalidInputError("stream URL must not contain whitespace")
    return url


def redact_url(url: str) -> str:
    """Hide the password component of a URL for logs and API responses."""
    parts = urlsplit(url)
    if parts.password is None:
        return url
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    user = parts.username or ""
    return urlunsplit((parts.scheme, f"{user}:***@{host}", parts.path, parts.query, parts.fragment))


@dataclass(frozen=True, slots=True)
class ReconnectPolicy:
    initial_delay: float = 0.5
    max_delay: float = 30.0
    multiplier: float = 2.0
    jitter: float = 0.1
    max_attempts: int | None = None

    def __post_init__(self) -> None:
        if self.initial_delay < 0 or self.max_delay < self.initial_delay:
            raise InvalidInputError("reconnect delays must satisfy 0 <= initial <= max")
        if self.multiplier < 1.0:
            raise InvalidInputError("backoff multiplier must be >= 1")
        if not 0.0 <= self.jitter <= 1.0:
            raise InvalidInputError("jitter must be within [0, 1]")
        if self.max_attempts is not None and self.max_attempts < 1:
            raise InvalidInputError("max_attempts must be >= 1 when set")

    def delay(self, attempt: int, rng: random.Random | None = None) -> float:
        """Backoff before reconnect ``attempt`` (1-based); attempt 1 has no delay."""
        if attempt <= 1:
            return 0.0
        exponent = attempt - 2
        if exponent > 60:  # avoid float overflow for long outages
            base = self.max_delay
        else:
            base = min(self.initial_delay * self.multiplier**exponent, self.max_delay)
        if self.jitter and rng is not None:
            base *= 1.0 + rng.uniform(-self.jitter, self.jitter)
        return max(0.0, min(base, self.max_delay))


def opencv_stream_capture(open_timeout_s: float, read_timeout_s: float) -> CaptureFactory:
    """Factory creating FFmpeg-backed ``cv2.VideoCapture`` objects with timeouts."""
    cv2: Any = load_cv2()

    def factory(url: str) -> Capture:
        params = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
            int(open_timeout_s * 1000),
            cv2.CAP_PROP_READ_TIMEOUT_MSEC,
            int(read_timeout_s * 1000),
        ]
        return cv2.VideoCapture(url, cv2.CAP_FFMPEG, params)

    return factory


class _LatestFrameSlot:
    """Mailbox holding the newest frame plus a terminal error.

    A newer frame replaces an unread one (live streams must not build up
    latency); a terminal error is only delivered after the pending frame.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._frame: tuple[str, Any, float] | None = None
        self._error: tuple[str, Any, float] | None = None
        self.overwritten = 0

    def put(self, kind: str, payload: Any) -> None:
        with self._cond:
            item = (kind, payload, time.time())
            if kind == "frame":
                if self._frame is not None:
                    self.overwritten += 1
                self._frame = item
            else:
                self._error = item
            self._cond.notify_all()

    def take(self, timeout: float) -> tuple[str, Any, float] | None:
        with self._cond:
            if self._frame is None and self._error is None:
                self._cond.wait(timeout)
            if self._frame is not None:
                item, self._frame = self._frame, None
                return item
            item, self._error = self._error, None
            return item


class _ReaderThread(threading.Thread):
    def __init__(self, cap: Capture, slot: _LatestFrameSlot, name: str) -> None:
        super().__init__(name=name, daemon=True)
        self.cap = cap
        self.slot = slot
        self.stop_event = threading.Event()

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                ok, image = self.cap.read()
            except Exception as exc:  # capture backends may raise arbitrary errors
                self.slot.put("error", f"capture read raised {type(exc).__name__}: {exc}")
                return
            if self.stop_event.is_set():
                return
            if not ok or image is None:
                self.slot.put("error", "capture returned no frame")
                return
            self.slot.put("frame", image)


class RtspSource(FrameSource):
    kind = "rtsp"

    def __init__(
        self,
        source_id: str,
        url: str,
        *,
        capture_factory: CaptureFactory | None = None,
        policy: ReconnectPolicy | None = None,
        frame_timeout: float = 5.0,
        open_timeout: float = 10.0,
        rng: random.Random | None = None,
        wait: Callable[[float], bool] | None = None,
    ) -> None:
        super().__init__(source_id)
        self.url = validate_stream_url(url)
        if frame_timeout <= 0 or open_timeout <= 0:
            raise InvalidInputError("timeouts must be positive")
        self.policy = policy or ReconnectPolicy()
        self.frame_timeout = frame_timeout
        self.open_timeout = open_timeout
        self._factory = capture_factory
        self._rng = rng or random.Random()
        self._stop = threading.Event()
        self._wait = wait or self._stop.wait
        self._cap: Capture | None = None
        self._reader: _ReaderThread | None = None
        self._slot = _LatestFrameSlot()
        self._index = 0
        self._fps: float | None = None
        self._size: tuple[int, int] | None = None
        self._backoff_log: list[float] = []

    # -- connection management ----------------------------------------------------

    def _capture_factory(self) -> CaptureFactory:
        if self._factory is None:
            self._factory = opencv_stream_capture(self.open_timeout, self.frame_timeout)
        return self._factory

    def _connect_once(self) -> None:
        factory = self._capture_factory()
        try:
            cap = factory(self.url)
        except Exception as exc:
            raise SourceError(f"capture factory failed: {type(exc).__name__}: {exc}") from exc
        if not cap.isOpened():
            cap.release()
            raise SourceError(f"could not open stream {redact_url(self.url)}")
        fps = float(cap.get(CAP_PROP_FPS) or 0.0)
        self._fps = fps if 0 < fps <= 240 else None
        w, h = int(cap.get(CAP_PROP_FRAME_WIDTH) or 0), int(cap.get(CAP_PROP_FRAME_HEIGHT) or 0)
        self._size = (w, h) if w > 0 and h > 0 else self._size
        self._cap = cap
        self._slot = _LatestFrameSlot()
        self._reader = _ReaderThread(cap, self._slot, f"rtsp-reader-{self.source_id}")
        self._reader.start()

    def _connect_with_backoff(self, *, reconnect: bool) -> None:
        # Failures that preceded this cycle (e.g. a stalled stream) count towards the
        # backoff so a camera that connects and immediately drops cannot spin hot.
        with self._lock:
            prior = self._health.consecutive_failures if reconnect else 0
        attempt = 0
        while not self._stop.is_set():
            attempt += 1
            if self.policy.max_attempts is not None and attempt > self.policy.max_attempts:
                with self._lock:
                    cause = self._health.last_error
                message = f"giving up after {self.policy.max_attempts} connection attempts"
                if cause:
                    message += f" (last error: {cause})"
                self._record_error(message, SourceState.FAILED)
                raise SourceError(message, details={"source_id": self.source_id})
            delay = self.policy.delay(attempt + prior, self._rng)
            if delay > 0:
                self._backoff_log.append(delay)
                if self._wait(delay) or self._stop.is_set():
                    break
            try:
                self._connect_once()
            except SourceError as exc:
                self._record_error(exc.message)
                continue
            if reconnect:
                with self._lock:
                    self._health.reconnects += 1
            self._set_state(SourceState.OPEN)
            return
        raise SourceError("source closed while connecting", details={"source_id": self.source_id})

    def _teardown(self) -> None:
        reader, cap = self._reader, self._cap
        self._reader, self._cap = None, None
        if reader is not None:
            reader.stop_event.set()
        if cap is not None:
            # Releasing a broken capture must never mask the original error.
            with contextlib.suppress(Exception):
                cap.release()
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=1.0)

    # -- FrameSource hooks ----------------------------------------------------------

    def _open(self) -> None:
        self._stop.clear()
        self._connect_with_backoff(reconnect=False)

    def _read(self) -> Frame | None:
        while not self._stop.is_set():
            if self._cap is None:
                self._set_state(SourceState.RECONNECTING)
                try:
                    self._connect_with_backoff(reconnect=True)
                except SourceError:
                    if self._stop.is_set():
                        return None
                    raise
            item = self._slot.take(self.frame_timeout)
            if self._stop.is_set():
                return None
            if item is None:
                self._record_error(f"frame timeout: no frame within {self.frame_timeout:.1f}s")
                self._teardown()
                continue
            kind, payload, _ = item
            if kind == "error":
                self._record_error(str(payload))
                self._teardown()
                continue
            image: np.ndarray = payload
            with self._lock:
                self._health.dropped_frames = self._slot.overwritten
            frame = Frame(bgr_to_frame_image(image), self._index, time.time(), self.source_id)
            self._index += 1
            return frame
        return None

    def _close(self) -> None:
        self._stop.set()
        self._teardown()

    def stop(self) -> None:
        """Interrupt a blocked read/backoff from another thread."""
        self._stop.set()

    def info(self) -> SourceInfo:
        width, height = self._size if self._size else (None, None)
        return SourceInfo(
            kind=self.kind,
            uri=redact_url(self.url),
            width=width,
            height=height,
            fps=self._fps,
            frame_count=None,
            live=True,
        )

    @property
    def backoff_history(self) -> list[float]:
        return list(self._backoff_log)
