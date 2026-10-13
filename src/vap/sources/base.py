"""Frame-source abstraction shared by synthetic, file, image-sequence and RTSP sources."""

from __future__ import annotations

import abc
import enum
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

from vap.errors import SourceError
from vap.imaging.frame import Frame


class SourceState(enum.StrEnum):
    IDLE = "idle"
    CONNECTING = "connecting"
    OPEN = "open"
    RECONNECTING = "reconnecting"
    EXHAUSTED = "exhausted"
    FAILED = "failed"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class SourceInfo:
    kind: str
    uri: str
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    frame_count: int | None = None
    live: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SourceHealth:
    state: SourceState = SourceState.IDLE
    frames_read: int = 0
    reconnects: int = 0
    consecutive_failures: int = 0
    dropped_frames: int = 0
    last_error: str | None = None
    last_frame_at: float | None = None
    opened_at: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def healthy(self) -> bool:
        return self.state in (SourceState.OPEN, SourceState.EXHAUSTED)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["state"] = self.state.value
        out["healthy"] = self.healthy
        if self.last_frame_at is not None:
            out["last_frame_age_s"] = round(max(time.time() - self.last_frame_at, 0.0), 3)
        return out


class FrameSource(abc.ABC):
    """Pull-based source of RGB frames.

    Subclasses implement :meth:`_open`, :meth:`_read` and :meth:`_close`.
    :meth:`read` returns ``None`` once a finite source is exhausted.
    """

    kind: str = "abstract"

    def __init__(self, source_id: str) -> None:
        self.source_id = source_id
        self._health = SourceHealth()
        self._lock = threading.Lock()
        self._opened = False

    # -- template methods ---------------------------------------------------------

    @abc.abstractmethod
    def _open(self) -> None: ...

    @abc.abstractmethod
    def _read(self) -> Frame | None: ...

    @abc.abstractmethod
    def _close(self) -> None: ...

    @abc.abstractmethod
    def info(self) -> SourceInfo: ...

    # -- public API ---------------------------------------------------------------

    def open(self) -> None:
        if self._opened:
            return
        self._set_state(SourceState.CONNECTING)
        try:
            self._open()
        except SourceError as exc:
            self._record_error(str(exc), SourceState.FAILED)
            raise
        self._opened = True
        with self._lock:
            self._health.opened_at = time.time()
        self._set_state(SourceState.OPEN)

    def read(self) -> Frame | None:
        if not self._opened:
            raise SourceError("source is not open", details={"source_id": self.source_id})
        frame = self._read()
        with self._lock:
            if frame is None:
                if self._health.state not in (SourceState.CLOSED, SourceState.FAILED):
                    self._health.state = SourceState.EXHAUSTED
            else:
                self._health.frames_read += 1
                self._health.consecutive_failures = 0
                self._health.last_frame_at = time.time()
        if frame is not None and not frame.source_id:
            frame.source_id = self.source_id
        return frame

    def close(self) -> None:
        if self._opened:
            self._close()
        self._opened = False
        self._set_state(SourceState.CLOSED)

    def health(self) -> SourceHealth:
        with self._lock:
            snapshot = SourceHealth(**{f: getattr(self._health, f) for f in SourceHealth.__slots__})
            snapshot.extra = dict(self._health.extra)
            return snapshot

    @property
    def is_open(self) -> bool:
        return self._opened

    def __enter__(self) -> FrameSource:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __iter__(self) -> Iterator[Frame]:
        if not self._opened:
            self.open()
        while True:
            frame = self.read()
            if frame is None:
                return
            yield frame

    # -- helpers for subclasses ---------------------------------------------------

    def _set_state(self, state: SourceState) -> None:
        with self._lock:
            self._health.state = state

    def _record_error(self, message: str, state: SourceState | None = None) -> None:
        with self._lock:
            self._health.last_error = message
            self._health.consecutive_failures += 1
            if state is not None:
                self._health.state = state
