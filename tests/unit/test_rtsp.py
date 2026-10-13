"""RTSP adapter behaviour against fake capture backends (no real camera)."""

from __future__ import annotations

import random
import threading
import time

import pytest

from tests.fakes import FakeCapture, ScriptedFactory
from vap.errors import InvalidInputError, SourceError
from vap.sources.base import SourceState
from vap.sources.rtsp import ReconnectPolicy, RtspSource, redact_url, validate_stream_url

URL = "rtsp://user:secret@camera.local:554/stream1"


def make_source(factory: ScriptedFactory, **kwargs: object) -> tuple[RtspSource, list[float]]:
    waits: list[float] = []

    def wait(delay: float) -> bool:
        waits.append(delay)
        return False

    policy = kwargs.pop("policy", ReconnectPolicy(initial_delay=0.5, max_delay=4.0, jitter=0.0))
    src = RtspSource(
        "cam",
        URL,
        capture_factory=factory,
        policy=policy,  # type: ignore[arg-type]
        wait=wait,
        frame_timeout=kwargs.pop("frame_timeout", 1.0),  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )
    return src, waits


class TestPolicy:
    def test_exponential_with_cap(self) -> None:
        p = ReconnectPolicy(initial_delay=0.5, max_delay=3.0, multiplier=2.0, jitter=0.0)
        assert [p.delay(n) for n in range(1, 7)] == [0.0, 0.5, 1.0, 2.0, 3.0, 3.0]

    def test_jitter_bounds(self) -> None:
        p = ReconnectPolicy(initial_delay=1.0, max_delay=100, jitter=0.2)
        rng = random.Random(1)
        values = [p.delay(3, rng) for _ in range(200)]
        assert min(values) >= 1.6 and max(values) <= 2.4
        assert len(set(values)) > 50

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"initial_delay": -1},
            {"initial_delay": 5, "max_delay": 1},
            {"multiplier": 0.5},
            {"jitter": 2},
            {"max_attempts": 0},
        ],
    )
    def test_validation(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(InvalidInputError):
            ReconnectPolicy(**kwargs)  # type: ignore[arg-type]


class TestUrls:
    def test_validate(self) -> None:
        assert validate_stream_url("rtsps://cam/x") == "rtsps://cam/x"
        for bad in ["", "file:///etc/passwd", "rtsp:///nohost", "rtsp://cam/a b"]:
            with pytest.raises(InvalidInputError):
                validate_stream_url(bad)

    def test_redact(self) -> None:
        assert redact_url(URL) == "rtsp://user:***@camera.local:554/stream1"
        assert redact_url("rtsp://cam/x") == "rtsp://cam/x"


class TestRtspSource:
    def test_reads_frames_and_reports_health(self) -> None:
        factory = ScriptedFactory([lambda: FakeCapture(fps=30, size=(40, 30))])
        src, waits = make_source(factory)
        src.open()
        frame = src.read()
        assert frame is not None and frame.width == 40
        assert frame.image[0, 0].tolist()[0] == 7  # BGR -> RGB conversion
        info = src.info()
        assert info.live and info.fps == 30 and "secret" not in info.uri
        health = src.health()
        assert health.state == SourceState.OPEN and health.frames_read == 1 and health.healthy
        assert factory.urls == [URL] and waits == []
        src.close()
        assert factory.created[0].released.is_set()
        assert src.health().state == SourceState.CLOSED

    def test_initial_connect_retries_with_backoff(self) -> None:
        factory = ScriptedFactory(
            [
                lambda: FakeCapture(opened=False),
                lambda: FakeCapture(opened=False),
                lambda: FakeCapture(),
            ]
        )
        src, waits = make_source(factory)
        src.open()
        assert waits == [0.5, 1.0]
        assert src.read() is not None
        assert src.health().reconnects == 0
        src.close()

    def test_reconnects_after_stream_ends(self) -> None:
        factory = ScriptedFactory(
            [lambda: FakeCapture(frames=2), lambda: FakeCapture(frames=50, start_index=100)]
        )
        src, waits = make_source(factory)
        src.open()
        values = []
        for _ in range(4):
            frame = src.read()
            assert frame is not None
            values.append(int(frame.image[0, 0, 2]))
        assert values == [0, 1, 100, 101]
        health = src.health()
        assert health.reconnects == 1 and health.last_error == "capture returned no frame"
        assert waits == [0.5]  # one failure before the reconnect -> initial delay
        src.close()

    def test_frame_timeout_triggers_reconnect(self) -> None:
        factory = ScriptedFactory([lambda: FakeCapture(block_after=1), lambda: FakeCapture()])
        src, _ = make_source(factory, frame_timeout=0.2)
        src.open()
        assert src.read() is not None
        started = time.monotonic()
        assert src.read() is not None
        assert time.monotonic() - started >= 0.2
        health = src.health()
        assert health.reconnects == 1 and "frame timeout" in (health.last_error or "")
        src.close()

    def test_exception_in_capture_is_contained(self) -> None:
        factory = ScriptedFactory([lambda: FakeCapture(raise_after=0), lambda: FakeCapture()])
        src, _ = make_source(factory)
        src.open()
        assert src.read() is not None
        assert "decoder exploded" in (src.health().last_error or "")
        src.close()

    def test_gives_up_after_max_attempts(self) -> None:
        factory = ScriptedFactory([], default=lambda: FakeCapture(opened=False))
        policy = ReconnectPolicy(initial_delay=0.1, max_delay=1.0, jitter=0.0, max_attempts=3)
        src, waits = make_source(factory, policy=policy)
        with pytest.raises(SourceError, match="3 connection attempts"):
            src.open()
        assert len(factory.urls) == 3 and waits == [0.1, 0.2]
        assert src.health().state == SourceState.FAILED

    def test_factory_exception_is_wrapped(self) -> None:
        def boom(_: str) -> FakeCapture:
            raise OSError("ffmpeg missing")

        policy = ReconnectPolicy(initial_delay=0.0, max_delay=0.0, jitter=0.0, max_attempts=1)
        src = RtspSource("cam", URL, capture_factory=boom, policy=policy)
        with pytest.raises(SourceError):
            src.open()
        assert "ffmpeg missing" in (src.health().last_error or "")

    def test_stop_interrupts_backoff(self) -> None:
        factory = ScriptedFactory(
            [lambda: FakeCapture(frames=1)], default=lambda: FakeCapture(opened=False)
        )
        policy = ReconnectPolicy(initial_delay=30.0, max_delay=30.0, jitter=0.0)
        src = RtspSource("cam", URL, capture_factory=factory, policy=policy, frame_timeout=1.0)
        src.open()
        assert src.read() is not None
        result: list[object] = []
        t = threading.Thread(target=lambda: result.append(src.read()))
        t.start()
        time.sleep(0.2)
        started = time.monotonic()
        src.stop()
        t.join(2.0)
        assert not t.is_alive() and time.monotonic() - started < 2.0
        assert result == [None]
        src.close()

    def test_invalid_construction(self) -> None:
        with pytest.raises(InvalidInputError):
            RtspSource("cam", "http://", capture_factory=ScriptedFactory([]))
        with pytest.raises(InvalidInputError):
            RtspSource("cam", URL, frame_timeout=0)
