from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from http import HTTPStatus

import pytest

from minecraft_ai import operator_server as operator
from minecraft_ai.platforms.bedrock_x11 import CapturedFrame


def _frame(frame_id: int = 1) -> CapturedFrame:
    return CapturedFrame(frame_id, time.monotonic_ns(), 1, 1, b"\0\0\0\xff")


def _snapshot(frame: CapturedFrame | None = None) -> operator._FrameSnapshot:
    frame = _frame() if frame is None else frame
    return operator._FrameSnapshot(
        frame=frame,
        dhash="0" * 16,
        hud_complete=True,
        survival_hud=True,
        full_jpeg=b"full-jpeg",
        public_jpeg=b"public-jpeg",
        captured_ns=frame.captured_ns,
        width=frame.width,
        height=frame.height,
    )


@pytest.fixture
def snapshot_service(monkeypatch) -> Iterator[Callable[[], threading.Thread]]:
    """Keep producer state and threads local to each regression."""
    mutex = threading.RLock()
    monkeypatch.setattr(operator, "_frame_snapshot_mutex", mutex)
    monkeypatch.setattr(operator, "_frame_snapshot_ready", threading.Condition(mutex))
    monkeypatch.setattr(operator, "_latest_frame_snapshot", None)
    monkeypatch.setattr(operator, "_frame_snapshot_service_running", False)
    monkeypatch.setattr(operator, "_frame_snapshot_thread", None)
    monkeypatch.setattr(operator, "_frame_snapshot_stop", None)
    monkeypatch.setattr(operator, "FRAME_SERVICE_TARGET_INTERVAL_S", 0.001)
    threads: list[threading.Thread] = []
    stops: list[threading.Event] = []

    def launch() -> threading.Thread:
        stop = threading.Event()
        stops.append(stop)
        operator._frame_snapshot_stop = stop
        operator._frame_snapshot_service_running = True
        thread = threading.Thread(target=operator._frame_snapshot_loop, args=(stop,), daemon=True)
        operator._frame_snapshot_thread = thread
        threads.append(thread)
        thread.start()
        return thread

    yield launch
    for stop in stops:
        stop.set()
    with operator._frame_snapshot_ready:
        operator._frame_snapshot_service_running = False
        operator._frame_snapshot_ready.notify_all()
    for thread in threads:
        thread.join(timeout=3)
        assert not thread.is_alive(), "snapshot producer leaked from regression"


def test_slow_encoding_does_not_block_cached_snapshot_reader(monkeypatch, snapshot_service) -> None:
    cached = _snapshot()
    operator._latest_frame_snapshot = cached
    encoding = threading.Event()
    release_encoding = threading.Event()
    reader_done = threading.Event()
    read: list[operator._FrameSnapshot | None] = []

    def build(frame: CapturedFrame) -> operator._FrameSnapshot:
        encoding.set()
        assert release_encoding.wait(3)
        return _snapshot(frame)

    def read_cached() -> None:
        read.append(operator._latest_frame_snapshot_or_none(0))
        reader_done.set()

    monkeypatch.setattr(operator, "_capture_live_bedrock_frame", _frame)
    monkeypatch.setattr(operator, "_build_frame_snapshot", build)
    snapshot_service()
    reader = threading.Thread(target=read_cached, daemon=True)
    try:
        assert encoding.wait(2)
        reader.start()
        assert reader_done.wait(0.5), "JPEG encoding held the shared reader lock"
        assert read == [cached]
    finally:
        release_encoding.set()
        if reader.ident is not None:
            reader.join(timeout=2)


def test_encoding_exception_recovers_to_next_snapshot(monkeypatch, snapshot_service) -> None:
    calls = 0
    built = 0
    next_capture = threading.Event()
    release_capture = threading.Event()
    good = _snapshot(_frame(2))

    def capture() -> CapturedFrame | None:
        nonlocal calls
        calls += 1
        if calls <= 2:
            return _frame(calls)
        next_capture.set()
        assert release_capture.wait(3)
        return None

    def build(frame: CapturedFrame) -> operator._FrameSnapshot:
        nonlocal built
        built += 1
        if built == 1:
            raise ValueError("temporary JPEG encoding failure")
        assert frame.frame_id == 2
        return good

    monkeypatch.setattr(operator, "_capture_live_bedrock_frame", capture)
    monkeypatch.setattr(operator, "_build_frame_snapshot", build)
    snapshot_service()
    try:
        assert next_capture.wait(2), "producer did not survive the encoding failure"
        assert operator._latest_frame_snapshot_or_none(0) is good
        assert built == 2
    finally:
        operator._frame_snapshot_stop.set()
        release_capture.set()


@pytest.mark.parametrize("raises", [False, True])
def test_capture_failure_invalidates_cached_snapshot(monkeypatch, snapshot_service, raises) -> None:
    operator._latest_frame_snapshot = _snapshot()
    calls = 0
    next_capture = threading.Event()
    release_capture = threading.Event()

    def capture() -> CapturedFrame | None:
        nonlocal calls
        calls += 1
        if calls == 1:
            if raises:
                raise RuntimeError("capture temporarily unavailable")
            return None
        next_capture.set()
        assert release_capture.wait(3)
        return None

    monkeypatch.setattr(operator, "_capture_live_bedrock_frame", capture)
    snapshot_service()
    try:
        assert next_capture.wait(2)
        with operator._frame_snapshot_mutex:
            assert operator._latest_frame_snapshot is None
    finally:
        operator._frame_snapshot_stop.set()
        release_capture.set()


@pytest.mark.parametrize("stale", [False, True], ids=["absent", "stale"])
@pytest.mark.parametrize("endpoint", ["readiness", "frame"])
def test_running_service_never_falls_back_to_request_thread_capture(
    monkeypatch, snapshot_service, stale, endpoint
) -> None:
    operator._frame_snapshot_service_running = True
    operator._frame_snapshot_stop = threading.Event()
    monkeypatch.setattr(operator, "FRAME_SERVICE_WAIT_TIMEOUT_S", 0)
    if stale:
        frame = _frame()
        operator._latest_frame_snapshot = _snapshot(
            CapturedFrame(
                frame.frame_id,
                time.monotonic_ns() - operator.FRAME_SERVICE_STALE_NS - 1_000_000,
                frame.width,
                frame.height,
                frame.bgra,
            )
        )

    def inline_capture() -> CapturedFrame:
        pytest.fail("HTTP/readiness must not queue behind the background capture")

    monkeypatch.setattr(operator, "_capture_live_bedrock_frame", inline_capture)
    if endpoint == "readiness":
        assert operator._readiness_capture() == (None, False)
    else:
        responses: list[tuple[HTTPStatus, dict[str, str]]] = []
        handler = object.__new__(operator.OperatorRequestHandler)
        handler.path = "/api/frame.png?size=public"
        handler._send_json = lambda status, payload: responses.append((status, payload))
        handler._send_bytes = lambda *_args, **_kwargs: pytest.fail("must not serve stale JPEG")
        handler._get_frame()
        assert len(responses) == 1
        assert responses[0][0] == HTTPStatus.SERVICE_UNAVAILABLE


def test_stop_wakes_waiting_snapshot_reader(monkeypatch, snapshot_service) -> None:
    waiting = threading.Event()
    reader_done = threading.Event()
    read: list[operator._FrameSnapshot | None] = []
    operator._frame_snapshot_service_running = True
    operator._frame_snapshot_stop = threading.Event()

    class ObservedCondition(threading.Condition):
        def wait(self, timeout: float | None = None) -> bool:
            waiting.set()
            return super().wait(timeout)

    monkeypatch.setattr(
        operator, "_frame_snapshot_ready", ObservedCondition(operator._frame_snapshot_mutex)
    )

    def read_waiting() -> None:
        read.append(operator._latest_frame_snapshot_or_none(30))
        reader_done.set()

    reader = threading.Thread(target=read_waiting, daemon=True)
    reader.start()
    try:
        assert waiting.wait(2)
        operator._stop_frame_snapshot_service()
        assert reader_done.wait(0.5), "shutdown left the reader waiting for its full timeout"
        assert read == [None]
    finally:
        with operator._frame_snapshot_ready:
            operator._frame_snapshot_service_running = False
            operator._latest_frame_snapshot = _snapshot()
            operator._frame_snapshot_ready.notify_all()
        reader.join(timeout=2)
        assert not reader.is_alive()


def test_stopped_generation_cannot_publish_over_replacement(monkeypatch, snapshot_service) -> None:
    capturing = threading.Event()
    release_capture = threading.Event()

    def capture() -> CapturedFrame:
        capturing.set()
        assert release_capture.wait(3)
        return _frame(1)

    monkeypatch.setattr(operator, "_capture_live_bedrock_frame", capture)
    monkeypatch.setattr(operator, "_build_frame_snapshot", _snapshot)
    thread = snapshot_service()
    try:
        assert capturing.wait(2)
        replacement = _snapshot(_frame(2))
        with operator._frame_snapshot_ready:
            operator._frame_snapshot_stop.set()
            operator._frame_snapshot_stop = threading.Event()
            operator._latest_frame_snapshot = replacement
        release_capture.set()
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert operator._latest_frame_snapshot is replacement
    finally:
        release_capture.set()


def test_spectator_budget_spaces_capture_starts_without_catchup_or_timestamp_refresh(monkeypatch):
    clock = [0]
    starts = []
    waits = []
    costs_ns = [20_000_000, 310_000_000, 20_000_000]
    mutex = threading.RLock()
    monkeypatch.setattr(operator, "_frame_snapshot_mutex", mutex)
    monkeypatch.setattr(operator, "_frame_snapshot_ready", threading.Condition(mutex))
    monkeypatch.setattr(operator, "_latest_frame_snapshot", None)
    monkeypatch.setattr(operator.time, "monotonic_ns", lambda: clock[0])
    assert operator.FRAME_SERVICE_MAX_FPS == 4
    assert operator.FRAME_SERVICE_TARGET_INTERVAL_S == .25

    class Stop:
        def is_set(self):
            return len(starts) >= 3

        def wait(self, seconds):
            waits.append(seconds)
            clock[0] += round(seconds * 1_000_000_000)
            return False

    def capture():
        starts.append(clock[0])
        return CapturedFrame(len(starts), clock[0], 1, 1, b"\0\0\0\xff")

    def encode(frame):
        clock[0] += costs_ns[frame.frame_id - 1]
        return _snapshot(frame)

    monkeypatch.setattr(operator, "_capture_live_bedrock_frame", capture)
    monkeypatch.setattr(operator, "_build_frame_snapshot", encode)
    operator._frame_snapshot_loop(Stop())
    assert starts == [0, 250_000_000, 560_000_000]
    assert waits == [pytest.approx(.23)]
    assert operator._latest_frame_snapshot.captured_ns == 250_000_000
    assert operator._latest_frame_snapshot.frame.frame_id == 2


def test_many_viewers_reuse_snapshot_without_recapture_or_timestamp_refresh(monkeypatch):
    frame = CapturedFrame(7, 1_000_000_000, 1, 1, b"\0\0\0\xff")
    snapshot = _snapshot(frame)
    monkeypatch.setattr(operator, "_latest_frame_snapshot", snapshot)
    monkeypatch.setattr(operator, "_frame_snapshot_service_running", True)
    monkeypatch.setattr(operator.time, "monotonic_ns", lambda: 1_100_000_000)
    monkeypatch.setattr(
        operator, "_capture_live_bedrock_frame", lambda: pytest.fail("extra capture")
    )
    monkeypatch.setattr(
        operator, "_build_frame_snapshot", lambda *_args: pytest.fail("extra encode")
    )
    for _ in range(100):
        result = operator._latest_frame_snapshot_or_none(0)
        assert result is snapshot
        assert result.captured_ns == 1_000_000_000
        assert result.full_jpeg is snapshot.full_jpeg
