from __future__ import annotations

import socket
import threading
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from minecraft_ai.platforms.bedrock_x11 import IsolatedX11Capture, IsolationError
from minecraft_ai.platforms.x11_image_reply import (
    MAX_IMAGE_REPLY_BYTES,
    MIN_COALESCED_REPLY_BYTES,
    ImageReplyError,
    coalesce_image_reply,
)


class _Socket:
    def __init__(self, chunks: list[bytes], timeout: float | None = None) -> None:
        self.chunks = list(chunks)
        self.timeout = timeout
        self.calls: list[tuple[int, int, float | None]] = []
        self.closed = False

    def gettimeout(self) -> float | None:
        return self.timeout

    def settimeout(self, value: float | None) -> None:
        self.timeout = value

    def recv(self, count: int, flags: int = 0) -> bytes:
        self.calls.append((count, flags, self.timeout))
        return self.chunks.pop(0)

    def close(self) -> None:
        self.closed = True


def _protocol(raw: Any, size: int = MIN_COALESCED_REPLY_BYTES) -> Any:
    return SimpleNamespace(socket=raw, recv_packet_len=size, data_recv=b"H" * 32)


def test_large_fragmented_reply_is_joined_once_without_consuming_next_packet() -> None:
    body = bytes(range(256)) * 4096
    raw, peer = socket.socketpair()
    raw.settimeout(0.8)
    peer.settimeout(0.8)
    protocol = _protocol(raw, len(body) + 32)
    errors: list[BaseException] = []

    def send() -> None:
        try:
            for offset in range(0, len(body), 8192):
                peer.sendall(body[offset:offset + 8192])
            peer.sendall(b"next-reply")
        except BaseException as exc:
            errors.append(exc)

    writer = threading.Thread(target=send, daemon=True)
    writer.start()
    try:
        with coalesce_image_reply(protocol, deadline_ns=time.monotonic_ns() + 1_000_000_000):
            assert protocol.socket.fileno() == raw.fileno()
            result = protocol.socket.recv(len(body))
        assert result == body
        assert protocol.socket is raw
        assert raw.gettimeout() == 0.8
        assert raw.recv(10) == b"next-reply"
    finally:
        raw.close()
        peer.close()
        writer.join(timeout=1)
    assert not writer.is_alive()
    assert not errors


def test_fragmented_header_and_interleaved_small_reply_keep_single_read_semantics() -> None:
    raw = _Socket([b"head", b"reply", b"image"])
    protocol = _protocol(raw, 0)
    protocol.data_recv = b""
    with coalesce_image_reply(protocol, deadline_ns=time.monotonic_ns() + 1_000_000_000):
        assert protocol.socket.recv(4096) == b"head"
        protocol.data_recv = b"head"
        assert protocol.socket.recv(4096) == b"reply"
        protocol.recv_packet_len = 32
        assert protocol.socket.recv(4096) == b"image"
    assert len(raw.calls) == 3
    assert raw.gettimeout() is None


def test_xlib_private_bytesview_is_supported_without_copying_buffered_data() -> None:
    # Xlib uses a private object exposing len(), not bytes or memoryview.
    class View:
        def __len__(self) -> int:
            return MIN_COALESCED_REPLY_BYTES - 6

    raw = _Socket([b"abc", b"def"])
    protocol = _protocol(raw)
    protocol.data_recv = View()
    with coalesce_image_reply(protocol, deadline_ns=time.monotonic_ns() + 1_000_000_000):
        assert protocol.socket.recv(4096) == b"abcdef"
    assert [call[0] for call in raw.calls] == [6, 3]


@pytest.mark.parametrize("timeout,flags", [(None, socket.MSG_PEEK), (0.0, 0), (0.01, 0)])
def test_flags_and_existing_shorter_or_nonblocking_timeout_are_preserved(
    timeout: float | None, flags: int,
) -> None:
    raw = _Socket([b"data"], timeout)
    protocol = _protocol(raw, 32)
    with coalesce_image_reply(protocol, deadline_ns=time.monotonic_ns() + 1_000_000_000):
        assert protocol.socket.recv(4096, flags) == b"data"
    assert raw.calls[0][1] == flags
    assert raw.gettimeout() == timeout
    if timeout is not None:
        assert raw.calls[0][2] == timeout
    else:
        assert 0 < raw.calls[0][2] <= 1


def test_absolute_deadline_is_not_refreshed_by_fragments(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [1_000_000_000]
    monkeypatch.setattr(
        "minecraft_ai.platforms.x11_image_reply.time.monotonic_ns", lambda: clock[0],
    )
    raw = _Socket([b"ab", b"cd", b"ef"], 5.0)
    original = raw.recv

    def fragment(count: int, flags: int = 0) -> bytes:
        result = original(count, flags)
        clock[0] += 200_000_000
        return result

    raw.recv = fragment  # type: ignore[method-assign]
    protocol = _protocol(raw)
    protocol.data_recv = b"H" * (MIN_COALESCED_REPLY_BYTES - 6)
    with pytest.raises(ImageReplyError, match="budget expired"):
        with coalesce_image_reply(protocol, deadline_ns=1_500_000_000):
            protocol.socket.recv(4096)
    assert [round(call[2], 2) for call in raw.calls] == [0.5, 0.3, 0.1]
    assert raw.gettimeout() == 5.0
    assert protocol.socket is raw


def test_slow_body_exits_on_real_socket_timeout_and_restores_socket() -> None:
    raw, peer = socket.socketpair()
    protocol = _protocol(raw)
    peer.sendall(b"fragment")
    start = time.monotonic()
    try:
        with pytest.raises(ImageReplyError, match="timed out"):
            with coalesce_image_reply(protocol, deadline_ns=time.monotonic_ns() + 30_000_000):
                protocol.socket.recv(MIN_COALESCED_REPLY_BYTES)
        assert time.monotonic() - start < 1.0
        assert protocol.socket is raw
        assert raw.gettimeout() is None
    finally:
        raw.close()
        peer.close()


@pytest.mark.parametrize("case", ["expired_entry", "expired_before_header", "oversize", "eof"])
def test_refusals_restore_exact_socket(case: str, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [1]
    monkeypatch.setattr(
        "minecraft_ai.platforms.x11_image_reply.time.monotonic_ns", lambda: clock[0],
    )
    raw = _Socket([b""])
    protocol = _protocol(raw)
    deadline = 100
    with pytest.raises(ImageReplyError):
        if case == "expired_entry":
            clock[0] = 100
        with coalesce_image_reply(protocol, deadline_ns=deadline):
            if case == "expired_before_header":
                clock[0] = 100
                protocol.recv_packet_len = 0
            elif case == "oversize":
                protocol.recv_packet_len = MAX_IMAGE_REPLY_BYTES + 1
            protocol.socket.recv(MAX_IMAGE_REPLY_BYTES)
    assert protocol.socket is raw
    assert raw.gettimeout() is None
    assert len(raw.calls) == (1 if case == "eof" else 0)


def test_nested_wrapper_refused_without_replacing_owner() -> None:
    raw = _Socket([])
    protocol = _protocol(raw)
    with coalesce_image_reply(protocol, deadline_ns=time.monotonic_ns() + 1_000_000_000):
        outer = protocol.socket
        with pytest.raises(ImageReplyError, match="nested"):
            with coalesce_image_reply(protocol, deadline_ns=time.monotonic_ns() + 1_000_000_000):
                pytest.fail("nested owner admitted")
        assert protocol.socket is outer
    assert protocol.socket is raw


def test_expired_completion_is_rejected_even_if_read_finished_before_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [1]
    monkeypatch.setattr(
        "minecraft_ai.platforms.x11_image_reply.time.monotonic_ns", lambda: clock[0],
    )
    raw = _Socket([b"response"])
    protocol = _protocol(raw, 32)
    with pytest.raises(ImageReplyError, match="budget expired"):
        with coalesce_image_reply(protocol, deadline_ns=100):
            assert protocol.socket.recv(4096) == b"response"
            clock[0] = 100
    assert protocol.socket is raw
    assert raw.gettimeout() is None


def test_wrapper_restores_on_unrelated_exception_and_does_not_clobber_replacement() -> None:
    raw = _Socket([])
    protocol = _protocol(raw)
    with pytest.raises(ValueError, match="fixture"):
        with coalesce_image_reply(protocol, deadline_ns=time.monotonic_ns() + 1_000_000_000):
            raise ValueError("fixture")
    assert protocol.socket is raw
    replacement = object()
    with pytest.raises(ImageReplyError, match="ownership changed"):
        with coalesce_image_reply(protocol, deadline_ns=time.monotonic_ns() + 1_000_000_000):
            protocol.socket = replacement
    assert protocol.socket is replacement


def test_capture_timeout_discards_connection_and_forbids_reuse_or_fallback() -> None:
    raw = _Socket([b""])
    protocol = _protocol(raw)
    capture = object.__new__(IsolatedX11Capture)
    capture._reply_protocol = protocol
    capture._reply_socket = raw
    capture._reply_failed = False
    capture._capture_budget_ms = 500
    capture._frame_id = 0
    capture.target_window_id = 7
    capture._X = SimpleNamespace(ZPixmap=2)
    capture._bounds = lambda: {"left": 0, "top": 0, "width": 4, "height": 4}  # type: ignore[method-assign]
    capture._content_rect = lambda *_args: None  # type: ignore[method-assign]
    window = SimpleNamespace(get_image=lambda *_args: protocol.socket.recv(MAX_IMAGE_REPLY_BYTES))
    display = Mock()
    display.create_resource_object.return_value = window
    capture._display = display
    capture._mss_module = Mock()
    with pytest.raises(IsolationError, match="ended before"):
        capture.capture()
    assert capture._reply_failed
    assert raw.closed
    assert protocol.socket is raw
    assert capture._frame_id == 0
    capture._mss_module.mss.assert_not_called()
    display.screen.assert_not_called()
    with pytest.raises(IsolationError, match="reconnect required"):
        capture.capture()
    capture.close()
    display.close.assert_not_called()  # Xlib flush-on-close must not reuse poison.
    assert len(raw.calls) == 1


@pytest.mark.parametrize("failure", ["reset", "restore", "eof"])
def test_partial_failure_is_fatal_and_never_falls_back(failure: str) -> None:
    raw = _Socket([b"fragment", b""])
    protocol = _protocol(raw)
    original_recv = raw.recv
    original_settimeout = raw.settimeout
    calls = [0]

    def read(count: int, flags: int = 0) -> bytes:
        calls[0] += 1
        if calls[0] == 2 and failure == "reset":
            raise ConnectionResetError("fixture reset after partial body")
        return original_recv(count, flags)

    def settimeout(value: float | None) -> None:
        if failure == "restore" and calls[0] == 1 and value is None:
            raise OSError("fixture restore failure")
        original_settimeout(value)

    raw.recv = read  # type: ignore[method-assign]
    raw.settimeout = settimeout  # type: ignore[method-assign]
    capture = object.__new__(IsolatedX11Capture)
    capture._reply_protocol = protocol
    capture._reply_socket = raw
    capture._reply_failed = False
    drawable = Mock()
    drawable.get_image.side_effect = lambda: protocol.socket.recv(MAX_IMAGE_REPLY_BYTES)
    with pytest.raises(IsolationError):
        capture._get_image(drawable, deadline_ns=time.monotonic_ns() + 1_000_000_000)
    assert capture._reply_failed
    assert raw.closed
    assert protocol.socket is raw
    if failure != "restore":
        assert raw.gettimeout() is None
    with pytest.raises(IsolationError, match="reconnect required"):
        capture._get_image(drawable, deadline_ns=time.monotonic_ns() + 1_000_000_000)
    assert drawable.get_image.call_count == 1


def test_socket_ownership_change_discards_only_original_connection() -> None:
    original = _Socket([])
    replacement = _Socket([])
    capture = object.__new__(IsolatedX11Capture)
    capture._reply_protocol = _protocol(replacement)
    capture._reply_socket = original
    capture._reply_failed = False
    drawable = Mock()
    with pytest.raises(IsolationError, match="reconnect required"):
        capture._get_image(drawable, deadline_ns=time.monotonic_ns() + 1_000_000_000)
    assert original.closed
    assert not replacement.closed
    assert capture._reply_protocol.socket is replacement
    assert capture._reply_failed
    drawable.get_image.assert_not_called()
