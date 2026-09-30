from __future__ import annotations

import json
import os
import struct
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from minecraft_ai import operator_server as operator
from minecraft_ai.agent_lifecycle import AgentProcess
from minecraft_ai.platforms.bedrock_x11 import (
    CapturedFrame,
    ImageCaptureTimeout,
    IsolatedX11Capture,
    IsolationError,
)
from minecraft_ai.platforms.frame_cache import PublishedFrameCapture, read_agent_frame


def _owner() -> AgentProcess:
    return AgentProcess(
        pid=os.getpid(), started_ns=123, display=":2", window_id=42,
        instance_id="bedrock:1.26.52.3:x11:42", role="generalist",
        proc_start_ticks=456, command_sha256="a" * 64,
    )


def _frame() -> CapturedFrame:
    return CapturedFrame(4, time.monotonic_ns(), 2, 1, b"\x01\x02\x03\xff" * 2)


def _publish(path: Path, owner: AgentProcess | None = None) -> PublishedFrameCapture:
    source = Mock()
    source.capture.return_value = _frame()
    publisher = PublishedFrameCapture(source, owner or _owner(), path=path)
    assert publisher.capture() is source.capture.return_value
    return publisher


def test_complete_pixels_and_original_capture_time_are_shared(tmp_path: Path) -> None:
    path = tmp_path / "frame"
    publisher = _publish(path)
    assert read_agent_frame(_owner(), path=path) == publisher.source.capture.return_value
    assert path.stat().st_mode & 0o777 == 0o600
    publisher.close()
    publisher.source.close.assert_called_once()
    assert not path.exists()


@pytest.mark.parametrize("field,value", [
    ("pid", 999), ("started_ns", 999), ("proc_start_ticks", 999),
    ("command_sha256", "b" * 64), ("display", ":0"), ("window_id", 99),
    ("instance_id", "bedrock:old"), ("allow_host_capture", True),
])
def test_pixels_cannot_cross_agent_generation_or_capture_target(tmp_path, field, value) -> None:
    path = tmp_path / "frame"
    _publish(path)
    assert read_agent_frame(replace(_owner(), **{field: value}), path=path) is None


def _alter_header(path: Path, updates: dict[str, object]) -> None:
    payload = path.read_bytes()
    header_len = struct.unpack("!I", payload[:4])[0]
    metadata = json.loads(payload[4:4 + header_len])
    metadata.update(updates)
    header = json.dumps(metadata).encode()
    path.write_bytes(struct.pack("!I", len(header)) + header + payload[4 + header_len:])


@pytest.mark.parametrize("updates", [
    {"captured_ns": 0}, {"captured_ns": 2**63}, {"captured_ns": 1},
    {"frame_id": -1}, {"frame_id": True}, {"width": False}, {"height": 0},
    {"width": 1}, {"width": 10**12}, {"schema_version": 2},
    {"schema_version": True},
])
def test_stale_future_incomplete_or_malformed_frames_are_unavailable(tmp_path, updates) -> None:
    path = tmp_path / "frame"
    _publish(path)
    _alter_header(path, updates)
    assert read_agent_frame(_owner(), path=path) is None


@pytest.mark.parametrize("kind", ["symlink", "fifo", "public", "truncated", "oversize-header"])
def test_frame_read_is_bounded_and_never_follows_special_files(tmp_path, kind) -> None:
    path = tmp_path / "frame"
    if kind == "symlink":
        original = tmp_path / "original"
        _publish(original)
        path.symlink_to(original)
    elif kind == "fifo":
        os.mkfifo(path)
    else:
        _publish(path)
        if kind == "public":
            path.chmod(0o644)
        elif kind == "truncated":
            path.write_bytes(path.read_bytes()[:-1])
        else:
            path.write_bytes(struct.pack("!I", 8192))
    assert read_agent_frame(_owner(), path=path) is None


def test_publication_is_throttled_and_failure_does_not_change_capture(tmp_path) -> None:
    path = tmp_path / "frame"
    publisher = _publish(path)
    identity = path.stat().st_ino
    publisher.capture()
    assert path.stat().st_ino == identity
    publisher._last_publish_ns = 0
    publisher._publish = Mock(side_effect=OSError("no space"))
    assert publisher.capture() is publisher.source.capture.return_value
    publisher._publish.assert_called_once()


def test_old_capture_close_cannot_remove_replacement_generation(tmp_path) -> None:
    path = tmp_path / "frame"
    old = _publish(path)
    new_owner = replace(_owner(), started_ns=999)
    new = _publish(path, new_owner)
    old.close()
    assert read_agent_frame(new_owner, path=path) == new.source.capture.return_value
    new.close()


@pytest.mark.parametrize("cached", [None, _frame()])
def test_live_dashboard_never_creates_competing_capture(monkeypatch, cached) -> None:
    owner = _owner()
    monkeypatch.setattr(operator.AgentProcess, "load", lambda: owner)
    monkeypatch.setattr(operator, "agent_alive", lambda process: process == owner)
    monkeypatch.setattr(operator.BedrockSession, "load", Mock(side_effect=FileNotFoundError))
    reader = Mock(return_value=cached)
    monkeypatch.setattr(operator, "read_agent_frame", reader)
    direct = Mock(side_effect=AssertionError("a live agent owns game capture"))
    monkeypatch.setattr(operator, "create_bedrock_capture", direct)
    monkeypatch.setattr(operator, "_close_live_bedrock_capture", Mock())
    assert operator._capture_live_bedrock_frame() is cached
    reader.assert_called_once_with(owner)
    direct.assert_not_called()


def _private_source(results: list) -> IsolatedX11Capture:
    source = object.__new__(IsolatedX11Capture)
    source.display_name = ":2"
    source.target_window_id = 42
    source.capture = Mock(side_effect=results)
    source.reconnect_after_image_timeout = Mock()
    source.close = Mock()
    return source


def _timeout() -> ImageCaptureTimeout:
    return ImageCaptureTimeout("image reply read timed out")


def test_timeout_drops_frame_then_reopens_once_on_next_capture(tmp_path: Path) -> None:
    frame = _frame()
    source = _private_source([_timeout(), frame])
    path = tmp_path / "frame"
    publisher = PublishedFrameCapture(source, _owner(), path=path)
    with pytest.raises(ImageCaptureTimeout):
        publisher.capture()
    assert not path.exists() and publisher._published_identity is None
    assert publisher._last_publish_ns == 0
    source.capture.assert_called_once()
    source.reconnect_after_image_timeout.assert_not_called()
    assert publisher.capture() is frame
    source.reconnect_after_image_timeout.assert_called_once()
    assert source.capture.call_count == 2
    assert read_agent_frame(_owner(), path=path) == frame
    assert publisher._consecutive_image_timeouts == 0


def test_three_consecutive_timeouts_exhaust_two_reopens_without_publishing(tmp_path: Path) -> None:
    source = _private_source([_timeout(), _timeout(), _timeout()])
    path = tmp_path / "frame"
    publisher = PublishedFrameCapture(source, _owner(), path=path)
    for _ in range(2):
        with pytest.raises(ImageCaptureTimeout):
            publisher.capture()
    with pytest.raises(IsolationError, match="budget exhausted") as caught:
        publisher.capture()
    assert type(caught.value) is IsolationError
    assert source.capture.call_count == 3
    assert source.reconnect_after_image_timeout.call_count == 2
    assert not publisher._image_reconnect_pending and not path.exists()


def test_fresh_success_resets_bounded_timeout_allowance(tmp_path: Path) -> None:
    first = _frame()
    second = replace(first, frame_id=5, captured_ns=first.captured_ns + 1)
    source = _private_source([_timeout(), _timeout(), first, _timeout(), _timeout(), second])
    publisher = PublishedFrameCapture(source, _owner(), path=tmp_path / "frame")
    for frame in (first, second):
        for _ in range(2):
            with pytest.raises(ImageCaptureTimeout):
                publisher.capture()
        assert publisher.capture() is frame
        assert publisher._consecutive_image_timeouts == 0
    assert source.reconnect_after_image_timeout.call_count == 4


@pytest.mark.parametrize("kind", ["stale", "future", "malformed", "replayed"])
def test_unusable_success_does_not_reset_reconnect_allowance(tmp_path: Path, kind: str) -> None:
    first = _frame()
    bad = {
        "stale": replace(first, frame_id=5, captured_ns=1),
        "future": replace(first, frame_id=5, captured_ns=2**63),
        "malformed": replace(first, frame_id=5, captured_ns=first.captured_ns + 1, bgra=b"bad"),
        "replayed": first,
    }[kind]
    source = _private_source([first, _timeout(), _timeout(), bad, _timeout()])
    publisher = PublishedFrameCapture(source, _owner(), path=tmp_path / "frame")
    assert publisher.capture() is first
    for _ in range(2):
        with pytest.raises(ImageCaptureTimeout):
            publisher.capture()
    assert publisher.capture() is bad  # Original invalid pixels/timestamp remain unchanged.
    assert publisher._consecutive_image_timeouts == 2
    with pytest.raises(IsolationError, match="budget exhausted"):
        publisher.capture()
    assert source.reconnect_after_image_timeout.call_count == 2


@pytest.mark.parametrize("fault", [IsolationError("ownership changed"), OSError("reset")])
def test_unclassified_faults_remain_fatal_and_do_not_schedule_reconnect(tmp_path, fault) -> None:
    source = _private_source([fault])
    publisher = PublishedFrameCapture(source, _owner(), path=tmp_path / "frame")
    with pytest.raises(type(fault)) as caught:
        publisher.capture()
    assert caught.value is fault
    assert not publisher._image_reconnect_pending
    source.reconnect_after_image_timeout.assert_not_called()


@pytest.mark.parametrize("kind", ["display", "window", "host", "owner", "source"])
def test_pending_reconnect_refuses_changed_capture_owner(tmp_path: Path, kind: str) -> None:
    source = _private_source([_timeout(), _frame()])
    publisher = PublishedFrameCapture(source, _owner(), path=tmp_path / "frame")
    with pytest.raises(ImageCaptureTimeout):
        publisher.capture()
    if kind == "display":
        source.display_name = ":3"
    elif kind == "window":
        source.target_window_id = 99
    elif kind == "host":
        publisher.owner = replace(publisher.owner, allow_host_capture=True)
    elif kind == "owner":
        publisher.owner = replace(publisher.owner, started_ns=124)
    else:
        publisher.source = _private_source([_frame()])
    with pytest.raises(IsolationError, match="owner or private target changed"):
        publisher.capture()
    source.reconnect_after_image_timeout.assert_not_called()
    assert source.capture.call_count == 1


def test_non_x11_source_cannot_grant_transient_timeout_admission(tmp_path: Path) -> None:
    source = Mock()
    source.capture.side_effect = _timeout()
    publisher = PublishedFrameCapture(source, _owner(), path=tmp_path / "frame")
    with pytest.raises(IsolationError, match="owner or private target changed") as caught:
        publisher.capture()
    assert type(caught.value) is IsolationError and not publisher._image_reconnect_pending
    source.reconnect_after_image_timeout.assert_not_called()


def test_failed_reopen_consumes_pending_attempt_without_inline_retry(tmp_path: Path) -> None:
    source = _private_source([_timeout(), IsolationError("connection remains poisoned")])
    source.reconnect_after_image_timeout.side_effect = IsolationError("exact target unavailable")
    publisher = PublishedFrameCapture(source, _owner(), path=tmp_path / "frame")
    with pytest.raises(ImageCaptureTimeout):
        publisher.capture()
    with pytest.raises(IsolationError, match="exact target unavailable"):
        publisher.capture()
    assert not publisher._image_reconnect_pending
    with pytest.raises(IsolationError, match="connection remains poisoned"):
        publisher.capture()
    source.reconnect_after_image_timeout.assert_called_once()
    assert source.capture.call_count == 2
