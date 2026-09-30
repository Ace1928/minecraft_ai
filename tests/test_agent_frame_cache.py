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
from minecraft_ai.platforms.bedrock_x11 import CapturedFrame
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
