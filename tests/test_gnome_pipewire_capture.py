from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from minecraft_ai.platforms.bedrock_x11 import HostMonitorBinding, IsolationError, ScreenRect
from minecraft_ai.platforms.gnome_pipewire_capture import (
    MutterPipeWireCapture,
    create_bedrock_capture,
)


def _binding() -> HostMonitorBinding:
    return HostMonitorBinding(
        display=":0",
        output_name="DP-2",
        monitor=ScreenRect(x=1920, y=0, width=1920, height=1080),
        window_id=42,
        bound_ns=1,
    )


def test_capture_factory_uses_x11_without_host_monitor_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    calls: list[tuple[str, int, bool]] = []

    def fake_x11(display: str, window_id: int, *, allow_host: bool) -> object:
        calls.append((display, window_id, allow_host))
        return sentinel

    monkeypatch.setattr(
        "minecraft_ai.platforms.gnome_pipewire_capture.IsolatedX11Capture",
        fake_x11,
    )

    capture = create_bedrock_capture(":12", 7)

    assert capture is sentinel
    assert calls == [(":12", 7, False)]


def test_capture_factory_uses_mutter_only_for_exact_host_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _binding()
    sentinel = object()
    calls: list[tuple[HostMonitorBinding, tuple[int, int, int, int]]] = []
    monkeypatch.setattr(
        "minecraft_ai.platforms.gnome_pipewire_capture.resolve_host_monitor_content_rect",
        lambda display, value: (0, 22, 1920, 1058),
    )

    def fake_mutter(
        value: HostMonitorBinding,
        *,
        content_rect: tuple[int, int, int, int],
    ) -> object:
        calls.append((value, content_rect))
        return sentinel

    monkeypatch.setattr(
        "minecraft_ai.platforms.gnome_pipewire_capture.MutterPipeWireCapture",
        fake_mutter,
    )

    capture = create_bedrock_capture(
        ":0",
        42,
        allow_host=True,
        host_monitor_binding=binding,
    )

    assert capture is sentinel
    assert calls == [(binding, (0, 22, 1920, 1058))]


@pytest.mark.parametrize(
    ("display", "window_id", "allow_host"),
    [(":0", 42, False), (":1", 42, True), (":0", 43, True)],
)
def test_capture_factory_rejects_invalid_host_binding_scope(
    display: str,
    window_id: int,
    allow_host: bool,
) -> None:
    with pytest.raises(IsolationError):
        create_bedrock_capture(
            display,
            window_id,
            allow_host=allow_host,
            host_monitor_binding=_binding(),
        )


def test_shared_memory_worker_protocol_transfers_frame_without_pixel_pipe(
    tmp_path: Path,
) -> None:
    helper = tmp_path / "fake_capture_worker.py"
    helper.write_text(
        """
import json
import sys
import time
from multiprocessing import shared_memory

block = shared_memory.SharedMemory(create=True, size=8)
print(json.dumps({
    "ok": True,
    "event": "ready",
    "shm_name": block.name,
    "width": 2,
    "height": 1,
    "size": 8,
}), flush=True)
frame_id = 0
try:
    for line in sys.stdin:
        command = json.loads(line)
        if command["command"] == "close":
            break
        frame_id += 1
        block.buf[:8] = bytes((1, 2, 3, 4, 5, 6, 7, 8))
        print(json.dumps({
            "ok": True,
            "event": "frame",
            "frame_id": frame_id,
            "captured_ns": time.monotonic_ns(),
            "bytes": 8,
        }), flush=True)
finally:
    block.close()
    block.unlink()
""".strip(),
        encoding="utf-8",
    )
    binding = HostMonitorBinding(
        display=":0",
        output_name="DP-2",
        monitor=ScreenRect(x=0, y=0, width=2, height=1),
        window_id=42,
        bound_ns=time.monotonic_ns(),
    )
    guard = _GeometryGuard()
    capture = MutterPipeWireCapture(
        binding,
        content_rect=(0, 0, 2, 1),
        worker_command=(sys.executable, "-u", str(helper)),
        startup_timeout_s=2,
        frame_timeout_s=1,
        _geometry_guard=guard,
    )
    try:
        frame = capture.capture()
    finally:
        capture.close()

    assert frame.frame_id == 1
    assert (frame.width, frame.height) == (2, 1)
    assert frame.bgra == bytes((1, 2, 3, 4, 5, 6, 7, 8))
    assert guard.validations == 1
    assert guard.closed


class _SilentProcess:
    """Minimal subprocess double proving startup timeout is fail-closed."""

    def __init__(self, stdout: Any) -> None:
        self.stdin = None
        self.stdout = stdout
        self.returncode: int | None = None
        self.terminated = False

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            raise subprocess.TimeoutExpired("silent-worker", 0.0 if timeout is None else timeout)
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.returncode = -9


class _GeometryGuard:
    def __init__(self) -> None:
        self.validations = 0
        self.closed = False

    def validate(self) -> None:
        self.validations += 1

    def close(self) -> None:
        self.closed = True


def test_worker_startup_timeout_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    read_descriptor, write_descriptor = os.pipe()
    unreadable = os.fdopen(read_descriptor, "r", encoding="utf-8")
    process = _SilentProcess(unreadable)
    monkeypatch.setattr(
        "minecraft_ai.platforms.gnome_pipewire_capture._spawn_worker",
        lambda command: process,
    )
    binding = HostMonitorBinding(
        display=":0",
        output_name="DP-2",
        monitor=ScreenRect(x=0, y=0, width=2, height=1),
        window_id=42,
        bound_ns=1,
    )
    guard = _GeometryGuard()
    try:
        with pytest.raises(IsolationError, match="timed out"):
            MutterPipeWireCapture(
                binding,
                content_rect=(0, 0, 2, 1),
                worker_command=("unused",),
                startup_timeout_s=0.01,
                _geometry_guard=guard,
            )
    finally:
        os.close(write_descriptor)
        unreadable.close()

    assert process.terminated
    assert guard.closed
