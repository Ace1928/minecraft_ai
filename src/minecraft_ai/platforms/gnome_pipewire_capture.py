from __future__ import annotations

import importlib
import json
import mmap
import os
import re
import select
import subprocess
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, TextIO, cast

from .bedrock_x11 import (
    CapturedFrame,
    HostMonitorBinding,
    IsolationError,
    IsolatedX11Capture,
    resolve_host_monitor_content_rect,
    validate_host_monitor_window,
)


_SYSTEM_PYTHON = Path("/usr/bin/python3")
_STARTUP_TIMEOUT_S = 8.0
_FRAME_TIMEOUT_S = 1.5
_SHUTDOWN_TIMEOUT_S = 1.0
_SHM_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


class _GeometryGuard(Protocol):
    def validate(self) -> None: ...

    def close(self) -> None: ...


class _HostMonitorGeometryGuard:
    """Keep the immutable host-monitor proof live for every captured frame."""

    def __init__(self, binding: HostMonitorBinding) -> None:
        try:
            display_module = importlib.import_module("Xlib.display")
            self._display: Any = display_module.Display(binding.display)
        except Exception as exc:
            raise IsolationError(f"cannot open host display {binding.display}: {exc}") from exc
        self._binding = binding
        try:
            self.validate()
        except Exception:
            self.close()
            raise

    def validate(self) -> None:
        validate_host_monitor_window(
            self._display,
            self._binding,
            target_window_id=self._binding.window_id,
        )

    def close(self) -> None:
        try:
            self._display.close()
        except Exception:
            pass


def _spawn_worker(command: Sequence[str]) -> subprocess.Popen[str]:
    return subprocess.Popen(
        list(command),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        bufsize=1,
        close_fds=True,
        start_new_session=True,
    )


def _positive_int(payload: object, key: str) -> int:
    if not isinstance(payload, dict):
        raise IsolationError("invalid GNOME capture worker response")
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise IsolationError(f"invalid GNOME capture worker {key}")
    return value


class MutterPipeWireCapture:
    """Monitor-scoped Mutter/PipeWire capture with shared-memory frame transfer.

    The subprocess intentionally runs under the distribution Python because
    GNOME's GI and dbus bindings are host packages and do not belong in the
    application's portable virtual environment.  Commands and metadata use a
    line-delimited JSON control channel; full-resolution frames never traverse
    a pipe.
    """

    def __init__(
        self,
        binding: HostMonitorBinding,
        *,
        content_rect: tuple[int, int, int, int],
        worker_command: Sequence[str] | None = None,
        startup_timeout_s: float = _STARTUP_TIMEOUT_S,
        frame_timeout_s: float = _FRAME_TIMEOUT_S,
        _geometry_guard: _GeometryGuard | None = None,
    ) -> None:
        x, y, width, height = content_rect
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise IsolationError("invalid host-monitor capture content rectangle")
        if x + width > binding.monitor.width or y + height > binding.monitor.height:
            raise IsolationError("host-monitor capture content rectangle exceeds bound output")
        if startup_timeout_s <= 0 or frame_timeout_s <= 0:
            raise ValueError("capture worker timeouts must be positive")
        command = (
            list(worker_command)
            if worker_command is not None
            else self._default_command(
                binding,
                content_rect,
            )
        )
        self.display_name = binding.display
        self.target_window_id = binding.window_id
        self.output_name = binding.output_name
        self._expected_width = width
        self._expected_height = height
        self._frame_timeout_s = frame_timeout_s
        self._lock = threading.Lock()
        self._closed = False
        self._process: subprocess.Popen[str] | None = None
        self._shared_memory: mmap.mmap | None = None
        self._last_frame_id = 0
        self._geometry_guard = (
            _HostMonitorGeometryGuard(binding) if _geometry_guard is None else _geometry_guard
        )
        try:
            self._process = _spawn_worker(command)
            response = self._read_response(timeout_s=startup_timeout_s)
            if response.get("event") != "ready":
                raise IsolationError("GNOME capture worker did not report ready")
            actual_width = _positive_int(response, "width")
            actual_height = _positive_int(response, "height")
            size = _positive_int(response, "size")
            shm_name = response.get("shm_name")
            if (
                not isinstance(shm_name, str)
                or not shm_name
                or _SHM_NAME.fullmatch(shm_name) is None
            ):
                raise IsolationError("GNOME capture worker omitted shared-memory name")
            expected_size = width * height * 4
            if (actual_width, actual_height, size) != (width, height, expected_size):
                raise IsolationError(
                    "GNOME capture worker geometry does not match bound Minecraft drawable"
                )
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(Path("/dev/shm") / shm_name, flags)
            try:
                if os.fstat(descriptor).st_size < expected_size:
                    raise IsolationError("GNOME capture worker shared memory is undersized")
                self._shared_memory = mmap.mmap(
                    descriptor,
                    expected_size,
                    access=mmap.ACCESS_READ,
                )
            finally:
                os.close(descriptor)
        except Exception:
            self.close()
            raise

    @staticmethod
    def _default_command(
        binding: HostMonitorBinding,
        content_rect: tuple[int, int, int, int],
    ) -> list[str]:
        if not _SYSTEM_PYTHON.is_file():
            raise IsolationError("/usr/bin/python3 is required for GNOME PipeWire capture")
        worker = Path(__file__).with_name("_gnome_pipewire_worker.py")
        if not worker.is_file():
            raise IsolationError("GNOME PipeWire capture worker is not installed")
        x, y, width, height = content_rect
        return [
            str(_SYSTEM_PYTHON),
            "-u",
            str(worker),
            "--output",
            binding.output_name,
            "--monitor-width",
            str(binding.monitor.width),
            "--monitor-height",
            str(binding.monitor.height),
            "--crop-x",
            str(x),
            "--crop-y",
            str(y),
            "--crop-width",
            str(width),
            "--crop-height",
            str(height),
        ]

    def _read_response(self, *, timeout_s: float) -> dict[str, object]:
        process = self._process
        if process is None or process.stdout is None:
            raise IsolationError("GNOME capture worker has no response channel")
        ready, _, _ = select.select([process.stdout], [], [], timeout_s)
        if not ready:
            if process.poll() is not None:
                raise IsolationError(
                    f"GNOME capture worker exited with status {process.returncode}"
                )
            raise IsolationError("GNOME capture worker response timed out")
        line = process.stdout.readline()
        if not line:
            raise IsolationError("GNOME capture worker closed its response channel")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise IsolationError("GNOME capture worker returned invalid JSON") from exc
        if not isinstance(response, dict):
            raise IsolationError("GNOME capture worker response must be an object")
        if response.get("ok") is not True:
            error = response.get("error")
            detail = error if isinstance(error, str) and error else "unknown worker failure"
            raise IsolationError(f"GNOME capture worker failed: {detail}")
        return cast(dict[str, object], response)

    def _send(self, payload: dict[str, object]) -> None:
        process = self._process
        if process is None or process.stdin is None or process.poll() is not None:
            raise IsolationError("GNOME capture worker is not running")
        try:
            process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise IsolationError("GNOME capture worker command channel failed") from exc

    def capture(self) -> CapturedFrame:
        with self._lock:
            if self._closed:
                raise IsolationError("GNOME capture source is closed")
            self._geometry_guard.validate()
            self._send(
                {
                    "command": "capture",
                    "timeout_ms": max(1, int(self._frame_timeout_s * 1000)),
                }
            )
            response = self._read_response(timeout_s=self._frame_timeout_s + 0.25)
            if response.get("event") != "frame":
                raise IsolationError("GNOME capture worker returned an unexpected event")
            frame_id = _positive_int(response, "frame_id")
            captured_ns = _positive_int(response, "captured_ns")
            size = _positive_int(response, "bytes")
            expected_size = self._expected_width * self._expected_height * 4
            if size != expected_size or frame_id <= self._last_frame_id:
                raise IsolationError("GNOME capture worker returned invalid frame metadata")
            block = self._shared_memory
            if block is None or block.size() < size:
                raise IsolationError("GNOME capture shared memory is unavailable")
            pixels = bytes(block[:size])
            self._last_frame_id = frame_id
            return CapturedFrame(
                frame_id=frame_id,
                captured_ns=captured_ns,
                width=self._expected_width,
                height=self._expected_height,
                bgra=pixels,
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            process = self._process
            block = self._shared_memory
            self._shared_memory = None
            if process is not None and process.poll() is None:
                try:
                    self._send({"command": "close"})
                except (IsolationError, OSError):
                    pass
                try:
                    process.wait(timeout=_SHUTDOWN_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=_SHUTDOWN_TIMEOUT_S)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=_SHUTDOWN_TIMEOUT_S)
            for stream in (
                None if process is None else process.stdin,
                None if process is None else process.stdout,
            ):
                if stream is not None:
                    try:
                        cast(TextIO, stream).close()
                    except OSError:
                        pass
            if block is not None:
                block.close()
            self._geometry_guard.close()


def create_bedrock_capture(
    display_name: str,
    target_window_id: int,
    *,
    allow_host: bool = False,
    host_monitor_binding: HostMonitorBinding | None = None,
) -> IsolatedX11Capture | MutterPipeWireCapture:
    """Select PipeWire only for a proven host-monitor session."""
    if host_monitor_binding is None:
        return IsolatedX11Capture(
            display_name,
            target_window_id,
            allow_host=allow_host,
        )
    if not allow_host:
        raise IsolationError("host-monitor capture requires explicit host-display access")
    if host_monitor_binding.display != display_name:
        raise IsolationError("host-monitor capture display does not match binding")
    if host_monitor_binding.window_id != target_window_id:
        raise IsolationError("host-monitor capture window does not match binding")
    content_rect = resolve_host_monitor_content_rect(display_name, host_monitor_binding)
    return MutterPipeWireCapture(host_monitor_binding, content_rect=content_rect)
