from __future__ import annotations

import importlib
import json
import mmap
import os
import queue
import re
import subprocess
import sys
import threading
from collections.abc import Sequence
from multiprocessing import resource_tracker, shared_memory
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
        self._shared_memory: mmap.mmap | shared_memory.SharedMemory | None = None
        self._responses: queue.Queue[str | Exception | None] = queue.Queue()
        self._reader_thread: threading.Thread | None = None
        self._last_frame_id = 0
        self._geometry_guard = (
            _HostMonitorGeometryGuard(binding) if _geometry_guard is None else _geometry_guard
        )
        try:
            self._process = _spawn_worker(command)
            if self._process.stdout is None:
                raise IsolationError("GNOME capture worker has no response channel")
            self._reader_thread = threading.Thread(
                target=self._read_worker_stdout,
                args=(self._process.stdout,),
                name="minecraft-ai-pipewire-metadata",
                daemon=True,
            )
            self._reader_thread.start()
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
            self._shared_memory = _attach_shared_memory(shm_name, expected_size)
        except Exception:
            self.close()
            raise

    def _read_worker_stdout(self, stream: TextIO) -> None:
        """Move blocking pipe reads off the realtime thread on every platform."""
        try:
            while True:
                line = stream.readline()
                if not line:
                    self._responses.put(None)
                    return
                self._responses.put(line)
        except Exception as exc:
            self._responses.put(exc)

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
        try:
            item = self._responses.get(timeout=timeout_s)
        except queue.Empty:
            if process.poll() is not None:
                raise IsolationError(
                    f"GNOME capture worker exited with status {process.returncode}"
                ) from None
            raise IsolationError("GNOME capture worker response timed out") from None
        if item is None:
            raise IsolationError("GNOME capture worker closed its response channel")
        if isinstance(item, Exception):
            raise IsolationError("GNOME capture worker response channel failed") from item
        line = item
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
            if block is None or _shared_memory_size(block) < size:
                raise IsolationError("GNOME capture shared memory is unavailable")
            pixels = _read_shared_memory(block, size)
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
            reader_thread = self._reader_thread
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
            if reader_thread is not None:
                reader_thread.join(timeout=_SHUTDOWN_TIMEOUT_S)
            reader_alive = reader_thread is not None and reader_thread.is_alive()
            for stream, owned_by_reader in (
                (None if process is None else process.stdin, False),
                (None if process is None else process.stdout, True),
            ):
                if stream is not None:
                    # TextIOWrapper.close can wait forever on a concurrent
                    # blocking readline. A daemon reader owns that stream until
                    # EOF, so a broken process double must not deadlock cleanup.
                    if owned_by_reader and reader_alive:
                        continue
                    try:
                        cast(TextIO, stream).close()
                    except OSError:
                        pass
            if block is not None:
                block.close()
            self._geometry_guard.close()


def _attach_shared_memory(
    name: str,
    expected_size: int,
) -> mmap.mmap | shared_memory.SharedMemory:
    """Attach safely on Linux production and portably in protocol tests."""
    if sys.platform.startswith("linux"):
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(Path("/dev/shm") / name, flags)
        try:
            if os.fstat(descriptor).st_size < expected_size:
                raise IsolationError("GNOME capture worker shared memory is undersized")
            return mmap.mmap(descriptor, expected_size, access=mmap.ACCESS_READ)
        finally:
            os.close(descriptor)
    block = shared_memory.SharedMemory(name=name, create=False)
    if block.size < expected_size:
        block.close()
        raise IsolationError("GNOME capture worker shared memory is undersized")
    # The worker owns unlinking. An attaching process must not later remove the
    # producer's allocation merely because Python's legacy tracker saw it.
    try:
        resource_tracker.unregister(getattr(block, "_name", name), "shared_memory")
    except (KeyError, ValueError):
        pass
    return block


def _shared_memory_size(block: mmap.mmap | shared_memory.SharedMemory) -> int:
    return block.size() if isinstance(block, mmap.mmap) else block.size


def _read_shared_memory(block: mmap.mmap | shared_memory.SharedMemory, size: int) -> bytes:
    if isinstance(block, mmap.mmap):
        return bytes(block[:size])
    view = block.buf
    if view is None:
        raise IsolationError("GNOME capture shared memory is closed")
    return bytes(view[:size])


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
