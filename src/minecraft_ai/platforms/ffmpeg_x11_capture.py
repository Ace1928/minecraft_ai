"""Low-latency, window-scoped X11 capture through one persistent FFmpeg process."""

from __future__ import annotations

import importlib
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Protocol, cast

from .bedrock_x11 import (
    CapturedFrame,
    ImageCaptureTimeout,
    IsolationError,
    _crop_bgra,
    _wine_content_rect,
    require_isolated_display,
)


class _ReadableStream(Protocol):
    def readinto(self, buffer: Any, /) -> int | None: ...


def _read_exact(stream: _ReadableStream, size: int) -> bytes | None:
    """Read one raw frame, returning ``None`` only for a clean pre-frame EOF."""

    frame = bytearray(size)
    view = memoryview(frame)
    offset = 0
    while offset < size:
        count = stream.readinto(view[offset:])
        if count is None:
            continue
        if count == 0:
            if offset == 0:
                return None
            raise EOFError("FFmpeg X11 stream ended within a raw frame")
        offset += count
    return bytes(frame)


class FfmpegX11Capture:
    """Continuously retain the newest complete frame from one exact X11 window.

    FFmpeg performs the X11 pixel transfer in native code. A sole reader thread
    drains the raw stream so a slower policy loop receives the newest complete
    frame instead of accumulating a FIFO of stale screenshots.
    """

    def __init__(
        self,
        display_name: str,
        target_window_id: int,
        *,
        host_display: str | None = None,
        allow_host: bool = False,
        capture_budget_ms: int = 500,
        frame_rate: int = 20,
    ) -> None:
        if type(capture_budget_ms) is not int or not 100 <= capture_budget_ms <= 5000:
            raise IsolationError("FFmpeg X11 capture budget must be between 100 and 5000 ms")
        if type(frame_rate) is not int or not 5 <= frame_rate <= 60:
            raise IsolationError("FFmpeg X11 frame rate must be between 5 and 60 Hz")
        require_isolated_display(display_name, host_display, allow_host=allow_host)
        binary = shutil.which("ffmpeg")
        if binary is None or not Path(binary).is_absolute():
            raise IsolationError("FFmpeg is required for ffmpeg-x11 capture")
        try:
            display_module = importlib.import_module("Xlib.display")
        except ImportError as exc:
            raise IsolationError("python-xlib is required for FFmpeg X11 geometry checks") from exc

        self.display_name = display_name
        self.target_window_id = target_window_id
        self._capture_budget_ms = capture_budget_ms
        self._display: Any = display_module.Display(display_name)
        bounds = self._bounds()
        self._source_bounds = bounds
        self._content = self._content_rect(bounds["width"], bounds["height"])
        self._last_geometry_check_ns = time.monotonic_ns()
        self._geometry_check_interval_ns = 1_000_000_000
        self._frame_size = bounds["width"] * bounds["height"] * 4
        command = [
            binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            "x11grab",
            "-framerate",
            str(frame_rate),
            "-window_id",
            str(target_window_id),
            "-video_size",
            f"{bounds['width']}x{bounds['height']}",
            "-i",
            display_name,
            "-pix_fmt",
            "bgra",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
                close_fds=True,
            )
        except OSError as exc:
            self._display.close()
            raise IsolationError("could not start FFmpeg X11 capture") from exc
        if self._process.stdout is None:
            self._process.kill()
            self._display.close()
            raise IsolationError("FFmpeg X11 capture did not expose a pixel stream")

        self._condition = threading.Condition()
        self._closed = False
        self._latest_sequence = 0
        self._returned_sequence = 0
        self._latest_bgra: bytes | None = None
        self._latest_captured_ns = 0
        self._reader_error: BaseException | None = None
        self._reader = threading.Thread(
            target=self._read_frames,
            name="minecraft-ffmpeg-x11-capture",
            daemon=True,
        )
        self._reader.start()

    def _bounds(self) -> dict[str, int]:
        try:
            root = self._display.screen().root
            window = self._display.create_resource_object("window", self.target_window_id)
            geometry = window.get_geometry()
            translated = root.translate_coords(window, 0, 0)
        except Exception as exc:
            raise IsolationError("cannot resolve FFmpeg Bedrock window geometry") from exc
        width = int(geometry.width)
        height = int(geometry.height)
        if width <= 0 or height <= 0:
            raise IsolationError("FFmpeg Bedrock target has invalid geometry")
        return {
            "left": int(translated.x),
            "top": int(translated.y),
            "width": width,
            "height": height,
        }

    def _content_rect(self, width: int, height: int) -> tuple[int, int, int, int] | None:
        return _wine_content_rect(self._display, self.target_window_id, width, height)

    def _read_frames(self) -> None:
        assert self._process.stdout is not None
        try:
            while True:
                frame = _read_exact(
                    cast(_ReadableStream, self._process.stdout),
                    self._frame_size,
                )
                if frame is None:
                    code = self._process.poll()
                    raise EOFError(f"FFmpeg X11 capture exited with code {code}")
                captured_ns = time.monotonic_ns()
                with self._condition:
                    if self._closed:
                        return
                    self._latest_sequence += 1
                    self._latest_bgra = frame
                    self._latest_captured_ns = captured_ns
                    self._condition.notify_all()
        except BaseException as exc:
            with self._condition:
                if not self._closed:
                    self._reader_error = exc
                    self._condition.notify_all()

    def capture(self) -> CapturedFrame:
        with self._condition:
            first = self._returned_sequence == 0
            timeout_s = max(5.0, self._capture_budget_ms / 1000) if first else (
                self._capture_budget_ms / 1000
            )
            deadline = time.monotonic() + timeout_s
            while self._latest_sequence <= self._returned_sequence:
                if self._reader_error is not None:
                    raise IsolationError("FFmpeg X11 pixel stream failed") from self._reader_error
                if self._closed:
                    raise IsolationError("FFmpeg X11 capture is closed")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    # A live, exact private FFmpeg stream can miss one frame
                    # deadline under CPU pressure without losing its capture
                    # identity. Classify only this wait expiry as recoverable;
                    # EOF, geometry, ownership and stream faults remain fatal.
                    raise ImageCaptureTimeout("FFmpeg X11 frame deadline expired")
                self._condition.wait(remaining)
            sequence = self._latest_sequence
            bgra = self._latest_bgra
            captured_ns = self._latest_captured_ns
            self._returned_sequence = sequence
        if bgra is None or len(bgra) != self._frame_size:
            raise IsolationError("FFmpeg X11 returned an incomplete BGRA frame")

        bounds = self._source_bounds
        content = self._content
        now_ns = time.monotonic_ns()
        if now_ns - self._last_geometry_check_ns >= self._geometry_check_interval_ns:
            bounds = self._bounds()
            content = self._content_rect(bounds["width"], bounds["height"])
            if bounds != self._source_bounds or content != self._content:
                raise IsolationError("Minecraft window geometry changed during FFmpeg capture")
            self._last_geometry_check_ns = now_ns
        width = bounds["width"]
        height = bounds["height"]
        if content is not None:
            bgra = _crop_bgra(bgra, source_width=width, rect=content)
            _, _, width, height = content
        return CapturedFrame(
            frame_id=sequence,
            captured_ns=captured_ns,
            width=width,
            height=height,
            bgra=bgra,
        )

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify_all()
        process = self._process
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2.0)
        if process.stdout is not None:
            process.stdout.close()
        self._reader.join(timeout=2.0)
        try:
            self._display.close()
        except Exception:
            pass
