from __future__ import annotations

import io
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from minecraft_ai.platforms import ffmpeg_x11_capture
from minecraft_ai.platforms.ffmpeg_x11_capture import FfmpegX11Capture, _read_exact


class _Process:
    def __init__(self, stream: bytes) -> None:
        self.stdout = io.BytesIO(stream)
        self.returncode: int | None = None
        self.command: list[str] | None = None
        self.kwargs: dict[str, object] | None = None

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9

    def wait(self, timeout: float) -> int:
        assert timeout == 2.0
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def test_read_exact_distinguishes_clean_and_partial_eof() -> None:
    assert _read_exact(io.BytesIO(b"abcd"), 4) == b"abcd"
    assert _read_exact(io.BytesIO(b""), 4) is None
    with pytest.raises(EOFError, match="within a raw frame"):
        _read_exact(io.BytesIO(b"abc"), 4)


def test_persistent_ffmpeg_capture_returns_newest_complete_exact_window_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = bytes(range(32))
    process = _Process(frame)
    popen_calls: list[tuple[list[str], dict[str, object]]] = []

    def popen(command: list[str], **kwargs: object) -> _Process:
        process.command = command
        process.kwargs = kwargs
        popen_calls.append((command, kwargs))
        return process

    window = Mock()
    window.get_geometry.return_value = SimpleNamespace(width=4, height=2)
    root = Mock()
    root.translate_coords.return_value = SimpleNamespace(x=11, y=13)
    display = Mock()
    display.create_resource_object.return_value = window
    display.screen.return_value.root = root
    modules = {"Xlib.display": SimpleNamespace(Display=lambda _name: display)}
    monkeypatch.setattr(ffmpeg_x11_capture.importlib, "import_module", modules.__getitem__)
    monkeypatch.setattr(ffmpeg_x11_capture.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(ffmpeg_x11_capture.subprocess, "Popen", popen)
    monkeypatch.setattr(ffmpeg_x11_capture, "require_isolated_display", Mock())
    monkeypatch.setattr(ffmpeg_x11_capture, "_wine_content_rect", lambda *_args: None)

    capture = FfmpegX11Capture(":2", 42, capture_budget_ms=500)
    observed = capture.capture()

    assert observed.frame_id == 1
    assert observed.width == 4 and observed.height == 2
    assert observed.bgra == frame
    assert len(popen_calls) == 1
    command, kwargs = popen_calls[0]
    assert command[command.index("-window_id") + 1] == "42"
    assert command[command.index("-video_size") + 1] == "4x2"
    assert command[command.index("-i") + 1] == ":2"
    assert "shell" not in kwargs
    assert kwargs["stdout"] is ffmpeg_x11_capture.subprocess.PIPE
    capture.close()
    display.close.assert_called_once()
