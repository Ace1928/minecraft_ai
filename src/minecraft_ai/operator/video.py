"""Bounded local HLS qualification from one private game and its dedicated audio.

This module does not route audio, publish media, or run a persistent service.
Only a successfully completed qualification may be considered for later review;
same-user changes to audio routing are detected, not an OS security boundary.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from minecraft_ai.platforms.bedrock_session import (
    BedrockSession,
    require_autonomous_input_isolation,
)
from minecraft_ai.platforms.bedrock_x11 import _new_wine_geometry

GAME_SINK = "minecraft_ai_game"
GAME_MONITOR = GAME_SINK + ".monitor"
_uid_reader = getattr(os, "getuid", None)
PULSE_SERVER = f"unix:/run/user/{_uid_reader()}/pulse/native" if _uid_reader is not None else ""
PROC_ROOT = Path("/proc")
MAX_DURATION_S = 60.0


class VideoQualificationError(RuntimeError):
    """A redacted, operator-readable capture qualification failure."""


def _current_uid() -> int:
    if _uid_reader is None:
        raise VideoQualificationError("local video qualification requires Linux")
    return int(_uid_reader())


@dataclass(frozen=True)
class GameCapture:
    display: str
    window_id: int
    width: int
    height: int
    pid: int
    process_start_ticks: int
    executable: str
    session_created_ns: int


def _read_bounded(path: Path, limit: int) -> bytes:
    with path.open("rb") as stream:
        value = stream.read(limit + 1)
    if len(value) > limit:
        raise VideoQualificationError("game process identity exceeds the observation limit")
    return value


def _process_identity(pid: int, display: str) -> tuple[int, str]:
    if pid <= 1:
        raise VideoQualificationError("game process identity is missing")
    directory = PROC_ROOT / str(pid)
    if directory.stat().st_uid != _current_uid():
        raise VideoQualificationError("game process is not owned by the current user")
    before = _read_bounded(directory / "stat", 8192).rsplit(b")", 1)[1].split()
    command = _read_bounded(directory / "cmdline", 65536).lower().replace(b"\\", b"/")
    environment = _read_bounded(directory / "environ", 262144).split(b"\0")
    executable = os.readlink(directory / "exe")
    after = _read_bounded(directory / "stat", 8192).rsplit(b")", 1)[1].split()
    if (
        len(before) < 20
        or len(after) < 20
        or before[0] in {b"Z", b"X"}
        or after[0] in {b"Z", b"X"}
        or before[19] != after[19]
        or not any(arg.endswith(b"/minecraft.windows.exe") for arg in command.split(b"\0"))
        or os.fsencode("DISPLAY=" + display) not in environment
        or Path(executable).name not in {"wine-preloader", "wine64-preloader", "wine", "wine64"}
    ):
        raise VideoQualificationError("game process identity is not qualified")
    return int(before[19]), executable


def inspect_game_capture() -> GameCapture:
    """Read current managed ownership, exact window geometry, and game PID."""
    _current_uid()
    session = BedrockSession.load()
    require_autonomous_input_isolation(session)
    window_id = session.find_window()
    if window_id is None:
        raise VideoQualificationError("private Minecraft window is unavailable")
    display_module = importlib.import_module("Xlib.display")
    connection: Any = display_module.Display(session.display)
    try:
        window, _client, _geometry = _new_wine_geometry(connection)
        if int(window.id) != window_id or window_id == int(connection.screen().root.id):
            raise VideoQualificationError("capture target is not the exact Minecraft window")
        geometry = window.get_geometry()
        width, height = int(geometry.width), int(geometry.height)
        root_geometry = connection.screen().root.get_geometry()
        position = connection.screen().root.translate_coords(window, 0, 0)
        if (
            width < 320 or height < 180 or width * height > 8_294_400
            or position.x < 0 or position.y < 0
            or position.x + width > root_geometry.width
            or position.y + height > root_geometry.height
        ):
            raise VideoQualificationError("game capture geometry is invalid or clipped")
        atom = connection.intern_atom("_NET_WM_PID", only_if_exists=True)
        prop = window.get_full_property(atom, 0) if atom else None
        if prop is None or len(prop.value) != 1:
            raise VideoQualificationError("Minecraft window has no exact process identity")
        pid = int(prop.value[0])
        start_ticks, executable = _process_identity(pid, session.display)
        return GameCapture(
            session.display, window_id, width, height, pid,
            start_ticks, executable, session.created_ns,
        )
    finally:
        connection.close()


def _pactl_json(*arguments: str) -> Any:
    try:
        result = subprocess.run(
            ["pactl", "--server", PULSE_SERVER, "--format=json", *arguments],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            check=True, timeout=3,
        )
        if len(result.stdout) > 2_000_000:
            raise VideoQualificationError("audio inventory exceeds the observation limit")
        return json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise VideoQualificationError("audio inventory could not be verified") from exc


def validate_game_audio(
    game: GameCapture,
    monitor: str,
    sinks: Any,
    inputs: Any,
    server: Any,
) -> None:
    """Reject default/mixed monitors and every unrelated dedicated-sink occupant."""
    if monitor != GAME_MONITOR:
        raise VideoQualificationError("only the explicitly dedicated game monitor is allowed")
    if not isinstance(sinks, list) or not isinstance(inputs, list) or not isinstance(server, dict):
        raise VideoQualificationError("audio inventory is malformed")
    selected = [sink for sink in sinks if isinstance(sink, dict) and sink.get("name") == GAME_SINK]
    if len(selected) != 1:
        raise VideoQualificationError("dedicated game audio sink is unavailable")
    sink = selected[0]
    if (
        sink.get("monitor_source") != monitor
        or type(sink.get("index")) is not int
        or not isinstance(server.get("default_sink_name"), str)
        or not isinstance(server.get("default_source_name"), str)
        or server.get("default_sink_name") == GAME_SINK
        or server.get("default_source_name") == monitor
    ):
        raise VideoQualificationError("dedicated audio monitor must not be a desktop default")
    occupants = []
    for item in inputs:
        if not isinstance(item, dict) or type(item.get("sink")) is not int:
            raise VideoQualificationError("audio stream inventory is malformed")
        if item["sink"] == sink["index"]:
            occupants.append(item)
    if not occupants:
        raise VideoQualificationError("dedicated audio sink has no verified Minecraft stream")
    for item in occupants:
        properties = item.get("properties")
        if not isinstance(properties, dict) or (
            properties.get("application.name") != "Minecraft"
            or str(properties.get("application.process.id")) != str(game.pid)
            or properties.get("window.x11.display") != game.display
            or properties.get("application.process.binary") != Path(game.executable).name
        ):
            raise VideoQualificationError("dedicated audio sink contains an unrelated stream")
    if _process_identity(game.pid, game.display) != (game.process_start_ticks, game.executable):
        raise VideoQualificationError("game process changed during audio qualification")


def verify_game_audio(game: GameCapture, monitor: str) -> None:
    validate_game_audio(
        game, monitor, _pactl_json("list", "sinks"),
        _pactl_json("list", "sink-inputs"), _pactl_json("info"),
    )


def validate_output_directory(directory: Path) -> Path:
    """Use a caller-created private temporary directory without following links."""
    if not directory.is_absolute() or directory.resolve() != directory:
        raise VideoQualificationError("output must be an absolute path without symbolic links")
    allowed = [Path(tempfile.gettempdir()).resolve()]
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime:
        allowed.append(Path(runtime).resolve())
    if not any(directory != base and directory.is_relative_to(base) for base in allowed):
        raise VideoQualificationError("output must be a task-scoped temporary directory")
    metadata = directory.stat()
    if not directory.is_dir() or metadata.st_uid != _current_uid() or metadata.st_mode & 0o077:
        raise VideoQualificationError("output directory must already exist and be owner-only")
    return directory


def ffmpeg_command(
    game: GameCapture, monitor: str, generation: Path, duration_s: float
) -> list[str]:
    if not math.isfinite(duration_s) or not 1 <= duration_s <= MAX_DURATION_S:
        raise VideoQualificationError("capture duration must be between one and sixty seconds")
    if monitor != GAME_MONITOR or game.window_id <= 0:
        raise VideoQualificationError("explicit game window and dedicated monitor are required")
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
        "-filter_threads", "1",
        "-thread_queue_size", "8", "-f", "x11grab", "-framerate", "24",
        "-video_size", f"{game.width}x{game.height}",
        "-window_id", str(game.window_id), "-draw_mouse", "0", "-i", game.display,
        "-thread_queue_size", "64", "-f", "pulse", "-server", PULSE_SERVER,
        "-sample_rate", "48000",
        "-channels", "2", "-i", monitor,
        "-map", "0:v:0", "-map", "1:a:0", "-map_metadata", "-1",
        "-vf", "scale=1280:720:flags=fast_bilinear",
        "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
        "-threads", "2", "-pix_fmt", "yuv420p", "-b:v", "2000k",
        "-maxrate", "2400k", "-bufsize", "4800k", "-g", "48", "-keyint_min", "48",
        "-sc_threshold", "0", "-c:a", "aac", "-b:a", "96k", "-ar", "48000", "-ac", "2",
        "-t", str(duration_s), "-f", "hls", "-hls_time", "2", "-hls_list_size", "6",
        "-hls_delete_threshold", "2", "-hls_start_number_source", "epoch_us",
        "-hls_flags", "delete_segments+independent_segments+temp_file+program_date_time",
        "-hls_segment_filename", str(generation / "segment-%d.ts"),
        str(generation / "index.m3u8"),
    ]


def qualify_video(directory: Path, monitor: str, duration_s: float) -> Path:
    directory = validate_output_directory(directory)
    game = inspect_game_capture()
    verify_game_audio(game, monitor)
    generation = directory / ("video-" + uuid.uuid4().hex)
    command = ffmpeg_command(game, monitor, generation, duration_s)
    if shutil.which("ffmpeg") is None:
        raise VideoQualificationError("FFmpeg is unavailable")
    generation.mkdir(mode=0o700, exist_ok=False)
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + duration_s + 8
        while process.poll() is None:
            if time.monotonic() >= deadline:
                raise VideoQualificationError("bounded encoder deadline exceeded")
            if inspect_game_capture() != game:
                raise VideoQualificationError("private game capture identity changed")
            verify_game_audio(game, monitor)
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
        if process.returncode != 0:
            raise VideoQualificationError("encoder did not complete successfully")
        if inspect_game_capture() != game:
            raise VideoQualificationError("private game capture identity changed")
        verify_game_audio(game, monitor)
        playlist = generation / "index.m3u8"
        if not playlist.is_file() or not any(generation.glob("segment-*.ts")):
            raise VideoQualificationError("encoder produced no complete HLS media")
        return playlist
    except (OSError, subprocess.SubprocessError) as exc:
        raise VideoQualificationError("local video qualification failed") from exc
    finally:
        # This owns only its encoder: never signal a game, supervisor, or audio server.
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audio-monitor", choices=[GAME_MONITOR], required=True)
    parser.add_argument("--duration-s", type=float, default=10)
    args = parser.parse_args()
    try:
        playlist = qualify_video(args.output_dir, args.audio_monitor, args.duration_s)
    except Exception:
        # Exception strings, subprocess commands, private paths and audio inventories stay local.
        print(json.dumps({"qualified": False, "error": "local video qualification failed"}))
        return 1
    print(json.dumps({"qualified": True, "playlist": str(playlist), "duration_s": args.duration_s}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
