from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_ai.operator import video

linux_only = pytest.mark.skipif(
    sys.platform != "linux", reason="Linux process/output qualification"
)


@pytest.fixture
def game():
    return video.GameCapture(":71", 123, 1920, 1080, 456, 789, "/wine/wine-preloader", 999)


@pytest.fixture
def audio(game, monkeypatch):
    monkeypatch.setattr(
        video, "_process_identity", lambda pid, display: (game.process_start_ticks, game.executable)
    )
    return (
        [{"index": 10, "name": video.GAME_SINK, "monitor_source": video.GAME_MONITOR}],
        [{"sink": 10, "properties": {
            "application.name": "Minecraft", "application.process.id": str(game.pid),
            "application.process.binary": "wine-preloader", "window.x11.display": game.display,
        }}],
        {"default_sink_name": "speakers", "default_source_name": "microphone"},
    )


def test_exact_game_only_audio_is_accepted(game, audio):
    sinks, inputs, server = audio
    inputs.append({"sink": 11, "properties": {"application.name": "Desktop browser"}})
    video.validate_game_audio(game, video.GAME_MONITOR, sinks, inputs, server)


@pytest.mark.parametrize("monitor", ["default", "@DEFAULT_MONITOR@", "speakers.monitor", ""])
def test_default_and_mixed_audio_sources_are_rejected(game, audio, monitor):
    with pytest.raises(video.VideoQualificationError, match="explicitly dedicated"):
        video.validate_game_audio(game, monitor, *audio)


@pytest.mark.parametrize("property_name,value", [
    ("application.name", "Browser"),
    ("application.process.id", "457"),
    ("application.process.binary", "browser"),
    ("window.x11.display", ":0"),
])
def test_unrelated_dedicated_sink_occupant_is_rejected(game, audio, property_name, value):
    sinks, inputs, server = audio
    extra = {"sink": 10, "properties": dict(inputs[0]["properties"])}
    extra["properties"][property_name] = value
    inputs.append(extra)
    with pytest.raises(video.VideoQualificationError, match="unrelated"):
        video.validate_game_audio(game, video.GAME_MONITOR, sinks, inputs, server)


@pytest.mark.parametrize("key,value", [
    ("default_sink_name", video.GAME_SINK),
    ("default_source_name", video.GAME_MONITOR),
    ("default_source_name", None),
])
def test_game_sink_must_not_be_a_desktop_default(game, audio, key, value):
    sinks, inputs, server = audio
    server[key] = value
    with pytest.raises(video.VideoQualificationError, match="desktop default"):
        video.validate_game_audio(game, video.GAME_MONITOR, sinks, inputs, server)


def test_missing_game_stream_is_rejected(game, audio):
    sinks, _, server = audio
    with pytest.raises(video.VideoQualificationError, match="no verified Minecraft"):
        video.validate_game_audio(game, video.GAME_MONITOR, sinks, [], server)


def test_reused_game_pid_is_rejected(game, audio, monkeypatch):
    monkeypatch.setattr(video, "_process_identity", lambda *_args: (790, game.executable))
    with pytest.raises(video.VideoQualificationError, match="process changed"):
        video.validate_game_audio(game, video.GAME_MONITOR, *audio)


def test_host_or_unqualified_session_is_rejected_before_window_lookup(monkeypatch):
    monkeypatch.setattr(video, "_current_uid", lambda: 1000)
    session = SimpleNamespace(find_window=lambda: pytest.fail("unqualified window inspected"))
    monkeypatch.setattr(video.BedrockSession, "load", lambda: session)

    def reject(_session):
        raise video.VideoQualificationError("not isolated")

    monkeypatch.setattr(video, "require_autonomous_input_isolation", reject)
    with pytest.raises(video.VideoQualificationError, match="not isolated"):
        video.inspect_game_capture()


def test_missing_unix_uid_fails_closed_before_live_session_access(monkeypatch):
    monkeypatch.setattr(video, "_uid_reader", None)
    monkeypatch.setattr(
        video.BedrockSession, "load", lambda: pytest.fail("unsupported OS must not inspect a game")
    )
    with pytest.raises(video.VideoQualificationError, match="requires Linux"):
        video.inspect_game_capture()


def test_generic_fallback_window_is_not_accepted_as_minecraft(monkeypatch):
    monkeypatch.setattr(video, "_current_uid", lambda: 1000)
    session = SimpleNamespace(display=":71", find_window=lambda: 124)
    connection = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(video.BedrockSession, "load", lambda: session)
    monkeypatch.setattr(video, "require_autonomous_input_isolation", lambda _session: None)
    monkeypatch.setattr(
        video.importlib, "import_module",
        lambda _name: SimpleNamespace(Display=lambda _: connection),
    )
    monkeypatch.setattr(
        video, "_new_wine_geometry", lambda _display: (SimpleNamespace(id=123), None, ())
    )
    with pytest.raises(video.VideoQualificationError, match="exact Minecraft"):
        video.inspect_game_capture()


@pytest.mark.parametrize("process_display", [":71", ":0"])
@linux_only
def test_process_identity_requires_real_game_command_and_private_display(
    tmp_path, monkeypatch, process_display
):
    directory = tmp_path / "456"
    directory.mkdir()
    fields = [b"S", b"1", *([b"0"] * 17), b"789"]
    (directory / "stat").write_bytes(b"456 (Minecraft MAIN) " + b" ".join(fields))
    (directory / "cmdline").write_bytes(b"C:\\game\\Minecraft.Windows.exe\0")
    (directory / "environ").write_bytes(("DISPLAY=" + process_display).encode() + b"\0")
    (directory / "exe").symlink_to("/wine/wine-preloader")
    monkeypatch.setattr(video, "PROC_ROOT", tmp_path)
    if process_display == ":71":
        assert video._process_identity(456, ":71") == (789, "/wine/wine-preloader")
    else:
        with pytest.raises(video.VideoQualificationError, match="not qualified"):
            video._process_identity(456, ":71")


def test_command_is_window_scoped_and_uses_bounded_cpu_hls(game, tmp_path):
    command = video.ffmpeg_command(game, video.GAME_MONITOR, tmp_path, 10)
    assert command[command.index("-window_id") + 1] == "123"
    assert [command[i + 1] for i, arg in enumerate(command) if arg == "-i"] == [
        ":71", video.GAME_MONITOR,
    ]
    assert command[command.index("-server") + 1] == video.PULSE_SERVER
    assert command[command.index("-threads") + 1] == "2"
    assert command[command.index("-filter_threads") + 1] == "1"
    assert command[command.index("-thread_queue_size") + 1] == "8"
    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-c:a") + 1] == "aac"
    assert command[command.index("-framerate") + 1] == "24"
    assert command[command.index("-vf") + 1].startswith("scale=1280:720:")
    assert command[command.index("-hls_list_size") + 1] == "6"
    assert "temp_file" in command[command.index("-hls_flags") + 1]
    assert "delete_segments" in command[command.index("-hls_flags") + 1]
    assert "-nostdin" in command and "-n" in command
    assert "-y" not in command
    assert not any(value in command for value in ("-hwaccel", "-vaapi_device", "default"))


@pytest.mark.parametrize("duration", [0, -1, 60.01, float("inf"), float("nan")])
def test_capture_duration_is_bounded(game, tmp_path, duration):
    with pytest.raises(video.VideoQualificationError, match="sixty"):
        video.ffmpeg_command(game, video.GAME_MONITOR, tmp_path, duration)


@linux_only
def test_output_must_be_existing_private_task_directory(tmp_path):
    tmp_path.chmod(0o700)
    assert video.validate_output_directory(tmp_path) == tmp_path
    tmp_path.chmod(0o755)
    with pytest.raises(video.VideoQualificationError, match="owner-only"):
        video.validate_output_directory(tmp_path)
    with pytest.raises(video.VideoQualificationError):
        video.validate_output_directory(Path("/tmp"))


@linux_only
def test_output_symbolic_link_is_refused(tmp_path):
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(video.VideoQualificationError, match="symbolic"):
        video.validate_output_directory(link)


def test_pactl_errors_do_not_reveal_subprocess_output(monkeypatch):
    def fail(*_args, **_kwargs):
        raise subprocess.CalledProcessError(1, "private command", output=b"private token")

    monkeypatch.setattr(video.subprocess, "run", fail)
    with pytest.raises(video.VideoQualificationError) as error:
        video._pactl_json("list", "sinks")
    assert str(error.value) == "audio inventory could not be verified"


@linux_only
def test_changed_capture_terminates_only_owned_encoder(game, tmp_path, monkeypatch):
    tmp_path.chmod(0o700)
    observations = iter([game, replace(game, window_id=124)])
    monkeypatch.setattr(video, "inspect_game_capture", lambda: next(observations))
    monkeypatch.setattr(video, "verify_game_audio", lambda *_args: None)
    monkeypatch.setattr(video.shutil, "which", lambda _name: "/usr/bin/ffmpeg")

    class Encoder:
        returncode = None
        terminated = False

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def wait(self, timeout):
            return self.returncode

    encoder = Encoder()
    monkeypatch.setattr(video.subprocess, "Popen", lambda *_args, **_kwargs: encoder)
    with pytest.raises(video.VideoQualificationError, match="identity changed"):
        video.qualify_video(tmp_path, video.GAME_MONITOR, 5)
    assert encoder.terminated
    assert len(list(tmp_path.iterdir())) == 1


@linux_only
def test_existing_generation_is_never_overwritten(game, tmp_path, monkeypatch):
    tmp_path.chmod(0o700)
    generation = tmp_path / "video-existing"
    generation.mkdir(mode=0o700)
    playlist = generation / "index.m3u8"
    playlist.write_text("preserve existing evidence")
    monkeypatch.setattr(video, "inspect_game_capture", lambda: game)
    monkeypatch.setattr(video, "verify_game_audio", lambda *_args: None)
    monkeypatch.setattr(video.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(video.uuid, "uuid4", lambda: SimpleNamespace(hex="existing"))
    monkeypatch.setattr(
        video.subprocess, "Popen", lambda *_args, **_kwargs: pytest.fail("must not start encoder")
    )
    with pytest.raises(FileExistsError):
        video.qualify_video(tmp_path, video.GAME_MONITOR, 5)
    assert playlist.read_text() == "preserve existing evidence"


def test_failed_cli_redacts_arbitrary_exception(monkeypatch, tmp_path, capsys):
    def fail(*_args):
        raise RuntimeError("secret token in a private path")

    monkeypatch.setattr(video, "qualify_video", fail)
    monkeypatch.setattr("sys.argv", [
        "video", "--output-dir", str(tmp_path), "--audio-monitor", video.GAME_MONITOR,
    ])
    assert video.main() == 1
    response = json.loads(capsys.readouterr().out)
    assert response == {"qualified": False, "error": "local video qualification failed"}
