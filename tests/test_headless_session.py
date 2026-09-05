from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import minecraft_ai.platforms.bedrock_session as sessions
from minecraft_ai.cli import _require_autonomous_isolated_session
from minecraft_ai.platforms.bedrock_x11 import IsolationError
from minecraft_ai.platforms.weston_seat import HeadlessSeatArtifact


def _verified_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> sessions.BedrockSession:
    command = tuple(
        sessions._weston_command(
            weston="/usr/bin/weston",
            wayland_socket="minecraft-ai-test",
            width=1920,
            height=1080,
            fullscreen=False,
            compositor_log=tmp_path / "weston.log",
            seat_module=tmp_path / "seat.so",
        )
    )
    monkeypatch.setattr(sessions, "_IS_LINUX", True)
    monkeypatch.setattr(sessions, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(sessions, "_linux_process_identity", lambda _pid: (99, command))
    monkeypatch.setattr(sessions, "require_loaded_headless_seat", lambda *_args: None)
    return sessions.BedrockSession(
        display=":71",
        host_display=":0",
        xserver_pid=300,
        launcher_pid=200,
        width=1920,
        height=1080,
        created_ns=1,
        launcher_command=("bedrock-on-linux", "play"),
        xserver_proc_start_ticks=99,
        xserver_command_sha256=sessions._command_sha256(command),
        mode="weston",
        input_isolation=sessions.HEADLESS_INPUT_ISOLATION,
        wayland_socket="minecraft-ai-test",
        compositor_log=str(tmp_path / "weston.log"),
        input_isolation_module=str(tmp_path / "seat.so"),
        input_isolation_module_sha256="module",
        input_isolation_source_sha256="source",
    )


def test_verified_headless_identity_passes_cli_and_descriptor_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _verified_session(tmp_path, monkeypatch)
    session.persist(tmp_path / "session.json")
    loaded = sessions.BedrockSession.load(tmp_path / "session.json")
    assert loaded == session
    _require_autonomous_isolated_session(loaded)


@pytest.mark.parametrize("mutation", ["legacy", "parent-backend", "extra-module", "reused-pid"])
def test_adoption_cannot_promote_host_transport_or_changed_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    session = _verified_session(tmp_path, monkeypatch)
    ticks, command = sessions._linux_process_identity(300)  # type: ignore[misc]
    if mutation == "legacy":
        session = replace(session, input_isolation="unverified")
    elif mutation == "parent-backend":
        command = tuple(
            "--backend=wayland" if arg == "--backend=headless" else arg for arg in command
        )
        # Even a descriptor whose command hash was relabeled to match cannot pass.
        session = replace(session, xserver_command_sha256=sessions._command_sha256(command))
    elif mutation == "extra-module":
        command = (*command, "--modules=/tmp/input-forwarder.so")
        session = replace(session, xserver_command_sha256=sessions._command_sha256(command))
    else:
        ticks += 1
    monkeypatch.setattr(sessions, "_linux_process_identity", lambda _pid: (ticks, command))
    with pytest.raises(IsolationError):
        sessions.require_autonomous_input_isolation(session)


def test_legacy_liveness_is_separate_from_authority_to_avoid_automatic_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = replace(_verified_session(tmp_path, monkeypatch), input_isolation="unverified")
    monkeypatch.setattr(
        sessions, "_session_process_state", lambda *_args, **_kwargs: "verified-live"
    )
    monkeypatch.setattr(sessions, "_x_socket", lambda _display: tmp_path)
    assert sessions.bedrock_session_alive(session)
    with pytest.raises(IsolationError, match="existing session is preserved"):
        sessions.require_autonomous_input_isolation(session)


def test_launch_without_host_display_drops_inherited_host_handles_and_verifies_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []
    order: list[str] = []

    class Child:
        def __init__(self, pid: int) -> None:
            self.pid = pid

    def popen(command: list[str], **kwargs: object) -> Child:
        calls.append((command, kwargs))
        return Child(300 + len(calls))

    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.delenv("DISPLAY", raising=False)
    for name in ("WAYLAND_DISPLAY", "WAYLAND_SOCKET", "XAUTHORITY", "BOL_DISPLAY"):
        monkeypatch.setenv(name, "host-handle")
    monkeypatch.setattr(sessions, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(sessions.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(sessions.subprocess, "Popen", popen)
    monkeypatch.setattr(sessions, "_wait_for_weston_xwayland", lambda *_args: ":71")
    monkeypatch.setattr(
        sessions, "_required_process_identity", lambda *_args, **_kwargs: (1, "hash")
    )
    monkeypatch.setattr(
        sessions,
        "build_headless_seat_module",
        lambda: HeadlessSeatArtifact(
            str(tmp_path / "seat.so"),
            "module",
            "source",
        ),
    )
    monkeypatch.setattr(
        sessions, "require_autonomous_input_isolation", lambda _session: order.append("verify")
    )
    monkeypatch.setattr(
        sessions, "_prepare_new_isolated_bedrock_geometry", lambda _session: order.append("prepare")
    )
    monkeypatch.setattr(
        sessions.BedrockSession, "persist", lambda _session: order.append("persist")
    )
    session = sessions.launch_isolated_bedrock_session(
        launcher_command=("bedrock-on-linux", "play")
    )
    assert session.host_display == ""
    assert session.input_isolation == sessions.HEADLESS_INPUT_ISOLATION
    assert not session.compositor_fullscreen
    assert order == ["verify", "prepare", "persist"]
    compositor_env = calls[0][1]["env"]
    launcher_env = calls[1][1]["env"]
    assert isinstance(compositor_env, dict) and isinstance(launcher_env, dict)
    for name in ("DISPLAY", "WAYLAND_DISPLAY", "WAYLAND_SOCKET", "XAUTHORITY", "BOL_DISPLAY"):
        assert name not in compositor_env
    assert launcher_env["DISPLAY"] == ":71"
    assert launcher_env["WAYLAND_DISPLAY"] == session.wayland_socket
    assert "WAYLAND_SOCKET" not in launcher_env


def test_missing_weston_never_falls_back_to_host_fed_xephyr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/tmp/test-runtime")
    monkeypatch.setattr(sessions.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        sessions, "launch_xephyr_bedrock_session", lambda **_kwargs: pytest.fail("fallback")
    )
    with pytest.raises(IsolationError, match="Weston is not installed"):
        sessions.launch_isolated_bedrock_session()


@pytest.mark.parametrize(
    "remaining", ["compositor", "launcher-group", "compositor-group", "socket"]
)
def test_dead_launcher_does_not_make_remaining_game_resources_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    remaining: str,
) -> None:
    session = _verified_session(tmp_path, monkeypatch)

    def process_probe(pid: int, signal: int) -> None:
        assert signal == 0
        if remaining == "compositor" and pid == session.xserver_pid:
            return
        raise ProcessLookupError

    def group_probe(pid: int, signal: int) -> None:
        assert signal == 0
        if remaining == "launcher-group" and pid == session.launcher_pid:
            return
        if remaining == "compositor-group" and pid == session.xserver_pid:
            return
        raise ProcessLookupError

    socket = tmp_path / "X71"
    if remaining == "socket":
        socket.touch()
    monkeypatch.setattr(sessions.os, "kill", process_probe)
    monkeypatch.setattr(sessions, "_signal_process_group", group_probe)
    monkeypatch.setattr(sessions, "_x_socket", lambda _display: socket)
    assert not sessions.bedrock_session_resources_absent(session)


@pytest.mark.parametrize("failure_at", ["pid", "group", "socket"])
@pytest.mark.parametrize("failure", [PermissionError, OSError])
def test_resource_probe_errors_never_authorize_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_at: str,
    failure: type[OSError],
) -> None:
    session = _verified_session(tmp_path, monkeypatch)

    def process_probe(_pid: int, _signal: int) -> None:
        raise failure() if failure_at == "pid" else ProcessLookupError()

    def group_probe(_pid: int, _signal: int) -> None:
        raise failure() if failure_at == "group" else ProcessLookupError()

    def socket_probe(_path: Path) -> None:
        raise failure()

    monkeypatch.setattr(sessions.os, "kill", process_probe)
    monkeypatch.setattr(sessions, "_signal_process_group", group_probe)
    monkeypatch.setattr(Path, "lstat", socket_probe)
    assert not sessions.bedrock_session_resources_absent(session)


@pytest.mark.parametrize("mode", ["weston", "xephyr", "direct", "host-monitor"])
def test_proven_absent_resources_allow_fresh_launch_without_probing_host_socket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    session = replace(_verified_session(tmp_path, monkeypatch), mode=mode)

    def absent(_pid: int, _signal: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr(sessions.os, "kill", absent)
    monkeypatch.setattr(sessions, "_signal_process_group", absent)
    if mode in {"direct", "host-monitor"}:
        session = replace(session, display=":0", xserver_pid=0)
        monkeypatch.setattr(sessions, "_x_socket", lambda _display: pytest.fail("host socket"))
    else:
        monkeypatch.setattr(sessions, "_x_socket", lambda _display: tmp_path / "absent-X71")
    assert sessions.bedrock_session_resources_absent(session)
