from __future__ import annotations

import signal
import sys
import threading
from dataclasses import replace
from pathlib import Path

import pytest

import minecraft_ai.platforms.bedrock_session as sessions
from minecraft_ai.platforms.bedrock_session import BedrockSession
from minecraft_ai.platforms.bedrock_x11 import IsolationError


def _session(*, mode: str = "direct", with_identity: bool = True) -> BedrockSession:
    launcher_command = ("/usr/bin/bedrock-on-linux", "play")
    launcher_observed = ("python3", "/usr/bin/bedrock-on-linux", "play")
    return BedrockSession(
        display=":71" if mode in {"weston", "xephyr"} else ":0",
        host_display=":0",
        xserver_pid=300 if mode in {"weston", "xephyr"} else 0,
        launcher_pid=200,
        width=1280,
        height=720,
        created_ns=10,
        launcher_command=launcher_command,
        launcher_proc_start_ticks=22 if with_identity else None,
        launcher_command_sha256=(
            sessions._command_sha256(launcher_observed) if with_identity else None
        ),
        xserver_proc_start_ticks=(33 if with_identity and mode in {"weston", "xephyr"} else None),
        xserver_command_sha256=(
            sessions._command_sha256(("/usr/bin/weston", "--xwayland"))
            if with_identity and mode == "weston"
            else sessions._command_sha256(("/usr/bin/Xephyr", ":71"))
            if with_identity and mode == "xephyr"
            else None
        ),
        mode=mode,
    )


def test_launcher_identity_waits_for_shebang_exec_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pre_exec = ("python", "-m", "minecraft_ai", "bedrock", "launch")
    stable = ("python3", "/usr/bin/bedrock-on-linux", "play")
    identities = iter(((91, pre_exec), (91, stable)))
    monkeypatch.setattr(sessions, "_IS_LINUX", True)
    monkeypatch.setattr(sessions, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(
        sessions,
        "_linux_process_identity",
        lambda _pid: next(identities),
    )

    start_ticks, digest = sessions._required_process_identity(
        123,
        label="Bedrock launcher",
        expected_program="bedrock-on-linux",
        required_argument="play",
    )

    assert start_ticks == 91
    assert digest == sessions._command_sha256(stable)


def test_legacy_live_descriptor_is_preserved_without_signaling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = tmp_path / "bedrock-session.json"
    session = _session(with_identity=False)
    monkeypatch.setattr(sessions, "BEDROCK_SESSION_FILE", descriptor)
    monkeypatch.setattr(sessions, "_pid_alive", lambda _pid: True)
    signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(
        sessions.os,
        "killpg",
        lambda pid, sig: signals.append((pid, sig)),
        raising=False,
    )
    session.persist()

    with pytest.raises(IsolationError, match="identity is unverifiable"):
        sessions.stop_bedrock_session()

    assert signals == []
    assert BedrockSession.load(descriptor) == session


def test_legacy_dead_leader_cannot_authorize_orphan_group_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = tmp_path / "bedrock-session.json"
    session = _session(with_identity=False)
    signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(sessions, "BEDROCK_SESSION_FILE", descriptor)
    monkeypatch.setattr(sessions, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(sessions, "_process_group_alive", lambda _pid: True)
    monkeypatch.setattr(
        sessions.os,
        "killpg",
        lambda pid, sent_signal: signals.append((pid, sent_signal)),
        raising=False,
    )
    session.persist()

    with pytest.raises(IsolationError, match="without modern identity metadata"):
        sessions.stop_bedrock_session()

    assert signals == []
    assert BedrockSession.load(descriptor) == session


def test_malformed_descriptor_is_preserved_and_stop_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = tmp_path / "bedrock-session.json"
    descriptor.write_text("{not-json", encoding="utf-8")
    monkeypatch.setattr(sessions, "BEDROCK_SESSION_FILE", descriptor)

    with pytest.raises(IsolationError, match="descriptor is unreadable"):
        sessions.stop_bedrock_session()

    assert descriptor.read_text(encoding="utf-8") == "{not-json"


def test_mismatched_live_descriptor_is_preserved_without_signaling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = tmp_path / "bedrock-session.json"
    session = _session()
    monkeypatch.setattr(sessions, "BEDROCK_SESSION_FILE", descriptor)
    monkeypatch.setattr(sessions, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(
        sessions,
        "_linux_process_identity",
        lambda _pid: (999, ("/usr/bin/unrelated",)),
    )
    signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(
        sessions.os,
        "killpg",
        lambda pid, sig: signals.append((pid, sig)),
        raising=False,
    )
    session.persist()

    with pytest.raises(IsolationError, match="identity is unverifiable or changed"):
        sessions.stop_bedrock_session()

    assert signals == []
    assert BedrockSession.load(descriptor) == session


def test_stop_rechecks_identity_immediately_before_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = tmp_path / "bedrock-session.json"
    session = _session()
    matching = ("python3", "/usr/bin/bedrock-on-linux", "play")
    identities = iter(((22, matching), (999, ("/usr/bin/unrelated",))))
    monkeypatch.setattr(sessions, "BEDROCK_SESSION_FILE", descriptor)
    monkeypatch.setattr(sessions, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(sessions, "_linux_process_identity", lambda _pid: next(identities))
    monkeypatch.setattr(sessions, "find_minecraft_window", lambda *_args, **_kwargs: None)
    signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(
        sessions.os,
        "killpg",
        lambda pid, sig: signals.append((pid, sig)),
        raising=False,
    )
    session.persist()

    with pytest.raises(IsolationError, match="identity changed"):
        sessions.stop_bedrock_session()

    assert signals == []
    assert descriptor.exists()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="requires Linux procfs")
def test_weston_persist_failure_cleans_both_fresh_process_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[object] = []
    cleaned: list[int] = []

    class _Child:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def poll(self) -> None:
            return None

    def fake_popen(*_args: object, **_kwargs: object) -> _Child:
        child = _Child(300 + len(created))
        created.append(child)
        return child

    monkeypatch.setattr(sessions, "RUNTIME_DIR", tmp_path)
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr(sessions.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(sessions.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(sessions, "_wait_for_weston_xwayland", lambda *_args: ":71")
    monkeypatch.setattr(sessions, "require_isolated_display", lambda *_args: None)
    monkeypatch.setattr(
        sessions,
        "_required_process_identity",
        lambda *_args, **_kwargs: (1, "digest"),
    )
    monkeypatch.setattr(
        sessions,
        "_terminate_spawned_process_group",
        lambda child: cleaned.append(child.pid) or True,
    )
    monkeypatch.setattr(
        BedrockSession,
        "persist",
        lambda _self, _path=None: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        sessions.launch_weston_bedrock_session(
            launcher_command=("/usr/bin/bedrock-on-linux", "play")
        )

    assert len(created) == 2
    assert cleaned == [301, 300]


def test_failed_residual_cleanup_preserves_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = tmp_path / "bedrock-session.json"
    session = _session(mode="weston")
    monkeypatch.setattr(sessions, "BEDROCK_SESSION_FILE", descriptor)
    monkeypatch.setattr(
        sessions,
        "_session_process_state",
        lambda *_args, **_kwargs: "verified-live",
    )
    monkeypatch.setattr(sessions, "find_minecraft_window", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        sessions,
        "_terminate_verified_session_group",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        sessions,
        "_terminate_private_display_processes",
        lambda _session, **_kwargs: (_ for _ in ()).throw(IsolationError("clients remain")),
    )
    session.persist()

    with pytest.raises(IsolationError, match="clients remain"):
        sessions.stop_bedrock_session()

    assert BedrockSession.load(descriptor) == session


def test_dead_nested_leaders_with_private_clients_fail_closed_without_signaling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = tmp_path / "bedrock-session.json"
    session = _session(mode="weston")
    swept: list[BedrockSession] = []
    monkeypatch.setattr(sessions, "BEDROCK_SESSION_FILE", descriptor)
    monkeypatch.setattr(sessions, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(sessions, "_process_group_alive", lambda _pid: False)
    monkeypatch.setattr(
        sessions,
        "_private_display_processes",
        lambda _session, *, require_wayland_nonce=False: (
            {} if require_wayland_nonce else {777: 1}
        ),
    )
    monkeypatch.setattr(
        sessions,
        "_terminate_private_display_processes",
        lambda selected, **_kwargs: swept.append(selected),
    )
    session.persist()

    with pytest.raises(IsolationError, match="private Bedrock clients remain"):
        sessions.stop_bedrock_session()

    assert swept == []
    assert descriptor.exists()


def test_dead_weston_uses_only_immutable_wayland_nonce_for_client_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = tmp_path / "bedrock-session.json"
    session = replace(_session(mode="weston"), wayland_socket="minecraft-ai-unique")
    cleanup_calls: list[tuple[dict[int, int], bool]] = []
    monkeypatch.setattr(sessions, "BEDROCK_SESSION_FILE", descriptor)
    monkeypatch.setattr(sessions, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(sessions, "_process_group_alive", lambda _pid: False)
    monkeypatch.setattr(
        sessions,
        "_private_display_processes",
        lambda _session, *, require_wayland_nonce=False: (
            {777: 12} if require_wayland_nonce else {777: 12, 888: 13}
        ),
    )
    monkeypatch.setattr(
        sessions,
        "_terminate_private_display_processes",
        lambda _session, *, expected, require_wayland_nonce=False: cleanup_calls.append(
            (expected, require_wayland_nonce)
        ),
    )
    session.persist()

    sessions.stop_bedrock_session()

    assert cleanup_calls == [({777: 12}, True)]
    assert not descriptor.exists()


def test_dead_leader_with_uncleanable_group_preserves_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = tmp_path / "bedrock-session.json"
    session = _session()
    monkeypatch.setattr(sessions, "BEDROCK_SESSION_FILE", descriptor)
    monkeypatch.setattr(sessions, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(sessions, "_process_group_alive", lambda pid: pid == 200)
    monkeypatch.setattr(
        sessions,
        "_terminate_orphaned_process_group",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            IsolationError("orphaned launcher remains")
        ),
    )
    session.persist()

    with pytest.raises(IsolationError, match="orphaned launcher remains"):
        sessions.stop_bedrock_session()

    assert BedrockSession.load(descriptor) == session


def test_graceful_window_close_precedes_and_avoids_process_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = tmp_path / "bedrock-session.json"
    session = _session()
    leader_state = "verified-live"
    calls: list[str] = []
    monkeypatch.setattr(sessions, "BEDROCK_SESSION_FILE", descriptor)
    monkeypatch.setattr(
        sessions,
        "_session_process_state",
        lambda *_args, **_kwargs: leader_state,
    )
    monkeypatch.setattr(sessions, "find_minecraft_window", lambda *_args, **_kwargs: 42)

    def close_window(*_args: object, **_kwargs: object) -> None:
        nonlocal leader_state
        calls.append("window-close")
        leader_state = "dead"

    monkeypatch.setattr(sessions, "request_window_close", close_window)
    monkeypatch.setattr(sessions, "_process_group_alive", lambda _pid: False)
    monkeypatch.setattr(
        sessions,
        "_private_display_processes",
        lambda _session, **_kwargs: {},
    )
    monkeypatch.setattr(
        sessions,
        "_terminate_verified_session_group",
        lambda *_args, **_kwargs: pytest.fail("graceful shutdown must avoid process signal"),
    )
    session.persist()

    sessions.stop_bedrock_session()

    assert calls == ["window-close"]
    assert not descriptor.exists()


def test_compositor_identity_change_skips_private_client_sweep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = tmp_path / "bedrock-session.json"
    session = _session(mode="weston")
    sweeps: list[bool] = []
    monkeypatch.setattr(sessions, "BEDROCK_SESSION_FILE", descriptor)
    monkeypatch.setattr(
        sessions,
        "_session_process_state",
        lambda *_args, **_kwargs: "verified-live",
    )
    monkeypatch.setattr(sessions, "find_minecraft_window", lambda *_args, **_kwargs: None)

    def terminate(_session: BedrockSession, *, launcher: bool) -> None:
        if not launcher:
            raise IsolationError("compositor identity changed")

    monkeypatch.setattr(sessions, "_terminate_verified_session_group", terminate)
    monkeypatch.setattr(
        sessions,
        "_terminate_private_display_processes",
        lambda _session: sweeps.append(True),
    )
    session.persist()

    with pytest.raises(IsolationError, match="compositor identity changed"):
        sessions.stop_bedrock_session()

    assert sweeps == []
    assert descriptor.exists()


def test_private_client_spawn_after_snapshot_is_not_signaled_and_retains_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = tmp_path / "bedrock-session.json"
    session = _session(mode="weston")
    scans = iter(
        (
            {777: 12},
            {777: 12},
            {777: 12, 888: 13},
            {888: 13},
        )
    )
    last_scan = {888: 13}
    signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(sessions, "BEDROCK_SESSION_FILE", descriptor)
    monkeypatch.setattr(
        sessions,
        "_session_process_state",
        lambda *_args, **_kwargs: "verified-live",
    )
    monkeypatch.setattr(sessions, "find_minecraft_window", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        sessions,
        "_terminate_verified_session_group",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        sessions,
        "_private_display_processes",
        lambda *_args, **_kwargs: next(scans, last_scan),
    )
    monkeypatch.setattr(
        sessions.os,
        "pidfd_open",
        lambda pid, _flags: pid + 1000,
        raising=False,
    )
    monkeypatch.setattr(sessions.os, "close", lambda _fd: None)
    monkeypatch.setattr(
        sessions.signal,
        "pidfd_send_signal",
        lambda pidfd, sent_signal: signals.append((pidfd - 1000, sent_signal)),
        raising=False,
    )
    monkeypatch.setattr(
        sessions.os,
        "kill",
        lambda _pid, _signal: pytest.fail("numeric PID signaling is forbidden"),
    )
    session.persist()

    with pytest.raises(IsolationError, match="unexpected private Bedrock display processes"):
        sessions.stop_bedrock_session()

    assert signals == [(777, signal.SIGTERM)]
    assert BedrockSession.load(descriptor) == session


def test_private_client_identity_change_after_pidfd_open_is_not_signaled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session(mode="weston")
    scans = iter(({777: 12}, {}, {}))
    signals: list[signal.Signals] = []
    closed: list[int] = []
    monkeypatch.setattr(
        sessions,
        "_private_display_processes",
        lambda *_args, **_kwargs: next(scans, {}),
    )
    monkeypatch.setattr(
        sessions.os, "pidfd_open", lambda _pid, _flags: 17, raising=False
    )
    monkeypatch.setattr(sessions.os, "close", lambda fd: closed.append(fd))
    monkeypatch.setattr(
        sessions.signal,
        "pidfd_send_signal",
        lambda _fd, sent_signal: signals.append(sent_signal),
        raising=False,
    )
    monkeypatch.setattr(
        sessions.os,
        "kill",
        lambda _pid, _signal: pytest.fail("numeric PID signaling is forbidden"),
    )

    sessions._terminate_private_display_processes(session, expected={777: 12})

    assert signals == []
    assert closed == [17]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="requires POSIX flock")
def test_bedrock_kernel_lock_rejects_concurrent_lifecycle_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sessions, "BEDROCK_LIFECYCLE_LOCK", tmp_path / "bedrock.lock")

    with sessions.bedrock_lifecycle_lock():
        with pytest.raises(IsolationError, match="already active"):
            with sessions.bedrock_lifecycle_lock():
                pytest.fail("concurrent Bedrock lifecycle operation acquired the lock")


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="requires POSIX flock")
def test_bedrock_kernel_lock_can_wait_for_inflight_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sessions, "BEDROCK_LIFECYCLE_LOCK", tmp_path / "bedrock.lock")
    first_locked = threading.Event()
    release_first = threading.Event()

    def hold_first_lock() -> None:
        with sessions.bedrock_lifecycle_lock():
            first_locked.set()
            assert release_first.wait(timeout=1.0)

    holder = threading.Thread(target=hold_first_lock)
    holder.start()
    assert first_locked.wait(timeout=1.0)
    threading.Timer(0.05, release_first.set).start()

    with sessions.bedrock_lifecycle_lock(wait_timeout_s=1.0):
        pass

    holder.join(timeout=1.0)
    assert not holder.is_alive()
