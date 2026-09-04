from __future__ import annotations

from pathlib import Path
import json
import signal
import sys

import pytest

import minecraft_ai.emergency as emergency
from minecraft_ai.safety import SupervisorState
from minecraft_ai.supervisor import Supervisor


def _control_payload(command: tuple[str, ...]) -> dict[str, object]:
    return {
        "host": "127.0.0.1",
        "port": 12345,
        "token": "secret",
        "pid": 4242,
        "session_id": "session",
        "proc_start_ticks": 9876,
        "command_sha256": emergency._command_sha256(command),
    }


def test_emergency_latch_persists_reason_and_clears(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    latch = tmp_path / "EMERGENCY_STOP"
    monkeypatch.setattr(emergency, "EMERGENCY_STOP_FILE", latch)

    assert emergency.emergency_stop_latched() is False
    emergency.engage_emergency_stop("operator-test")

    assert emergency.emergency_stop_latched() is True
    assert emergency.emergency_reason() == "operator-test"

    emergency.clear_emergency_stop()
    assert emergency.emergency_stop_latched() is False
    assert emergency.emergency_reason() is None


def test_supervisor_refuses_start_while_latched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    latch = tmp_path / "EMERGENCY_STOP"
    monkeypatch.setattr(emergency, "EMERGENCY_STOP_FILE", latch)
    emergency.engage_emergency_stop("do-not-start")

    supervisor = Supervisor()
    with pytest.raises(RuntimeError, match="emergency stop is latched"):
        supervisor.start()

    assert supervisor.state == SupervisorState.STOPPED
    assert supervisor.motor.lease is None
    assert supervisor.status()["emergency_stop_latched"] is True
    assert supervisor.last_fault == "do-not-start"


def test_latch_blocks_resume_and_rearming(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    latch = tmp_path / "EMERGENCY_STOP"
    monkeypatch.setattr(emergency, "EMERGENCY_STOP_FILE", latch)

    supervisor = Supervisor()
    supervisor.start()
    supervisor.pause()
    emergency.engage_emergency_stop("operator-stop")

    with pytest.raises(RuntimeError, match="emergency stop is latched"):
        supervisor.resume()

    assert supervisor.state == SupervisorState.PAUSED
    assert supervisor.motor.lease is None


def test_latch_while_armed_prevents_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    latch = tmp_path / "EMERGENCY_STOP"
    monkeypatch.setattr(emergency, "EMERGENCY_STOP_FILE", latch)

    supervisor = Supervisor()
    supervisor.start()
    supervisor.arm("fake-instance")
    emergency.engage_emergency_stop("operator-stop")

    with pytest.raises(RuntimeError, match="emergency stop is latched"):
        supervisor.activate()

    assert supervisor.state == SupervisorState.FAILSAFE
    assert supervisor.motor.lease is None
    assert supervisor.backend.held_keys == set()
    assert supervisor.backend.held_buttons == set()


def test_legacy_supervisor_descriptor_is_never_signaled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = tmp_path / "control.json"
    control.write_text(json.dumps({"pid": 4242}), encoding="utf-8")
    monkeypatch.setattr(emergency, "CONTROL_FILE", control)
    monkeypatch.setattr(emergency, "_IS_LINUX", True)
    signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(emergency.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(
        emergency.os, "pidfd_open", lambda _pid, _flags: 11, raising=False
    )

    assert emergency.terminate_registered_supervisor() is False
    assert signals == []


def test_supervisor_identity_is_rechecked_immediately_before_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = (sys.executable, "-m", "minecraft_ai.supervisor", "--role", "builder")
    control = tmp_path / "control.json"
    control.write_text(json.dumps(_control_payload(command)), encoding="utf-8")
    identities = iter(((9876, command), None))
    monkeypatch.setattr(emergency, "CONTROL_FILE", control)
    monkeypatch.setattr(emergency, "_IS_LINUX", True)
    monkeypatch.setattr(
        emergency, "_linux_process_identity", lambda _pid: next(identities)
    )
    signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(emergency.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(
        emergency.os, "pidfd_open", lambda _pid, _flags: 11, raising=False
    )
    monkeypatch.setattr(emergency.os, "close", lambda _fd: None)
    monkeypatch.setattr(
        emergency.signal,
        "pidfd_send_signal",
        lambda _fd, sent: signals.append((4242, sent)),
        raising=False,
    )

    assert emergency.terminate_registered_supervisor() is False
    assert signals == []


def test_verified_supervisor_is_terminated_without_signaling_reused_pid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = (sys.executable, "-m", "minecraft_ai.supervisor", "--role", "builder")
    control = tmp_path / "control.json"
    control.write_text(json.dumps(_control_payload(command)), encoding="utf-8")
    running = True
    monkeypatch.setattr(emergency, "CONTROL_FILE", control)
    monkeypatch.setattr(emergency, "_IS_LINUX", True)
    monkeypatch.setattr(
        emergency,
        "_linux_process_identity",
        lambda _pid: (9876, command) if running else None,
    )
    signals: list[tuple[int, signal.Signals]] = []

    def fake_pidfd_signal(_fd: int, sent: signal.Signals) -> None:
        nonlocal running
        signals.append((4242, sent))
        running = False

    monkeypatch.setattr(
        emergency.os, "pidfd_open", lambda _pid, _flags: 11, raising=False
    )
    monkeypatch.setattr(emergency.os, "close", lambda _fd: None)
    monkeypatch.setattr(
        emergency.signal, "pidfd_send_signal", fake_pidfd_signal, raising=False
    )
    monkeypatch.setattr(
        emergency.os,
        "kill",
        lambda _pid, _sig: pytest.fail("numeric PID signaling is forbidden"),
    )

    assert emergency.terminate_registered_supervisor() is True
    assert signals == [(4242, signal.SIGTERM)]


def test_supervisor_identity_change_after_pidfd_open_sends_no_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = (sys.executable, "-m", "minecraft_ai.supervisor", "--role", "builder")
    control = tmp_path / "control.json"
    control.write_text(json.dumps(_control_payload(command)), encoding="utf-8")
    identities = iter(((9876, command), None))
    signals: list[signal.Signals] = []
    monkeypatch.setattr(emergency, "CONTROL_FILE", control)
    monkeypatch.setattr(emergency, "_IS_LINUX", True)
    monkeypatch.setattr(
        emergency, "_linux_process_identity", lambda _pid: next(identities)
    )
    monkeypatch.setattr(
        emergency.os, "pidfd_open", lambda _pid, _flags: 11, raising=False
    )
    monkeypatch.setattr(emergency.os, "close", lambda _fd: None)
    monkeypatch.setattr(
        emergency.signal,
        "pidfd_send_signal",
        lambda _fd, sent: signals.append(sent),
        raising=False,
    )
    monkeypatch.setattr(
        emergency.os,
        "kill",
        lambda _pid, _sig: pytest.fail("numeric PID signaling is forbidden"),
    )

    assert emergency.terminate_registered_supervisor() is False
    assert signals == []
