from __future__ import annotations

from pathlib import Path

import pytest

import minecraft_ai.emergency as emergency
from minecraft_ai.safety import SupervisorState
from minecraft_ai.supervisor import Supervisor


def test_emergency_latch_persists_reason_and_clears(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    latch = tmp_path / "EMERGENCY_STOP"
    monkeypatch.setattr(emergency, "EMERGENCY_STOP_FILE", latch)

    assert emergency.emergency_stop_latched() is False
    emergency.engage_emergency_stop("operator-test")

    assert emergency.emergency_stop_latched() is True
    assert emergency.emergency_reason() == "operator-test"

    emergency.clear_emergency_stop()
    assert emergency.emergency_stop_latched() is False
    assert emergency.emergency_reason() is None


def test_supervisor_refuses_start_while_latched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_latch_blocks_resume_and_rearming(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_latch_while_armed_prevents_activation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    latch = tmp_path / "EMERGENCY_STOP"
    monkeypatch.setattr(emergency, "EMERGENCY_STOP_FILE", latch)

    supervisor = Supervisor()
    supervisor.start()
    supervisor.arm("fake-instance")
    emergency.engage_emergency_stop("operator-stop")

    supervisor.activate()

    assert supervisor.state == SupervisorState.FAILSAFE
    assert supervisor.motor.lease is None
    assert supervisor.backend.held_keys == set()
    assert supervisor.backend.held_buttons == set()
