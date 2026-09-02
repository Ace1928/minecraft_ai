from __future__ import annotations

from minecraft_ai.safety import MotorAction, SupervisorState
from minecraft_ai.supervisor import Supervisor


def test_supervisor_starts_safe_idle() -> None:
    supervisor = Supervisor()
    supervisor.start()
    assert supervisor.state == SupervisorState.SAFE_IDLE
    assert supervisor.status()["live_capable"] is False
    assert supervisor.status()["motor_lease_active"] is False


def test_pause_revokes_active_lease_and_releases_input() -> None:
    supervisor = Supervisor()
    supervisor.start()
    lease_info = supervisor.arm("fake-instance")
    lease_id = str(lease_info["lease_id"])
    supervisor.motor.apply(lease_id, MotorAction(sequence=0, keys_down=("w",)))
    assert "w" in supervisor.backend.held_keys

    supervisor.pause()

    assert supervisor.state == SupervisorState.PAUSED
    assert not supervisor.backend.held_keys
    assert supervisor.motor.lease is None


def test_fault_enters_failsafe_and_releases_input() -> None:
    supervisor = Supervisor()
    supervisor.start()
    lease_info = supervisor.arm("fake-instance")
    lease_id = str(lease_info["lease_id"])
    supervisor.motor.apply(
        lease_id,
        MotorAction(sequence=0, keys_down=("w",), buttons_down=("left",)),
    )

    supervisor.fail("capture-crash")

    assert supervisor.state == SupervisorState.FAILSAFE
    assert supervisor.last_fault == "capture-crash"
    assert not supervisor.backend.held_keys
    assert not supervisor.backend.held_buttons
    assert supervisor.motor.lease is None


def test_release_inputs_preserves_running_lease() -> None:
    supervisor = Supervisor()
    supervisor.start()
    lease_info = supervisor.arm("fake-instance")
    lease_id = str(lease_info["lease_id"])
    supervisor.activate()
    supervisor.motor.apply(
        lease_id,
        MotorAction(sequence=0, keys_down=("w",), buttons_down=("left",)),
    )

    result = supervisor.release_inputs(lease_id)

    assert result == {"released": True, "lease_active": True}
    assert supervisor.state == SupervisorState.RUNNING
    assert supervisor.motor.lease is not None
    assert not supervisor.backend.held_keys
    assert not supervisor.backend.held_buttons


def test_stop_from_failsafe_reaches_stopped() -> None:
    supervisor = Supervisor()
    supervisor.start()
    supervisor.fail("test")
    supervisor.stop()
    assert supervisor.state == SupervisorState.STOPPED
    assert not supervisor.backend.held_keys
    assert not supervisor.backend.held_buttons


def test_resume_only_returns_to_safe_idle() -> None:
    supervisor = Supervisor()
    supervisor.start()
    supervisor.pause()
    supervisor.resume()
    assert supervisor.state == SupervisorState.SAFE_IDLE
    assert supervisor.motor.lease is None
