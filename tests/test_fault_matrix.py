from __future__ import annotations

import pytest

from minecraft_ai.safety import MotorAction, SupervisorState
from minecraft_ai.supervisor import Supervisor


FAULTS = (
    "stale-frame-stream",
    "target-process-death",
    "target-identity-mismatch",
    "ipc-disconnect",
    "deadline-miss",
    "malformed-motor-action",
    "scope-regression",
    "supervisor-heartbeat-miss",
    "cognition-crash",
    "motor-crash",
    "capture-crash",
    "minecraft-crash",
    "target-window-replacement",
    "focus-change",
    "suspend-resume",
)


@pytest.mark.parametrize("fault", FAULTS)
def test_fault_releases_all_input_and_enters_failsafe(fault: str) -> None:
    supervisor = Supervisor()
    supervisor.start()
    lease_info = supervisor.arm("fake-instance")
    supervisor.activate()
    lease_id = str(lease_info["lease_id"])
    supervisor.motor.apply(
        lease_id,
        MotorAction(
            sequence=0,
            keys_down=("w", "shift"),
            buttons_down=("left",),
            duration_ms=25,
        ),
    )
    assert supervisor.state == SupervisorState.RUNNING
    assert supervisor.backend.held_keys == {"w", "shift"}
    assert supervisor.backend.held_buttons == {"left"}

    supervisor.fail(fault)

    status = supervisor.status()
    assert supervisor.state == SupervisorState.FAILSAFE
    assert status["last_fault"] == fault
    assert status["motor_lease_active"] is False
    assert supervisor.motor.lease is None
    assert supervisor.backend.held_keys == set()
    assert supervisor.backend.held_buttons == set()


def test_running_requires_a_live_lease() -> None:
    supervisor = Supervisor()
    supervisor.start()
    supervisor.arm("fake-instance")
    supervisor.motor.revoke("test-revoke-before-activate")

    supervisor.activate()

    assert supervisor.state == SupervisorState.FAILSAFE
    assert supervisor.last_fault == "activate-without-live-lease"
    assert supervisor.motor.lease is None
    assert supervisor.backend.held_keys == set()
    assert supervisor.backend.held_buttons == set()


def test_arm_then_activate_reaches_running() -> None:
    supervisor = Supervisor()
    supervisor.start()
    supervisor.arm("fake-instance")

    assert supervisor.state == SupervisorState.ARMED
    assert supervisor.motor.lease is not None

    supervisor.activate()

    assert supervisor.state == SupervisorState.RUNNING
    assert supervisor.motor.lease is not None
