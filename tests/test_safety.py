from __future__ import annotations

import time

import pytest

from minecraft_ai.safety import (
    FakeInputBackend,
    MotorAction,
    MotorGate,
    MotorRejected,
    SupervisorState,
    TransitionError,
    validate_transition,
)


def test_valid_lifecycle_transitions() -> None:
    path = [
        SupervisorState.STOPPED,
        SupervisorState.STARTING,
        SupervisorState.SAFE_IDLE,
        SupervisorState.ARMED,
        SupervisorState.RUNNING,
        SupervisorState.PAUSED,
        SupervisorState.SAFE_IDLE,
        SupervisorState.STOPPING,
        SupervisorState.STOPPED,
    ]
    for current, target in zip(path, path[1:], strict=True):
        validate_transition(current, target)


def test_invalid_transition_rejected() -> None:
    with pytest.raises(TransitionError):
        validate_transition(SupervisorState.STOPPED, SupervisorState.RUNNING)


def test_revoke_releases_all_held_input() -> None:
    backend = FakeInputBackend()
    gate = MotorGate(backend)
    lease = gate.issue(session_id="test", target_instance="minecraft")
    gate.apply(
        lease.lease_id,
        MotorAction(sequence=0, keys_down=("w", "shift"), buttons_down=("left",)),
    )
    assert backend.held_keys == {"w", "shift"}
    assert backend.held_buttons == {"left"}

    gate.revoke("test-stop")

    assert backend.held_keys == set()
    assert backend.held_buttons == set()
    assert gate.lease is None
    assert backend.release_count >= 2  # issue clears stale state; revoke clears active state


def test_replayed_action_fails_closed() -> None:
    backend = FakeInputBackend()
    gate = MotorGate(backend)
    lease = gate.issue(session_id="test", target_instance="minecraft")
    gate.apply(lease.lease_id, MotorAction(sequence=5, keys_down=("w",)))
    assert "w" in backend.held_keys

    with pytest.raises(MotorRejected):
        gate.apply(lease.lease_id, MotorAction(sequence=5, keys_down=("a",)))

    assert backend.held_keys == set()
    assert backend.held_buttons == set()
    assert gate.lease is None
    assert gate.revocation_reason == "replay-or-out-of-order-action"


def test_action_duration_limit_fails_closed() -> None:
    backend = FakeInputBackend()
    gate = MotorGate(backend)
    lease = gate.issue(
        session_id="test",
        target_instance="minecraft",
        max_action_duration_ms=50,
    )
    gate.apply(lease.lease_id, MotorAction(sequence=0, keys_down=("w",), duration_ms=10))

    with pytest.raises(MotorRejected):
        gate.apply(lease.lease_id, MotorAction(sequence=1, keys_down=("w",), duration_ms=51))

    assert not backend.held_keys
    assert gate.lease is None


def test_expired_lease_fails_closed() -> None:
    backend = FakeInputBackend()
    gate = MotorGate(backend)
    lease = gate.issue(session_id="test", target_instance="minecraft", ttl_ms=50)
    gate.apply(lease.lease_id, MotorAction(sequence=0, buttons_down=("left",)))
    time.sleep(0.06)

    assert gate.check_expiry()
    assert gate.lease is None
    assert backend.held_buttons == set()
    assert gate.revocation_reason == "watchdog-expired"


def test_wrong_lease_id_releases_existing_input() -> None:
    backend = FakeInputBackend()
    gate = MotorGate(backend)
    lease = gate.issue(session_id="test", target_instance="minecraft")
    gate.apply(lease.lease_id, MotorAction(sequence=0, keys_down=("w",)))

    with pytest.raises(MotorRejected):
        gate.apply("wrong-lease", MotorAction(sequence=1, keys_down=("a",)))

    assert backend.held_keys == set()


def test_action_model_rejects_unbounded_mouse_and_duration() -> None:
    with pytest.raises(ValueError):
        MotorAction(sequence=0, mouse_dx=5000)
    with pytest.raises(ValueError):
        MotorAction(sequence=0, duration_ms=1001)
