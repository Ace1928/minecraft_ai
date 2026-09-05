from __future__ import annotations

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
    for current, target in zip(path[:-1], path[1:], strict=True):
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


def test_partial_backend_failure_revokes_and_releases_immediately() -> None:
    class _PartialFailureBackend(FakeInputBackend):
        def apply(self, action: MotorAction) -> None:
            self.held_keys.add("w")
            raise RuntimeError("simulated XTEST failure")

    backend = _PartialFailureBackend()
    gate = MotorGate(backend)
    lease = gate.issue(session_id="test", target_instance="minecraft")

    with pytest.raises(RuntimeError, match="simulated XTEST failure"):
        gate.apply(lease.lease_id, MotorAction(sequence=0, keys_down=("w",)))

    assert backend.held_keys == set()
    assert backend.held_buttons == set()
    assert backend.lease_id is None
    assert backend.release_count == 2
    assert gate.lease is None
    assert gate.revocation_reason == "backend-apply-failed"


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

    assert not gate.check_expiry(lease.expires_monotonic_ns - 1)
    assert gate.check_expiry(lease.expires_monotonic_ns)
    assert gate.lease is None
    assert backend.held_buttons == set()
    assert gate.revocation_reason == "watchdog-expired"


def test_accepted_action_refresh_is_atomic_bounded_and_still_expires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock_ns = [1_000_000_000]
    monkeypatch.setattr(
        "minecraft_ai.safety.time.monotonic_ns",
        lambda: clock_ns[0],
    )

    class _DelayedAcceptBackend(FakeInputBackend):
        def apply(self, action: MotorAction) -> None:
            super().apply(action)
            # Simulate backend I/O crossing the old deadline while MotorGate's
            # lock prevents the watchdog from racing accepted actuation.
            clock_ns[0] += 60_000_000

    backend = _DelayedAcceptBackend()
    gate = MotorGate(backend)
    lease = gate.issue(session_id="test", target_instance="minecraft", ttl_ms=50)

    gate.apply(
        lease.lease_id,
        MotorAction(sequence=0, keys_down=("w",)),
        accepted_action_ttl_ms=5_000,
    )

    refreshed = gate.lease
    assert refreshed is not None
    assert refreshed.lease_id == lease.lease_id
    assert refreshed.expires_monotonic_ns > lease.expires_monotonic_ns
    assert not gate.check_expiry(refreshed.expires_monotonic_ns - 1)
    assert gate.check_expiry(refreshed.expires_monotonic_ns)
    assert backend.held_keys == set()
    assert gate.revocation_reason == "watchdog-expired"


def test_accepted_action_refresh_cannot_exceed_fixed_ttl_cap() -> None:
    backend = FakeInputBackend()
    gate = MotorGate(backend)
    lease = gate.issue(session_id="test", target_instance="minecraft")

    with pytest.raises(MotorRejected, match="ttl outside safety bounds"):
        gate.apply(
            lease.lease_id,
            MotorAction(sequence=0, keys_down=("w",)),
            accepted_action_ttl_ms=5_001,
        )

    assert backend.actions == []
    assert gate.lease == lease


def test_wrong_lease_id_releases_existing_input() -> None:
    backend = FakeInputBackend()
    gate = MotorGate(backend)
    lease = gate.issue(session_id="test", target_instance="minecraft")
    gate.apply(lease.lease_id, MotorAction(sequence=0, keys_down=("w",)))

    with pytest.raises(MotorRejected):
        gate.apply("wrong-lease", MotorAction(sequence=1, keys_down=("a",)))

    assert backend.held_keys == set()


@pytest.mark.parametrize("clear_fails", [False, True])
def test_revoke_clears_backend_even_when_release_fails(
    monkeypatch: pytest.MonkeyPatch, clear_fails: bool,
) -> None:
    backend = FakeInputBackend()
    gate = MotorGate(backend)
    gate.issue(session_id="test", target_instance="minecraft")
    release_error = RuntimeError("synthetic release failure")
    callbacks = []

    def release() -> None:
        assert gate.lease is None
        callbacks.append("release")
        raise release_error

    def clear() -> None:
        assert gate.lease is None
        callbacks.append("clear")
        backend.lease_id = None
        if clear_fails:
            raise OSError("synthetic clear failure")

    monkeypatch.setattr(backend, "release_all", release)
    monkeypatch.setattr(backend, "clear_lease", clear)
    with pytest.raises(RuntimeError) as caught:
        gate.revoke("test-stop")
    assert caught.value is release_error
    assert callbacks == ["release", "clear"]
    assert gate.lease is None and backend.lease_id is None
    assert gate.revocation_reason == "test-stop"
    if clear_fails:
        assert release_error.__notes__ == ["motor lease clearing also failed: OSError"]


@pytest.mark.parametrize("operation", ["issue", "renew", "refresh", "apply"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_backend_failure_preserves_original_error_and_clears_capability(
    monkeypatch: pytest.MonkeyPatch, operation: str, cleanup_fails: bool,
) -> None:
    backend = FakeInputBackend()
    gate = MotorGate(backend)
    lease = gate.issue(session_id="test", target_instance="minecraft")
    gate.apply(lease.lease_id, MotorAction(sequence=0, keys_down=("w",)))
    original_error = RuntimeError("synthetic backend failure")
    failure_started = False
    cleanup = []
    original_release = backend.release_all
    original_clear = backend.clear_lease
    original_apply = backend.apply
    original_bind = backend.bind_lease

    def bind(value) -> None:
        nonlocal failure_started
        if operation in {"issue", "renew"}:
            assert gate.lease is None
        original_bind(value)
        failure_started = True
        raise original_error

    def apply(action) -> None:
        nonlocal failure_started
        original_apply(action)
        failure_started = True
        raise original_error

    def release() -> None:
        assert gate.lease is None
        original_release()
        if failure_started:
            cleanup.append("release")
            if cleanup_fails:
                raise OSError("synthetic cleanup release failure")

    def clear() -> None:
        assert gate.lease is None
        original_clear()
        if failure_started:
            cleanup.append("clear")
            if cleanup_fails:
                raise ValueError("synthetic cleanup clear failure")

    monkeypatch.setattr(backend, "bind_lease", bind)
    monkeypatch.setattr(backend, "release_all", release)
    monkeypatch.setattr(backend, "clear_lease", clear)
    with pytest.raises(RuntimeError) as caught:
        if operation == "issue":
            gate.issue(session_id="test", target_instance="minecraft")
        elif operation == "renew":
            gate.renew(lease.lease_id)
        else:
            if operation == "apply":
                monkeypatch.setattr(backend, "apply", apply)
            gate.apply(
                lease.lease_id, MotorAction(sequence=1, keys_down=("a",)),
                accepted_action_ttl_ms=750,
            )
    assert caught.value is original_error
    assert gate.lease is None and backend.lease_id is None
    assert not backend.held_keys
    assert cleanup == ["release", "clear"]
    assert gate.revocation_reason == {
        "issue": "lease-issue-failed", "renew": "lease-renewal-failed",
        "refresh": "accepted-action-refresh-failed", "apply": "backend-apply-failed",
    }[operation]
    if cleanup_fails:
        assert original_error.__notes__ == ["motor cleanup also failed: OSError"]


def test_invalid_lease_clears_gate_before_failing_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeInputBackend()
    gate = MotorGate(backend)
    gate.issue(session_id="test", target_instance="minecraft")

    def release() -> None:
        assert gate.lease is None
        raise OSError("synthetic release failure")

    monkeypatch.setattr(backend, "release_all", release)
    with pytest.raises(MotorRejected, match="invalid or missing motor lease"):
        gate.renew("wrong-lease")
    assert gate.lease is None and backend.lease_id is None
    assert gate.revocation_reason == "invalid-or-missing-lease"


@pytest.mark.parametrize("operation", ["issue-ttl", "issue-duration", "renew"])
def test_invalid_lease_parameters_do_not_release_existing_input(operation: str) -> None:
    backend = FakeInputBackend()
    gate = MotorGate(backend)
    lease = gate.issue(session_id="test", target_instance="minecraft")
    gate.apply(lease.lease_id, MotorAction(sequence=0, keys_down=("w",)))
    releases = backend.release_count
    with pytest.raises(MotorRejected):
        if operation == "renew":
            gate.renew(lease.lease_id, ttl_ms=5001)
        else:
            gate.issue(
                session_id="test", target_instance="minecraft",
                ttl_ms=5001 if operation == "issue-ttl" else 750,
                max_action_duration_ms=1001 if operation == "issue-duration" else 250,
            )
    assert gate.lease == lease and backend.lease_id == lease.lease_id
    assert backend.held_keys == {"w"} and backend.release_count == releases


def test_successful_renew_keeps_held_input_and_sequence() -> None:
    backend = FakeInputBackend()
    gate = MotorGate(backend)
    lease = gate.issue(session_id="test", target_instance="minecraft")
    gate.apply(lease.lease_id, MotorAction(sequence=4, keys_down=("w",)))
    releases = backend.release_count
    renewed = gate.renew(lease.lease_id)
    assert renewed.lease_id == lease.lease_id and gate.revocation_reason == "active"
    assert backend.held_keys == {"w"} and backend.release_count == releases
    with pytest.raises(MotorRejected, match="monotonically"):
        gate.apply(lease.lease_id, MotorAction(sequence=4))


def test_input_release_failure_revokes_capability_and_preserves_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeInputBackend()
    gate = MotorGate(backend)
    lease = gate.issue(session_id="test", target_instance="minecraft")
    gate.apply(lease.lease_id, MotorAction(sequence=0, keys_down=("w",)))
    error = OSError("synthetic release failure")

    def release() -> None:
        raise error

    monkeypatch.setattr(backend, "release_all", release)
    with pytest.raises(OSError) as caught:
        gate.release_inputs(lease.lease_id)
    assert caught.value is error
    assert gate.lease is None and backend.lease_id is None
    assert gate.revocation_reason == "input-release-failed"
    # Revoked authority is not a claim that failed physical release succeeded.
    assert backend.held_keys == {"w"}


def test_action_model_rejects_unbounded_mouse_and_duration() -> None:
    with pytest.raises(ValueError):
        MotorAction(sequence=0, mouse_dx=5000)
    with pytest.raises(ValueError):
        MotorAction(sequence=0, duration_ms=1001)
    with pytest.raises(ValueError):
        MotorAction(sequence=0, camera_semantics="screen")


def test_absolute_cursor_requires_a_complete_gui_only_contract() -> None:
    action = MotorAction(
        sequence=0,
        cursor_x=0.25,
        cursor_y=0.75,
        camera_semantics="cursor",
        buttons_down=("right",),
        buttons_up=("right",),
    )
    assert action.action_kinds() == frozenset({"button", "mouse"})

    with pytest.raises(ValueError, match="supplied together"):
        MotorAction(sequence=0, cursor_x=0.25, camera_semantics="cursor")
    with pytest.raises(ValueError, match="cursor semantics"):
        MotorAction(sequence=0, cursor_x=0.25, cursor_y=0.75)
    with pytest.raises(ValueError, match="cannot be combined"):
        MotorAction(
            sequence=0,
            cursor_x=0.25,
            cursor_y=0.75,
            mouse_dx=1,
            camera_semantics="cursor",
        )
