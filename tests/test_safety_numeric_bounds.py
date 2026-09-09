from __future__ import annotations

from typing import cast

import pytest

from minecraft_ai.safety import FakeInputBackend, MotorAction, MotorGate, MotorRejected


@pytest.mark.parametrize("operation", ["issue", "renew", "refresh"])
@pytest.mark.parametrize(
    "invalid",
    [49, 5001, True, False, 750.0, float("nan"), float("inf"), -float("inf"), "750"],
)
def test_invalid_ttl_cannot_replace_or_refresh_a_valid_lease(
    operation: str, invalid: object,
) -> None:
    backend = FakeInputBackend()
    gate = MotorGate(backend)
    lease = gate.issue(session_id="test", target_instance="minecraft")
    gate.apply(lease.lease_id, MotorAction(sequence=0, keys_down=("w",)))
    releases = backend.release_count
    actions = list(backend.actions)

    # Deliberately exercise callers that do not honour Python type annotations.
    ttl = cast(int, invalid)
    with pytest.raises(MotorRejected):
        if operation == "issue":
            gate.issue(session_id="test", target_instance="minecraft", ttl_ms=ttl)
        elif operation == "renew":
            gate.renew(lease.lease_id, ttl_ms=ttl)
        else:
            gate.apply(
                lease.lease_id,
                MotorAction(sequence=1, keys_down=("a",)),
                accepted_action_ttl_ms=ttl,
            )

    assert gate.lease == lease
    assert backend.lease_id == lease.lease_id
    assert backend.held_keys == {"w"}
    assert backend.release_count == releases
    assert backend.actions == actions
    # Rejection must not replace the original deadline with NaN or infinity.
    assert gate.check_expiry(lease.expires_monotonic_ns)
    assert not backend.held_keys


@pytest.mark.parametrize("operation", ["issue", "renew"])
def test_none_is_not_a_lease_ttl(operation: str) -> None:
    gate = MotorGate(FakeInputBackend())
    lease = gate.issue(session_id="test", target_instance="minecraft")
    with pytest.raises(MotorRejected):
        if operation == "issue":
            gate.issue(
                session_id="test", target_instance="minecraft", ttl_ms=cast(int, None),
            )
        else:
            gate.renew(lease.lease_id, ttl_ms=cast(int, None))
    assert gate.lease == lease


@pytest.mark.parametrize(
    "invalid",
    [0, 1001, True, False, 250.0, float("nan"), float("inf"), -float("inf"), "250", None],
)
def test_invalid_duration_cannot_weaken_or_replace_existing_authority(invalid: object) -> None:
    backend = FakeInputBackend()
    gate = MotorGate(backend)
    lease = gate.issue(session_id="test", target_instance="minecraft")
    gate.apply(lease.lease_id, MotorAction(sequence=0, keys_down=("w",)))
    releases = backend.release_count
    with pytest.raises(MotorRejected):
        gate.issue(
            session_id="test",
            target_instance="minecraft",
            max_action_duration_ms=cast(int, invalid),
        )
    assert gate.lease == lease
    assert backend.lease_id == lease.lease_id
    assert backend.held_keys == {"w"}
    assert backend.release_count == releases


@pytest.mark.parametrize(
    "invalid",
    [-1, True, False, 0.0, float("nan"), float("inf"), -float("inf"), "0", None],
)
def test_invalid_first_sequence_is_rejected_before_backend_mutation(invalid: object) -> None:
    backend = FakeInputBackend()
    gate = MotorGate(backend)
    lease = gate.issue(session_id="test", target_instance="minecraft")
    gate.apply(lease.lease_id, MotorAction(sequence=0, keys_down=("w",)))
    releases = backend.release_count
    with pytest.raises(MotorRejected):
        gate.issue(
            session_id="test", target_instance="minecraft", first_sequence=cast(int, invalid),
        )
    assert gate.lease == lease
    assert backend.lease_id == lease.lease_id
    assert backend.held_keys == {"w"}
    assert backend.release_count == releases


@pytest.mark.parametrize("ttl_ms", [50, 5000])
@pytest.mark.parametrize("duration_ms", [1, 1000])
@pytest.mark.parametrize("first_sequence", [0, 7])
def test_integer_boundaries_retain_expiry_duration_and_replay_checks(
    ttl_ms: int, duration_ms: int, first_sequence: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [1_000_000_000]
    monkeypatch.setattr("minecraft_ai.safety.time.monotonic_ns", lambda: clock[0])
    backend = FakeInputBackend()
    gate = MotorGate(backend)
    lease = gate.issue(
        session_id="test", target_instance="minecraft", ttl_ms=ttl_ms,
        max_action_duration_ms=duration_ms, first_sequence=first_sequence,
    )
    assert type(lease.expires_monotonic_ns) is int
    assert lease.expires_monotonic_ns == clock[0] + ttl_ms * 1_000_000
    gate.apply(
        lease.lease_id,
        MotorAction(sequence=first_sequence, keys_down=("w",), duration_ms=duration_ms),
    )
    renewed = gate.renew(lease.lease_id, ttl_ms=ttl_ms)
    assert renewed.max_action_duration_ms == duration_ms
    assert renewed.first_sequence == first_sequence
    with pytest.raises(MotorRejected, match="monotonically"):
        gate.apply(lease.lease_id, MotorAction(sequence=first_sequence))
    assert not backend.held_keys


def test_none_refresh_sentinel_keeps_the_original_deadline() -> None:
    gate = MotorGate(FakeInputBackend())
    lease = gate.issue(session_id="test", target_instance="minecraft")
    gate.apply(lease.lease_id, MotorAction(sequence=0), accepted_action_ttl_ms=None)
    assert gate.lease == lease
