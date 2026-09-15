"""Comprehensive unit tests for Bounded Paid Directions Gateway."""

from __future__ import annotations

import time
import pytest

from minecraft_ai.directions import (
    CapacityExceededError,
    ConflictError,
    ControlUnavailableError,
    DirectionsGateway,
    DirectionsRequest,
    DirectionsState,
    InvalidInstructionError,
    MinecraftDirectionsAdapter,
)
from minecraft_ai.episodes import RuntimeEvent, RuntimeEventKind
from minecraft_ai.storage import StateDatabase


@pytest.fixture
def mock_supervisor():
    return {
        "state": "RUNNING",
        "live_capable": True,
        "motor_lease_active": True,
        "operator_pause_latched": False,
        "emergency_stop_latched": False,
        "session_id": "test-session-1234",
        "release_count": 5,
        "motor_target_instance": "bedrock:test:14680065",
        "agent": {"instance_id": "bedrock:test:14680065"},
    }


@pytest.fixture
def gateway(tmp_path, mock_supervisor):
    db_path = tmp_path / "state.sqlite3"
    return DirectionsGateway(
        db_path,
        supervisor_status_fn=lambda: mock_supervisor,
        queue_capacity=4,
        per_member_limit=2,
    )


def test_discovery_reports_correct_readiness_and_epoch(gateway, mock_supervisor):
    discovery = gateway.discover()
    assert discovery.available is True
    assert discovery.session_id == "test-session-1234"
    assert discovery.control_epoch == 5
    assert discovery.motor_lease_active is True
    assert discovery.queue_capacity == 4
    assert discovery.queue_depth == 0
    assert len(discovery.supported_instructions) >= 2


def test_discovery_unavailable_when_paused(gateway, mock_supervisor):
    mock_supervisor["state"] = "PAUSED"
    discovery = gateway.discover()
    assert discovery.available is False
    assert discovery.readiness_reason == "operator_paused"


def test_submit_creates_receipt_and_dispatches_operator_message(gateway):
    req = DirectionsRequest(
        request_id="req-001",
        member_id="member-alpha",
        server_id="bedrock:test:14680065",
        expected_session_id="test-session-1234",
        expected_epoch=5,
        instruction_id="open_observe_close_inventory",
    )
    receipt = gateway.submit(req)
    assert receipt.request_id == "req-001"
    assert receipt.state == DirectionsState.QUEUED
    assert receipt.replayed is False
    assert receipt.session_id == "test-session-1234"
    assert receipt.epoch == 5
    assert receipt.message_id is not None

    # Verify operator message was queued in state database
    with StateDatabase(gateway.db_path) as db:
        msgs = db.load_operator_messages(limit=10)
        assert len(msgs) == 1
        assert msgs[0].message_id == receipt.message_id
        assert msgs[0].text == "open inventory, observe it, close it; no movement or attack"


def test_idempotency_returns_replayed_receipt(gateway):
    req = DirectionsRequest(
        request_id="req-idem-1",
        member_id="member-alpha",
        server_id="bedrock:test:14680065",
        expected_session_id="test-session-1234",
        expected_epoch=5,
        instruction_id="open_observe_close_inventory",
    )
    receipt1 = gateway.submit(req)
    assert receipt1.replayed is False

    # Second submit with identical payload
    receipt2 = gateway.submit(req)
    assert receipt2.replayed is True
    assert receipt2.request_id == receipt1.request_id
    assert receipt2.message_id == receipt1.message_id


def test_conflicting_payload_raises_conflict_error(gateway):
    req1 = DirectionsRequest(
        request_id="req-conflict-1",
        member_id="member-alpha",
        server_id="bedrock:test:14680065",
        expected_session_id="test-session-1234",
        expected_epoch=5,
        instruction_id="open_observe_close_inventory",
    )
    gateway.submit(req1)

    req2 = DirectionsRequest(
        request_id="req-conflict-1",
        member_id="member-alpha",
        server_id="bedrock:test:14680065",
        expected_session_id="test-session-1234",
        expected_epoch=5,
        instruction_id="explore_forward",  # Changed instruction!
    )
    with pytest.raises(ConflictError):
        gateway.submit(req2)


def test_stale_epoch_or_session_rejected(gateway):
    req = DirectionsRequest(
        request_id="req-stale",
        member_id="member-alpha",
        server_id="bedrock:test:14680065",
        expected_session_id="old-session",
        expected_epoch=5,
        instruction_id="open_observe_close_inventory",
    )
    with pytest.raises(ControlUnavailableError, match="Session mismatch"):
        gateway.submit(req)

    req_epoch = DirectionsRequest(
        request_id="req-stale-epoch",
        member_id="member-alpha",
        server_id="bedrock:test:14680065",
        expected_session_id="test-session-1234",
        expected_epoch=4,  # Live is 5
        instruction_id="open_observe_close_inventory",
    )
    with pytest.raises(ControlUnavailableError, match="Control epoch mismatch"):
        gateway.submit(req_epoch)


def test_unsupported_instruction_rejected(gateway):
    req = DirectionsRequest(
        request_id="req-bad-inst",
        member_id="member-alpha",
        server_id="bedrock:test:14680065",
        expected_session_id="test-session-1234",
        expected_epoch=5,
        instruction_id="fly_to_the_moon",
    )
    with pytest.raises(InvalidInstructionError):
        gateway.submit(req)


def test_per_member_quota_enforced(gateway):
    # Member limit is 2
    for i in range(2):
        gateway.submit(
            DirectionsRequest(
                request_id=f"member-limit-{i}",
                member_id="member-spam",
                server_id="bedrock:test:14680065",
                expected_session_id="test-session-1234",
                expected_epoch=5,
                instruction_id="open_observe_close_inventory",
            )
        )

    # Third should fail with CapacityExceededError
    with pytest.raises(CapacityExceededError, match="Member active direction limit"):
        gateway.submit(
            DirectionsRequest(
                request_id="member-limit-overflow",
                member_id="member-spam",
                server_id="bedrock:test:14680065",
                expected_session_id="test-session-1234",
                expected_epoch=5,
                instruction_id="open_observe_close_inventory",
            )
        )


def test_cancel_direction(gateway):
    req = DirectionsRequest(
        request_id="req-cancel-1",
        member_id="member-alpha",
        server_id="bedrock:test:14680065",
        expected_session_id="test-session-1234",
        expected_epoch=5,
        instruction_id="open_observe_close_inventory",
    )
    gateway.submit(req)
    receipt = gateway.cancel("req-cancel-1", member_id="member-alpha")
    assert receipt.state == DirectionsState.CANCELLED
    assert receipt.reason_code == "cancelled_by_user"


def test_outcome_verification_success(gateway):
    req = DirectionsRequest(
        request_id="req-verify-success",
        member_id="member-alpha",
        server_id="bedrock:test:14680065",
        expected_session_id="test-session-1234",
        expected_epoch=5,
        instruction_id="open_observe_close_inventory",
    )
    receipt = gateway.submit(req)
    msg_id = receipt.message_id
    now = time.time_ns()

    # Simulate runtime recording successful events
    with StateDatabase(gateway.db_path) as db:
        db.connection.execute(
            "INSERT INTO trajectories(trajectory_id, started_ns, source_type, game_version, payload) VALUES ('traj-1', 1, 'agent', '1.26', '{}')"
        )
        db.connection.commit()
        # Event 1: open_inventory succeeded
        db.save_runtime_event(
            RuntimeEvent(
                event_id="evt-open-1",
                kind=RuntimeEventKind.SKILL_SUCCEEDED,
                observed_ns=now + 100,
                trajectory_id="traj-1",
                payload={
                    "run_id": "run-open",
                    "skill_id": "open_inventory",
                    "context_key": f"operator:{msg_id}",
                    "outcome": "succeeded",
                    "duration_ms": 500.0,
                    "parameters_json": '{"allow_attack":false}',
                },
            )
        )
        # Event 2: close_open_inventory succeeded
        db.save_runtime_event(
            RuntimeEvent(
                event_id="evt-close-1",
                kind=RuntimeEventKind.SKILL_SUCCEEDED,
                observed_ns=now + 200,
                trajectory_id="traj-1",
                payload={
                    "run_id": "run-close",
                    "skill_id": "close_open_inventory",
                    "context_key": "scene-recovery",
                    "outcome": "succeeded",
                    "duration_ms": 300.0,
                    "parameters_json": "{}",
                },
            )
        )

    # Check status -> gateway should verify events and transition to SUCCEEDED
    updated = gateway.status("req-verify-success")
    assert updated.state == DirectionsState.SUCCEEDED
    assert updated.outcome is not None
    assert updated.outcome.success is True
    assert updated.outcome.duration_ms == 800.0
    assert len(updated.outcome.observed_steps) == 2


def test_adapter_discovery_and_validation(gateway):
    adapter = MinecraftDirectionsAdapter(gateway.db_path, gateway=gateway)
    disc = adapter.discover()
    assert disc.available is True

    # Validate catalog text
    inst_id = adapter.validate("open inventory, observe it, close it; no movement or attack")
    assert inst_id == "open_observe_close_inventory"

    with pytest.raises(ValueError, match="does not match"):
        adapter.validate("fly around and make diamonds")
