from __future__ import annotations

import threading
import time
import json
from pathlib import Path

import pytest

import minecraft_ai.supervisor as supervisor_module
from minecraft_ai.emergency import engage_emergency_stop
from minecraft_ai.safety import MotorAction, MotorLease, SupervisorState
from minecraft_ai.supervisor import (
    ControlEndpoint,
    Supervisor,
    _bounded_camera_calibration_deltas,
)


class _PhysicalFakeBackend:
    backend_id = "physical-fake"
    live_capable = True

    def __init__(self, display_name: str, target_window_id: int) -> None:
        self.display_name = display_name
        self.target_window_id = target_window_id
        self.held_keys: set[str] = set()
        self.held_buttons: set[str] = set()
        self.release_count = 0
        self.lease_id: str | None = None
        self.actions: list[MotorAction] = []

    def bind_lease(self, lease: MotorLease) -> None:
        self.lease_id = lease.lease_id

    def apply(self, action: MotorAction) -> None:
        if self.lease_id is None:
            raise RuntimeError("backend has no lease")
        self.actions.append(action)

    def release_all(self) -> None:
        self.held_keys.clear()
        self.held_buttons.clear()
        self.release_count += 1

    def clear_lease(self) -> None:
        self.lease_id = None


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


@pytest.mark.parametrize("contents", ["{}", '{"paused":false}', "not-json", ""])
def test_any_existing_operator_pause_marker_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contents: str,
) -> None:
    marker = tmp_path / "OPERATOR_PAUSE"
    monkeypatch.setattr(supervisor_module, "OPERATOR_PAUSE_FILE", marker)
    marker.write_text(contents, encoding="utf-8")

    assert supervisor_module.operator_pause_latched() is True
    supervisor_module.clear_operator_pause()
    assert supervisor_module.operator_pause_latched() is False


def test_camera_calibration_stops_on_operator_pause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "OPERATOR_PAUSE"
    monkeypatch.setattr(supervisor_module, "OPERATOR_PAUSE_FILE", marker)
    monkeypatch.setattr("minecraft_ai.supervisor.time.sleep", lambda _seconds: None)

    class _PauseAfterFirstMotion(_PhysicalFakeBackend):
        def apply(self, action: MotorAction) -> None:
            super().apply(action)
            if len(self.actions) == 1:
                supervisor_module.latch_operator_pause()

    supervisor = Supervisor()
    supervisor.start()
    backend = _PauseAfterFirstMotion(":2", 42)
    supervisor.replace_backend(backend)

    with pytest.raises(RuntimeError, match="interlock"):
        supervisor.calibrate_world_camera(
            pitch_counts_per_degree=47.9638888889,
            calibration_id="calibration-sha256",
        )

    assert len(backend.actions) == 1
    assert supervisor.motor.lease is None
    assert supervisor.motor.revocation_reason == "operator-pause"


def test_renew_observes_durable_pause_and_releases_held_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "OPERATOR_PAUSE"
    monkeypatch.setattr(supervisor_module, "OPERATOR_PAUSE_FILE", marker)
    supervisor = Supervisor()
    supervisor.start()
    lease = supervisor.arm("fake-instance")
    lease_id = str(lease["lease_id"])
    supervisor.activate()
    supervisor.motor.apply(lease_id, MotorAction(sequence=0, keys_down=("w",)))
    supervisor_module.latch_operator_pause()

    with pytest.raises(RuntimeError, match="operator pause is latched"):
        supervisor.renew(lease_id)

    assert supervisor.state == SupervisorState.PAUSED
    assert supervisor.motor.lease is None
    assert supervisor.backend.held_keys == set()


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


def test_supervisor_tracks_only_accepted_world_camera_motion() -> None:
    supervisor = Supervisor()
    supervisor.start()
    lease_info = supervisor.arm("fake-instance")
    lease_id = str(lease_info["lease_id"])
    supervisor.activate()

    first = supervisor.apply_motor_action(
        lease_id,
        MotorAction(sequence=0, mouse_dy=-30).model_dump(mode="json"),
    )
    supervisor.apply_motor_action(
        lease_id,
        MotorAction(
            sequence=1,
            mouse_dy=20,
            camera_semantics="cursor",
        ).model_dump(mode="json"),
    )
    last = supervisor.apply_motor_action(
        lease_id,
        MotorAction(sequence=2, mouse_dy=7).model_dump(mode="json"),
    )

    assert first["world_camera"] == {
        "estimated_pitch_units": -30,
        "accepted_updates": 1,
        "origin_calibrated": False,
        "pitch_counts_per_degree": None,
        "calibration_id": None,
    }
    assert last["world_camera"] == {
        "estimated_pitch_units": -23,
        "accepted_updates": 2,
        "origin_calibrated": False,
        "pitch_counts_per_degree": None,
        "calibration_id": None,
    }

    assert supervisor.status()["world_camera"] == {
        "estimated_pitch_units": -23,
        "accepted_updates": 2,
        "origin_calibrated": False,
        "pitch_counts_per_degree": None,
        "calibration_id": None,
    }


def test_accepted_runtime_motor_action_refreshes_same_lease_to_safety_cap() -> None:
    supervisor = Supervisor()
    supervisor.start()
    lease_info = supervisor.arm("fake-instance")
    lease_id = str(lease_info["lease_id"])
    original = supervisor.motor.lease
    assert original is not None
    supervisor.activate()

    supervisor.apply_motor_action(
        lease_id,
        MotorAction(sequence=0, keys_down=("w",)).model_dump(mode="json"),
    )

    refreshed = supervisor.motor.lease
    assert refreshed is not None
    assert refreshed.lease_id == lease_id
    assert refreshed.expires_monotonic_ns > original.expires_monotonic_ns
    assert supervisor.state == SupervisorState.RUNNING


def test_same_physical_backend_replacement_preserves_camera_state() -> None:
    supervisor = Supervisor()
    supervisor.start()
    first = _PhysicalFakeBackend(":2", 42)
    supervisor.replace_backend(first)
    supervisor.world_camera_pitch_units = -317
    supervisor.world_camera_updates = 9
    supervisor.world_camera_origin_calibrated = True
    supervisor.world_camera_pitch_counts_per_degree = 47.96
    supervisor.world_camera_calibration_id = "profile"

    supervisor.replace_backend(
        _PhysicalFakeBackend(":2", 42),
        preserve_world_camera=True,
    )

    assert supervisor.status()["world_camera"] == {
        "estimated_pitch_units": -317,
        "accepted_updates": 9,
        "origin_calibrated": True,
        "pitch_counts_per_degree": 47.96,
        "calibration_id": "profile",
    }


def test_new_physical_backend_replacement_resets_camera_state() -> None:
    supervisor = Supervisor()
    supervisor.start()
    supervisor.replace_backend(_PhysicalFakeBackend(":2", 42))
    supervisor.world_camera_pitch_units = 281
    supervisor.world_camera_updates = 5

    supervisor.replace_backend(_PhysicalFakeBackend(":3", 43))

    assert supervisor.status()["world_camera"] == {
        "estimated_pitch_units": 0,
        "accepted_updates": 0,
        "origin_calibrated": False,
        "pitch_counts_per_degree": None,
        "calibration_id": None,
    }


def test_camera_calibration_homes_then_establishes_measured_horizon(monkeypatch) -> None:
    monkeypatch.setattr("minecraft_ai.supervisor.time.sleep", lambda _seconds: None)
    deltas = _bounded_camera_calibration_deltas(47.9638888889)

    assert all(0 < abs(delta) <= 96 for delta in deltas)
    assert sum(delta for delta in deltas if delta < 0) == -9593
    assert sum(delta for delta in deltas if delta > 0) == 4317
    assert deltas[0] < 0
    assert deltas[-1] > 0

    supervisor = Supervisor()
    supervisor.start()
    backend = _PhysicalFakeBackend(":2", 42)
    supervisor.replace_backend(backend)

    status = supervisor.calibrate_world_camera(
        pitch_counts_per_degree=47.9638888889,
        calibration_id="calibration-sha256",
    )

    assert sum(action.mouse_dy for action in backend.actions) == -5276
    assert status["state"] == "SAFE_IDLE"
    assert status["motor_lease_active"] is False
    assert status["world_camera"] == {
        "estimated_pitch_units": 0,
        "accepted_updates": 0,
        "origin_calibrated": True,
        "pitch_counts_per_degree": 47.9638888889,
        "calibration_id": "calibration-sha256",
    }


def _established_camera_supervisor(backend: _PhysicalFakeBackend) -> Supervisor:
    supervisor = Supervisor()
    supervisor.start()
    supervisor.replace_backend(backend)
    supervisor.world_camera_pitch_units = 117
    supervisor.world_camera_updates = 9
    supervisor.world_camera_origin_calibrated = True
    supervisor.world_camera_pitch_counts_per_degree = 47.96
    supervisor.world_camera_calibration_id = "previous-measured-profile"
    supervisor._persist_status()
    return supervisor


@pytest.mark.parametrize(
    "interruption",
    ["return-pause", "backend-failure", "settle-pause", "settle-stop", "cleanup-pause"],
)
def test_recalibration_interruption_invalidates_origin_but_preserves_measured_profile(
    monkeypatch: pytest.MonkeyPatch, interruption: str
) -> None:
    class _InterruptedBackend(_PhysicalFakeBackend):
        def apply(self, action: MotorAction) -> None:
            # Both in-memory and persisted status must become invalid before
            # any calibration input, not only after a failure is handled.
            assert not supervisor.world_camera_origin_calibrated
            persisted = json.loads(supervisor_module.STATUS_FILE.read_text())
            assert persisted["world_camera"]["origin_calibrated"] is False
            super().apply(action)
            if interruption == "return-pause" and action.mouse_dy > 0:
                supervisor_module.latch_operator_pause()
            if interruption == "backend-failure" and len(self.actions) == 2:
                raise RuntimeError("synthetic camera backend failure")

    backend = _InterruptedBackend(":2", 42)
    supervisor = _established_camera_supervisor(backend)
    # Conftest isolates every runtime path; opt this fake instance into writing
    # only that temporary status artifact without creating a control server.
    monkeypatch.setattr(supervisor, "_owns_control_file", lambda: True)
    settles = 0

    def sleep(seconds: float) -> None:
        nonlocal settles
        if seconds == 0.1:
            settles += 1
            if settles == 2 and interruption == "settle-pause":
                supervisor_module.latch_operator_pause()
            if settles == 2 and interruption == "settle-stop":
                engage_emergency_stop("test-final-settle")

    monkeypatch.setattr(supervisor_module.time, "sleep", sleep)
    original_revoke = supervisor.motor.revoke

    def revoke(reason: str) -> None:
        original_revoke(reason)
        if reason == "camera-calibration-complete" and interruption == "cleanup-pause":
            supervisor_module.latch_operator_pause()

    monkeypatch.setattr(supervisor.motor, "revoke", revoke)
    with pytest.raises(RuntimeError):
        supervisor.calibrate_world_camera(
            pitch_counts_per_degree=1.0, calibration_id="replacement-profile"
        )

    assert backend.actions
    assert supervisor.motor.lease is None
    assert supervisor.status()["world_camera"] == {
        "estimated_pitch_units": 117,
        "accepted_updates": 9,
        "origin_calibrated": False,
        "pitch_counts_per_degree": 47.96,
        "calibration_id": "previous-measured-profile",
    }
    persisted = json.loads(supervisor_module.STATUS_FILE.read_text())
    assert persisted["world_camera"]["origin_calibrated"] is False
    if interruption == "settle-stop":
        assert supervisor.state == SupervisorState.FAILSAFE
        assert supervisor.last_fault == "emergency-stop-latched"
    elif interruption.endswith("pause"):
        assert supervisor.motor.revocation_reason == "operator-pause"


@pytest.mark.parametrize(
    ("pitch", "identity"),
    [(0.0, "valid"), (101.0, "valid"), (float("nan"), "valid"), (1.0, ""), (1.0, "x" * 129)],
)
def test_invalid_recalibration_request_preserves_established_origin(pitch, identity) -> None:
    backend = _PhysicalFakeBackend(":2", 42)
    supervisor = _established_camera_supervisor(backend)
    before = supervisor.status()["world_camera"]

    with pytest.raises(ValueError):
        supervisor.calibrate_world_camera(pitch_counts_per_degree=pitch, calibration_id=identity)

    assert not backend.actions
    assert supervisor.motor.lease is None
    assert supervisor.status()["world_camera"] == before


def test_recalibration_blocked_by_existing_interlock_preserves_established_origin() -> None:
    backend = _PhysicalFakeBackend(":2", 42)
    supervisor = _established_camera_supervisor(backend)
    before = supervisor.status()["world_camera"]
    releases_before = backend.release_count
    supervisor_module.latch_operator_pause()

    with pytest.raises(RuntimeError):
        supervisor.calibrate_world_camera(pitch_counts_per_degree=1.0, calibration_id="new")

    assert not backend.actions
    assert backend.release_count == releases_before
    assert supervisor.motor.lease is None
    assert supervisor.status()["world_camera"] == before


@pytest.mark.parametrize("issue_fails", [False, True])
def test_recalibration_invalidates_before_backend_release_during_lease_issue(
    monkeypatch: pytest.MonkeyPatch, issue_fails: bool
) -> None:
    backend = _PhysicalFakeBackend(":2", 42)
    supervisor = _established_camera_supervisor(backend)
    before = supervisor.status()["world_camera"]
    monkeypatch.setattr(supervisor, "_owns_control_file", lambda: True)
    monkeypatch.setattr(supervisor_module.time, "sleep", lambda _seconds: None)
    original_release = backend.release_all
    releases = 0

    def release() -> None:
        nonlocal releases
        releases += 1
        assert not supervisor.world_camera_origin_calibrated
        assert supervisor.world_camera_pitch_counts_per_degree == 47.96
        assert supervisor.world_camera_calibration_id == "previous-measured-profile"
        persisted = json.loads(supervisor_module.STATUS_FILE.read_text())
        assert persisted["world_camera"]["origin_calibrated"] is False
        original_release()
        if issue_fails and releases == 1:
            raise RuntimeError("synthetic issue-time backend release failure")

    monkeypatch.setattr(backend, "release_all", release)
    if issue_fails:
        with pytest.raises(RuntimeError, match="issue-time"):
            supervisor.calibrate_world_camera(pitch_counts_per_degree=1.0, calibration_id="new")
        assert not backend.actions
        assert supervisor.status()["world_camera"] == {**before, "origin_calibrated": False}
    else:
        supervisor.calibrate_world_camera(pitch_counts_per_degree=1.0, calibration_id="new")
        assert backend.actions
        assert supervisor.world_camera_origin_calibrated
    assert releases >= 1
    assert supervisor.motor.lease is None


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


def test_resume_retires_failsafe_generation_for_clean_recovery() -> None:
    supervisor = Supervisor()
    supervisor.start()
    supervisor.fail("controller-crashed")

    supervisor.resume()

    assert supervisor.state == SupervisorState.STOPPED
    assert supervisor.motor.lease is None
    assert supervisor._stop.is_set()


@pytest.mark.parametrize(
    ("later_command", "expected_state"),
    [("pause", SupervisorState.PAUSED), ("stop", SupervisorState.STOPPED)],
)
def test_resume_cannot_clear_later_durable_stop_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    later_command: str,
    expected_state: SupervisorState,
) -> None:
    marker = tmp_path / "OPERATOR_PAUSE"
    monkeypatch.setattr(supervisor_module, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(supervisor_module, "OPERATOR_PAUSE_FILE", marker)
    supervisor = Supervisor()
    supervisor.start()
    supervisor_module.latch_operator_pause()
    supervisor.pause()
    supervisor._endpoint = ControlEndpoint(
        host="127.0.0.1",
        port=1,
        token="secret",
        pid=1,
        session_id="race-test",
    )

    entered_clear = threading.Event()
    permit_clear = threading.Event()
    original_clear = supervisor_module.clear_operator_pause

    def blocking_clear() -> None:
        entered_clear.set()
        assert permit_clear.wait(timeout=2.0)
        original_clear()

    class _Connection:
        def __init__(self, command: str) -> None:
            self.payload = {
                "token": "secret",
                "command": command,
                "persistent_intent": True,
            }

        def close(self) -> None:
            return

    monkeypatch.setattr(supervisor_module, "clear_operator_pause", blocking_clear)
    monkeypatch.setattr(
        supervisor_module,
        "_recv_json_line",
        lambda connection: connection.payload,
    )
    monkeypatch.setattr(supervisor_module, "_send_json_line", lambda *_args: None)

    resume_thread = threading.Thread(
        target=supervisor._handle_connection,
        args=(_Connection("resume"),),
    )
    later_thread = threading.Thread(
        target=supervisor._handle_connection,
        args=(_Connection(later_command),),
    )
    resume_thread.start()
    assert entered_clear.wait(timeout=2.0)
    later_thread.start()
    time.sleep(0.05)
    assert later_thread.is_alive()
    permit_clear.set()
    resume_thread.join(timeout=2.0)
    later_thread.join(timeout=2.0)

    assert not resume_thread.is_alive()
    assert not later_thread.is_alive()
    assert supervisor.state == expected_state
    assert marker.exists()


@pytest.mark.parametrize(
    ("command", "expected_state"),
    [("pause", SupervisorState.PAUSED), ("stop", SupervisorState.STOPPED)],
)
def test_controlled_shutdown_allows_agent_cleanup_to_disarm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    expected_state: SupervisorState,
) -> None:
    marker = tmp_path / "OPERATOR_PAUSE"
    missing_agent = tmp_path / "agent-process.json"
    monkeypatch.setattr(supervisor_module, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(supervisor_module, "OPERATOR_PAUSE_FILE", marker)
    monkeypatch.setattr(supervisor_module, "AGENT_FILE", missing_agent)
    supervisor = Supervisor()
    supervisor.start()
    supervisor._endpoint = ControlEndpoint(
        host="127.0.0.1",
        port=1,
        token="secret",
        pid=1,
        session_id="graceful-stop-test",
    )

    cleanup_completed = threading.Event()

    def graceful_agent_stop(*, timeout_s: float) -> bool:
        assert timeout_s == supervisor_module.GRACEFUL_AGENT_STOP_TIMEOUT_S
        cleanup = threading.Thread(
            target=lambda: (supervisor.disarm("agent-cleanup"), cleanup_completed.set())
        )
        cleanup.start()
        cleanup.join(timeout=1.0)
        assert not cleanup.is_alive(), "supervisor lock blocked agent cleanup disarm"
        return True

    class _Connection:
        payload = {
            "token": "secret",
            "command": command,
            "persistent_intent": True,
        }

        def close(self) -> None:
            return

    responses: list[dict[str, object]] = []
    monkeypatch.setattr(supervisor_module, "stop_agent_process", graceful_agent_stop)
    monkeypatch.setattr(
        supervisor_module,
        "_recv_json_line",
        lambda connection: connection.payload,
    )
    monkeypatch.setattr(
        supervisor_module,
        "_send_json_line",
        lambda _connection, payload: responses.append(payload),
    )

    supervisor._handle_connection(_Connection())  # type: ignore[arg-type]

    assert cleanup_completed.is_set()
    assert supervisor.state == expected_state
    assert marker.exists()
    assert responses[-1]["ok"] is True
