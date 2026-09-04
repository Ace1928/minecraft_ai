from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

import minecraft_ai.supervisor as supervisor_module
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

    supervisor.apply_motor_action(
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
    supervisor.apply_motor_action(
        lease_id,
        MotorAction(sequence=2, mouse_dy=7).model_dump(mode="json"),
    )

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
