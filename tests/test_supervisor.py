from __future__ import annotations

from minecraft_ai.safety import MotorAction, MotorLease, SupervisorState
from minecraft_ai.supervisor import Supervisor


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

    def bind_lease(self, lease: MotorLease) -> None:
        self.lease_id = lease.lease_id

    def apply(self, action: MotorAction) -> None:
        if self.lease_id is None:
            raise RuntimeError("backend has no lease")

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
    }


def test_same_physical_backend_replacement_preserves_camera_state() -> None:
    supervisor = Supervisor()
    supervisor.start()
    first = _PhysicalFakeBackend(":2", 42)
    supervisor.replace_backend(first)
    supervisor.world_camera_pitch_units = -317
    supervisor.world_camera_updates = 9

    supervisor.replace_backend(
        _PhysicalFakeBackend(":2", 42),
        preserve_world_camera=True,
    )

    assert supervisor.status()["world_camera"] == {
        "estimated_pitch_units": -317,
        "accepted_updates": 9,
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
