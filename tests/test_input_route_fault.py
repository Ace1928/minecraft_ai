from __future__ import annotations

import json

import pytest

import minecraft_ai.cli as cli_module
import minecraft_ai.supervisor as supervisor_module
from minecraft_ai.safety import FakeInputBackend, InputRouteUnavailable, MotorAction, MotorRejected
from minecraft_ai.supervisor import Supervisor
from minecraft_ai.platforms.bedrock_x11 import IsolationError


class _RouteFaultBackend(FakeInputBackend):
    display_name = ":2"
    target_window_id = 42

    def apply(self, action: MotorAction) -> None:
        raise InputRouteUnavailable("Minecraft pointer routing could not be verified")


@pytest.mark.parametrize("calibration", [False, True])
def test_route_fault_survives_runtime_wrapping_and_retirement(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], calibration: bool
) -> None:
    supervisor = Supervisor()
    monkeypatch.setattr(supervisor, "_owns_control_file", lambda: True)
    supervisor.start()
    supervisor.replace_backend(_RouteFaultBackend(live_capable=True))

    with pytest.raises(InputRouteUnavailable):
        if calibration:
            supervisor.calibrate_world_camera(pitch_counts_per_degree=1, calibration_id="test")
        else:
            lease = supervisor.arm("fake-instance")
            supervisor.activate()
            supervisor.apply_motor_action(
                lease["lease_id"], MotorAction(sequence=0, mouse_dy=1).model_dump()
            )

    assert supervisor.motor.lease is None
    assert supervisor.status()["fault_code"] == "input-route-unverified"
    supervisor.fail("agent-runtime:RuntimeError:wrapped transport failure")
    supervisor.disarm()
    supervisor.stop()
    persisted = json.loads(supervisor_module.STATUS_FILE.read_text())
    assert persisted["state"] == "STOPPED"
    assert persisted["fault_code"] == "input-route-unverified"

    monkeypatch.setattr(cli_module, "supervisor_alive", lambda: False)
    monkeypatch.setattr(cli_module, "_agent_payload", lambda: {"alive": False})
    cli_module.status()
    status = json.loads(capsys.readouterr().out)
    assert status["fault_code"] == "input-route-unverified"
    assert status["supervisor_reachable"] is False
    assert Supervisor().status()["fault_code"] is None


def test_generic_rejection_is_not_reclassified_as_input_route_failure() -> None:
    class _OtherFailure(FakeInputBackend):
        def apply(self, action: MotorAction) -> None:
            # Even identical text cannot manufacture a typed route failure.
            raise MotorRejected("Minecraft pointer routing could not be verified")

    supervisor = Supervisor()
    supervisor.start()
    supervisor.replace_backend(_OtherFailure())
    lease = supervisor.arm("fake-instance")
    supervisor.activate()
    with pytest.raises(MotorRejected):
        supervisor.apply_motor_action(lease["lease_id"], MotorAction(sequence=0).model_dump())
    assert supervisor.status()["fault_code"] is None


def test_live_legacy_session_reports_isolation_failure_without_changing_liveness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = object()
    monkeypatch.setattr(cli_module.BedrockSession, "load", lambda: session)
    monkeypatch.setattr(cli_module, "bedrock_session_alive", lambda _session: True)
    monkeypatch.setattr(cli_module, "bedrock_session_resources_absent", lambda _session: False)

    def unverified(_session: object) -> None:
        assert _session is session
        raise IsolationError("host seat remains reachable")

    monkeypatch.setattr(cli_module, "require_autonomous_input_isolation", unverified)
    result = cli_module._input_isolation_status()
    assert result == {"verified": False, "reason": "host seat remains reachable"}


def test_missing_session_does_not_latch_existing_game_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing() -> None:
        raise FileNotFoundError()

    monkeypatch.setattr(cli_module.BedrockSession, "load", missing)
    assert cli_module._input_isolation_status()["verified"] is None


def test_calibration_route_fault_is_contained_before_fallible_status_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = Supervisor()
    supervisor.start()
    supervisor.replace_backend(_RouteFaultBackend(live_capable=True))

    def reject_fault_publication() -> None:
        if supervisor.fault_code is not None:
            raise OSError("simulated status disk failure")

    monkeypatch.setattr(supervisor, "_persist_status", reject_fault_publication)

    with pytest.raises(OSError, match="status disk failure"):
        supervisor.calibrate_world_camera(pitch_counts_per_degree=1, calibration_id="test")

    assert supervisor.state.value == "FAILSAFE"
    assert supervisor.motor.lease is None
    assert supervisor.fault_code == "input-route-unverified"
