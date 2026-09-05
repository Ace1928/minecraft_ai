"""Guarded agent-only recovery never grants fresh operator permission."""

from contextlib import contextmanager

import pytest

import minecraft_ai.emergency as emergency_module
import minecraft_ai.supervisor as supervisor_module
from minecraft_ai.safety import SupervisorState
from minecraft_ai.supervisor import ControlEndpoint, Supervisor


class _Connection:
    def __init__(self, payload):
        self.payload = payload
        self.responses = []
        self.closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def reload_supervisor(monkeypatch):
    supervisor = Supervisor()
    supervisor.start()
    supervisor.pause()
    supervisor.world_camera_pitch_units = 64
    supervisor.world_camera_updates = 19
    supervisor.world_camera_origin_calibrated = True
    supervisor.world_camera_calibration_id = "retained-camera-profile"
    supervisor._endpoint = ControlEndpoint(
        host="127.0.0.1",
        port=1,
        token="reload-test-secret",
        pid=1,
        session_id=supervisor.session_id,
    )
    forbidden_calls = []

    def forbid(name):
        def called(*_args, **_kwargs):
            forbidden_calls.append(name)
            raise RuntimeError(f"forbidden reload side effect: {name}")

        return called

    monkeypatch.setattr(supervisor, "resume", forbid("resume"))
    monkeypatch.setattr(supervisor, "stop", forbid("stop"))
    monkeypatch.setattr(supervisor_module, "clear_operator_pause", forbid("clear_pause"))
    monkeypatch.setattr(supervisor_module, "stop_agent_process", forbid("stop_agent"))
    monkeypatch.setattr(supervisor_module, "_recv_json_line", lambda connection: connection.payload)
    monkeypatch.setattr(
        supervisor_module,
        "_send_json_line",
        lambda connection, response: connection.responses.append(response),
    )

    def request(**overrides):
        connection = _Connection(
            {
                "token": "reload-test-secret",
                "command": "resume-for-agent-reload",
                "session_id": supervisor.session_id,
                **overrides,
            }
        )
        supervisor._handle_connection(connection)
        assert connection.closed
        assert len(connection.responses) == 1
        assert forbidden_calls == []
        return connection.responses[0]

    return supervisor, request


def test_guarded_reload_resumes_only_unarmed_paused_generation(reload_supervisor):
    supervisor, request = reload_supervisor
    before = supervisor.status()
    backend = supervisor.backend
    release_count = backend.release_count

    response = request()

    assert response["ok"] is True
    result = response["result"]
    assert result["state"] == "SAFE_IDLE"
    assert result["session_id"] == before["session_id"]
    assert result["agent_reload_resume_supported"] is True
    assert result["world_camera"] == before["world_camera"]
    assert result["motor_lease_active"] is False
    assert supervisor.motor.lease is None
    assert supervisor.backend is backend and backend.release_count == release_count
    assert not supervisor._stop.is_set()


@pytest.mark.parametrize("marker_kind", ["pause", "emergency"])
def test_guarded_reload_refuses_and_preserves_durable_intent(reload_supervisor, marker_kind):
    supervisor, request = reload_supervisor
    marker = (
        supervisor_module.OPERATOR_PAUSE_FILE
        if marker_kind == "pause"
        else emergency_module.EMERGENCY_STOP_FILE
    )
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_bytes(b"original operator intent")

    response = request(persistent_intent=False, clear_pause=True)

    assert response["ok"] is False
    assert "is latched" in response["error"]
    assert supervisor.state == SupervisorState.PAUSED
    assert marker.read_bytes() == b"original operator intent"
    assert not supervisor._stop.is_set()


@pytest.mark.parametrize(
    "state",
    [
        SupervisorState.FAILSAFE,
        SupervisorState.STOPPED,
        SupervisorState.STOPPING,
        SupervisorState.RUNNING,
        SupervisorState.ARMED,
        SupervisorState.SAFE_IDLE,
    ],
)
def test_guarded_reload_never_recovers_or_stops_another_state(reload_supervisor, state):
    supervisor, request = reload_supervisor
    supervisor.state = state

    response = request()

    assert response["ok"] is False
    assert "cannot resume agent reload" in response["error"]
    assert supervisor.state == state
    assert not supervisor._stop.is_set()


@pytest.mark.parametrize("session_id", [None, "", "other-generation", True, 1])
def test_guarded_reload_requires_exact_session_identity(reload_supervisor, session_id):
    supervisor, request = reload_supervisor

    response = request(session_id=session_id)

    assert response["ok"] is False
    assert "session mismatch" in response["error"]
    assert supervisor.state == SupervisorState.PAUSED


def test_guarded_reload_session_identity_does_not_bypass_ipc_authentication(reload_supervisor):
    supervisor, request = reload_supervisor

    response = request(token="incorrect-token")

    assert response == {"ok": False, "error": "unauthorized"}
    assert supervisor.state == SupervisorState.PAUSED


def test_guarded_reload_refuses_existing_lease_without_revoking_it(reload_supervisor):
    supervisor, request = reload_supervisor
    lease = supervisor.motor.issue(session_id=supervisor.session_id, target_instance="fake-target")

    response = request()

    assert response["ok"] is False
    assert "revoked motor lease" in response["error"]
    assert supervisor.state == SupervisorState.PAUSED
    assert supervisor.motor.lease is lease


@pytest.mark.parametrize("descriptor_kind", ["malformed-file", "dangling-symlink"])
def test_guarded_reload_does_not_delete_or_ignore_unretired_descriptor(
    reload_supervisor,
    descriptor_kind,
):
    supervisor, request = reload_supervisor
    descriptor = supervisor_module.AGENT_FILE
    descriptor.parent.mkdir(parents=True, exist_ok=True)
    if descriptor_kind == "malformed-file":
        descriptor.write_bytes(b"unverified agent descriptor")
    else:
        try:
            descriptor.symlink_to(descriptor.with_name("absent-descriptor"))
        except OSError:
            pytest.skip("symlink creation is unavailable")

    response = request()

    assert response["ok"] is False
    assert "confirmed agent retirement" in response["error"]
    assert supervisor.state == SupervisorState.PAUSED
    if descriptor_kind == "malformed-file":
        assert descriptor.read_bytes() == b"unverified agent descriptor"
    else:
        assert descriptor.is_symlink()


@pytest.mark.parametrize("marker_kind", ["pause", "emergency"])
def test_guarded_reload_rechecks_intent_after_acquiring_transaction_lock(
    reload_supervisor,
    monkeypatch,
    marker_kind,
):
    supervisor, request = reload_supervisor
    original_lock = supervisor_module.operator_intent_lock

    @contextmanager
    def raced_intent_lock():
        with original_lock():
            marker = (
                supervisor_module.OPERATOR_PAUSE_FILE
                if marker_kind == "pause"
                else emergency_module.EMERGENCY_STOP_FILE
            )
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_bytes(b"intent raced preflight")
            yield

    monkeypatch.setattr(supervisor_module, "operator_intent_lock", raced_intent_lock)
    assert not supervisor_module.operator_pause_latched()
    assert not supervisor_module.emergency_stop_latched()

    response = request()

    assert response["ok"] is False
    assert "is latched" in response["error"]
    assert supervisor.state == SupervisorState.PAUSED
    assert not supervisor._stop.is_set()
