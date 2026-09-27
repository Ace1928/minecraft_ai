from __future__ import annotations

import threading
import time
import json
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from minecraft_ai.builtin_skills import build_bootstrap_skill_library
from minecraft_ai.directions import (
    ControlUnavailableError,
    DirectionsGateway,
    DirectionsRequest,
    DirectionsState,
    InvalidInstructionError,
    MinecraftDirectionsAdapter,
)
from minecraft_ai.directions.runtime import tick_inventory_direction
from minecraft_ai.execution import SkillExecutor
from minecraft_ai.motor import BootstrapMotorPolicy
from minecraft_ai.perception import FrameState, PerceptionBlackboard, PerceptionFact
from minecraft_ai.perception_service import BEDROCK_HUD_SAFETY_SOURCE
from minecraft_ai.runtime import AgentRuntime
from minecraft_ai.runtime_support.helpers import _active_operator_messages, _terminal_run_event
from minecraft_ai.safety import MotorAction
from minecraft_ai.skills import SkillOutcome, SkillRun
from minecraft_ai.social import OperatorMessageStatus
from minecraft_ai.storage import StateDatabase


@pytest.fixture
def control():
    return dict(
        state="RUNNING",
        live_capable=True,
        motor_lease_active=True,
        session_id="test-session",
        release_count=5,
        motor_target_instance="test-world",
    )


@pytest.fixture
def gateway(tmp_path, control):
    return DirectionsGateway(tmp_path / "state.sqlite3", supervisor_status_fn=lambda: control)


def request(**changes):
    values = dict(
        request_id="test-request",
        member_id="local-operator-qualification",
        server_id="test-world",
        expected_session_id="test-session",
        expected_epoch=5,
        instruction_id="open_observe_close_inventory",
        deadline_s=25,
    )
    values.update(changes)
    return DirectionsRequest(**values)


def begun(gateway):
    receipt = gateway.submit_qualification(request())
    gateway.claim(receipt.request_id)
    params = gateway.start_step(
        receipt.request_id, receipt.attempt_id, "open_inventory", "open-run"
    )
    run = SkillRun(
        run_id="open-run",
        skill_id="open_inventory",
        started_ns=time.monotonic_ns(),
        context_key=f"operator:{receipt.message_id}",
        parameters=params,
    )
    return receipt, run


def frame(frame_id=1):
    return SimpleNamespace(frame_id=frame_id, captured_ns=time.monotonic_ns())


def toggle(sequence=1):
    return MotorAction(sequence=sequence, keys_down=("e",), keys_up=("e",), duration_ms=75)


def supervisor_acceptance(sequence=1):
    return {
        "accepted_sequence": sequence,
        "accepted_monotonic_ns": time.monotonic_ns(),
        "lease_active": True,
    }


def test_private_qualification_does_not_enable_paid_or_free_form_admission(gateway):
    assert gateway.discover().available is False
    with pytest.raises(ControlUnavailableError):
        gateway.submit(request())
    for overrides in (
        {"instruction_text": "open inventory and attack"},
        {"arguments": {"allow_attack": True}},
        {"instruction_id": "explore_forward"},
    ):
        with pytest.raises(InvalidInstructionError):
            gateway.submit_qualification(request(**overrides))
    adapter = MinecraftDirectionsAdapter(gateway=gateway)
    with pytest.raises(ValueError):
        adapter.validate("open inventory, observe it, close it; no movement or attack; then attack")
    receipt = gateway.submit_qualification(request())
    with StateDatabase(gateway.db_path) as db:
        message = db.load_operator_messages()[0]
        assert message.direction_request_id == receipt.request_id
        assert message.direction_attempt_id == receipt.attempt_id
        assert _active_operator_messages((message,)) == ()
    assert gateway.discover().available is False


def test_receipt_failure_rolls_back_queue_and_authority_revision(gateway):
    with StateDatabase(gateway.db_path) as db:
        before = db.operator_revision()
        db.connection.execute(
            "CREATE TRIGGER reject_receipt BEFORE INSERT ON paid_directions "
            "BEGIN SELECT RAISE(ABORT, 'injected failure'); END"
        )
        db.connection.commit()
    with pytest.raises(Exception, match="injected failure"):
        gateway.submit_qualification(request())
    with StateDatabase(gateway.db_path) as db:
        assert db.load_operator_messages() == ()
        assert db.operator_revision() == before
        assert db.connection.execute("SELECT COUNT(*) FROM paid_directions").fetchone()[0] == 0


def test_queued_cancel_revokes_authority_and_offline_retry_reconciles(gateway, control):
    receipt = gateway.submit_qualification(request())
    cancelled = gateway.cancel(receipt.request_id)
    assert cancelled.state == DirectionsState.CANCELLED
    assert cancelled.accepted_action_count == 0
    assert gateway.claim(receipt.request_id) is None
    with StateDatabase(gateway.db_path) as db:
        assert db.load_operator_messages()[0].status == OperatorMessageStatus.ARCHIVED
    control["state"] = "PAUSED"
    replay = gateway.submit_qualification(request())
    assert replay.replayed and replay.state == DirectionsState.CANCELLED


@pytest.mark.parametrize(
    "change",
    [
        {"keys_down": ("w",)},
        {"keys_down": ("e", "e")},
        {"buttons_down": ("left",)},
        {"mouse_dx": 1},
        {"cursor_x": 0.5},
        {"keys_down": ("e",), "duration_ms": 800},
    ],
)
def test_wire_boundary_rejects_every_non_inventory_action(gateway, change):
    _, run = begun(gateway)
    with pytest.raises(InvalidInstructionError):
        with gateway.motor_authority(run, toggle().model_copy(update=change), frame()):
            pytest.fail("unsafe action reached dispatch")


def test_cancel_waits_for_dispatch_boundary_then_blocks_all_later_actions(gateway):
    receipt, run = begun(gateway)
    dispatch_entered, cancel_started, allow_dispatch, cancel_done = (
        threading.Event() for _ in range(4)
    )
    results = []

    def dispatch():
        with gateway.motor_authority(run, toggle(), frame()) as accepted:
            dispatch_entered.set()
            assert allow_dispatch.wait(2)
            accepted(supervisor_acceptance())

    def cancel():
        cancel_started.set()
        results.append(gateway.cancel(receipt.request_id))
        cancel_done.set()

    sender = threading.Thread(target=dispatch)
    sender.start()
    assert dispatch_entered.wait(2)
    canceller = threading.Thread(target=cancel)
    canceller.start()
    assert cancel_started.wait(2)
    assert not cancel_done.wait(0.05)
    allow_dispatch.set()
    sender.join(2)
    canceller.join(2)
    assert cancel_done.is_set() and results[0].accepted_action_count == 1
    with pytest.raises(ControlUnavailableError):
        with gateway.motor_authority(run, toggle(2), frame(2)):
            pytest.fail("cancelled attempt dispatched again")


def test_uncertain_transport_and_restart_never_redispatch(gateway):
    receipt, run = begun(gateway)
    with pytest.raises(OSError):
        with gateway.motor_authority(run, toggle(), frame()):
            raise OSError("lost acknowledgement")
    assert gateway.status(receipt.request_id).state == DirectionsState.UNKNOWN
    assert gateway.claim(receipt.request_id) is None


def test_restarted_running_attempt_becomes_unknown(gateway):
    receipt, _ = begun(gateway)
    assert gateway.claim(receipt.request_id) is None
    assert gateway.status(receipt.request_id).state == DirectionsState.UNKNOWN


def test_safe_release_keeps_lease_generation_but_new_lease_revokes(gateway, control):
    control["motor_lease_id"] = "actual-motor-lease"
    epoch = gateway.discover().control_epoch
    receipt = gateway.submit_qualification(request(expected_epoch=epoch))
    control["release_count"] += 1
    assert gateway.discover().control_epoch == epoch
    assert gateway.claim(receipt.request_id) is not None
    control["motor_lease_id"] = "replacement-motor-lease"
    assert gateway.status(receipt.request_id).state == DirectionsState.CANCELLED


def test_private_http_qualification_keeps_origin_guard_and_refuses_raw_commands(
    gateway, monkeypatch
):
    from minecraft_ai.operator import server
    import minecraft_ai.directions

    monkeypatch.setattr(server, "app_paths", lambda: SimpleNamespace(state_db=gateway.db_path))
    monkeypatch.setattr(minecraft_ai.directions, "DirectionsGateway", lambda *_a: gateway)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.OperatorRequestHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_port}"
    payload = dict(
        request_id="browser-check",
        server_id="test-world",
        expected_session_id="test-session",
        expected_epoch=5,
    )

    def post(body, *, suffix="", origin=base):
        req = urllib.request.Request(
            base + "/api/directions/qualification" + suffix,
            data=json.dumps(body).encode(),
            method="POST",
            headers={"Content-Type": "application/json", "Origin": origin},
        )
        with urllib.request.urlopen(req, timeout=2) as response:
            return json.load(response)

    try:
        for body, origin in (
            (payload, "https://evil.example:443"),
            ({**payload, "instruction_text": "attack"}, base),
        ):
            with pytest.raises(urllib.error.HTTPError) as caught:
                post(body, origin=origin)
            assert caught.value.code == 400
        assert gateway.next_pending() is None
        admitted = post(payload)
        assert admitted["state"] == "queued"
        with urllib.request.urlopen(base + "/api/directions/qualification", timeout=2) as response:
            status = json.load(response)
        assert status["commands_enabled"] is False and not status["discovery"]["available"]
        assert status["receipt"]["request_id"] == payload["request_id"]
        assert post({"request_id": payload["request_id"]}, suffix="/cancel")["state"] == "cancelled"
        with StateDatabase(gateway.db_path) as db:
            assert db.load_operator_messages()[0].status == OperatorMessageStatus.ARCHIVED
    finally:
        httpd.shutdown()
        httpd.server_close()


class RuntimeHarness:
    """Real skill executor and AgentRuntime motor boundary; no game input backend."""

    _send_motor = AgentRuntime._send_motor

    def __init__(self, gateway, database):
        self.state_db, self._direction_gateway = database, gateway
        self.skills = build_bootstrap_skill_library()
        self.executor = SkillExecutor(BootstrapMotorPolicy())
        self.blackboard = PerceptionBlackboard()
        self.perception = SimpleNamespace(last_capture=None)
        self._active_direction = None
        self._stop = threading.Event()
        self._input_release_pending_ns = None
        self._sequence = self._execution_revision = 0
        self._pending_decision = None
        self._cognition_requested = False
        self.lease_id = "fixture-lease"
        self.trajectory = None
        self.metrics = SimpleNamespace(motor_actions=0)
        self.runs = []
        self._note_keepalive_prediction = lambda *args: None

    def _start_skill(self, spec, *, source, **kwargs):
        return self.executor.start(spec, **kwargs)

    def _record_terminal_run(self, run, **kwargs):
        self.runs.append(run)

    def _release_and_reconcile_inputs(self):
        self.executor.notify_inputs_released(now_ns=time.monotonic_ns())
        return True

    def observe(self, frame_id, *, inventory):
        capture = frame(frame_id)
        self.perception.last_capture = capture
        self.blackboard.publish(
            FrameState(
                frame_id=frame_id,
                captured_ns=capture.captured_ns,
                instance_id="test-world",
                width=1920,
                height=1080,
                facts=tuple(
                    PerceptionFact(
                        key=key,
                        value=value,
                        confidence=0.995,
                        source=BEDROCK_HUD_SAFETY_SOURCE,
                        observed_ns=capture.captured_ns,
                        expires_after_ms=1000,
                    )
                    for key, value in (
                        ("scene.playable", not inventory),
                        ("scene.ui_overlay", inventory),
                        ("scene.inventory_overlay", inventory),
                    )
                ),
            )
        )


def test_existing_executor_completes_two_attributed_pulses_and_retires_once(gateway, monkeypatch):
    import minecraft_ai.runtime as runtime_module

    sent = []
    monkeypatch.setattr(runtime_module, "operator_pause_latched", lambda: False)
    monkeypatch.setattr(
        runtime_module,
        "send_command",
        lambda command, **kw: sent.append(kw) or supervisor_acceptance(kw["action"]["sequence"]),
    )
    receipt = gateway.submit_qualification(request())
    with StateDatabase(gateway.db_path) as database:
        runtime = RuntimeHarness(gateway, database)
        for frame_id, inventory in enumerate((False, False, True, True, False), start=1):
            runtime.observe(frame_id, inventory=inventory)
            assert tick_inventory_direction(runtime)
        result = gateway.status(receipt.request_id)
        assert result.state == DirectionsState.SUCCEEDED
        assert result.accepted_action_count == 2
        assert result.outcome is not None and len(result.outcome.observed_steps) == 2
        positive = [entry["action"] for entry in sent if entry["action"]["keys_down"]]
        assert len(positive) == 2 and all(a["keys_down"] == ["e"] for a in positive)
        assert all(not a["buttons_down"] and not a["mouse_dx"] for a in positive)
        assert not tick_inventory_direction(runtime)
        for run in runtime.runs:
            event = _terminal_run_event(run, observed_ns=time.time_ns(), trajectory_id=None)
            assert event.payload["direction_request_id"] == receipt.request_id
            assert event.payload["direction_attempt_id"] == receipt.attempt_id
        assert database.load_operator_messages()[0].status == OperatorMessageStatus.ARCHIVED
    assert not gateway.discover().available


@pytest.mark.parametrize(
    "response",
    [
        None,
        {},
        {"accepted_sequence": -1},
        {"accepted_sequence": True},
        {"lease_active": False},
        {"accepted_monotonic_ns": 0},
        {"accepted_monotonic_ns": 2**63 - 1},
    ],
)
def test_runtime_never_counts_unconfirmed_supervisor_actions(gateway, monkeypatch, response):
    import minecraft_ai.runtime as runtime_module

    def dispatch(command, **kwargs):
        if not response:
            return response
        return {**supervisor_acceptance(kwargs["action"]["sequence"]), **response}

    monkeypatch.setattr(runtime_module, "operator_pause_latched", lambda: False)
    monkeypatch.setattr(runtime_module, "send_command", dispatch)
    receipt, run = begun(gateway)
    with StateDatabase(gateway.db_path) as database:
        runtime = RuntimeHarness(gateway, database)
        runtime.observe(1, inventory=False)
        runtime.executor.start(
            runtime.skills.get(run.skill_id),
            run_id=run.run_id,
            context_key=run.context_key,
            parameters=run.parameters,
        )
        with pytest.raises(ControlUnavailableError, match="acceptance is unverified"):
            runtime._send_motor(toggle())
        result = gateway.status(receipt.request_id)
        assert result.state == DirectionsState.UNKNOWN
        assert result.accepted_action_count == 0
        assert result.outcome is None
        step = database.connection.execute(
            "SELECT action_sequence, action_monotonic_ns FROM paid_direction_steps WHERE run_id=?",
            (run.run_id,),
        ).fetchone()
        assert tuple(step) == (None, None)


@pytest.mark.parametrize("fault", ["wrong-run", "wrong-attempt", "pre-action", "wrong-source"])
def test_misattributed_observation_never_becomes_success(gateway, fault):
    receipt, run = begun(gateway)
    capture = frame()
    with gateway.motor_authority(run, toggle(), capture) as accepted:
        accepted(supervisor_acceptance())
    after = frame(2)
    fact = PerceptionFact(
        key="scene.inventory_overlay",
        value=True,
        confidence=0.995,
        observed_ns=after.captured_ns,
        source=BEDROCK_HUD_SAFETY_SOURCE,
    )
    if fault == "pre-action":
        fact = fact.model_copy(update={"observed_ns": capture.captured_ns})
    elif fault == "wrong-source":
        fact = fact.model_copy(update={"source": "untrusted-model"})
    run = run.model_copy(
        update={
            "outcome": SkillOutcome.SUCCEEDED,
            "run_id": "unrelated" if fault == "wrong-run" else run.run_id,
        }
    )
    if fault == "wrong-attempt":
        with pytest.raises(ControlUnavailableError):
            gateway.finish_step(
                receipt.request_id,
                "different-attempt",
                run,
                SimpleNamespace(fact=lambda *args, **kw: fact),
                after,
            )
    else:
        assert not gateway.finish_step(
            receipt.request_id,
            receipt.attempt_id,
            run,
            SimpleNamespace(fact=lambda *args, **kw: fact),
            after,
        )
    assert gateway.status(receipt.request_id).state != DirectionsState.SUCCEEDED
