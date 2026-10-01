"""Total operator attempts use mock clocks/policies/supervisor, never a game."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

import minecraft_ai.runtime as runtime_module
from minecraft_ai.control.operator_budget import OperatorAttempt
from minecraft_ai.execution import ExecutionTick
from minecraft_ai.execution import SkillExecutor
from minecraft_ai.builtin_skills import build_bootstrap_skill_library
from minecraft_ai.runtime import AgentRuntime
from minecraft_ai.runtime_support.types import SkillStartSource
from minecraft_ai.safety import MotorAction
from minecraft_ai.skills import SkillOutcome, SkillRun, SkillSpec
from minecraft_ai.social import OperatorMessage, OperatorMessageStatus
from minecraft_ai.storage import StateDatabase


def message(**changes):
    return OperatorMessage.model_validate(
        {
            "message_id": "bounded",
            "created_ns": 1_000_000_000,
            "text": "explore_forward",
            "kind": "correction",
            "execution_budget": {"timeout_ms": 1000, "max_skills": 2},
            **changes,
        }
    )


@pytest.mark.parametrize(
    "budget",
    [
        {"timeout_ms": True, "max_skills": 1},
        {"timeout_ms": "1000", "max_skills": 1},
        {"timeout_ms": 1000, "max_skills": 0},
        {"timeout_ms": 300001, "max_skills": 1},
        {"timeout_ms": 1000, "max_skills": 17},
        {"timeout_ms": 1000, "max_skills": 1, "allow_attack": True},
    ],
)
def test_closed_integer_budget(budget):
    with pytest.raises(ValidationError):
        message(execution_budget=budget)


@pytest.mark.parametrize(
    "changes",
    [
        {"kind": "question"},
        {"direction_request_id": "paid"},
        {"direction_attempt_id": "paid-attempt"},
    ],
)
def test_no_reply_or_paid_direction_budget_override(changes):
    with pytest.raises(ValidationError):
        message(**changes)


def test_admission_charges_queue_age_and_children_share_one_clock_count():
    request = message()
    owner = OperatorAttempt.admit(request, wall_ns=1_600_000_000, monotonic_ns=10_000_000_000)
    assert owner.deadline_ns == 10_400_000_000
    assert owner.start(now_ns=10_100_000_000) is None
    owner.bind(request.model_copy(update={"status": OperatorMessageStatus.ACKNOWLEDGED}))
    assert owner.start(now_ns=10_200_000_000) is None
    assert owner.start(now_ns=10_300_000_000) == "operator.attempt_skill_limit"
    assert owner.skills_started == 2 and owner.deadline_ns == 10_400_000_000


def test_expired_queue_future_clock_and_lost_consumed_owner_refuse():
    request = message()
    assert (
        OperatorAttempt.admit(request, wall_ns=2_000_000_000, monotonic_ns=3).terminal_reason
        == "operator.attempt_deadline"
    )
    assert (
        OperatorAttempt.admit(request, wall_ns=999_999_999, monotonic_ns=3).terminal_reason
        == "operator.attempt_clock_invalid"
    )
    consumed = request.model_copy(update={"status": OperatorMessageStatus.ACKNOWLEDGED})
    assert (
        OperatorAttempt.admit(consumed, wall_ns=1_100_000_000, monotonic_ns=3).terminal_reason
        == "operator.attempt_owner_not_retained"
    )


def test_same_request_cannot_extend_budget_by_changing_content():
    request = message()
    owner = OperatorAttempt.admit(request, wall_ns=request.created_ns, monotonic_ns=10)
    owner.bind(message(execution_budget={"timeout_ms": 2000, "max_skills": 3}))
    assert owner.check(now_ns=11) == "operator.attempt_authority_changed"
    assert owner.deadline_ns == 1_000_000_010 and owner.max_skills == 2


def test_budget_roundtrip_and_change_invalidate_sampled_operator_authority(tmp_path):
    with StateDatabase(tmp_path / "state.sqlite") as database:
        request = message()
        database.save_operator_message(request)
        assert database.load_operator_messages() == (request,)
        revision = database.operator_revision()
        database.save_operator_message(
            message(execution_budget={"timeout_ms": 2000, "max_skills": 2})
        )
        assert database.operator_revision() == revision + 1


class Executor:
    def __init__(self):
        self.run = None
        self.starts = []
        self.cancelled = []
        self.policy = SimpleNamespace(policy_id="mock", reset=Mock())
        self.notify_inputs_released = Mock()

    def start(self, spec, **kwargs):
        self.starts.append(spec.skill_id)
        self.run = SkillRun(
            run_id=kwargs["run_id"],
            skill_id=spec.skill_id,
            started_ns=0,
            context_key=kwargs.get("context_key", "default"),
        )
        return self.run

    def expire_operator_attempt(self, **kwargs):
        self.cancelled.append(self.run.run_id)
        self.run = self.run.model_copy(
            update={
                "outcome": SkillOutcome.TIMED_OUT,
                "failure_reason": kwargs["reason"],
                "ended_ns": kwargs["now_ns"],
            }
        )

    def push_child(self, spec, **kwargs):
        return self.start(spec, **kwargs)


def runtime(monkeypatch):
    clock = [10_000_000_000]
    monkeypatch.setattr(runtime_module.time, "monotonic_ns", lambda: clock[0])
    value = object.__new__(AgentRuntime)
    value._operator_attempts = {"operator:bounded": OperatorAttempt("fixture", 11_000_000_000, 2)}
    value.executor = Executor()
    value._request_policy_warm = Mock()
    value._warm_plan_specialists = Mock()
    value.on_skill_run_started = Mock()
    value._record_terminal_run = Mock()
    value._release_and_reconcile_inputs = Mock(return_value=True)
    value._plan_graph = None
    value._execution_revision = 0
    value._input_release_pending_ns = None
    value._stop = SimpleNamespace(is_set=lambda: False)
    value._sequence = 0
    value.lease_id = "mock-lease"
    monkeypatch.setattr(runtime_module, "operator_pause_latched", lambda: False)
    return value, clock


def start(
    value, skill_id, run_id, source=SkillStartSource.RECOVERY, context_key="operator:bounded"
):
    spec = SkillSpec(skill_id=skill_id, version=1, name=skill_id)
    return value._start_skill(spec, source=source, run_id=run_id, context_key=context_key)


def test_initial_recovery_and_plan_starts_share_cap_before_policy_work(monkeypatch):
    value, clock = runtime(monkeypatch)
    start(value, "explore_forward", "first", SkillStartSource.COGNITION)
    clock[0] += 100_000_000
    start(value, "traverse_visible_obstacle", "second")
    clock[0] += 100_000_000
    refused = start(value, "survey_surroundings", "third", SkillStartSource.PLAN)
    assert refused.outcome == SkillOutcome.TIMED_OUT
    assert refused.failure_reason == "operator.attempt_skill_limit"
    assert value.executor.starts == ["explore_forward", "traverse_visible_obstacle"]
    assert value._request_policy_warm.call_count == 2
    value._release_and_reconcile_inputs.assert_called_once()


def test_expired_start_never_warms_policy_and_independent_safety_still_starts(monkeypatch):
    value, clock = runtime(monkeypatch)
    clock[0] = 11_000_000_000
    refused = start(value, "explore_forward", "late", SkillStartSource.COGNITION)
    assert refused.failure_reason == "operator.attempt_deadline"
    value._request_policy_warm.assert_not_called()
    safety = start(value, "respawn_after_death", "safety", context_key="scene-recovery")
    assert safety.outcome == SkillOutcome.RUNNING
    assert value.executor.starts == ["respawn_after_death"]


def test_slow_policy_late_action_and_recovery_are_suppressed_with_physical_release(monkeypatch):
    value, clock = runtime(monkeypatch)
    run = start(value, "explore_forward", "first")
    result = ExecutionTick(
        run=run,
        action=MotorAction(sequence=0, keys_down=("w",)),
        recovery_skills=("traverse_visible_obstacle",),
    )
    clock[0] = 11_000_000_001
    expired = value._expire_operator_result(result)
    assert expired.action is None and expired.recovery_skills == ()
    assert expired.policy_proposal is None and expired.outcome_verification is None
    assert expired.run.failure_reason == "operator.attempt_deadline"
    value._release_and_reconcile_inputs.assert_called_once()
    assert value.executor.cancelled == [run.run_id]
    assert value.executor.run.failure_reason == "operator.attempt_deadline"
    assert value.executor.run.outcome == SkillOutcome.TIMED_OUT
    value.executor.policy.reset.assert_not_called()


def test_final_actuation_gate_suppresses_expired_input_without_stopping_game(monkeypatch):
    value, clock = runtime(monkeypatch)
    run = start(value, "explore_forward", "first")
    clock[0] = 11_000_000_001
    wire = Mock()
    monkeypatch.setattr(runtime_module, "send_command", wire)
    assert (
        value._send_motor(
            MotorAction(sequence=0, keys_down=("w",)), execution=ExecutionTick(run=run, action=None)
        )
        is False
    )
    wire.assert_not_called()
    assert not value._stop.is_set()


def test_expiration_at_second_wire_gate_has_no_accepted_action_credit(monkeypatch):
    value, _ = runtime(monkeypatch)
    run = start(value, "explore_forward", "first")
    value._operator_motor_admitted = Mock(side_effect=[True, False])
    value.blackboard = Mock()
    value.trajectory = Mock()
    value.metrics = SimpleNamespace(motor_actions=0)
    value.executor.policy.observe_accepted_action = Mock()
    value.executor.policy.restore_world_camera_state = Mock()
    monkeypatch.setattr(runtime_module, "_accepted_action_provenance", Mock())
    wire = Mock()
    monkeypatch.setattr(runtime_module, "send_command", wire)
    result = value._send_motor(
        MotorAction(sequence=0, keys_down=("w",)), execution=ExecutionTick(run=run, action=None)
    )
    assert result is False
    assert value._sequence == 0 and value.metrics.motor_actions == 0
    wire.assert_not_called()
    value.trajectory.record_accepted.assert_not_called()
    value.executor.policy.observe_accepted_action.assert_not_called()
    value.executor.policy.restore_world_camera_state.assert_not_called()


def test_idle_camera_suppression_does_not_restore_a_move_that_was_not_sent(monkeypatch):
    value, _ = runtime(monkeypatch)
    value._unresolved_operator_directive_waiting = Mock(return_value=False)
    value._authoritative_world_camera_pitch_units = Mock(return_value=400)
    value._send_motor = Mock(return_value=False)
    restore = Mock()
    monkeypatch.setattr(runtime_module, "_headroom_reorient_mouse_dy", lambda pitch: -100)
    monkeypatch.setattr(runtime_module, "_restore_policy_world_camera", restore)
    assert value._keepalive_horizon_reorient() is False
    restore.assert_not_called()


def test_terminal_tick_at_expired_wire_boundary_records_timeout_not_old_success(monkeypatch):
    value, clock = runtime(monkeypatch)
    run = start(value, "explore_forward", "first")
    terminal = run.model_copy(update={"outcome": SkillOutcome.SUCCEEDED, "ended_ns": clock[0]})
    value.executor.run = terminal
    clock[0] = 11_000_000_001
    assert value._operator_motor_admitted(ExecutionTick(run=terminal, action=None)) is False
    recorded = value._record_terminal_run.call_args.args[0]
    assert recorded.outcome == SkillOutcome.TIMED_OUT
    assert recorded.failure_reason == "operator.attempt_deadline"
    assert value.executor.run == recorded


def test_failed_release_keeps_existing_positive_input_gate_armed(monkeypatch):
    value, clock = runtime(monkeypatch)
    run = start(value, "explore_forward", "first")
    value._release_and_reconcile_inputs = AgentRuntime._release_and_reconcile_inputs.__get__(value)
    calls = []
    monkeypatch.setattr(
        runtime_module,
        "send_command",
        lambda command, **kwargs: calls.append(command) or {"released": False},
    )
    clock[0] = 11_000_000_001
    result = value._expire_operator_result(
        ExecutionTick(run=run, action=MotorAction(sequence=0, keys_down=("w",)))
    )
    assert result.action is None and value._input_release_pending_ns == clock[0]
    assert calls == ["release-inputs"]
    value.executor.notify_inputs_released.assert_not_called()
    with pytest.raises(RuntimeError, match="release acknowledgement"):
        value._send_motor(MotorAction(sequence=1, keys_down=("w",)))


def test_headroom_camera_phase_has_same_deadline_even_without_active_child(monkeypatch):
    value, clock = runtime(monkeypatch)
    value._headroom_recovery = SimpleNamespace(context_key="operator:bounded")
    value._clear_headroom_recovery = Mock(
        side_effect=lambda recovery: setattr(value, "_headroom_recovery", None)
    )
    clock[0] = 11_000_000_001
    assert value._expire_current_operator_attempt()
    assert value._headroom_recovery is None
    value._release_and_reconcile_inputs.assert_called_once()


def test_nested_options_use_the_same_counter_before_warming(monkeypatch):
    value, _ = runtime(monkeypatch)
    start(value, "explore_forward", "parent")
    child = value._push_child_skill(
        SkillSpec(skill_id="survey_surroundings", version=1, name="survey"),
        run_id="child",
        context_key="operator:bounded",
    )
    refused = value._push_child_skill(
        SkillSpec(skill_id="survey_surroundings", version=1, name="survey"),
        run_id="third",
        context_key="operator:bounded",
    )
    assert child.outcome == SkillOutcome.RUNNING
    assert refused.failure_reason == "operator.attempt_skill_limit"
    assert value.executor.starts == ["explore_forward", "survey_surroundings"]
    assert value._request_policy_warm.call_count == 2
    value._release_and_reconcile_inputs.assert_called_once()
    assert value.executor.run.outcome == SkillOutcome.TIMED_OUT


def test_expiration_removes_real_suspended_parent_without_policy_callbacks(monkeypatch):
    value, clock = runtime(monkeypatch)
    policy = SimpleNamespace(
        policy_id="mock",
        reset=Mock(side_effect=AssertionError("no policy reset")),
        status=Mock(side_effect=AssertionError("no policy status")),
    )
    value.executor = SkillExecutor(policy)
    skills = build_bootstrap_skill_library()
    parent = value._start_skill(
        skills.get("explore_forward"),
        source=SkillStartSource.COGNITION,
        run_id="parent",
        context_key="operator:bounded",
        now_ns=clock[0],
    )
    child = value._push_child_skill(
        skills.get("survey_surroundings"),
        run_id="child",
        context_key=parent.context_key,
        now_ns=clock[0],
    )
    assert value.executor.parent_run().run_id == parent.run_id
    clock[0] = 11_000_000_001
    expired = value._expire_operator_result(ExecutionTick(run=child, action=None))
    assert value.executor.run == expired.run
    assert value.executor.parent_run() is None
    with pytest.raises(RuntimeError, match="no suspended parent"):
        value.executor.resume_parent()
    value._release_and_reconcile_inputs.assert_called_once()
    policy.reset.assert_not_called()
    policy.status.assert_not_called()


def test_matching_headroom_child_expiration_releases_once(monkeypatch):
    value, clock = runtime(monkeypatch)
    start(value, "traverse_visible_obstacle", "child")
    value._headroom_recovery = SimpleNamespace(context_key="operator:bounded")
    value._clear_headroom_recovery = Mock(
        side_effect=lambda recovery: setattr(value, "_headroom_recovery", None)
    )
    clock[0] = 11_000_000_001
    assert value._expire_current_operator_attempt()
    value._release_and_reconcile_inputs.assert_called_once()
    assert value.executor.run.failure_reason == "operator.attempt_deadline"
    value._record_terminal_run.assert_called_once()


def test_expired_headroom_owner_does_not_timeout_independent_safety_run(monkeypatch):
    value, clock = runtime(monkeypatch)
    safety = start(value, "respawn_after_death", "safety", context_key="scene-recovery")
    value._headroom_recovery = SimpleNamespace(context_key="operator:bounded")
    value._clear_headroom_recovery = Mock(
        side_effect=lambda recovery: setattr(value, "_headroom_recovery", None)
    )
    clock[0] = 11_000_000_001
    assert value._operator_motor_admitted(None) is False
    assert value.executor.run == safety and safety.outcome == SkillOutcome.RUNNING
    value._record_terminal_run.assert_not_called()
    value._release_and_reconcile_inputs.assert_called_once()


def test_owner_filter_refuses_expired_queue_and_lost_acknowledged_owner(monkeypatch):
    value, _ = runtime(monkeypatch)
    value._operator_attempts = {}
    value._persist_operator_message_status = Mock(return_value=True)
    monkeypatch.setattr(runtime_module.time, "time_ns", lambda: 2_000_000_000)
    assert value._remember_operator_attempts((message(),)) == ()
    assert value._persist_operator_message_status.call_args.args[:2] == (
        "bounded",
        OperatorMessageStatus.ACKNOWLEDGED,
    )
    assert (
        "operator.attempt_deadline"
        in value._persist_operator_message_status.call_args.kwargs["response_text"]
    )
    value._operator_attempts = {}
    request = message(kind="instruction", status=OperatorMessageStatus.ACKNOWLEDGED)
    assert value._remember_operator_attempts((request,)) == ()
    assert (
        value._operator_attempts["operator:bounded"].terminal_reason
        == "operator.attempt_owner_not_retained"
    )


def test_expired_owners_are_not_evicted_while_old_decisions_may_reference_them(monkeypatch):
    value, clock = runtime(monkeypatch)
    value._operator_attempts = {
        f"operator:{number}": OperatorAttempt("fixture", 1, 1) for number in range(257)
    }
    monkeypatch.setattr(runtime_module.time, "time_ns", lambda: 2_000_000_000)
    assert value._remember_operator_attempts(()) == ()
    assert len(value._operator_attempts) == 257
    assert (
        value._operator_attempts["operator:0"].check(now_ns=clock[0]) == "operator.attempt_deadline"
    )


def test_unbounded_message_path_does_not_admit_a_new_clock(monkeypatch):
    value, _ = runtime(monkeypatch)
    value._operator_attempts = {}
    monkeypatch.setattr(
        runtime_module.time, "time_ns", Mock(side_effect=AssertionError("no new clock"))
    )
    request = message(execution_budget=None)
    assert value._remember_operator_attempts((request,)) == (request,)


def test_message_endpoint_accepts_only_declared_budget_without_socket_or_game(
    tmp_path, monkeypatch
):
    import minecraft_ai.operator.server as server_module

    handler = object.__new__(server_module.OperatorRequestHandler)
    handler._send_json = Mock()
    path = tmp_path / "endpoint.sqlite"
    monkeypatch.setattr(server_module, "app_paths", lambda: SimpleNamespace(state_db=path))
    handler._post_message(
        {
            "text": "explore_forward",
            "kind": "correction",
            "execution_budget": {"timeout_ms": 12000, "max_skills": 2},
        }
    )
    with StateDatabase(path) as database:
        requests = database.load_operator_messages()
    assert len(requests) == 1
    assert requests[0].execution_budget.timeout_ms == 12000
    assert requests[0].execution_budget.max_skills == 2
    for bad in (
        {"timeout_ms": True, "max_skills": 2},
        {"timeout_ms": 12000, "max_skills": 2, "allow_attack": True},
    ):
        with pytest.raises(ValidationError):
            handler._post_message({"text": "explore_forward", "execution_budget": bad})
    with pytest.raises(ValueError, match="unsupported fields"):
        handler._post_message({"text": "explore_forward", "direction_request_id": "fake-paid"})
    with StateDatabase(path) as database:
        assert len(database.load_operator_messages()) == 1


def test_no_numerical_backend_was_loaded_by_these_controls():
    assert not ({"torch", "numpy", "transformers", "erais"} & set(sys.modules))
