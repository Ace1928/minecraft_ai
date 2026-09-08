"""Warm-controller choice feedback; no real inference or supervisor inputs."""
from __future__ import annotations

import sqlite3
import threading
import time
from collections import deque
from types import SimpleNamespace

import pytest

from minecraft_ai.action_levels import ActionLevel
from minecraft_ai.builtin_skills import build_bootstrap_skill_library
from minecraft_ai.execution import ExecutionTick, SkillExecutor
from minecraft_ai.memory import MemoryStore
from minecraft_ai.motor import BootstrapMotorPolicy, MotorIntent
from minecraft_ai.outcome_verifier import (
    OutcomeKind, OutcomeSignal, OutcomeStatus, OutcomeVerification,
)
from minecraft_ai.perception import FrameState, PerceptionBlackboard, PerceptionFact
from minecraft_ai.runtime_support.helpers import _accepted_action_provenance
from minecraft_ai.runtime import AgentRuntime, RuntimeMetrics
from minecraft_ai.runtime_support.types import SkillStartSource
from minecraft_ai.safety import MotorAction
from minecraft_ai.skills import SkillFailureCode, SkillOutcome, SkillRun, SkillStats
from minecraft_ai.storage import OperatorContextSnapshot, StateDatabase
from minecraft_ai.trajectory import ActionOrigin


def _runtime(monkeypatch: pytest.MonkeyPatch) -> tuple[AgentRuntime, list[int], list[dict]]:
    clock = [1_000_000_000_000]
    monkeypatch.setattr(time, "monotonic_ns", lambda: clock[0])
    monkeypatch.setattr("minecraft_ai.runtime.operator_pause_latched", lambda: False)
    monkeypatch.setattr("minecraft_ai.runtime.emergency_stop_latched", lambda: False)
    accepted = []

    def send(command: str, **payload: object) -> dict:
        assert command == "motor-action"
        accepted.append(payload)
        return {"world_camera": {"origin_calibrated": True, "estimated_pitch_units": 0}}

    monkeypatch.setattr("minecraft_ai.runtime.send_command", send)
    runtime = object.__new__(AgentRuntime)
    runtime.blackboard = PerceptionBlackboard()
    runtime.blackboard.publish(FrameState(
        frame_id=1, captured_ns=clock[0], instance_id="test-world", width=32, height=32,
        facts=tuple(PerceptionFact(
            key=key, value=value, confidence=0.995, observed_ns=clock[0],
            source="safety:bedrock-hud-v1:not-training-label", expires_after_ms=30_000,
        ) for key, value in (("scene.mode", "world"), ("scene.playable", True))),
    ))
    runtime.skills = build_bootstrap_skill_library()
    runtime.skills.stats[("explore_forward", "explore-keepalive")] = SkillStats(
        failures=20, consecutive_failures=20,
    )
    runtime.executor = SkillExecutor(BootstrapMotorPolicy())
    runtime.metrics = RuntimeMetrics()
    runtime.memories = MemoryStore()
    runtime._recorded_run_ids = set()
    runtime._recorded_run_order = deque()
    runtime._recent_skill_runs = deque()
    runtime._stop = threading.Event()
    runtime._pending_operator_status_updates = {}
    runtime.lease_id = "test-only"
    return runtime, clock, accepted


def _start(runtime: AgentRuntime, *, source: SkillStartSource = SkillStartSource.KEEPALIVE,
           run_id: str = "run-one", context: str = "explore-keepalive") -> SkillRun:
    return runtime._start_skill(
        runtime.skills.get("traverse_level_ground"), source=source,
        run_id=run_id, context_key=context,
    )


def _prediction(runtime: AgentRuntime, clock: list[int], prediction_id: str,
                *, variant: str = "prediction") -> None:
    clock[0] += 10_000_000
    run = runtime.executor.run
    assert run is not None
    intent = MotorIntent(skill_id=run.skill_id, mode="traverse", episode_id=run.run_id,
                         action_level=ActionLevel.MOTION)
    condition = intent.model_dump(mode="json")
    if variant == "old-episode":
        condition["episode_id"] = "retired-run"
    if variant == "wrong-skill":
        condition["skill_id"] = "gather_nearby_wood"
    causal = {
        "policy_id": "test-body", "action_kind": (
            "prediction_hold" if variant == "hold" else "prediction"
        ),
        "request_id": prediction_id, "prediction_id": prediction_id,
        "source_frame_id": 1, "source_captured_ns": (
            run.started_ns - 1 if variant == "old-frame" else clock[0]
        ),
        "condition": condition,
    }
    route = "semantic" if variant == "other-route" else "raw_motion"
    status = {
        "active_route": route,
        "primary" if route == "semantic" else route: {
            "policy_id": "test-body", "model_version": "test-v1",
            "last_action_provenance": causal,
            "last_prediction": {
                "keys": ["w"] if variant == "filtered-movement" else [], "buttons": [],
                "suppressed_actions": ["forward"] if variant == "suppressed" else [],
            },
        },
    }
    if variant == "missing-route":
        status.update(status.pop("raw_motion"))
    elif variant == "direct":
        status = status["raw_motion"]
    elif variant == "foreign-component":
        status["grounded"] = status.pop("raw_motion")
    action = MotorAction(sequence=runtime._sequence, mouse_dx=1,
                         keys_down=("w",) if variant == "actual-movement" else ())
    tick = ExecutionTick(
        run=run, action=action, motor_intent=intent, policy_status=status,
        action_origin=ActionOrigin.RESET if variant == "reset" else ActionOrigin.POLICY,
    )
    runtime._send_motor(action, execution=tick)


def _finish(runtime: AgentRuntime, clock: list[int], *, collision: bool = False) -> SkillRun:
    clock[0] += 3_000_000_000
    code = (SkillFailureCode.LOCOMOTION_STALLED if collision
            else SkillFailureCode.CONTROLLER_STARVATION)
    run = runtime.executor.cancel().run.model_copy(update={
        "outcome": SkillOutcome.FAILED, "failure_code": code, "failure_reason": code.value,
    })
    verification = OutcomeVerification(
        run_id=run.run_id, kind=OutcomeKind.TRAVERSAL, status=OutcomeStatus.STALLED,
        signal=(OutcomeSignal.LOCOMOTION_STALLED if collision
                else OutcomeSignal.CONTROLLER_STARVATION),
        observed_ns=clock[0], confidence=0.96, reason="test-only verifier result",
    )
    runtime._record_terminal_run(run, outcome_verification=verification)
    return run


def test_three_predictions_rotate_once_without_reclassifying_starvation(monkeypatch) -> None:
    runtime, clock, accepted = _runtime(monkeypatch)
    _start(runtime)
    for index in range(10):
        _prediction(runtime, clock, str(index))
    evidence = runtime._keepalive_prediction_evidence
    assert evidence is not None and len(evidence.prediction_ids) == 3
    run = _finish(runtime, clock)
    assert run.failure_code == SkillFailureCode.CONTROLLER_STARVATION
    stats = runtime.skills.stats[(run.skill_id, run.context_key)]
    assert stats.failures == 1 and stats.consecutive_failures == 0
    assert runtime._traversal_escalation_pending is False
    assert runtime._explore_keep_alive().skill_id == "explore_forward"
    assert runtime._explore_keep_alive().skill_id == "traverse_level_ground"
    assert len(accepted) == 10  # Selection does not emit an action.


@pytest.mark.parametrize("variant", (
    "cold", "two", "same-id", "hold", "old-episode", "old-frame", "wrong-skill", "reset",
    "filtered-movement", "suppressed", "other-route", "rejected", "collision",
    "missing-route", "foreign-component", "actual-movement",
))
def test_unqualified_starvation_cannot_rotate(monkeypatch, variant: str) -> None:
    runtime, clock, _ = _runtime(monkeypatch)
    _start(runtime)
    if variant == "other-route":
        _prediction(runtime, clock, "initial")
    if variant == "rejected":
        def reject(*_args: object, **_kwargs: object) -> dict:
            raise RuntimeError("supervisor rejected input")
        monkeypatch.setattr("minecraft_ai.runtime.send_command", reject)
    for index in range(0 if variant == "cold" else 2 if variant == "two" else 3):
        pred_variant = "prediction" if variant in {
            "two", "same-id", "rejected", "collision",
        } else variant
        if variant == "rejected":
            with pytest.raises(RuntimeError, match="supervisor rejected"):
                _prediction(runtime, clock, str(index))
        else:
            _prediction(runtime, clock, "repeated" if variant == "same-id" else str(index),
                        variant=pred_variant)
    _finish(runtime, clock, collision=variant == "collision")
    assert runtime._keepalive_rotation_hint is None
    assert runtime._explore_keep_alive().skill_id == "traverse_level_ground"


@pytest.mark.parametrize("change", (
    "operator", "operator-revision", "plan", "execution", "pause", "emergency", "unsafe",
    "headroom", "perception", "escalation", "running",
))
def test_rotation_cannot_outlive_context_or_bypass_safety(monkeypatch, change: str) -> None:
    runtime, clock, accepted = _runtime(monkeypatch)
    _start(runtime)
    for index in range(3):
        _prediction(runtime, clock, str(index))
    _finish(runtime, clock)
    assert runtime._keepalive_rotation_hint is not None
    if change == "operator":
        runtime._pending_operator_message_ids = ("new-operator",)
    elif change == "operator-revision":
        runtime.state_db = SimpleNamespace(load_operator_messages=lambda **_: (),
                                          load_operator_context=lambda **_:
                                          OperatorContextSnapshot(
                                              revision=1, messages=(), target=None,
                                          ))
    elif change == "plan":
        runtime._plan_started_ns += 1
    elif change == "execution":
        runtime._execution_revision += 1
    elif change == "pause":
        monkeypatch.setattr("minecraft_ai.runtime.operator_pause_latched", lambda: True)
    elif change == "emergency":
        monkeypatch.setattr("minecraft_ai.runtime.emergency_stop_latched", lambda: True)
    elif change == "unsafe":
        clock[0] += 50_000_000_000  # Expired HUD is not WORLD permission.
    elif change == "headroom":
        runtime._headroom_recovery = object()
    elif change == "perception":
        runtime._cognition_perception_probe = object()
    elif change == "escalation":
        runtime._traversal_escalation_pending = True
    else:
        runtime.executor.start(runtime.skills.get("explore_forward"), run_id="new-owner")
    assert runtime._explore_keep_alive() is None
    assert runtime._keepalive_rotation_hint is None
    assert len(accepted) == 3


@pytest.mark.parametrize("source,context", (
    (SkillStartSource.COGNITION, "explore-keepalive"),
    (SkillStartSource.KEEPALIVE, "operator:task"),
))
def test_only_autonomous_keepalive_start_collects_evidence(monkeypatch, source, context) -> None:
    runtime, _, _ = _runtime(monkeypatch)
    _start(runtime, source=source, context=context)
    assert runtime._keepalive_prediction_evidence is None


def test_new_run_cannot_inherit_old_prediction_evidence(monkeypatch) -> None:
    runtime, clock, _ = _runtime(monkeypatch)
    _start(runtime)
    for index in range(3):
        _prediction(runtime, clock, str(index))
    runtime.executor.cancel()
    _start(runtime, run_id="new-run")
    _finish(runtime, clock)
    assert runtime._keepalive_rotation_hint is None


def test_direct_policy_predictions_have_the_same_bounded_rotation(monkeypatch) -> None:
    runtime, clock, _ = _runtime(monkeypatch)
    _start(runtime)
    for index in range(3):
        _prediction(runtime, clock, str(index), variant="direct")
    _finish(runtime, clock)
    assert runtime._explore_keep_alive().skill_id == "explore_forward"


def test_operator_context_read_failure_does_not_break_started_skill(monkeypatch) -> None:
    runtime, _, _ = _runtime(monkeypatch)

    def unavailable(**_kwargs: object) -> None:
        raise sqlite3.OperationalError("database is locked")

    runtime.state_db = SimpleNamespace(load_operator_context=unavailable)
    run = _start(runtime)
    assert run.outcome == SkillOutcome.RUNNING
    assert runtime.executor.run is run
    assert runtime._keepalive_prediction_evidence is None
    assert runtime._keepalive_rotation_hint is None


def test_pending_storage_transaction_does_not_break_started_skill(monkeypatch, tmp_path) -> None:
    runtime, _, _ = _runtime(monkeypatch)
    with StateDatabase(tmp_path / "state.sqlite") as database:
        runtime.state_db = database
        database.connection.execute("BEGIN")
        run = _start(runtime)
        assert runtime.executor.run is run and run.outcome == SkillOutcome.RUNNING
        assert runtime._keepalive_prediction_evidence is None
        assert database.connection.in_transaction


def test_foreign_accepted_provenance_cannot_borrow_selected_prediction(monkeypatch) -> None:
    runtime, clock, _ = _runtime(monkeypatch)

    def mismatched(*args, **kwargs):
        provenance = _accepted_action_provenance(*args, **kwargs)
        return provenance.model_copy(update={"source_frame_id": 999})

    monkeypatch.setattr("minecraft_ai.runtime._accepted_action_provenance", mismatched)
    _start(runtime)
    for index in range(3):
        _prediction(runtime, clock, str(index))
    _finish(runtime, clock)
    assert runtime._keepalive_rotation_hint is None
