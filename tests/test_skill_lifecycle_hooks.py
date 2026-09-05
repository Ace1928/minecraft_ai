"""Exact skill lifecycle attribution with no model, supervisor, or game I/O."""

from __future__ import annotations

import time
from collections.abc import Iterator
from concurrent.futures import Future
from contextlib import nullcontext
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

import minecraft_ai.runtime as runtime_module
from minecraft_ai.cognition import (
    CognitionDecision,
    DecisionModelOrigin,
    cognition_decision_sha256,
)
from minecraft_ai.execution import SkillExecutor
from minecraft_ai.model_requests import ModelRequestLifecycle, RequestBinding
from minecraft_ai.outcome_verifier import (
    OutcomeKind,
    OutcomeSignal,
    OutcomeStatus,
    OutcomeVerification,
)
from minecraft_ai.perception import FrameState, PerceptionBlackboard
from minecraft_ai.roles import get_role
from minecraft_ai.runtime import AgentRuntime, SkillDecisionOrigin, SkillStartSource
from minecraft_ai.safety import MotorAction
from minecraft_ai.skills import SkillLibrary, SkillOutcome, SkillRun, SkillSpec


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    board = PerceptionBlackboard()
    board.publish(FrameState(
        frame_id=1, captured_ns=time.monotonic_ns(), instance_id="lifecycle-test",
        width=32, height=32,
    ))
    skills = SkillLibrary()
    for name in ("probe-a", "probe-b"):
        skills.register(SkillSpec(skill_id=name, name=name))
    policy = SimpleNamespace(
        policy_id="synthetic-lifecycle-policy",
        reset=Mock(return_value=MotorAction(sequence=0)), close=Mock(),
    )
    model = SimpleNamespace(
        admit_bound_decision=Mock(return_value=None), discard_bound_request=Mock(),
    )
    database = SimpleNamespace(
        admit_operator_revision=Mock(side_effect=lambda revision: nullcontext(revision == 7)),
        load_operator_messages=Mock(return_value=[]), save_skill_stats=Mock(),
        save_runtime_event=Mock(), save_memory=Mock(), close=Mock(),
    )
    item = AgentRuntime(
        perception=SimpleNamespace(  # type: ignore[arg-type]
            instance_id="lifecycle-test", close=Mock(), active_vlm=None, last_capture=None,
        ),
        blackboard=board, executor=SkillExecutor(policy), skills=skills,
        role=get_role("generalist"), lease_id="lifecycle-test-lease",
        high_level=SimpleNamespace(model=model),  # type: ignore[arg-type]
        state_db=database,  # type: ignore[arg-type]
        telemetry=SimpleNamespace(publish=Mock()),  # type: ignore[arg-type]
    )
    forbidden = Mock(side_effect=AssertionError("lifecycle test attempted supervisor I/O"))
    monkeypatch.setattr(runtime_module, "send_command", forbidden)
    monkeypatch.setattr(runtime_module, "operator_pause_latched", lambda: False)
    monkeypatch.setattr(runtime_module, "emergency_stop_latched", lambda: False)
    monkeypatch.setattr(runtime_module, "operator_intent_lock", lambda **_: nullcontext())
    monkeypatch.setattr(item, "_send_motor", Mock())
    monkeypatch.setattr(item, "on_skill_run_started", Mock())
    monkeypatch.setattr(item, "on_skill_run_terminal", Mock())
    try:
        yield item
    finally:
        item._pool.shutdown(wait=False, cancel_futures=True)
        assert item._pool.wait_closed(1.0)
        forbidden.assert_not_called()
        database.close.assert_not_called()


def _prepared_decision(
    runtime: Any, name: str, *, skill_id: str | None = "probe-a", model_origin: bool = True,
    attempt_error: str | None = None,
) -> tuple[CognitionDecision, ModelRequestLifecycle]:
    now = time.monotonic_ns()
    snapshot = runtime.blackboard.cognition_snapshot(now_ns=now)
    request = ModelRequestLifecycle(RequestBinding.from_snapshot(
        snapshot, request_id=name, operator_revision=7,
        execution_revision=runtime._execution_revision,
        submitted_ns=now, deadline_ns=now + 60_000_000_000,
    ))
    attempt = request.start_attempt("synthetic-test")
    request.finish_attempt(attempt, error_type=attempt_error)
    request.mark_computation_complete()
    decision = CognitionDecision(
        skill_id=skill_id, chosen_goal_id="same-task", skill_parameters={"distance": 1},
    )
    if model_origin:
        decision._model_origin = DecisionModelOrigin(
            name, attempt, cognition_decision_sha256(decision),
        )
    return decision, request


def _consume(runtime: Any, decision: CognitionDecision, request: ModelRequestLifecycle) -> None:
    future: Future[CognitionDecision] = Future()
    future.set_result(decision)
    runtime._pending_decision = future
    runtime._pending_execution_revision = runtime._execution_revision
    runtime._bound_cognition_requests[future] = (request, runtime.high_level.model)
    runtime._consume_cognition_decision()


def _terminal(run: SkillRun, *, outcome: SkillOutcome = SkillOutcome.CANCELLED) -> SkillRun:
    return run.model_copy(update={
        "outcome": outcome, "ended_ns": run.started_ns + 1_000,
        "failure_reason": "synthetic-terminal",
    })


def _verification(run_id: str) -> OutcomeVerification:
    return OutcomeVerification(
        run_id=run_id, kind=OutcomeKind.TRAVERSAL, status=OutcomeStatus.STALLED,
        signal=OutcomeSignal.LOCOMOTION_STALLED, observed_ns=100,
        confidence=0.9, reason="synthetic typed locomotion stall",
    )


def test_origin_preserves_exact_accepted_request_attempt_and_rewrite(runtime: Any) -> None:
    decision, request = _prepared_decision(runtime, "rewrite-request")
    source = decision.model_origin
    assert source is not None
    decision = decision.model_copy(update={"instruction": "bounded rewritten instruction"})
    final_hash = cognition_decision_sha256(decision)
    assert request.accept(source.attempt_id, final_hash, lambda: None)

    origin = runtime._skill_decision_origin((request, runtime.high_level.model), decision)

    assert isinstance(origin, SkillDecisionOrigin)
    assert origin.request is request.binding
    assert origin.attempt_id == source.attempt_id
    assert origin.source_decision_sha256 == source.source_decision_sha256
    assert origin.final_decision_sha256 == final_hash
    assert origin.source_decision_sha256 != origin.final_decision_sha256
    with pytest.raises(FrozenInstanceError):
        origin.attempt_id = "replacement"  # type: ignore[misc]


def test_unaccepted_or_unbound_decision_has_no_start_origin(runtime: Any) -> None:
    decision, request = _prepared_decision(runtime, "pending-request")
    assert runtime._skill_decision_origin(None, decision) is None
    assert runtime._skill_decision_origin((request, runtime.high_level.model), decision) is None
    assert request.reject("synthetic rejection")
    assert runtime._skill_decision_origin((request, runtime.high_level.model), decision) is None


def test_same_action_later_admission_does_not_relabel_existing_run(runtime: Any) -> None:
    first, request_a = _prepared_decision(runtime, "request-a")
    _consume(runtime, first, request_a)
    original_run = runtime.executor.run
    assert original_run is not None
    first_start = runtime.on_skill_run_started.call_args.kwargs
    assert first_start["origin"].request is request_a.binding
    assert first_start["source"] == SkillStartSource.COGNITION

    second, request_b = _prepared_decision(runtime, "request-b")
    _consume(runtime, second, request_b)

    assert request_a.snapshot().disposition == request_b.snapshot().disposition == "accepted"
    assert runtime._last_decision is second
    assert runtime.executor.run is original_run
    runtime.on_skill_run_started.assert_called_once()
    runtime._record_terminal_run(_terminal(original_run))
    terminal = runtime.on_skill_run_terminal.call_args.kwargs
    assert terminal["run"].run_id == first_start["run"].run_id
    assert first_start["origin"].request.request_id == "request-a"
    assert terminal["outcome_verification"] is None


def test_replacement_records_old_terminal_before_new_start(runtime: Any) -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    runtime.on_skill_run_started.side_effect = lambda **kw: events.append(("start", kw))
    runtime.on_skill_run_terminal.side_effect = lambda **kw: events.append(("terminal", kw))
    first, request_a = _prepared_decision(runtime, "request-a")
    _consume(runtime, first, request_a)
    second, request_b = _prepared_decision(runtime, "request-b", skill_id="probe-b")
    _consume(runtime, second, request_b)

    assert [kind for kind, _ in events] == ["start", "terminal", "start"]
    old_start, old_terminal, new_start = (event for _, event in events)
    assert old_start["run"].run_id == old_terminal["run"].run_id
    assert old_terminal["run"].outcome == SkillOutcome.CANCELLED
    assert old_start["origin"].request is request_a.binding
    assert new_start["origin"].request is request_b.binding
    assert new_start["run"].run_id != old_start["run"].run_id


@pytest.mark.parametrize("attempt_error", [None, "RuntimeError"])
def test_adopted_non_model_fallback_starts_without_borrowing_attempt(
    runtime: Any, attempt_error: str | None,
) -> None:
    decision, request = _prepared_decision(
        runtime, "fallback-request", model_origin=False, attempt_error=attempt_error,
    )
    _consume(runtime, decision, request)

    assert runtime.executor.run is not None
    runtime.on_skill_run_started.assert_called_once()
    assert runtime.on_skill_run_started.call_args.kwargs["origin"] is None
    assert request.snapshot().reason == "non_model_decision"
    assert request.snapshot().attempts[0].error_type == attempt_error
    runtime.high_level.model.admit_bound_decision.assert_not_called()


def test_rejected_publication_creates_no_start(runtime: Any) -> None:
    decision, request = _prepared_decision(runtime, "rejected-publication")
    runtime.high_level.model.admit_bound_decision.side_effect = RuntimeError("synthetic rejection")
    _consume(runtime, decision, request)
    assert request.snapshot().disposition == "publication_failed"
    assert runtime.executor.run is None
    runtime.on_skill_run_started.assert_not_called()


@pytest.mark.parametrize("changed", ["request_id", "attempt_id", "final_digest"])
def test_accepted_request_does_not_lend_origin_to_mismatched_decision(
    runtime: Any, changed: str,
) -> None:
    decision, request = _prepared_decision(runtime, "accepted-request")
    source = decision.model_origin
    assert source is not None
    assert request.accept(source.attempt_id, cognition_decision_sha256(decision), lambda: None)
    if changed == "final_digest":
        decision = decision.model_copy(update={"skill_parameters": {"distance": 2}})
    else:
        decision._model_origin = DecisionModelOrigin(
            "different-request" if changed == "request_id" else source.request_id,
            "different-attempt" if changed == "attempt_id" else source.attempt_id,
            source.source_decision_sha256,
        )
    assert runtime._skill_decision_origin((request, runtime.high_level.model), decision) is None


def test_idle_admission_creates_no_skill_lifecycle(runtime: Any) -> None:
    decision, request = _prepared_decision(runtime, "idle-request", skill_id=None)
    _consume(runtime, decision, request)
    assert request.snapshot().disposition == "accepted"
    assert runtime.executor.run is None
    runtime.on_skill_run_started.assert_not_called()


@pytest.mark.parametrize("source", [
    SkillStartSource.RECOVERY, SkillStartSource.CONTINUATION,
    SkillStartSource.KEEPALIVE, SkillStartSource.BOOTSTRAP,
])
def test_non_cognition_start_has_explicit_source_without_inherited_origin(
    runtime: Any, source: SkillStartSource,
) -> None:
    runtime._last_decision, _ = _prepared_decision(runtime, "unrelated-latest-decision")
    run = runtime._start_skill(
        runtime.skills.get("probe-a"), source=source,
        parent_run_id="actual-parent", run_id="child-run", context_key="same-task",
    )
    event = runtime.on_skill_run_started.call_args.kwargs
    assert event["run"] == run and event["run"] is not run
    assert event["source"] == source
    assert event["origin"] is None
    assert event["parent_run_id"] == "actual-parent"


def test_failed_executor_start_emits_no_start_hook(runtime: Any) -> None:
    runtime.executor.start(runtime.skills.get("probe-a"), run_id="already-running")
    with pytest.raises(RuntimeError, match="already running"):
        runtime._start_skill(
            runtime.skills.get("probe-b"), source=SkillStartSource.COGNITION, run_id="not-started",
        )
    runtime.on_skill_run_started.assert_not_called()
    assert runtime.executor.run.run_id == "already-running"


def test_start_hook_cannot_mutate_executor_run_parameters(runtime: Any) -> None:
    def mutate(*, run: SkillRun, **_: object) -> None:
        run.parameters["distance"] = 999

    runtime.on_skill_run_started.side_effect = mutate
    run = runtime._start_skill(
        runtime.skills.get("probe-a"), source=SkillStartSource.COGNITION,
        run_id="mutation-start", parameters={"distance": 1},
    )
    assert run.parameters == runtime.executor.parameters == {"distance": 1}


def test_duplicate_unmatched_terminal_is_reported_once_without_invented_start(runtime: Any) -> None:
    runtime._last_decision, _ = _prepared_decision(runtime, "unrelated-current-decision")
    run = _terminal(SkillRun(run_id="unmatched", skill_id="probe-a", started_ns=1))
    runtime._record_terminal_run(run)
    runtime._record_terminal_run(run)
    runtime.on_skill_run_started.assert_not_called()
    runtime.on_skill_run_terminal.assert_called_once()
    event = runtime.on_skill_run_terminal.call_args.kwargs
    assert set(event) == {"run", "outcome_verification"}
    assert event["run"] == run and event["run"] is not run
    assert event["outcome_verification"] is None
    assert runtime.metrics.skill_cancellations == 1


@pytest.mark.parametrize("matching", [False, True])
def test_terminal_passes_only_verification_for_its_exact_run(runtime: Any, matching: bool) -> None:
    run = _terminal(SkillRun(run_id="terminal", skill_id="probe-a", started_ns=1),
                    outcome=SkillOutcome.FAILED)
    verification = _verification(run.run_id if matching else "different-run")
    runtime._record_terminal_run(run, outcome_verification=verification)
    event = runtime.on_skill_run_terminal.call_args.kwargs
    assert event["outcome_verification"] == (verification if matching else None)


def test_terminal_hook_cannot_mutate_recorded_history(runtime: Any) -> None:
    def mutate(*, run: SkillRun, **_: object) -> None:
        run.parameters["distance"] = 999

    runtime.on_skill_run_terminal.side_effect = mutate
    run = _terminal(SkillRun(
        run_id="terminal-mutation", skill_id="probe-a", started_ns=1,
        parameters={"distance": 1},
    ))
    runtime._record_terminal_run(run)
    assert run.parameters == {"distance": 1}
    assert runtime._recent_skill_runs[0].parameters == {"distance": 1}


def test_terminal_hook_runs_without_database(runtime: Any) -> None:
    runtime.state_db = None
    run = _terminal(SkillRun(run_id="without-db", skill_id="probe-a", started_ns=1))
    runtime._record_terminal_run(run)
    runtime.on_skill_run_terminal.assert_called_once()


def test_observer_failures_do_not_change_execution_or_leak_messages(
    runtime: Any, caplog: pytest.LogCaptureFixture,
) -> None:
    runtime.on_skill_run_started.side_effect = RuntimeError("PRIVATE-START-DETAIL")
    runtime.on_skill_run_terminal.side_effect = ValueError("PRIVATE-TERMINAL-DETAIL")
    run = runtime._start_skill(
        runtime.skills.get("probe-a"), source=SkillStartSource.COGNITION, run_id="observer-errors",
    )
    assert runtime.executor.run is run and run.outcome == SkillOutcome.RUNNING
    terminal = _terminal(run)
    runtime._record_terminal_run(terminal)
    runtime._record_terminal_run(terminal)
    runtime.on_skill_run_terminal.assert_called_once()
    assert runtime.metrics.skill_cancellations == 1
    assert "RuntimeError" in caplog.text and "ValueError" in caplog.text
    assert "PRIVATE-START-DETAIL" not in caplog.text
    assert "PRIVATE-TERMINAL-DETAIL" not in caplog.text


def test_actual_headroom_children_link_stall_then_mining_not_latest_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Reuse the existing 64px synthetic-scene fixture, not a live capture/VLM.
    from test_headroom_recovery import (
        _capture,
        _install_strict_monotonic_clock,
        _mining_success,
        _publish_frame,
        _publish_headroom_answer,
        _runtime_for_probe,
        _stall_result,
    )

    forbidden = Mock(side_effect=AssertionError("headroom lineage attempted supervisor I/O"))
    monkeypatch.setattr(runtime_module, "send_command", forbidden)
    monkeypatch.setattr(runtime_module, "operator_pause_latched", lambda: False)
    monkeypatch.setattr(runtime_module, "emergency_stop_latched", lambda: False)
    _install_strict_monotonic_clock(monkeypatch)
    runtime, perception, _ = _runtime_for_probe()
    runtime.on_skill_run_started = Mock()
    stalled = _stall_result("exact-obstacle-stall")
    assert runtime._route_headroom_terminal(stalled)
    recovery = runtime._headroom_recovery
    assert recovery is not None
    runtime._advance_headroom_recovery()
    runtime._advance_headroom_recovery()
    assert recovery.phase == "settle"
    for index in range(3):
        frame_id = perception.last_capture.frame_id + 1
        captured_ns = time.monotonic_ns()
        frame = (
            _capture(frame_id=frame_id, captured_ns=captured_ns, changed=True)
            if index == 0 else replace(
                perception.last_capture, frame_id=frame_id, captured_ns=captured_ns,
            )
        )
        perception.last_capture = frame
        _publish_frame(runtime.blackboard, frame)
        runtime._advance_headroom_recovery()
        if recovery.phase == "grounding":
            break
    assert recovery.phase == "grounding"
    _publish_headroom_answer(runtime.blackboard, recovery, perception.last_capture)
    perception.available = True
    runtime._advance_headroom_recovery()
    assert recovery.phase == "mining"
    mining_event = runtime.on_skill_run_started.call_args.kwargs
    assert mining_event["source"] == SkillStartSource.RECOVERY
    assert mining_event["parent_run_id"] == stalled.run.run_id
    assert mining_event["origin"] is None

    # Supply the existing synthetic terminal result; this tests lineage, not
    # physical mining or a new claim of verified gameplay.
    completed = _mining_success(mining_event["run"].run_id)
    runtime.executor._run = completed.run
    assert runtime._route_headroom_terminal(completed)
    retry_event = runtime.on_skill_run_started.call_args.kwargs
    assert runtime.on_skill_run_started.call_count == 2
    assert recovery.phase == "retry"
    assert retry_event["source"] == SkillStartSource.RECOVERY
    assert retry_event["parent_run_id"] == mining_event["run"].run_id
    assert retry_event["origin"] is None
    assert retry_event["run"].run_id != mining_event["run"].run_id
    forbidden.assert_not_called()
