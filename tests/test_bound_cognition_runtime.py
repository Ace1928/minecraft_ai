"""Bound cognition publication tests using manual Futures and metadata-only fakes."""
from __future__ import annotations

import concurrent.futures
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

import minecraft_ai.runtime as runtime_module
from minecraft_ai.action_levels import ActionLevel
from minecraft_ai.builtin_skills import build_bootstrap_skill_library
from minecraft_ai.cognition import (
    CognitionContext,
    CognitionDecision,
    DecisionModelOrigin,
    HighLevelController,
    cognition_decision_sha256,
)
from minecraft_ai.model_requests import ModelRequestLifecycle, RequestBinding
from minecraft_ai.models import ModelResponse
from minecraft_ai.perception import FrameState, PerceptionBlackboard, ScreenRegion, Track
from minecraft_ai.roles import get_role
from minecraft_ai.runtime import AgentRuntime
from minecraft_ai.skills import SkillOutcome
from minecraft_ai.social import OperatorMessage, OperatorMessageStatus
from minecraft_ai.storage import OperatorContextSnapshot, StateDatabase


class _IntentBarrier:
    held = False

    @contextmanager
    def lock(self, *, timeout_s: float) -> Iterator[None]:
        assert timeout_s > 0 and not self.held
        self.held = True
        try:
            yield
        finally:
            self.held = False


class _Database:
    revision = 7
    held = False
    revision_matches = True

    def __init__(self) -> None:
        self.history: tuple[OperatorMessage, ...] = ()
        self.loads: list[tuple[frozenset[OperatorMessageStatus], int, bool]] = []
        self.before_exit: Callable[[], None] | None = None
        self.exit_error: Exception | None = None

    def load_operator_context(
        self, *, statuses: set[OperatorMessageStatus], limit: int,
    ) -> OperatorContextSnapshot:
        self.loads.append((frozenset(statuses), limit, self.held))
        messages = tuple(message for message in self.history if message.status in statuses)[:limit]
        return OperatorContextSnapshot(self.revision, messages, None)

    @contextmanager
    def admit_operator_revision(self, revision: int) -> Iterator[bool]:
        assert not self.held
        self.held = True
        try:
            yield self.revision_matches and self.revision == revision
            if self.before_exit is not None:
                self.before_exit()
            if self.exit_error is not None:
                raise self.exit_error
        finally:
            self.held = False


class _Adapter:
    def __init__(self, intent: _IntentBarrier, database: _Database) -> None:
        self.intent = intent
        self.database = database
        self.published: list[dict[str, Any]] = []
        self.discarded: list[dict[str, Any]] = []
        self.publication_error: Exception | None = None
        self.publication_result: Any = None
        self.active_publication: dict[str, Any] | None = None
        self._previous_publication: dict[str, Any] | None = None

    def complete_bound_constrained(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("capture tests never invoke inference")

    def admit_bound_decision(self, **receipt: Any) -> Any:
        assert self.intent.held and self.database.held
        if self.publication_error is not None:
            raise self.publication_error
        self.published.append(receipt)
        self._previous_publication = self.active_publication
        self.active_publication = receipt
        return self.publication_result

    def discard_bound_request(self, **receipt: Any) -> None:
        self.discarded.append(receipt)
        if (receipt["reason"] == "publication_failed"
                and self.active_publication is not None
                and self.active_publication.get("request") == receipt["request"]):
            self.active_publication = self._previous_publication


class _Controller:
    def __init__(self, model: _Adapter) -> None:
        self.model = model
        self.before_finish: Callable[[ModelRequestLifecycle], None] | None = None
        self.error: Exception | None = None
        self.requests: list[ModelRequestLifecycle] = []

    def decide(
        self, snapshot: Any, context: CognitionContext, *, request: ModelRequestLifecycle,
    ) -> CognitionDecision:
        self.requests.append(request)
        assert snapshot.source_sha256 == request.binding.source_sha256
        attempt = request.start_attempt("synthetic_model")
        try:
            if self.before_finish is not None:
                self.before_finish(request)
            if self.error is not None:
                raise self.error
        except Exception as error:
            request.finish_attempt(attempt, type(error).__name__)
            raise
        request.finish_attempt(attempt)
        result = CognitionDecision(skill_id="walk")
        result._model_origin = DecisionModelOrigin(
            request.binding.request_id, attempt, cognition_decision_sha256(result),
        )
        return result


class _ManualPool:
    """Exercise real Future transitions and callbacks without starting threads."""

    def __init__(self) -> None:
        self.jobs: list[tuple[
            concurrent.futures.Future[CognitionDecision], Callable[[], CognitionDecision],
        ]] = []
        self.submission_error: Exception | None = None

    def submit(
        self, operation: Callable[[], CognitionDecision],
    ) -> concurrent.futures.Future[CognitionDecision]:
        if self.submission_error is not None:
            raise self.submission_error
        future: concurrent.futures.Future[CognitionDecision] = concurrent.futures.Future()
        self.jobs.append((future, operation))
        return future

    def run(self, index: int = 0) -> None:
        future, operation = self.jobs[index]
        if not future.set_running_or_notify_cancel():
            return
        try:
            result = operation()
        except BaseException as error:
            future.set_exception(error)
        else:
            future.set_result(result)


@dataclass
class _Harness:
    runtime: Any
    adapter: _Adapter
    controller: _Controller
    pool: _ManualPool
    database: _Database
    board: PerceptionBlackboard
    clock: list[int]
    flags: dict[str, bool]
    adopted: list[CognitionDecision]

    def submit(self) -> concurrent.futures.Future[CognitionDecision]:
        context = CognitionContext(
            role=get_role("generalist"), goals=(), memories=(), promises=(), wiki=(),
        )
        return self.runtime._submit_bound_cognition(
            context, self.board.cognition_snapshot(now_ns=self.clock[0]),
            operator_revision=self.database.revision,
        )

    def completed_candidate(
        self, *, attempts: int = 1,
    ) -> tuple[ModelRequestLifecycle, CognitionDecision]:
        snapshot = self.board.cognition_snapshot(now_ns=self.clock[0])
        binding = RequestBinding.from_snapshot(
            snapshot, request_id="candidate", operator_revision=self.database.revision,
            execution_revision=self.runtime._execution_revision,
            submitted_ns=self.clock[0], deadline_ns=self.clock[0] + 1_000_000,
        )
        request = ModelRequestLifecycle(binding, clock_ns=lambda: self.clock[0])
        for _ in range(attempts):
            selected = request.start_attempt("synthetic_model")
            request.finish_attempt(selected)
        request.mark_computation_complete()
        decision = CognitionDecision(skill_id="walk", plan_steps=("Observe nearby terrain",))
        decision._model_origin = DecisionModelOrigin(
            binding.request_id, selected, cognition_decision_sha256(decision),
        )
        return request, decision

    def admit(self, request: ModelRequestLifecycle, decision: CognitionDecision) -> bool:
        return self.runtime._admit_bound_cognition(
            (request, self.adapter), decision, adopt_plan=True,
        )


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> _Harness:
    now = time.monotonic_ns()
    clock = [now]
    flags = {"pause": False, "emergency": False, "feasible": True}
    intent = _IntentBarrier()
    database = _Database()
    adapter = _Adapter(intent, database)
    controller = _Controller(adapter)
    pool = _ManualPool()
    board = PerceptionBlackboard()
    board.publish(FrameState(
        frame_id=1, captured_ns=now, instance_id="world", width=16, height=16,
    ))
    runtime: Any = AgentRuntime.__new__(AgentRuntime)
    runtime.high_level = controller
    runtime.state_db = database
    runtime.blackboard = board
    runtime._execution_revision = 11
    runtime.cognition_request_timeout_ms = 1000
    runtime._bound_cognition_requests = {}
    runtime._pool = pool
    runtime._stop = threading.Event()
    runtime._input_release_pending_ns = None
    runtime._last_decision = None
    runtime._plan_steps = ()
    runtime._plan_goal_id = None
    runtime._plan_index = 0
    runtime._plan_started_ns = 0
    runtime._pending_decision = None
    runtime.skills = SimpleNamespace(specs={"walk": object()}, stats={}, get=lambda name: name)
    adopted: list[CognitionDecision] = []
    runtime._adopt_plan_if_revised = adopted.append
    monkeypatch.setattr(runtime_module, "operator_intent_lock", intent.lock)
    monkeypatch.setattr(runtime_module, "operator_pause_latched", lambda: flags["pause"])
    monkeypatch.setattr(runtime_module, "emergency_stop_latched", lambda: flags["emergency"])
    monkeypatch.setattr(runtime_module, "initiation_satisfied", lambda *args: flags["feasible"])
    monkeypatch.setattr(runtime_module, "time", SimpleNamespace(monotonic_ns=lambda: clock[0]))
    return _Harness(runtime, adapter, controller, pool, database, board, clock, flags, adopted)


def test_queued_cancellation_rejects_before_synchronous_done_callback(harness: _Harness) -> None:
    future = harness.submit()
    request, _ = harness.runtime._bound_cognition_requests[future]
    harness.runtime._reject_bound_cognition(future, "operator_preempted")
    assert request.snapshot().computation_completed_ns is None
    assert not harness.adapter.discarded
    assert future.cancel()
    assert not harness.runtime._bound_cognition_requests
    state = request.snapshot()
    assert state.reason == "operator_preempted" and state.attempts == ()
    assert state.computation_completed_ns is not None
    assert harness.adapter.discarded == [
        {"request": request.binding, "reason": "operator_preempted"},
    ]
    assert not harness.adapter.published and not harness.adopted


def test_external_queued_cancellation_has_zero_attempts_and_one_discard(harness: _Harness) -> None:
    future = harness.submit()
    request, _ = harness.runtime._bound_cognition_requests[future]
    assert future.cancel()
    harness.runtime._reject_bound_cognition(future, "later_poll")
    assert request.snapshot().attempts == ()
    assert request.snapshot().reason == "cancelled_before_start"
    assert len(harness.adapter.discarded) == 1 and not harness.adapter.published


def test_detached_running_work_keeps_exact_future_accounting(harness: _Harness) -> None:
    old_future = harness.submit()
    old_request, _ = harness.runtime._bound_cognition_requests[old_future]
    replacement = harness.submit()
    replacement_request, _ = harness.runtime._bound_cognition_requests[replacement]
    harness.runtime._pending_decision = replacement

    def detach(request: ModelRequestLifecycle) -> None:
        assert request is old_request
        assert request.snapshot().attempts[0].finished_ns is None
        harness.runtime._reject_bound_cognition(old_future, "operator_preempted")
        assert not old_future.cancel()  # Running work was not canceled.
        assert not harness.adapter.discarded
        assert request.snapshot().computation_completed_ns is None

    harness.controller.before_finish = detach
    harness.pool.run(0)
    assert old_future.done() and not old_future.cancelled()
    old_state = old_request.snapshot()
    assert old_state.attempts[0].finished_ns is not None
    assert old_state.computation_completed_ns is not None
    assert old_state.reason == "operator_preempted"
    assert harness.adapter.discarded == [
        {"request": old_request.binding, "reason": "operator_preempted"},
    ]
    assert harness.runtime._pending_decision is replacement
    assert harness.runtime._bound_cognition_requests[replacement][0] is replacement_request
    assert replacement_request.snapshot().disposition == "pending"
    assert replacement_request.snapshot().attempts == ()
    assert not replacement.done() and not harness.adapter.published


def test_successful_done_callback_accounts_without_publishing(harness: _Harness) -> None:
    future = harness.submit()
    request, _ = harness.runtime._bound_cognition_requests[future]
    harness.pool.run()
    assert future.result().model_origin is not None
    assert request.snapshot().computation_completed_ns is not None
    assert request.snapshot().disposition == "pending"
    assert not harness.adapter.published and not harness.adapter.discarded
    assert harness.runtime._last_decision is None and not harness.adopted


def test_worker_exception_accounts_paid_attempt_then_discards(harness: _Harness) -> None:
    harness.controller.error = ValueError("synthetic model failure")
    future = harness.submit()
    request, _ = harness.runtime._bound_cognition_requests[future]
    harness.pool.run()
    with pytest.raises(ValueError, match="synthetic model failure"):
        future.result()
    state = request.snapshot()
    assert state.reason == "computation_failed" and state.computation_completed_ns is not None
    assert len(state.attempts) == 1 and state.attempts[0].error_type == "ValueError"
    assert len(harness.adapter.discarded) == 1 and not harness.adapter.published


def test_queued_deadline_expiry_never_enters_controller(harness: _Harness) -> None:
    future = harness.submit()
    request, _ = harness.runtime._bound_cognition_requests[future]
    harness.clock[0] = request.binding.deadline_ns
    harness.pool.run()
    with pytest.raises(RuntimeError, match="expired while queued"):
        future.result()
    assert request.snapshot().reason == "deadline_before_start"
    assert request.snapshot().attempts == () and not harness.controller.requests
    assert len(harness.adapter.discarded) == 1 and not harness.adapter.published


def test_submission_failure_finishes_and_discards_without_a_future(harness: _Harness) -> None:
    harness.pool.submission_error = RuntimeError("synthetic closed executor")
    with pytest.raises(RuntimeError, match="synthetic closed executor"):
        harness.submit()
    assert not harness.runtime._bound_cognition_requests and not harness.pool.jobs
    assert len(harness.adapter.discarded) == 1
    assert harness.adapter.discarded[0]["reason"] == "submission_failed"
    assert not harness.adapter.published


@pytest.mark.parametrize("change", [
    "operator", "execution", "deadline", "stop", "pause", "emergency", "release_pending",
    "model", "instance", "no_frame", "preconditions", "unknown_skill",
])
@pytest.mark.parametrize("model_origin", [True, False])
def test_final_admission_rechecks_current_authority(
    harness: _Harness, change: str, model_origin: bool,
) -> None:
    request, decision = harness.completed_candidate()
    if not model_origin:
        decision._model_origin = None
    if change == "operator":
        harness.database.revision += 1
    elif change == "execution":
        harness.runtime._execution_revision += 1
    elif change == "deadline":
        harness.clock[0] = request.binding.deadline_ns
    elif change == "stop":
        harness.runtime._stop.set()
    elif change in {"pause", "emergency"}:
        harness.flags[change] = True
    elif change == "release_pending":
        harness.runtime._input_release_pending_ns = harness.clock[0]
    elif change == "model":
        harness.runtime.high_level = SimpleNamespace(model=object())
    elif change == "instance":
        # A blackboard belongs to one instance. A replacement world owns a new
        # board; the outstanding request still refers to the original world.
        harness.clock[0] += 1
        harness.board = PerceptionBlackboard()
        harness.board.publish(FrameState(
            frame_id=1, captured_ns=harness.clock[0], instance_id="other-world",
            width=16, height=16,
        ))
        harness.runtime.blackboard = harness.board
    elif change == "no_frame":
        harness.runtime.blackboard = PerceptionBlackboard()
    elif change == "preconditions":
        harness.flags["feasible"] = False
    elif change == "unknown_skill":
        decision = decision.model_copy(update={"skill_id": "unknown"})
    assert not harness.admit(request, decision)
    expected_disposition = "publication_failed" if model_origin else "rejected"
    assert request.snapshot().disposition == expected_disposition
    assert len(harness.adapter.discarded) == 1
    assert not harness.adapter.published and not harness.adopted
    assert harness.runtime._last_decision is None


def test_admission_selects_exact_attempt_and_reports_authority_rewrite(harness: _Harness) -> None:
    request, source = harness.completed_candidate(attempts=2)
    decision = source.model_copy(update={"instruction": "Follow current operator bounds"})
    assert harness.admit(request, decision)
    receipt, = harness.adapter.published
    assert receipt["request"] is request.binding
    assert receipt["attempt_id"] == "candidate:attempt-2"
    assert receipt["source_decision_sha256"] == cognition_decision_sha256(source)
    assert receipt["final_decision_sha256"] == cognition_decision_sha256(decision)
    assert receipt["rewritten"] is True
    assert receipt["final_decision"] == decision.model_dump(mode="json")
    assert "model_origin" not in receipt["final_decision"]
    assert harness.runtime._last_decision is decision and harness.adopted == [decision]
    assert request.snapshot().disposition == "accepted" and not harness.adapter.discarded


@pytest.mark.parametrize("replan", [False, True])
def test_full_consume_publishes_exact_missing_referent_rewrite_without_adopting_plan(
    harness: _Harness, replan: bool,
) -> None:
    runtime = harness.runtime
    request, original = harness.completed_candidate()
    source = original.model_copy(update={
        "ask_perception": ("target.visible", "obstacle.ahead"),
        "instruction": " \n\t ", "request_replan": replan,
        "chosen_goal_id": "new-goal", "skill_parameters": {"allow_attack": False},
    })
    assert original.model_origin is not None
    source._model_origin = DecisionModelOrigin(
        original.model_origin.request_id, original.model_origin.attempt_id,
        cognition_decision_sha256(source),
    )
    future: concurrent.futures.Future[CognitionDecision] = concurrent.futures.Future()
    future.set_result(source)
    runtime._pending_decision = future
    runtime._bound_cognition_requests[future] = request, harness.adapter
    runtime._pending_execution_revision = runtime._execution_revision
    runtime._pending_operator_message_ids = ()
    runtime._cognition_requested = False
    runtime._plan_steps, runtime._plan_goal_id = ("retain old plan",), "old-goal"
    runtime.executor = SimpleNamespace(run=None)
    runtime._operator_message_arrived_after_snapshot = Mock(return_value=False)
    runtime._queued_operator_message_waiting = Mock(return_value=False)
    runtime._start_skill = Mock(side_effect=AssertionError("no skill authorized"))
    runtime._consume_cognition_decision()

    expected = source.model_copy(update={"skill_id": None, "ask_perception": ()})
    receipt, = harness.adapter.published
    assert receipt["source_decision_sha256"] == cognition_decision_sha256(source)
    assert receipt["final_decision"] == expected.model_dump(mode="json")
    assert receipt["final_decision_sha256"] == cognition_decision_sha256(expected)
    assert receipt["rewritten"] is True
    assert request.snapshot().disposition == "accepted" and not harness.adapter.discarded
    assert runtime._last_decision == expected and not harness.adopted
    assert (runtime._plan_steps, runtime._plan_goal_id) == (("retain old plan",), "old-goal")
    assert runtime._cognition_perception_probe is None
    assert runtime._cognition_requested is replan
    runtime._start_skill.assert_not_called()


def test_new_ordinary_frame_does_not_perpetually_invalidate_slow_cognition(
    harness: _Harness,
) -> None:
    request, decision = harness.completed_candidate()
    harness.clock[0] += 1
    harness.board.publish(FrameState(
        frame_id=2, captured_ns=harness.clock[0], instance_id="world", width=16, height=16,
    ))
    assert request.binding.frame_id == 1
    assert harness.board.raw_latest().captured_ns > request.binding.captured_ns
    assert harness.admit(request, decision)
    assert harness.adapter.published[0]["rewritten"] is False


def test_repeated_admission_does_not_publish_or_adopt_twice(harness: _Harness) -> None:
    request, decision = harness.completed_candidate()
    assert harness.admit(request, decision)
    assert harness.admit(request, decision)
    assert len(harness.adapter.published) == 1 and harness.adopted == [decision]


def test_non_model_fallback_discards_prior_attempts_without_publishing(harness: _Harness) -> None:
    request, _ = harness.completed_candidate(attempts=2)
    fallback = CognitionDecision(request_replan=True, reasoning_summary="Model output unavailable")
    assert harness.admit(request, fallback)
    assert request.snapshot().reason == "non_model_decision"
    assert len(request.snapshot().attempts) == 2
    assert not harness.adapter.published and len(harness.adapter.discarded) == 1
    assert harness.runtime._last_decision is fallback and harness.adopted == [fallback]


def test_previously_rejected_fallback_cannot_be_adopted(harness: _Harness) -> None:
    request, _ = harness.completed_candidate()
    request.reject("operator_preempted")
    assert not harness.admit(request, CognitionDecision(request_replan=True))
    assert request.snapshot().reason == "operator_preempted"
    assert not harness.adapter.published and not harness.adopted
    assert harness.runtime._last_decision is None


@pytest.mark.parametrize("mismatch", ["request", "attempt"])
def test_wrong_selected_origin_cannot_publish(harness: _Harness, mismatch: str) -> None:
    request, decision = harness.completed_candidate()
    origin = decision.model_origin
    assert origin is not None
    decision._model_origin = DecisionModelOrigin(
        "another-request" if mismatch == "request" else origin.request_id,
        "candidate:attempt-999" if mismatch == "attempt" else origin.attempt_id,
        origin.source_decision_sha256,
    )
    assert not harness.admit(request, decision)
    assert not harness.adapter.published and not harness.adopted
    assert len(harness.adapter.discarded) == 1


def test_publication_error_is_terminal_and_does_not_adopt_decision(harness: _Harness) -> None:
    request, decision = harness.completed_candidate()
    harness.adapter.publication_error = RuntimeError("synthetic publication failure")
    assert not harness.admit(request, decision)
    assert request.snapshot().disposition == "publication_failed"
    assert request.snapshot().publication_error_type == "RuntimeError"
    assert not harness.adapter.published and not harness.adopted
    assert harness.runtime._last_decision is None and len(harness.adapter.discarded) == 1


@pytest.mark.parametrize("model_origin", [True, False])
@pytest.mark.parametrize("failure", ["database_exit", "plan_adoption"])
def test_failed_publication_restores_exact_prior_decision_and_plan(
    harness: _Harness, model_origin: bool, failure: str,
) -> None:
    request, source = harness.completed_candidate()
    decision = source.model_copy(update={"chosen_goal_id": "new-goal"})
    if not model_origin:
        decision._model_origin = None
    runtime = harness.runtime
    previous_decision = CognitionDecision(chosen_goal_id="prior-goal")
    previous_steps = ("Previous step one", "Previous step two", "Previous step three")
    previous = (previous_decision, previous_steps, "prior-goal", 1, 123)
    (
        runtime._last_decision, runtime._plan_steps, runtime._plan_goal_id,
        runtime._plan_index, runtime._plan_started_ns,
    ) = previous
    previous_publication = {"request": "previous-request", "decision": "previous-decision"}
    harness.adapter.active_publication = previous_publication
    adoption_calls: list[CognitionDecision] = []
    exit_checks: list[str] = []

    def adopt_then_maybe_fail(accepted: CognitionDecision) -> None:
        assert harness.database.held and harness.adapter.intent.held
        AgentRuntime._adopt_plan_if_revised(runtime, accepted)
        adoption_calls.append(accepted)
        assert runtime._last_decision is accepted
        assert runtime._plan_steps == accepted.plan_steps
        assert runtime._plan_goal_id == "new-goal" and runtime._plan_index == 0
        assert runtime._plan_started_ns == harness.clock[0]
        if failure == "plan_adoption":
            raise ValueError("synthetic failure after plan metadata changed")

    def check_before_database_exit() -> None:
        assert harness.database.held and harness.adapter.intent.held
        # Neither model acceptance nor fallback rejection precedes commit.
        assert request.snapshot().disposition == "pending"
        exit_checks.append("pending")

    runtime._adopt_plan_if_revised = adopt_then_maybe_fail
    harness.database.before_exit = check_before_database_exit
    if failure == "database_exit":
        harness.database.exit_error = ValueError("synthetic database commit failure")
    assert not harness.admit(request, decision)
    assert (
        runtime._last_decision, runtime._plan_steps, runtime._plan_goal_id,
        runtime._plan_index, runtime._plan_started_ns,
    ) == previous
    assert runtime._last_decision is previous_decision
    assert runtime._plan_steps is previous_steps
    assert adoption_calls == [decision]
    assert exit_checks == (["pending"] if failure == "database_exit" else [])
    assert not harness.database.held and not harness.adapter.intent.held
    state = request.snapshot()
    if model_origin:
        assert state.disposition == "publication_failed"
        assert state.publication_error_type == "ValueError"
        assert len(harness.adapter.published) == 1
        expected_reason = "publication_failed"
    else:
        assert state.disposition == "rejected" and state.reason == "ValueError"
        assert state.selected_attempt_id is None and state.publication_error_type is None
        assert not harness.adapter.published
        expected_reason = "ValueError"
    assert harness.adapter.discarded == [{"request": request.binding, "reason": expected_reason}]
    assert harness.adapter.active_publication is previous_publication
    assert not harness.admit(request, decision)
    assert adoption_calls == [decision] and len(harness.adapter.discarded) == 1
    assert harness.adapter.active_publication is previous_publication


def test_fallback_rejection_waits_for_successful_database_exit(harness: _Harness) -> None:
    request, _ = harness.completed_candidate()
    fallback = CognitionDecision(request_replan=True)
    exit_checks: list[str] = []

    def check_pending_at_exit() -> None:
        assert harness.database.held and harness.adapter.intent.held
        assert request.snapshot().disposition == "pending"
        assert not harness.adapter.discarded
        exit_checks.append("pending")

    harness.database.before_exit = check_pending_at_exit
    assert harness.admit(request, fallback)
    assert exit_checks == ["pending"]
    assert request.snapshot().reason == "non_model_decision"
    assert harness.runtime._last_decision is fallback
    assert not harness.adapter.published and len(harness.adapter.discarded) == 1


def test_fallback_losing_rejection_race_restores_prior_metadata(harness: _Harness) -> None:
    request, _ = harness.completed_candidate()
    previous = CognitionDecision(chosen_goal_id="prior-goal")
    harness.runtime._last_decision = previous
    harness.runtime._plan_steps = ("Keep prior plan",)
    harness.runtime._plan_goal_id = "prior-goal"
    harness.runtime._plan_index = 0
    harness.runtime._plan_started_ns = 123
    harness.runtime._adopt_plan_if_revised = lambda decision: AgentRuntime._adopt_plan_if_revised(
        harness.runtime, decision,
    )
    fallback = CognitionDecision(chosen_goal_id="fallback-goal", plan_steps=("Replacement plan",))

    def reject_at_exit() -> None:
        assert request.reject("operator_preempted")

    harness.database.before_exit = reject_at_exit
    assert not harness.admit(request, fallback)
    assert request.snapshot().reason == "operator_preempted"
    assert harness.runtime._last_decision is previous
    assert harness.runtime._plan_steps == ("Keep prior plan",)
    assert harness.runtime._plan_goal_id == "prior-goal"
    assert harness.runtime._plan_index == 0 and harness.runtime._plan_started_ns == 123
    assert not harness.adapter.published
    assert harness.adapter.discarded == [
        {"request": request.binding, "reason": "operator_preempted"},
    ]


def test_interrupted_fallback_restores_metadata_and_discards_before_propagating(
    harness: _Harness,
) -> None:
    request, _ = harness.completed_candidate()
    previous = CognitionDecision(chosen_goal_id="prior-goal")
    harness.runtime._last_decision = previous
    prior_plan = ((), None, 0, 0)
    fallback = CognitionDecision(chosen_goal_id="fallback-goal", plan_steps=("Interrupted plan",))

    def interrupt_after_adoption(decision: CognitionDecision) -> None:
        AgentRuntime._adopt_plan_if_revised(harness.runtime, decision)
        assert harness.runtime._plan_steps == ("Interrupted plan",)
        raise KeyboardInterrupt("synthetic interrupted fallback")

    harness.runtime._adopt_plan_if_revised = interrupt_after_adoption
    with pytest.raises(KeyboardInterrupt, match="synthetic interrupted fallback"):
        harness.admit(request, fallback)
    assert request.snapshot().disposition == "rejected"
    assert request.snapshot().reason == "KeyboardInterrupt"
    assert harness.runtime._last_decision is previous
    assert (
        harness.runtime._plan_steps, harness.runtime._plan_goal_id,
        harness.runtime._plan_index, harness.runtime._plan_started_ns,
    ) == prior_plan
    assert not harness.database.held and not harness.adapter.intent.held
    assert not harness.adapter.published
    assert harness.adapter.discarded == [
        {"request": request.binding, "reason": "KeyboardInterrupt"},
    ]


@pytest.mark.parametrize("result", [False, True, "accepted"])
def test_non_none_adapter_result_cannot_claim_admission(harness: _Harness, result: Any) -> None:
    request, decision = harness.completed_candidate()
    harness.adapter.publication_result = result
    assert not harness.admit(request, decision)
    assert request.snapshot().disposition == "publication_failed"
    assert request.snapshot().publication_error_type == "TypeError"
    assert not harness.adopted and harness.runtime._last_decision is None
    assert len(harness.adapter.discarded) == 1


def _capture_fixture(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[str], list[Any]]:
    runtime = harness.runtime
    runtime.role = get_role("generalist")
    runtime.custom_goals = []
    runtime.memories = SimpleNamespace(retrieve=lambda **kwargs: ())
    runtime.social = SimpleNamespace(active_promises=lambda: ())
    runtime._recent_skill_runs = ()
    runtime._plan_steps = ()
    runtime._plan_goal_id = None
    runtime._plan_index = 0
    runtime._plan_started_ns = 0
    events: list[str] = []
    snapshots: list[Any] = []
    original_snapshot = harness.board.cognition_snapshot
    original_context = runtime._cognition_context

    def merge_target() -> None:
        assert harness.database.held and harness.adapter.intent.held
        events.append("merge")

    def take_snapshot() -> Any:
        assert harness.database.held and harness.adapter.intent.held
        result = original_snapshot(now_ns=harness.clock[0])
        snapshots.append(result)
        events.append("snapshot")
        return result

    def reconcile(*, snapshot: Any) -> bool:
        assert not harness.database.held and not harness.adapter.intent.held
        assert snapshot is snapshots[0]
        events.append("reconcile")
        # A new ordinary frame cannot replace the frozen reconciliation input.
        harness.clock[0] += 1
        harness.board.publish(FrameState(
            frame_id=2, captured_ns=harness.clock[0], instance_id="world", width=16, height=16,
        ))
        assert snapshot.frame_id == 1
        assert harness.board.raw_latest().captured_ns > snapshot.captured_ns
        return True

    def make_context(operator: OperatorContextSnapshot, *, requires_wood: bool) -> CognitionContext:
        assert not harness.database.held and not harness.adapter.intent.held
        assert requires_wood is True
        events.append("context")
        return original_context(operator, requires_wood=requires_wood)

    runtime._merge_operator_target = merge_target
    runtime._planks_retry_requires_wood = reconcile
    runtime._cognition_context = make_context
    monkeypatch.setattr(harness.board, "cognition_snapshot", take_snapshot)
    return events, snapshots


@pytest.mark.parametrize("pending", [True, False])
def test_capture_filters_history_before_caps_and_reconciles_outside_authority_locks(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch, pending: bool,
) -> None:
    events, snapshots = _capture_fixture(harness, monkeypatch)
    archived = tuple(OperatorMessage(
        message_id=f"archived-{index}", created_ns=1000 + index, text="Old history",
        status=OperatorMessageStatus.ARCHIVED,
    ) for index in range(25))
    active = tuple(OperatorMessage(
        message_id=f"queued-{index}", created_ns=500 + index, text="Current instruction",
        status=OperatorMessageStatus.QUEUED,
    ) for index in range(23)) if pending else ()
    acknowledged = tuple(OperatorMessage(
        message_id=f"ack-{index}", created_ns=100 + index, text="Persistent instruction",
        status=OperatorMessageStatus.ACKNOWLEDGED,
    ) for index in range(23))
    harness.database.history = (*archived, *active, *acknowledged)
    context, snapshot, revision = harness.runtime._capture_bound_cognition_inputs()
    assert events == ["merge", "snapshot", "reconcile", "context"]
    assert snapshot is snapshots[0] and snapshot.frame_id == 1
    assert harness.board.raw_latest().frame_id == 2
    assert revision == harness.database.revision
    assert context.planks_retry_requires_wood is True
    assert 1 <= len(context.operator_messages) <= 20
    expected_status = (
        OperatorMessageStatus.QUEUED if pending else OperatorMessageStatus.ACKNOWLEDGED
    )
    assert all(message.status == expected_status for message in context.operator_messages)
    expected_loads = [(
        frozenset({OperatorMessageStatus.QUEUED, OperatorMessageStatus.DELIVERED}), 20, False,
    )]
    if not pending:
        expected_loads.append((frozenset({OperatorMessageStatus.ACKNOWLEDGED}), 20, True))
    assert harness.database.loads == expected_loads
    assert not harness.controller.requests and not harness.pool.jobs


@pytest.mark.parametrize("failure", ["revision", "snapshot"])
def test_failed_capture_never_runs_prerequisite_cleanup_or_builds_context(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    events, snapshots = _capture_fixture(harness, monkeypatch)
    if failure == "revision":
        harness.database.revision_matches = False
    else:
        def fail_snapshot() -> None:
            assert harness.database.held and harness.adapter.intent.held
            raise RuntimeError("synthetic snapshot serialization failure")
        monkeypatch.setattr(harness.board, "cognition_snapshot", fail_snapshot)
    with pytest.raises(RuntimeError):
        harness.runtime._capture_bound_cognition_inputs()
    assert "reconcile" not in events and "context" not in events
    assert not snapshots and not harness.pool.jobs
    assert not harness.database.held and not harness.adapter.intent.held


def test_optional_bound_detection_tolerates_controller_without_model(harness: _Harness) -> None:
    harness.runtime.high_level = SimpleNamespace()
    assert not harness.runtime._uses_bound_cognition()


_BOUND_HOOKS = (
    "complete_bound_constrained", "admit_bound_decision", "discard_bound_request",
)


@pytest.mark.parametrize("controller", [
    None, SimpleNamespace(model=None), SimpleNamespace(model=object()),
    SimpleNamespace(model=SimpleNamespace(complete=lambda messages: None)),
])
def test_optional_bound_detection_preserves_no_capability_legacy_mode(
    harness: _Harness, controller: object,
) -> None:
    harness.runtime.high_level = controller
    assert not harness.runtime._uses_bound_cognition()


def test_optional_bound_detection_accepts_complete_callable_capability(harness: _Harness) -> None:
    assert harness.runtime._uses_bound_cognition()
    assert not harness.adapter.published and not harness.adapter.discarded
    assert not harness.controller.requests and not harness.pool.jobs


@pytest.mark.parametrize("present", [
    (_BOUND_HOOKS[0],), (_BOUND_HOOKS[1],), (_BOUND_HOOKS[2],),
    _BOUND_HOOKS[:2], _BOUND_HOOKS[1:], (_BOUND_HOOKS[0], _BOUND_HOOKS[2]),
])
def test_optional_bound_detection_keeps_partial_capability_legacy(
    harness: _Harness, present: tuple[str, ...],
) -> None:
    hooks = {name: Mock(side_effect=AssertionError("must not invoke adapter")) for name in present}
    harness.runtime.high_level = SimpleNamespace(model=SimpleNamespace(**hooks))
    assert not harness.runtime._uses_bound_cognition()
    for hook in hooks.values():
        hook.assert_not_called()


@pytest.mark.parametrize("invalid_hook", _BOUND_HOOKS)
@pytest.mark.parametrize("invalid_value", [None, False, object()])
def test_optional_bound_detection_keeps_noncallable_capability_legacy(
    harness: _Harness, invalid_hook: str, invalid_value: object,
) -> None:
    hooks: dict[str, Any] = {
        name: Mock(side_effect=AssertionError("must not invoke adapter")) for name in _BOUND_HOOKS
    }
    hooks[invalid_hook] = invalid_value
    harness.runtime.high_level = SimpleNamespace(model=SimpleNamespace(**hooks))
    assert not harness.runtime._uses_bound_cognition()
    for name, hook in hooks.items():
        if name != invalid_hook:
            hook.assert_not_called()


def test_optional_bound_detection_keeps_null_hooks_legacy(harness: _Harness) -> None:
    harness.runtime.high_level = SimpleNamespace(
        model=SimpleNamespace(**dict.fromkeys(_BOUND_HOOKS)),
    )
    assert not harness.runtime._uses_bound_cognition()


def test_partial_capability_starts_real_controller_through_legacy_complete(
    harness: _Harness,
) -> None:
    runtime = harness.runtime
    legacy_complete = Mock(return_value=ModelResponse(
        text=CognitionDecision(reasoning_summary="Legacy idle.").model_dump_json(),
        model="synthetic-legacy", latency_ms=1.0,
    ))
    experimental_hook = Mock(side_effect=AssertionError("must not invoke partial bound hook"))
    model = SimpleNamespace(
        model_id="synthetic-legacy", complete=legacy_complete,
        complete_bound_constrained=experimental_hook,
    )
    controller = HighLevelController(model, build_bootstrap_skill_library())
    runtime.high_level = controller
    runtime.executor = SimpleNamespace(run=None)
    runtime._cognition_requested = True
    runtime.cognition_hz = 0.5
    runtime._last_cognition_ns = 0
    runtime.metrics = SimpleNamespace(cognition_calls=0)
    runtime._new_queued_operator_message_waiting = Mock(return_value=False)
    runtime._queued_operator_message_waiting = Mock(return_value=False)
    context = CognitionContext(
        role=get_role("generalist"), goals=(), memories=(), promises=(), wiki=(),
    )
    runtime._cognition_context = Mock(return_value=context)
    runtime._capture_bound_cognition_inputs = Mock(wraps=runtime._capture_bound_cognition_inputs)
    runtime._schedule_cognition_retry = Mock()

    def submit(operation: Any, *args: Any) -> concurrent.futures.Future[CognitionDecision]:
        future: concurrent.futures.Future[CognitionDecision] = concurrent.futures.Future()
        future.set_result(operation(*args))
        return future

    runtime._pool = SimpleNamespace(submit=Mock(side_effect=submit))
    runtime._start_cognition_if_due()
    runtime._pool.submit.assert_called_once_with(controller.decide, harness.board, context)
    legacy_complete.assert_called_once()
    experimental_hook.assert_not_called()
    runtime._capture_bound_cognition_inputs.assert_not_called()
    runtime._schedule_cognition_retry.assert_not_called()
    assert runtime._pending_decision.result().reasoning_summary == "Legacy idle."
    assert runtime._pending_decision.result().model_origin is None
    assert not runtime._bound_cognition_requests


@pytest.mark.parametrize("edited_field", ["label", "region", "attributes"])
def test_same_id_operator_target_edit_updates_snapshot(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch, edited_field: str,
) -> None:
    original = Track(
        track_id="selected-target", label="log", confidence=1.0,
        region=ScreenRegion(x=0.1, y=0.1, width=0.2, height=0.2),
        first_seen_ns=harness.clock[0], last_seen_ns=harness.clock[0],
        attributes={"source": "operator"},
    )
    update: dict[str, Any] = {
        "label": "stone",
        "region": ScreenRegion(x=0.6, y=0.3, width=0.2, height=0.2),
        "attributes": {"source": "operator", "description": "updated target"},
    }
    edited = original.model_copy(update={edited_field: update[edited_field]})
    assert harness.board.upsert_semantic_track(instance_id="world", track=original)
    before = harness.board.cognition_snapshot(now_ns=harness.clock[0])
    harness.runtime._last_operator_target_id = original.track_id
    harness.runtime.perception = SimpleNamespace(instance_id="world")
    monkeypatch.setattr(harness.database, "load_operator_target", lambda: edited, raising=False)
    upserts: list[Track] = []
    original_upsert = harness.board.upsert_semantic_track

    def record_upsert(*, instance_id: str, track: Track) -> bool:
        upserts.append(track)
        return original_upsert(instance_id=instance_id, track=track)

    monkeypatch.setattr(harness.board, "upsert_semantic_track", record_upsert)
    harness.runtime._merge_operator_target()
    after = harness.board.cognition_snapshot(now_ns=harness.clock[0])
    assert after.latest().tracks == (edited,)
    assert after.source_sha256 != before.source_sha256
    assert harness.runtime._last_operator_target_id == original.track_id
    harness.runtime._merge_operator_target()
    assert upserts == [edited]  # An unchanged repeat still avoids replacement.


def _stage_crafting_consume(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch, database: StateDatabase,
) -> tuple[ModelRequestLifecycle, concurrent.futures.Future[CognitionDecision], list[str]]:
    """Keep real consume/GUI-close logic; replace only its physical side effects."""
    message = OperatorMessage(message_id="instruction", created_ns=10, text="Go to the tree.")
    target = Track(
        track_id="target", label="oak_log", confidence=1.0,
        region=ScreenRegion(x=0.2, y=0.3, width=0.1, height=0.2),
        first_seen_ns=10, last_seen_ns=10, attributes={"source": "operator"},
    )
    database.save_operator_message(message)
    database.save_operator_target(target)
    harness.database.revision = database.operator_revision()
    harness.runtime.state_db = database
    request, decision = harness.completed_candidate()
    future: concurrent.futures.Future[CognitionDecision] = concurrent.futures.Future()
    future.set_result(decision)
    runtime = harness.runtime
    runtime._pending_decision = future
    runtime._bound_cognition_requests[future] = request, harness.adapter
    runtime._pending_execution_revision = runtime._execution_revision
    runtime._pending_operator_message_ids = (message.message_id,)
    runtime._pending_operator_message_kinds = {message.message_id: message.kind}
    runtime._cognition_requested = False
    active = SimpleNamespace(
        run_id="active-craft", skill_id="craft_wood_planks", outcome=SkillOutcome.RUNNING,
    )
    effects: list[str] = []

    def effect(name: str) -> None:
        assert not harness.adapter.intent.held
        assert not database.connection.in_transaction
        effects.append(name)

    def cancel() -> SimpleNamespace:
        effect("cancel")
        return SimpleNamespace(
            run=active, action=object(), recovery_skills=("close_open_inventory",),
        )

    def recovery(*args: Any) -> object:
        effect("select_recovery")
        return object()

    runtime.executor = SimpleNamespace(run=active, cancel=cancel)
    runtime.skills = SimpleNamespace(
        specs={"walk": object()}, get=lambda name: SimpleNamespace(action_level=ActionLevel.MOTION),
    )
    runtime._send_motor = lambda *args, **kwargs: effect("actuation")
    runtime._record_terminal_run = lambda *args: effect("record_terminal")
    runtime._start_recovery_skill = lambda *args: effect("start_recovery")
    monkeypatch.setattr(runtime_module, "_first_feasible_recovery", recovery)
    return request, future, effects


@pytest.mark.parametrize("change", [
    "deadline", "fallback_deadline", "same_instruction", "same_target", "execution",
    "stop", "pause", "emergency", "release_pending", "model", "instance", "no_frame",
])
def test_full_consume_rejects_stale_authority_before_crafting_side_effects(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, change: str,
) -> None:
    with StateDatabase(tmp_path / "consume.sqlite3") as database:
        request, future, effects = _stage_crafting_consume(harness, monkeypatch, database)
        runtime = harness.runtime
        active = runtime.executor.run
        if change in {"deadline", "fallback_deadline"}:
            harness.clock[0] = request.binding.deadline_ns
            if change == "fallback_deadline":
                future.result()._model_origin = None
        elif change == "same_instruction":
            message = database.load_operator_messages()[0]
            database.save_operator_message(message.model_copy(update={"text": "Stay here."}))
        elif change == "same_target":
            target = database.load_operator_target()
            assert target is not None
            database.save_operator_target(target.model_copy(update={"label": "stone"}))
        elif change == "execution":
            runtime._execution_revision += 1
            # Even a refreshed legacy marker cannot renew the bound request.
            runtime._pending_execution_revision = runtime._execution_revision
        elif change == "stop":
            runtime._stop.set()
        elif change in {"pause", "emergency"}:
            harness.flags[change] = True
        elif change == "release_pending":
            runtime._input_release_pending_ns = harness.clock[0]
        elif change == "model":
            runtime.high_level = SimpleNamespace(model=object())
        elif change == "instance":
            runtime.blackboard = PerceptionBlackboard()
            runtime.blackboard.publish(FrameState(
                frame_id=1, captured_ns=harness.clock[0], instance_id="other-world",
                width=16, height=16,
            ))
        elif change == "no_frame":
            runtime.blackboard = PerceptionBlackboard()
        # The old ID-only guard cannot distinguish any of these changes.
        assert not runtime._operator_message_arrived_after_snapshot()
        execution_revision = runtime._execution_revision
        runtime._consume_cognition()
        assert effects == []
        assert runtime.executor.run is active and active.outcome == SkillOutcome.RUNNING
        assert runtime._execution_revision == execution_revision
        assert runtime._pending_decision is None and not runtime._bound_cognition_requests
        assert runtime._pending_operator_message_ids == ()
        assert runtime._cognition_requested
        assert runtime._last_decision is None and not harness.adopted
        assert not harness.adapter.published and len(harness.adapter.discarded) == 1
        assert request.snapshot().disposition == "rejected"
        assert request.snapshot().reason == "consumption_authority_changed"
        assert not database.connection.in_transaction and not harness.adapter.intent.held


@pytest.mark.parametrize("failure", ["intent_lock", "database_exit"])
def test_full_consume_preflight_failure_never_mutates_active_crafting(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str,
) -> None:
    with StateDatabase(tmp_path / "consume.sqlite3") as database:
        request, _, effects = _stage_crafting_consume(harness, monkeypatch, database)
        if failure == "intent_lock":
            @contextmanager
            def unavailable_lock(*, timeout_s: float) -> Iterator[None]:
                assert timeout_s == 0.05
                raise TimeoutError("synthetic authority contention")
                yield  # pragma: no cover
            monkeypatch.setattr(runtime_module, "operator_intent_lock", unavailable_lock)
        else:
            original = database.admit_operator_revision

            @contextmanager
            def failed_exit(revision: int) -> Iterator[bool]:
                with original(revision) as current:
                    yield current
                    raise RuntimeError("synthetic transaction exit failure")
            monkeypatch.setattr(database, "admit_operator_revision", failed_exit)
        harness.runtime._consume_cognition()
        assert effects == []
        assert request.snapshot().disposition == "rejected"
        assert len(harness.adapter.discarded) == 1 and not harness.adapter.published
        assert not database.connection.in_transaction and not harness.adapter.intent.held


def test_full_consume_current_bound_authority_defers_crafting_close(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    with StateDatabase(tmp_path / "consume.sqlite3") as database:
        request, _, effects = _stage_crafting_consume(harness, monkeypatch, database)

        def forbidden_close(decision: CognitionDecision) -> bool:
            raise AssertionError("bound consumption must not invoke the mutating GUI-close helper")

        harness.runtime._close_crafting_gui_before_world_decision = forbidden_close
        harness.runtime._consume_cognition()
        assert effects == []
        assert request.snapshot().disposition == "rejected"
        assert request.snapshot().reason == "crafting_gui_close_deferred"
        assert harness.runtime.executor.run.outcome == SkillOutcome.RUNNING
        assert harness.runtime._cognition_requested
        assert harness.runtime._cognition_retry_not_before_ns > harness.clock[0]
        assert harness.runtime._last_decision is None and not harness.adopted
        assert len(harness.adapter.discarded) == 1 and not harness.adapter.published
        assert not harness.runtime._bound_cognition_requests


def test_full_consume_deadline_expiring_at_preflight_exit_cannot_close_crafting(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    with StateDatabase(tmp_path / "consume.sqlite3") as database:
        request, _, effects = _stage_crafting_consume(harness, monkeypatch, database)
        original = database.admit_operator_revision
        exits: list[int] = []

        @contextmanager
        def expire_at_exit(revision: int) -> Iterator[bool]:
            with original(revision) as current:
                yield current
            # The preflight already evaluated True, but has not returned yet.
            harness.clock[0] = request.binding.deadline_ns
            exits.append(revision)

        monkeypatch.setattr(database, "admit_operator_revision", expire_at_exit)
        execution_revision = harness.runtime._execution_revision
        harness.runtime._consume_cognition()
        assert exits == [request.binding.operator_revision]
        assert effects == []
        assert harness.runtime.executor.run.outcome == SkillOutcome.RUNNING
        assert harness.runtime._execution_revision == execution_revision
        assert request.snapshot().reason == "crafting_gui_close_deferred"
        assert request.snapshot().disposition == "rejected"
        assert len(harness.adapter.discarded) == 1 and not harness.adapter.published
        assert not harness.runtime._bound_cognition_requests and not harness.adopted


@pytest.mark.parametrize("edited", ["instruction", "target"])
def test_full_consume_same_id_edit_after_preflight_cannot_close_crafting(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, edited: str,
) -> None:
    with StateDatabase(tmp_path / "consume.sqlite3") as database:
        request, _, effects = _stage_crafting_consume(harness, monkeypatch, database)
        original = harness.runtime._preflight_bound_cognition
        passed: list[bool] = []

        def edit_after_preflight(record: tuple[ModelRequestLifecycle, object]) -> bool:
            result = original(record)
            assert result and not harness.adapter.intent.held
            assert not database.connection.in_transaction
            passed.append(result)
            if edited == "instruction":
                message = database.load_operator_messages()[0]
                database.save_operator_message(message.model_copy(update={"text": "Stay here."}))
            else:
                target = database.load_operator_target()
                assert target is not None
                database.save_operator_target(target.model_copy(update={"label": "stone"}))
            assert not harness.runtime._operator_message_arrived_after_snapshot()
            return result

        harness.runtime._preflight_bound_cognition = edit_after_preflight
        execution_revision = harness.runtime._execution_revision
        harness.runtime._consume_cognition()
        assert passed == [True]
        assert database.operator_revision() == request.binding.operator_revision + 1
        assert effects == []
        assert harness.runtime.executor.run.outcome == SkillOutcome.RUNNING
        assert harness.runtime._execution_revision == execution_revision
        assert request.snapshot().reason == "crafting_gui_close_deferred"
        assert request.snapshot().disposition == "rejected"
        assert len(harness.adapter.discarded) == 1 and not harness.adapter.published
        assert not harness.runtime._bound_cognition_requests and not harness.adopted


def test_full_consume_legacy_authority_retains_crafting_close_shortcut(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    with StateDatabase(tmp_path / "consume.sqlite3") as database:
        _, future, effects = _stage_crafting_consume(harness, monkeypatch, database)
        harness.runtime._bound_cognition_requests.pop(future)
        future.result()._model_origin = None
        harness.runtime._consume_cognition()
        assert effects == [
            "cancel", "actuation", "record_terminal", "select_recovery", "start_recovery",
        ]
        assert harness.runtime._last_decision is None and not harness.adopted
        assert not harness.adapter.published and not harness.adapter.discarded
        assert not harness.runtime._bound_cognition_requests
