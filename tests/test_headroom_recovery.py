from __future__ import annotations

import time
from collections import deque
from concurrent.futures import Future
from dataclasses import replace

import pytest

from minecraft_ai.builtin_skills import build_bootstrap_skill_library
from minecraft_ai.execution import ExecutionTick, SkillExecutor
from minecraft_ai.memory import MemoryStore
from minecraft_ai.motor import BootstrapMotorPolicy
from minecraft_ai.outcome_verifier import (
    OutcomeKind,
    OutcomeSignal,
    OutcomeStatus,
    OutcomeVerification,
)
from minecraft_ai.perception import (
    ActivePerceptionQuery,
    FrameState,
    PerceptionBlackboard,
    PerceptionFact,
    ScreenRegion,
    Track,
)
from minecraft_ai.perception_service import frame_dhash
from minecraft_ai.platforms.bedrock_x11 import CapturedFrame
from minecraft_ai.runtime import (
    AgentRuntime,
    RuntimeMetrics,
    _HeadroomRecovery,
    _headroom_clear_target,
    _headroom_deadline_ns,
    _headroom_retry_advances_plan,
    _verified_headroom_retry,
    _verified_obstacle_stall,
)
from minecraft_ai.safety import MotorAction
from minecraft_ai.skills import SkillFailureCode, SkillOutcome, SkillRun


_HASH_A = "0000000000000000"
_HASH_FAR = "ffffffffffffffff"


def _stall_result(run_id: str = "obstacle-stall") -> ExecutionTick:
    run = SkillRun(
        run_id=run_id,
        skill_id="traverse_visible_obstacle",
        context_key="explore-keepalive",
        started_ns=1,
        ended_ns=2,
        outcome=SkillOutcome.FAILED,
        failure_reason=SkillFailureCode.LOCOMOTION_STALLED.value,
        failure_code=SkillFailureCode.LOCOMOTION_STALLED,
        parameters={"allow_jump": True},
    )
    return ExecutionTick(
        run=run,
        action=None,
        outcome_verification=OutcomeVerification(
            run_id=run_id,
            kind=OutcomeKind.TRAVERSAL,
            status=OutcomeStatus.STALLED,
            signal=OutcomeSignal.LOCOMOTION_STALLED,
            observed_ns=2,
            confidence=0.92,
            reason="verified no displacement",
        ),
    )


def _publish_headroom_answer(
    board: PerceptionBlackboard,
    recovery: _HeadroomRecovery,
    *,
    kind: str = "dirt",
    dx: float = 0.0,
    track_x: float = 0.4,
    mixed_key: str | None = None,
    current_hash: str = _HASH_A,
) -> None:
    assert recovery.query_id is not None
    now = max(time.monotonic_ns(), recovery.query_started_ns + 1)
    source = f"vlm:test:{recovery.query_id}"
    values: dict[str, str | int | float | bool] = {
        "scene.mode": "world",
        "scene.playable": True,
        "danger.immediate": False,
        "target.visible": True,
        "target.kind": kind,
        "target.mineable": True,
        "target.dx": dx,
        "target.dy": 0.0,
        "scene.observation_dhash": recovery.query_frame_dhash or _HASH_A,
        "frame.dhash": current_hash,
    }
    facts = tuple(
        PerceptionFact(
            key=key,
            value=value,
            confidence=1.0,
            observed_ns=now,
            source=("vlm:test:other-query" if key == mixed_key else source),
            expires_after_ms=60_000,
        )
        for key, value in values.items()
        if key != "frame.dhash"
    ) + (
        PerceptionFact(
            key="frame.dhash",
            value=current_hash,
            confidence=1.0,
            observed_ns=now,
            source="bootstrap:frame",
            expires_after_ms=60_000,
        ),
    )
    board.merge_semantics(
        instance_id="bedrock:headroom",
        facts=facts,
        tracks=(
            Track(
                track_id=f"vlm:{recovery.query_id}:0",
                label=kind,
                confidence=0.95,
                region=ScreenRegion(x=track_x, y=0.4, width=0.2, height=0.2),
                first_seen_ns=now,
                last_seen_ns=now,
            ),
        ),
    )


def _board(*, captured_ns: int | None = None) -> PerceptionBlackboard:
    now = time.monotonic_ns() if captured_ns is None else captured_ns
    board = PerceptionBlackboard()
    board.publish(
        FrameState(
            frame_id=4,
            captured_ns=now,
            instance_id="bedrock:headroom",
            width=8,
            height=8,
        )
    )
    return board


def test_headroom_trigger_requires_exact_verified_obstacle_stall() -> None:
    exact = _stall_result()
    assert _verified_obstacle_stall(exact) is True

    no_verification = replace(exact, outcome_verification=None)
    assert _verified_obstacle_stall(no_verification) is False

    level_ground = replace(
        exact,
        run=exact.run.model_copy(update={"skill_id": "traverse_level_ground"}),
    )
    assert _verified_obstacle_stall(level_ground) is False


def test_headroom_deadline_respects_serialized_vlm_timeout_budget() -> None:
    class _Model:
        timeout_s = 75.0

    class _Worker:
        model = _Model()

    now = 12_000
    assert _headroom_deadline_ns(_Worker(), now_ns=now) == now + 180_000_000_000


@pytest.mark.parametrize(
    ("kind", "dx", "track_x", "mixed_key", "current_hash"),
    (
        ("stone", 0.0, 0.4, None, _HASH_A),
        ("dirt", 0.2, 0.4, None, _HASH_A),
        ("dirt", 0.0, 0.7, None, _HASH_A),
        ("dirt", 0.0, 0.4, "target.mineable", _HASH_A),
        ("dirt", 0.0, 0.4, None, _HASH_FAR),
    ),
)
def test_headroom_target_abstains_on_unsafe_or_incoherent_evidence(
    kind: str,
    dx: float,
    track_x: float,
    mixed_key: str | None,
    current_hash: str,
) -> None:
    board = _board()
    recovery = _HeadroomRecovery(
        context_key="explore-keepalive",
        traversal_parameters={},
        deadline_ns=time.monotonic_ns() + 60_000_000_000,
        phase="grounding",
        query_id="query-one",
        query_started_ns=time.monotonic_ns() - 1_000,
        query_frame_dhash=_HASH_A,
    )
    _publish_headroom_answer(
        board,
        recovery,
        kind=kind,
        dx=dx,
        track_x=track_x,
        mixed_key=mixed_key,
        current_hash=current_hash,
    )

    assert _headroom_clear_target(board, recovery, now_ns=time.monotonic_ns()) is None


def test_headroom_target_requires_post_request_evidence() -> None:
    board = _board()
    recovery = _HeadroomRecovery(
        context_key="explore-keepalive",
        traversal_parameters={},
        deadline_ns=time.monotonic_ns() + 60_000_000_000,
        phase="grounding",
        query_id="query-one",
        query_started_ns=time.monotonic_ns() - 1_000,
        query_frame_dhash=_HASH_A,
    )
    _publish_headroom_answer(board, recovery)
    assert _headroom_clear_target(board, recovery, now_ns=time.monotonic_ns()) is not None

    recovery.query_started_ns = time.monotonic_ns() + 1
    assert _headroom_clear_target(board, recovery, now_ns=time.monotonic_ns()) is None


class _Perception:
    def __init__(self, captured: CapturedFrame) -> None:
        self.active_vlm = object()
        self.last_capture = captured
        self.available = True
        self.requests: list[tuple[ActivePerceptionQuery, CapturedFrame | None]] = []

    def semantic_available(self) -> bool:
        return self.available

    def request_semantics(
        self,
        query: ActivePerceptionQuery,
        frame: CapturedFrame | None = None,
    ) -> bool:
        self.requests.append((query, frame))
        self.available = False
        return True


def _runtime_for_probe() -> tuple[AgentRuntime, _Perception, list[MotorAction]]:
    now = time.monotonic_ns()
    pixels = bytes(range(64)) * 4
    captured = CapturedFrame(
        # Capture-source IDs and blackboard IDs are independent in production.
        frame_id=41,
        captured_ns=now,
        width=8,
        height=8,
        bgra=pixels,
    )
    board = _board(captured_ns=now)
    board.merge_semantics(
        instance_id="bedrock:headroom",
        facts=(
            PerceptionFact(
                key="frame.dhash",
                value=frame_dhash(captured),
                confidence=1.0,
                observed_ns=time.monotonic_ns(),
                source="bootstrap:frame",
                expires_after_ms=60_000,
            ),
        ),
    )
    perception = _Perception(captured)
    runtime = object.__new__(AgentRuntime)
    runtime.blackboard = board
    runtime.perception = perception  # type: ignore[assignment]
    runtime.skills = build_bootstrap_skill_library()
    runtime.executor = SkillExecutor(BootstrapMotorPolicy())
    runtime.metrics = RuntimeMetrics()
    runtime.semantic_hz = 0.0
    runtime._headroom_recovery = None
    runtime._traversal_escalation_pending = False
    runtime._execution_revision = 0
    runtime._cognition_requested = False
    runtime._pending_decision = None
    runtime._sequence = 0
    sent: list[MotorAction] = []

    def _send(action: MotorAction, **_kwargs: object) -> None:
        sent.append(action)
        runtime._sequence += 1

    runtime._send_motor = _send  # type: ignore[method-assign]
    return runtime, perception, sent


def test_verified_stall_requests_one_event_query_at_zero_semantic_hz_then_mines() -> None:
    runtime, perception, sent = _runtime_for_probe()

    assert runtime._route_headroom_terminal(_stall_result()) is True
    recovery = runtime._headroom_recovery
    assert recovery is not None
    assert recovery.phase == "reorient"
    assert perception.requests == []

    runtime._advance_headroom_recovery()
    assert recovery.phase == "settle"
    assert len(sent) == 1
    reorientation = sent[0]
    assert isinstance(reorientation, MotorAction)
    assert reorientation.mouse_dy == 96
    assert not reorientation.keys_down
    assert not reorientation.buttons_down

    runtime._advance_headroom_recovery()
    assert perception.requests == []

    previous = runtime.blackboard.raw_latest()
    assert previous is not None
    next_ns = max(time.monotonic_ns(), previous.captured_ns + 1)
    next_capture = CapturedFrame(42, next_ns, 8, 8, perception.last_capture.bgra)
    perception.last_capture = next_capture
    runtime.blackboard.publish(
        FrameState(
            frame_id=5,
            captured_ns=next_ns,
            instance_id="bedrock:headroom",
            width=8,
            height=8,
        )
    )
    runtime._advance_headroom_recovery()
    assert recovery.phase == "settle"
    assert perception.requests == []

    changed_ns = max(time.monotonic_ns(), next_ns + 1)
    changed_capture = CapturedFrame(
        43,
        changed_ns,
        8,
        8,
        bytes(reversed(range(64))) * 4,
    )
    perception.last_capture = changed_capture
    runtime.blackboard.publish(
        FrameState(
            frame_id=6,
            captured_ns=changed_ns,
            instance_id="bedrock:headroom",
            width=8,
            height=8,
        )
    )
    runtime.blackboard.merge_semantics(
        instance_id="bedrock:headroom",
        facts=(
            PerceptionFact(
                key="frame.dhash",
                value=frame_dhash(changed_capture),
                confidence=1.0,
                observed_ns=changed_ns,
                source="bootstrap:frame",
                expires_after_ms=60_000,
            ),
        ),
    )
    runtime._advance_headroom_recovery()
    assert recovery.phase == "grounding"
    assert len(perception.requests) == 1
    query = perception.requests[0][0]
    assert query.skill_id == "mine_visible_block"
    assert query.output_keys == (
        "scene.mode",
        "scene.playable",
        "danger.immediate",
        "target.visible",
        "target.kind",
        "target.mineable",
        "target.dx",
        "target.dy",
    )

    runtime._advance_headroom_recovery()
    assert len(perception.requests) == 1

    _publish_headroom_answer(
        runtime.blackboard,
        recovery,
        current_hash=recovery.query_frame_dhash or _HASH_A,
    )
    perception.available = True
    runtime._advance_headroom_recovery()

    assert len(perception.requests) == 1
    assert recovery.phase == "mining"
    assert runtime.executor.run is not None
    assert runtime.executor.run.skill_id == "mine_visible_block"
    assert runtime.executor.parameters == {
        "target": "dirt",
        "target_track_id": f"vlm:{recovery.query_id}:0",
    }


def test_unsafe_stall_preserves_ordinary_recovery_route() -> None:
    runtime, perception, _ = _runtime_for_probe()
    now = time.monotonic_ns()
    runtime.blackboard.merge_semantics(
        instance_id="bedrock:headroom",
        facts=(
            PerceptionFact(
                key="danger.immediate",
                value=True,
                confidence=1.0,
                observed_ns=now,
                source="safety:test",
                expires_after_ms=5_000,
            ),
        ),
    )

    assert runtime._route_headroom_terminal(_stall_result()) is False
    assert runtime._headroom_recovery is None
    assert perception.requests == []


def test_expired_headroom_transaction_abstains_without_input_or_query() -> None:
    runtime, perception, sent = _runtime_for_probe()
    runtime._headroom_recovery = _HeadroomRecovery(
        context_key="explore-keepalive",
        traversal_parameters={},
        deadline_ns=time.monotonic_ns() - 1,
    )

    runtime._advance_headroom_recovery()

    assert runtime._headroom_recovery is None
    assert perception.requests == []
    assert sent == []
    assert runtime._traversal_escalation_pending is True
    assert runtime._cognition_requested is True


def test_unchanged_reorientation_settle_abstains_after_two_seconds() -> None:
    runtime, perception, sent = _runtime_for_probe()
    assert perception.last_capture is not None
    latest = runtime.blackboard.raw_latest()
    assert latest is not None
    runtime._headroom_recovery = _HeadroomRecovery(
        context_key="explore-keepalive",
        traversal_parameters={},
        deadline_ns=time.monotonic_ns() + 60_000_000_000,
        phase="settle",
        reoriented_frame_id=latest.frame_id,
        pre_reorient_dhash=frame_dhash(perception.last_capture),
        settle_deadline_ns=time.monotonic_ns() - 1,
    )

    runtime._advance_headroom_recovery()

    assert runtime._headroom_recovery is None
    assert perception.requests == []
    assert sent == []
    assert runtime._traversal_escalation_pending is True
    assert runtime._cognition_requested is True


def test_expired_headroom_transaction_cancels_and_releases_running_child() -> None:
    runtime, _, sent = _runtime_for_probe()
    runtime.executor.start(
        runtime.skills.get("mine_visible_block"),
        run_id="clear-one",
        context_key="explore-keepalive",
        parameters={"target": "dirt"},
    )
    runtime._headroom_recovery = _HeadroomRecovery(
        context_key="explore-keepalive",
        traversal_parameters={},
        deadline_ns=time.monotonic_ns() - 1,
        phase="mining",
        mining_run_id="clear-one",
    )
    recorded: list[SkillRun] = []
    runtime._record_terminal_run = (  # type: ignore[method-assign]
        lambda run, **_kwargs: recorded.append(run)
    )

    runtime._advance_headroom_recovery()

    assert runtime._headroom_recovery is None
    assert recorded[0].outcome == SkillOutcome.CANCELLED
    assert sent
    assert runtime._traversal_escalation_pending is True
    assert runtime._cognition_requested is True


def _mining_success(run_id: str) -> ExecutionTick:
    run = SkillRun(
        run_id=run_id,
        skill_id="mine_visible_block",
        context_key="explore-keepalive",
        started_ns=2,
        ended_ns=3,
        outcome=SkillOutcome.SUCCEEDED,
    )
    return ExecutionTick(
        run=run,
        action=None,
        outcome_verification=OutcomeVerification(
            run_id=run_id,
            kind=OutcomeKind.MINING,
            status=OutcomeStatus.SUCCEEDED,
            signal=OutcomeSignal.BLOCK_BROKEN,
            observed_ns=3,
            confidence=0.9,
            reason="bound dirt block disappeared after damage",
            target_kind="dirt",
        ),
    )


def _retry_progress(
    run_id: str,
    *,
    context_key: str = "explore-keepalive",
) -> ExecutionTick:
    run = SkillRun(
        run_id=run_id,
        skill_id="traverse_visible_obstacle",
        context_key=context_key,
        started_ns=3,
        ended_ns=4,
        outcome=SkillOutcome.SUCCEEDED,
    )
    return ExecutionTick(
        run=run,
        action=None,
        outcome_verification=OutcomeVerification(
            run_id=run_id,
            kind=OutcomeKind.TRAVERSAL,
            status=OutcomeStatus.PROGRESS,
            signal=OutcomeSignal.LOCOMOTION_PROGRESS,
            observed_ns=4,
            confidence=0.91,
            reason="verified displacement after the bounded retry",
        ),
    )


def test_verified_clear_retries_obstacle_once_and_retry_stall_does_not_loop() -> None:
    runtime, _, _ = _runtime_for_probe()
    recovery = _HeadroomRecovery(
        context_key="explore-keepalive",
        traversal_parameters={"allow_jump": True},
        deadline_ns=time.monotonic_ns() + 60_000_000_000,
        phase="mining",
        mining_run_id="clear-one",
    )
    runtime._headroom_recovery = recovery
    runtime._traversal_escalation_pending = True

    assert runtime._route_headroom_terminal(_mining_success("clear-one")) is True
    assert recovery.phase == "retry"
    assert recovery.retry_run_id is not None
    assert runtime.executor.run is not None
    assert runtime.executor.run.skill_id == "traverse_visible_obstacle"
    assert runtime.executor.parameters == {"allow_jump": True}
    assert runtime._traversal_escalation_pending is True

    retry_stall = _stall_result(recovery.retry_run_id)
    assert runtime._route_headroom_terminal(retry_stall) is True
    assert runtime._headroom_recovery is None
    assert runtime._traversal_escalation_pending is True
    assert runtime.executor.run is not None
    assert runtime.executor.run.run_id == retry_stall.run.run_id


def test_verified_retry_progress_clears_escalation_and_advances_plan_once() -> None:
    runtime, _, _ = _runtime_for_probe()
    recovery = _HeadroomRecovery(
        context_key="explore-keepalive",
        traversal_parameters={"allow_jump": True},
        deadline_ns=time.monotonic_ns() + 60_000_000_000,
        phase="retry",
        retry_run_id="retry-one",
    )
    runtime._headroom_recovery = recovery
    runtime._traversal_escalation_pending = True
    result = _retry_progress("retry-one")

    assert _verified_headroom_retry(result, recovery) is True
    assert runtime._route_headroom_terminal(result) is True
    assert runtime._headroom_recovery is None
    assert runtime._traversal_escalation_pending is False

    assert (
        _headroom_retry_advances_plan(
            result,
            recovery,
            plan_steps=("unrelated strategic work",),
            plan_index=0,
            plan_goal_id="strategic-goal",
        )
        is False
    )

    planned_recovery = _HeadroomRecovery(
        context_key="strategic-goal",
        traversal_parameters={},
        deadline_ns=time.monotonic_ns() + 60_000_000_000,
        phase="retry",
        retry_run_id="planned-retry",
    )
    planned_result = _retry_progress(
        "planned-retry",
        context_key="strategic-goal",
    )
    advances_plan = _headroom_retry_advances_plan(
        planned_result,
        planned_recovery,
        plan_steps=("clear the obstacle", "continue exploring"),
        plan_index=0,
        plan_goal_id="strategic-goal",
    )
    assert advances_plan is True

    recorder = object.__new__(AgentRuntime)
    recorder.skills = build_bootstrap_skill_library()
    recorder.memories = MemoryStore()
    recorder.state_db = None
    recorder.trajectory = None
    recorder.metrics = RuntimeMetrics()
    recorder._recorded_run_ids = set()
    recorder._recorded_run_order = deque(maxlen=4_096)
    recorder._recent_skill_runs = deque(maxlen=8)
    recorder._plan_steps = ("clear the obstacle", "continue exploring")
    recorder._plan_index = 0
    recorder._plan_goal_id = "strategic-goal"
    recorder._last_decision = None
    recorder._plan_step_completed_ns = 0
    recorder._record_terminal_run(
        planned_result.run,
        outcome_verification=planned_result.outcome_verification,
        advance_plan=advances_plan,
    )
    recorder._record_terminal_run(
        planned_result.run,
        outcome_verification=planned_result.outcome_verification,
        advance_plan=advances_plan,
    )

    assert recorder._plan_index == 1
    assert recorder.metrics.skill_successes == 1


def test_unverified_retry_success_keeps_escalation_latched() -> None:
    runtime, _, _ = _runtime_for_probe()
    recovery = _HeadroomRecovery(
        context_key="explore-keepalive",
        traversal_parameters={},
        deadline_ns=time.monotonic_ns() + 60_000_000_000,
        phase="retry",
        retry_run_id="retry-one",
    )
    runtime._headroom_recovery = recovery
    runtime._traversal_escalation_pending = True
    unverified = replace(_retry_progress("retry-one"), outcome_verification=None)

    assert _verified_headroom_retry(unverified, recovery) is False
    assert runtime._route_headroom_terminal(unverified) is True
    assert runtime._headroom_recovery is None
    assert runtime._traversal_escalation_pending is True


def test_retry_result_returning_after_transaction_deadline_is_cancelled() -> None:
    runtime, _, _ = _runtime_for_probe()
    recovery = _HeadroomRecovery(
        context_key="strategic-goal",
        traversal_parameters={},
        deadline_ns=time.monotonic_ns() - 1,
        phase="retry",
        retry_run_id="late-retry",
    )
    runtime._headroom_recovery = recovery
    runtime._traversal_escalation_pending = True
    late = _retry_progress("late-retry", context_key="strategic-goal")
    runtime.executor._run = late.run  # mirror the terminal executor state after tick

    expired, was_expired = runtime._expire_late_headroom_child(late)

    assert was_expired is True
    assert expired.run.outcome == SkillOutcome.CANCELLED
    assert expired.run.failure_reason == "headroom-transaction-expired"
    assert expired.outcome_verification is None
    assert expired.action is None or not expired.action.keys_down
    assert runtime._headroom_recovery is None
    assert runtime._traversal_escalation_pending is True
    assert runtime._cognition_requested is True
    assert (
        _headroom_retry_advances_plan(
            expired,
            recovery,
            plan_steps=("clear the obstacle",),
            plan_index=0,
            plan_goal_id="strategic-goal",
        )
        is False
    )


def test_operator_authority_cancels_running_headroom_child() -> None:
    runtime, _, _ = _runtime_for_probe()
    runtime.executor.start(
        runtime.skills.get("mine_visible_block"),
        run_id="clear-one",
        context_key="explore-keepalive",
        parameters={"target": "dirt"},
    )
    runtime._headroom_recovery = _HeadroomRecovery(
        context_key="explore-keepalive",
        traversal_parameters={},
        deadline_ns=time.monotonic_ns() + 60_000_000_000,
        phase="mining",
        mining_run_id="clear-one",
    )
    runtime._pending_decision = Future()
    runtime._queued_operator_message_waiting = lambda: True  # type: ignore[method-assign]
    runtime._preempt_pending_cognition_for_operator = lambda: False  # type: ignore[method-assign]
    recorded: list[SkillRun] = []
    sent = []
    runtime._record_terminal_run = (  # type: ignore[method-assign]
        lambda run, **_kwargs: recorded.append(run)
    )
    runtime._send_motor = lambda action, **_kwargs: sent.append(action)  # type: ignore[method-assign]

    runtime._start_cognition_if_due()

    assert runtime._headroom_recovery is None
    assert recorded[0].outcome == SkillOutcome.CANCELLED
    assert sent


def test_death_scene_preempts_running_headroom_child() -> None:
    runtime, _, _ = _runtime_for_probe()
    runtime.executor.start(
        runtime.skills.get("mine_visible_block"),
        run_id="clear-one",
        context_key="explore-keepalive",
        parameters={"target": "dirt"},
    )
    runtime._headroom_recovery = _HeadroomRecovery(
        context_key="explore-keepalive",
        traversal_parameters={},
        deadline_ns=time.monotonic_ns() + 60_000_000_000,
        phase="mining",
        mining_run_id="clear-one",
    )
    now = time.monotonic_ns()
    runtime.blackboard.merge_semantics(
        instance_id="bedrock:headroom",
        facts=(
            PerceptionFact(
                key="scene.death",
                value=True,
                confidence=1.0,
                observed_ns=now,
                source="safety:test",
                expires_after_ms=5_000,
            ),
        ),
    )
    recorded: list[SkillRun] = []
    sent = []
    runtime._record_terminal_run = (  # type: ignore[method-assign]
        lambda run, **_kwargs: recorded.append(run)
    )
    runtime._send_motor = lambda action, **_kwargs: sent.append(action)  # type: ignore[method-assign]

    runtime._route_observed_scene_recovery()

    assert runtime._headroom_recovery is None
    assert recorded[0].outcome == SkillOutcome.CANCELLED
    assert sent
    assert runtime.executor.run is not None
    assert runtime.executor.run.skill_id == "respawn_after_death"
