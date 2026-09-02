from __future__ import annotations

import time

from minecraft_ai.execution import SkillExecutor
from minecraft_ai.motor import BootstrapMotorPolicy, MotorIntent
from minecraft_ai.perception import FrameState, PerceptionBlackboard, PerceptionFact
from minecraft_ai.safety import MotorAction
from minecraft_ai.skills import SkillCondition, SkillOutcome, SkillSpec


class _IntentCapturePolicy:
    policy_id = "capture"

    def __init__(self) -> None:
        self.intent: MotorIntent | None = None

    def act(
        self,
        blackboard: PerceptionBlackboard,
        intent: MotorIntent,
        *,
        sequence: int,
    ) -> MotorAction:
        self.intent = intent
        return MotorAction(sequence=sequence)

    def reset(self) -> MotorAction:
        return MotorAction(sequence=1)


def _board(*facts: PerceptionFact) -> PerceptionBlackboard:
    board = PerceptionBlackboard()
    board.publish(
        FrameState(
            frame_id=1,
            captured_ns=100,
            instance_id="bedrock:test",
            width=1280,
            height=720,
            facts=facts,
        )
    )
    return board


def _fact(key: str, value: str | int | float | bool) -> PerceptionFact:
    return PerceptionFact(
        key=key,
        value=value,
        confidence=1.0,
        observed_ns=time.monotonic_ns(),
        source="test",
        expires_after_ms=1_000_000,
    )


def test_running_skill_emits_bounded_motor_action() -> None:
    policy = BootstrapMotorPolicy()
    executor = SkillExecutor(policy)
    spec = SkillSpec(
        skill_id="approach",
        name="Approach",
        policy_ref="approach",
        preconditions=(SkillCondition(key="target.visible", operator="truthy"),),
        success_conditions=(SkillCondition(key="target.near", operator="truthy"),),
    )
    board = _board(
        _fact("target.visible", True),
        _fact("target.near", False),
        _fact("target.dx", 0.5),
        _fact("target.dy", -0.25),
    )
    executor.start(spec, run_id="r1", now_ns=100)
    tick = executor.tick(board, sequence=1, now_ns=200)
    assert tick.run.outcome == SkillOutcome.RUNNING
    assert tick.action is not None
    assert tick.action.keys_down == ("w",)
    assert abs(tick.action.mouse_dx) <= policy.max_mouse_step
    assert abs(tick.action.mouse_dy) <= policy.max_mouse_step


def test_skill_contract_becomes_learned_policy_instruction() -> None:
    policy = _IntentCapturePolicy()
    executor = SkillExecutor(policy)
    spec = SkillSpec(
        skill_id="mine_log",
        name="Mine log",
        description="Approach and mine a visible tree log",
        policy_ref="mine",
        policy_instruction="mine log",
    )
    executor.start(
        spec,
        run_id="r1",
        parameters={"target": "oak_log", "wood": "oak"},
        now_ns=100,
    )

    executor.tick(_board(), sequence=1, now_ns=200)

    assert policy.intent is not None
    assert policy.intent.instruction == "mine log"
    assert policy.intent.target_label == "oak_log"
    assert executor.parameters == {"target": "oak_log", "wood": "oak"}
    assert executor.run is not None
    assert executor.run.parameters == {"target": "oak_log", "wood": "oak"}
    assert executor.instruction == (
        "Approach and mine a visible tree log. Parameters: target=oak_log, wood=oak"
    )


def test_skill_success_releases_held_input() -> None:
    policy = BootstrapMotorPolicy()
    executor = SkillExecutor(policy)
    spec = SkillSpec(
        skill_id="mine",
        name="Mine",
        policy_ref="mine",
        success_conditions=(SkillCondition(key="done", operator="truthy"),),
    )
    executor.start(spec, run_id="r1", now_ns=100)
    running = executor.tick(
        _board(_fact("done", False), _fact("target.visible", True)),
        sequence=1,
        now_ns=200,
    )
    assert running.action is not None
    assert "w" in running.action.keys_down
    assert "left" in running.action.buttons_down

    done = executor.tick(_board(_fact("done", True)), sequence=2, now_ns=300)
    assert done.run.outcome == SkillOutcome.SUCCEEDED
    assert done.action is not None
    assert "w" in done.action.keys_up
    assert "left" in done.action.buttons_up


def test_failure_requests_recovery_and_releases() -> None:
    policy = BootstrapMotorPolicy()
    executor = SkillExecutor(policy)
    spec = SkillSpec(
        skill_id="mine",
        name="Mine",
        policy_ref="mine",
        failure_conditions=(SkillCondition(key="danger", operator="truthy"),),
        recovery_skills=("retreat",),
    )
    executor.start(spec, run_id="r1", now_ns=100)
    tick = executor.tick(_board(_fact("danger", True)), sequence=1, now_ns=200)
    assert tick.run.outcome == SkillOutcome.FAILED
    assert tick.recovery_skills == ("retreat",)
    assert tick.action is not None


def test_precondition_is_checked_at_initiation_but_not_rechecked_as_invariant() -> None:
    policy = _IntentCapturePolicy()
    executor = SkillExecutor(policy)
    spec = SkillSpec(
        skill_id="track_target",
        name="Track target",
        preconditions=(SkillCondition(key="target.visible", operator="truthy"),),
        policy_ref="approach",
    )

    executor.start(spec, run_id="r1", now_ns=100)
    first = executor.tick(
        _board(_fact("target.visible", True)),
        sequence=1,
        now_ns=200,
    )
    continued = executor.tick(_board(), sequence=2, now_ns=300)

    assert first.run.outcome == SkillOutcome.RUNNING
    assert continued.run.outcome == SkillOutcome.RUNNING
    assert continued.action is not None


def test_missing_initiation_precondition_fails_closed() -> None:
    policy = _IntentCapturePolicy()
    executor = SkillExecutor(policy)
    spec = SkillSpec(
        skill_id="track_target",
        name="Track target",
        preconditions=(SkillCondition(key="target.visible", operator="truthy"),),
        policy_ref="approach",
    )

    executor.start(spec, run_id="r1", now_ns=100)
    tick = executor.tick(_board(), sequence=1, now_ns=200)

    assert tick.run.outcome == SkillOutcome.FAILED
    assert tick.run.failure_reason == "initiation-precondition-unsatisfied"


def test_invariant_loss_terminates_running_option() -> None:
    policy = _IntentCapturePolicy()
    executor = SkillExecutor(policy)
    spec = SkillSpec(
        skill_id="stay_safe",
        name="Stay safe",
        invariants=(SkillCondition(key="scene.playable", operator="truthy"),),
        policy_ref="explore",
    )

    executor.start(spec, run_id="r1", now_ns=100)
    running = executor.tick(
        _board(_fact("scene.playable", True)),
        sequence=1,
        now_ns=200,
    )
    stopped = executor.tick(
        _board(_fact("scene.playable", False)),
        sequence=2,
        now_ns=300,
    )

    assert running.run.outcome == SkillOutcome.RUNNING
    assert stopped.run.outcome == SkillOutcome.FAILED
    assert stopped.run.failure_reason == "invariant-lost"


def test_cancel_release_advances_sequence_before_next_skill() -> None:
    board = _board(_fact("done", False))
    policy = BootstrapMotorPolicy()
    executor = SkillExecutor(policy)
    first = SkillSpec(skill_id="first", name="First", policy_ref="explore")
    second = SkillSpec(skill_id="second", name="Second", policy_ref="navigate")

    executor.start(first, run_id="run-first", now_ns=100)
    action = executor.tick(board, sequence=0, now_ns=200).action
    assert action is not None
    assert action.sequence == 0

    release = executor.cancel(now_ns=300).action
    assert release is not None
    assert release.sequence == 1

    executor.start(second, run_id="run-second", now_ns=400)
    next_action = executor.tick(
        board,
        sequence=release.sequence + 1,
        now_ns=500,
    ).action
    assert next_action is not None
    assert next_action.sequence == 2


def test_exploration_has_no_implicit_camera_sweeps_or_periodic_jumps() -> None:
    policy = BootstrapMotorPolicy()
    board = _board(_fact("scene.playable", True))
    intent = MotorIntent(skill_id="explore", mode="explore")

    actions = [policy.act(board, intent, sequence=index) for index in range(80)]

    assert all(action.mouse_dx == 0 for action in actions)
    assert all(action.mouse_dy == 0 for action in actions)
    assert all("space" not in action.keys_down for action in actions)


def test_target_tracking_is_deadbanded_and_acceleration_limited() -> None:
    policy = BootstrapMotorPolicy()
    board = _board(_fact("target.visible", True), _fact("target.dx", 1.0))
    intent = MotorIntent(skill_id="approach", mode="approach")

    actions = [policy.act(board, intent, sequence=index) for index in range(8)]
    deltas = [action.mouse_dx for action in actions]

    assert deltas[0] <= policy.max_mouse_acceleration
    assert all(
        abs(current - previous) <= policy.max_mouse_acceleration
        for previous, current in zip(deltas, deltas[1:], strict=False)
    )
    assert max(map(abs, deltas)) <= policy.max_mouse_step


def test_explicit_unplayable_scene_releases_movement() -> None:
    policy = BootstrapMotorPolicy()
    intent = MotorIntent(skill_id="explore", mode="explore")
    moving = policy.act(_board(), intent, sequence=0)
    stopped = policy.act(
        _board(_fact("scene.playable", False)),
        intent,
        sequence=1,
    )

    assert "w" in moving.keys_down
    assert "w" in stopped.keys_up
