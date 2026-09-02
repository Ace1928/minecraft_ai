from __future__ import annotations

from minecraft_ai.execution import SkillExecutor
from minecraft_ai.motor import BootstrapMotorPolicy
from minecraft_ai.perception import FrameState, PerceptionBlackboard, PerceptionFact
from minecraft_ai.skills import SkillCondition, SkillOutcome, SkillSpec


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
        observed_ns=100,
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
    running = executor.tick(_board(_fact("done", False)), sequence=1, now_ns=200)
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
