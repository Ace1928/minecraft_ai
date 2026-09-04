from __future__ import annotations

import time
from typing import Any, cast

import pytest

from minecraft_ai.execution import SkillExecutor
from minecraft_ai.mining_control import MiningGuardDecision
from minecraft_ai.motor import MotorIntent
from minecraft_ai.perception import (
    FrameState,
    PerceptionBlackboard,
    PerceptionFact,
    ScreenRegion,
    Track,
)
from minecraft_ai.safety import MotorAction
from minecraft_ai.skills import SkillCondition, SkillFailureCode, SkillOutcome, SkillSpec
from minecraft_ai.trajectory import ActionOrigin


_HASH_A = "0000000000000000"
_HASH_B = "ffffffffffffffff"


class _MiningPolicy:
    policy_id = "outcome-test-policy"

    def __init__(self) -> None:
        self.last_sequence = -1
        self.held_left = False
        self.act_calls = 0
        self.observation_release_calls = 0
        self.perception_poll_calls = 0

    def act(
        self,
        _blackboard: PerceptionBlackboard,
        _intent: MotorIntent,
        *,
        sequence: int,
    ) -> MotorAction:
        self.act_calls += 1
        self.last_sequence = sequence
        self.held_left = True
        return MotorAction(sequence=sequence, buttons_down=("left",))

    def reset(self) -> MotorAction:
        self.last_sequence += 1
        buttons_up = ("left",) if self.held_left else ()
        self.held_left = False
        return MotorAction(sequence=max(0, self.last_sequence), buttons_up=buttons_up)

    def release_for_observation(self) -> MotorAction:
        self.observation_release_calls += 1
        return self.reset()

    def poll_perception(
        self,
        _blackboard: PerceptionBlackboard,
        _intent: MotorIntent,
    ) -> bool:
        self.perception_poll_calls += 1
        return False


class _ScriptedMiningGuard:
    def __init__(self, *failures: SkillFailureCode | None) -> None:
        self._failures = list(failures)
        self._held = False

    @property
    def held_keys(self) -> tuple[str, ...]:
        return ()

    @property
    def held_buttons(self) -> tuple[str, ...]:
        return ("left",) if self._held else ()

    def inspect(
        self,
        action: MotorAction,
        _blackboard: PerceptionBlackboard,
        _intent: MotorIntent,
        *,
        now_ns: int | None = None,
    ) -> MiningGuardDecision:
        del now_ns
        failure = self._failures.pop(0) if self._failures else None
        self._held = failure is None
        return MiningGuardDecision(
            action=action,
            failure_code=failure,
            force_release_left=failure is not None,
        )

    def reset(self) -> bool:
        held = self._held
        self._held = False
        return held


def _fact(
    key: str,
    value: str | int | float | bool,
    *,
    now_ns: int,
    source: str,
) -> PerceptionFact:
    return PerceptionFact(
        key=key,
        value=value,
        confidence=1.0,
        observed_ns=now_ns,
        source=source,
        expires_after_ms=60_000,
    )


def _board(
    now_ns: int,
    *,
    crosshair_hash: str,
    target_visible: bool,
    extra_facts: tuple[PerceptionFact, ...] = (),
) -> PerceptionBlackboard:
    board = PerceptionBlackboard()
    board.publish(
        FrameState(
            frame_id=1,
            captured_ns=now_ns,
            instance_id="bedrock:test",
            width=1280,
            height=720,
            facts=(
                _fact(
                    "frame.crosshair_dhash",
                    crosshair_hash,
                    now_ns=now_ns,
                    source="bootstrap:test-hash",
                ),
                _fact(
                    "target.visible",
                    target_visible,
                    now_ns=now_ns,
                    source="learned:test-target",
                ),
                _fact(
                    "target.kind",
                    "dirt",
                    now_ns=now_ns,
                    source="learned:test-target",
                ),
                _fact(
                    "target.exists_probability",
                    0.95 if target_visible else 0.05,
                    now_ns=now_ns,
                    source="learned:test-target",
                ),
                *extra_facts,
            ),
            tracks=(
                Track(
                    track_id="target:test",
                    label="dirt",
                    confidence=1.0,
                    region=ScreenRegion(x=0.4, y=0.4, width=0.2, height=0.2),
                    first_seen_ns=now_ns,
                    last_seen_ns=now_ns,
                    attributes={"tracking_source": "learned:test-target"},
                ),
            ),
        )
    )
    return board


def _executor(
    now_ns: int,
    *failures: SkillFailureCode | None,
    max_duration_ms: int = 20_000,
    success_conditions: tuple[SkillCondition, ...] = (),
) -> tuple[SkillExecutor, _MiningPolicy]:
    policy = _MiningPolicy()
    executor = SkillExecutor(policy)
    executor.start(
        SkillSpec(
            skill_id="mine_visible_block",
            name="Mine visible block",
            policy_ref="mine",
            recovery_skills=("reacquire_target",),
            success_conditions=success_conditions,
            max_duration_ms=max_duration_ms,
        ),
        run_id="mine-outcome",
        now_ns=now_ns,
    )
    executor._mining_guard = cast(Any, _ScriptedMiningGuard(*failures))
    return executor, policy


def test_verified_replacement_converts_target_changed_to_one_success() -> None:
    now = time.monotonic_ns()
    executor, policy = _executor(
        now,
        None,
        SkillFailureCode.MINING_TARGET_CHANGED,
    )

    started = executor.tick(
        _board(now, crosshair_hash=_HASH_A, target_visible=True),
        sequence=1,
        now_ns=now,
    )
    released = executor.tick(
        _board(now + 500_000_000, crosshair_hash=_HASH_B, target_visible=False),
        sequence=2,
        now_ns=now + 500_000_000,
    )
    gathering = executor.tick(
        _board(now + 850_000_000, crosshair_hash=_HASH_B, target_visible=False),
        sequence=4,
        now_ns=now + 850_000_000,
    )
    succeeded = executor.tick(
        _board(now + 900_000_000, crosshair_hash=_HASH_B, target_visible=False),
        sequence=4,
        now_ns=now + 900_000_000,
    )
    repeated = executor.tick(
        _board(now + 950_000_000, crosshair_hash=_HASH_B, target_visible=False),
        sequence=5,
        now_ns=now + 950_000_000,
    )

    assert started.run.outcome == SkillOutcome.RUNNING
    assert released.run.outcome == SkillOutcome.RUNNING
    assert released.action is not None and "left" in released.action.buttons_up
    assert released.action_origin == ActionOrigin.RESET
    assert gathering.run.outcome == SkillOutcome.RUNNING
    assert gathering.action is None
    assert succeeded.run.outcome == SkillOutcome.SUCCEEDED
    assert succeeded.run.failure_code is None
    assert succeeded.outcome_verification is not None
    assert succeeded.outcome_verification.signal.value == "block_broken"
    assert repeated.run == succeeded.run
    assert repeated.action is None
    assert policy.act_calls == 2
    assert policy.observation_release_calls == 1
    assert policy.perception_poll_calls == 2


def test_visual_change_without_target_loss_remains_target_changed_failure() -> None:
    now = time.monotonic_ns()
    executor, _policy = _executor(
        now,
        None,
        SkillFailureCode.MINING_TARGET_CHANGED,
    )

    executor.tick(
        _board(now, crosshair_hash=_HASH_A, target_visible=True),
        sequence=1,
        now_ns=now,
    )
    released = executor.tick(
        _board(now + 500_000_000, crosshair_hash=_HASH_B, target_visible=True),
        sequence=2,
        now_ns=now + 500_000_000,
    )
    failed = executor.tick(
        _board(now + 5_600_000_000, crosshair_hash=_HASH_B, target_visible=True),
        sequence=4,
        now_ns=now + 5_600_000_000,
    )

    assert released.run.outcome == SkillOutcome.RUNNING
    assert released.action is not None and "left" in released.action.buttons_up
    assert failed.run.outcome == SkillOutcome.FAILED
    assert failed.run.failure_code == SkillFailureCode.MINING_TARGET_CHANGED
    assert failed.recovery_skills == ("reacquire_target",)


@pytest.mark.parametrize(
    "guard_failure",
    (
        SkillFailureCode.MINING_VISUAL_STAGNATION,
        SkillFailureCode.MINING_LEASE_EXPIRED,
    ),
)
def test_progress_defers_recoverable_failure_for_two_delayed_target_losses(
    guard_failure: SkillFailureCode,
) -> None:
    now = time.monotonic_ns()
    executor, policy = _executor(now, None, None, None, guard_failure)
    for sequence, offset_ms in enumerate((0, 450, 500), start=1):
        executor.tick(
            _board(
                now + offset_ms * 1_000_000,
                crosshair_hash=_HASH_A if offset_ms == 0 else _HASH_B,
                target_visible=True,
            ),
            sequence=sequence,
            now_ns=now + offset_ms * 1_000_000,
        )
    released = executor.tick(
        _board(now + 550_000_000, crosshair_hash=_HASH_B, target_visible=True),
        sequence=4,
        now_ns=now + 550_000_000,
    )
    first = executor.tick(
        _board(now + 2_300_000_000, crosshair_hash=_HASH_B, target_visible=False),
        sequence=5,
        now_ns=now + 2_300_000_000,
    )
    succeeded = executor.tick(
        _board(now + 4_000_000_000, crosshair_hash=_HASH_B, target_visible=False),
        sequence=6,
        now_ns=now + 4_000_000_000,
    )

    assert released.run.outcome == SkillOutcome.RUNNING
    assert released.action is not None and "left" in released.action.buttons_up
    assert first.run.outcome == SkillOutcome.RUNNING
    assert first.action is None
    assert succeeded.run.outcome == SkillOutcome.SUCCEEDED
    assert succeeded.outcome_verification is not None
    assert succeeded.outcome_verification.signal.value == "block_broken"
    assert policy.act_calls == 4
    assert policy.observation_release_calls == 1
    assert policy.perception_poll_calls == 2


@pytest.mark.parametrize(
    "guard_failure",
    (
        SkillFailureCode.MINING_VISUAL_STAGNATION,
        SkillFailureCode.MINING_LEASE_EXPIRED,
    ),
)
def test_unconfirmed_recoverable_failure_exits_after_bounded_verification(
    guard_failure: SkillFailureCode,
) -> None:
    now = time.monotonic_ns()
    executor, _policy = _executor(now, None, None, None, guard_failure)
    for sequence, offset_ms in enumerate((0, 450, 500), start=1):
        executor.tick(
            _board(
                now + offset_ms * 1_000_000,
                crosshair_hash=_HASH_A if offset_ms == 0 else _HASH_B,
                target_visible=True,
            ),
            sequence=sequence,
            now_ns=now + offset_ms * 1_000_000,
        )
    released = executor.tick(
        _board(now + 550_000_000, crosshair_hash=_HASH_B, target_visible=True),
        sequence=4,
        now_ns=now + 550_000_000,
    )
    failed = executor.tick(
        _board(now + 5_600_000_000, crosshair_hash=_HASH_B, target_visible=True),
        sequence=5,
        now_ns=now + 5_600_000_000,
    )

    assert released.run.outcome == SkillOutcome.RUNNING
    assert failed.run.outcome == SkillOutcome.FAILED
    assert failed.run.failure_code == guard_failure
    assert failed.recovery_skills == ("reacquire_target",)


def test_static_emitted_attack_surfaces_typed_stall_and_releases() -> None:
    now = time.monotonic_ns()
    executor, _policy = _executor(now, None, None, None, None)

    tick = None
    for sequence, offset_ms in enumerate((0, 500, 1_000, 1_600), start=1):
        tick = executor.tick(
            _board(
                now + offset_ms * 1_000_000,
                crosshair_hash=_HASH_A,
                target_visible=True,
            ),
            sequence=sequence,
            now_ns=now + offset_ms * 1_000_000,
        )

    assert tick is not None
    assert tick.run.outcome == SkillOutcome.FAILED
    assert tick.run.failure_code == SkillFailureCode.MINING_VISUAL_STAGNATION
    assert tick.action is not None and "left" in tick.action.buttons_up
    assert tick.recovery_skills == ("reacquire_target",)


def test_stale_prior_target_broken_does_not_bypass_attack_verification() -> None:
    now = time.monotonic_ns()
    executor, policy = _executor(
        now,
        None,
        success_conditions=(SkillCondition(key="target.broken", operator="truthy"),),
    )
    stale_broken = _fact(
        "target.broken",
        True,
        now_ns=now,
        source="learned:test-target",
    )

    tick = executor.tick(
        _board(
            now,
            crosshair_hash=_HASH_A,
            target_visible=True,
            extra_facts=(stale_broken,),
        ),
        sequence=1,
        now_ns=now,
    )

    assert tick.run.outcome == SkillOutcome.RUNNING
    assert tick.outcome_verification is None
    assert policy.act_calls == 1


def test_death_ui_preempts_post_release_break_candidate() -> None:
    now = time.monotonic_ns()
    executor, _policy = _executor(
        now,
        None,
        SkillFailureCode.MINING_TARGET_CHANGED,
    )
    executor.tick(
        _board(now, crosshair_hash=_HASH_A, target_visible=True),
        sequence=1,
        now_ns=now,
    )
    executor.tick(
        _board(now + 500_000_000, crosshair_hash=_HASH_B, target_visible=False),
        sequence=2,
        now_ns=now + 500_000_000,
    )
    executor.tick(
        _board(now + 700_000_000, crosshair_hash=_HASH_B, target_visible=False),
        sequence=4,
        now_ns=now + 700_000_000,
    )
    death = _fact("scene.mode", "death", now_ns=now + 900_000_000, source="vlm:test")

    stopped = executor.tick(
        _board(
            now + 900_000_000,
            crosshair_hash=_HASH_B,
            target_visible=False,
            extra_facts=(death,),
        ),
        sequence=4,
        now_ns=now + 900_000_000,
    )

    assert stopped.run.outcome == SkillOutcome.FAILED
    assert stopped.run.failure_code == SkillFailureCode.MINING_UNSAFE_SCENE
    assert stopped.outcome_verification is None


def test_hard_skill_deadline_preempts_post_release_break_candidate() -> None:
    now = time.monotonic_ns()
    executor, _policy = _executor(
        now,
        None,
        SkillFailureCode.MINING_TARGET_CHANGED,
        max_duration_ms=800,
    )
    executor.tick(
        _board(now, crosshair_hash=_HASH_A, target_visible=True),
        sequence=1,
        now_ns=now,
    )
    executor.tick(
        _board(now + 500_000_000, crosshair_hash=_HASH_B, target_visible=False),
        sequence=2,
        now_ns=now + 500_000_000,
    )
    executor.tick(
        _board(now + 700_000_000, crosshair_hash=_HASH_B, target_visible=False),
        sequence=4,
        now_ns=now + 700_000_000,
    )

    stopped = executor.tick(
        _board(now + 900_000_000, crosshair_hash=_HASH_B, target_visible=False),
        sequence=4,
        now_ns=now + 900_000_000,
    )

    assert stopped.run.outcome == SkillOutcome.TIMED_OUT
    assert stopped.outcome_verification is None


@pytest.mark.parametrize(
    ("guard_failure", "expected_outcome"),
    (
        (SkillFailureCode.MINING_LEASE_EXPIRED, SkillOutcome.SUCCEEDED),
        (SkillFailureCode.MINING_VISUAL_STAGNATION, SkillOutcome.SUCCEEDED),
        (SkillFailureCode.MINING_CAMERA_CHANGED, SkillOutcome.FAILED),
        (SkillFailureCode.MINING_TOOL_CHANGED, SkillOutcome.FAILED),
    ),
)
def test_verified_break_precedence_respects_safety_boundary(
    guard_failure: SkillFailureCode,
    expected_outcome: SkillOutcome,
) -> None:
    now = time.monotonic_ns()
    executor, _policy = _executor(now, None, None, None, guard_failure)
    executor.tick(
        _board(now, crosshair_hash=_HASH_A, target_visible=True),
        sequence=1,
        now_ns=now,
    )
    executor.tick(
        _board(now + 500_000_000, crosshair_hash=_HASH_B, target_visible=True),
        sequence=2,
        now_ns=now + 500_000_000,
    )
    executor.tick(
        _board(now + 600_000_000, crosshair_hash=_HASH_B, target_visible=True),
        sequence=3,
        now_ns=now + 600_000_000,
    )
    broken = _fact(
        "target.broken",
        True,
        now_ns=now + 700_000_000,
        source="learned:test-target",
    )

    terminal = executor.tick(
        _board(
            now + 700_000_000,
            crosshair_hash=_HASH_B,
            target_visible=True,
            extra_facts=(broken,),
        ),
        sequence=4,
        now_ns=now + 700_000_000,
    )

    assert terminal.run.outcome == expected_outcome
    if expected_outcome == SkillOutcome.SUCCEEDED:
        assert terminal.outcome_verification is not None
        assert terminal.outcome_verification.signal.value == "block_broken"
    else:
        assert terminal.run.failure_code == guard_failure
        assert terminal.outcome_verification is None
