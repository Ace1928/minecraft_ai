from __future__ import annotations

import time
from typing import Any, cast

import pytest

from minecraft_ai.builtin_skills import build_bootstrap_skill_library
from minecraft_ai.execution import SkillExecutor
from minecraft_ai.mining_control import MiningGuardDecision
from minecraft_ai.motor import MotorIntent
from minecraft_ai.outcome_verifier import OutcomeSignal
from minecraft_ai.perception import (
    FrameState,
    PerceptionBlackboard,
    PerceptionFact,
    ScreenRegion,
    Track,
)
from minecraft_ai.perception_service import BEDROCK_HOTBAR_LOG_COUNT_SOURCE
from minecraft_ai.safety import MotorAction
from minecraft_ai.skills import SkillCondition, SkillFailureCode, SkillOutcome, SkillSpec
from minecraft_ai.trajectory import ActionOrigin


_HASH_A = "0000000000000000"
_HASH_B = "ffffffffffffffff"
_LUMA_A = "20" * 64
_LUMA_B = "80" * 16 + "20" * 48
_LUMA_C = "e0" * 16 + "20" * 48


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


class _TraversalPolicy:
    policy_id = "traversal-outcome-test-policy"

    def __init__(self, *actions: MotorAction) -> None:
        self.actions = list(actions)
        self.last_sequence = -1
        self.held_keys: set[str] = set()

    def act(
        self,
        _blackboard: PerceptionBlackboard,
        _intent: MotorIntent,
        *,
        sequence: int,
    ) -> MotorAction:
        self.last_sequence = sequence
        template = self.actions.pop(0) if self.actions else MotorAction(sequence=0)
        action = template.model_copy(update={"sequence": sequence})
        self.held_keys.update(action.keys_down)
        self.held_keys.difference_update(action.keys_up)
        return action

    def reset(self) -> MotorAction:
        self.last_sequence += 1
        action = MotorAction(
            sequence=max(0, self.last_sequence),
            keys_up=tuple(sorted(self.held_keys)),
        )
        self.held_keys.clear()
        return action


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


def _traversal_board(
    now_ns: int,
    *,
    luma_grid: str,
    frame_hash: str = _HASH_A,
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
                    "frame.dhash",
                    frame_hash,
                    now_ns=now_ns,
                    source="bootstrap:test-hash",
                ),
                _fact(
                    "frame.crosshair_dhash",
                    frame_hash,
                    now_ns=now_ns,
                    source="bootstrap:test-hash",
                ),
                _fact(
                    "frame.crosshair_luma_grid",
                    luma_grid,
                    now_ns=now_ns,
                    source="bootstrap:test-luma",
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


@pytest.mark.parametrize("before", [None, 0, 15])
def test_mining_preserves_pre_attack_hotbar_count_after_immediate_pickup(
    before: int | None,
) -> None:
    now = time.monotonic_ns()
    executor, _ = _executor(now, None, SkillFailureCode.MINING_TARGET_CHANGED)
    baseline = (
        None
        if before is None
        else PerceptionFact(
            key="inventory.hotbar.logs",
            value=before,
            confidence=0.995,
            observed_ns=now,
            source=BEDROCK_HOTBAR_LOG_COUNT_SOURCE,
            expires_after_ms=250,
        )
    )
    first = executor.tick(
        _board(
            now,
            crosshair_hash=_HASH_A,
            target_visible=True,
            extra_facts=() if baseline is None else (baseline,),
        ),
        sequence=1,
        now_ns=now,
    )
    assert first.action is not None and "left" in first.action.buttons_down
    assert executor.mining_hotbar_log_baseline == baseline
    for sequence, offset_ns in enumerate((500_000_000, 850_000_000, 900_000_000), start=2):
        # Break confirmation may settle after automatic pickup. None must also
        # stay frozen rather than inventing a post-break starting count.
        board = _board(
            now + offset_ns,
            crosshair_hash=_HASH_B,
            target_visible=False,
            extra_facts=(
                PerceptionFact(
                    key="inventory.hotbar.logs",
                    value=(before or 0) + 1,
                    confidence=0.995,
                    observed_ns=now + offset_ns,
                    source=BEDROCK_HOTBAR_LOG_COUNT_SOURCE,
                    expires_after_ms=250,
                ),
            ),
        )
        terminal = executor.tick(board, sequence=sequence, now_ns=now + offset_ns)
        assert executor.mining_hotbar_log_baseline == baseline
    assert terminal.run.outcome == SkillOutcome.SUCCEEDED
    assert terminal.outcome_verification is not None
    assert terminal.outcome_verification.signal == OutcomeSignal.BLOCK_BROKEN


def test_mining_does_not_reuse_recent_hotbar_fact_when_current_frame_abstains() -> None:
    now = time.monotonic_ns()
    executor, _ = _executor(now, None)
    previous = PerceptionFact(
        key="inventory.hotbar.logs",
        value=0,
        confidence=0.995,
        observed_ns=now - 1,
        source=BEDROCK_HOTBAR_LOG_COUNT_SOURCE,
        expires_after_ms=250,
    )
    board = _board(
        now,
        crosshair_hash=_HASH_A,
        target_visible=True,
        extra_facts=(previous,),
    )
    assert board.fact("inventory.hotbar.logs", now_ns=now) is not None
    first = executor.tick(board, sequence=1, now_ns=now)
    assert first.action is not None and "left" in first.action.buttons_down
    assert executor.mining_hotbar_log_baseline is None


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


@pytest.mark.parametrize(
    "skill_id",
    ("explore_forward", "traverse_level_ground", "traverse_visible_obstacle"),
)
def test_traversal_stall_fails_early_with_typed_evidence_and_full_release(
    skill_id: str,
) -> None:
    now = time.monotonic_ns()
    policy = _TraversalPolicy(
        MotorAction(sequence=0, keys_down=("w",)),
        MotorAction(sequence=0),
        MotorAction(sequence=0),
    )
    executor = SkillExecutor(policy)
    spec = build_bootstrap_skill_library().get(skill_id)
    executor.start(spec, run_id=f"traversal-{skill_id}", now_ns=now)

    executor.tick(
        _traversal_board(now, luma_grid=_LUMA_A),
        sequence=1,
        now_ns=now,
    )
    executor.tick(
        _traversal_board(
            now + 1_100_000_000,
            luma_grid=_LUMA_A,
            frame_hash=_HASH_B,
        ),
        sequence=2,
        now_ns=now + 1_100_000_000,
    )
    stopped = executor.tick(
        _traversal_board(now + 2_200_000_000, luma_grid=_LUMA_A),
        sequence=3,
        now_ns=now + 2_200_000_000,
    )

    assert stopped.run.outcome == SkillOutcome.FAILED
    assert stopped.run.failure_code == SkillFailureCode.LOCOMOTION_STALLED
    assert stopped.run.failure_reason == SkillFailureCode.LOCOMOTION_STALLED.value
    assert stopped.action is not None
    assert set(stopped.action.keys_up) == {"w", "a", "s", "d", "space"}
    assert stopped.outcome_verification is not None
    assert stopped.outcome_verification.signal.value == "locomotion_stalled"
    assert stopped.recovery_skills == spec.recovery_skills


def test_traversal_progress_is_evidence_without_finishing_the_skill() -> None:
    now = time.monotonic_ns()
    policy = _TraversalPolicy(
        MotorAction(sequence=0, keys_down=("w",)),
        MotorAction(sequence=0),
        MotorAction(sequence=0),
    )
    executor = SkillExecutor(policy)
    executor.start(
        build_bootstrap_skill_library().get("explore_forward"),
        run_id="traversal-progress",
        now_ns=now,
    )

    executor.tick(
        _traversal_board(now, luma_grid=_LUMA_A),
        sequence=1,
        now_ns=now,
    )
    executor.tick(
        _traversal_board(now + 300_000_000, luma_grid=_LUMA_B),
        sequence=2,
        now_ns=now + 300_000_000,
    )
    progressed = executor.tick(
        _traversal_board(now + 600_000_000, luma_grid=_LUMA_C),
        sequence=3,
        now_ns=now + 600_000_000,
    )

    assert progressed.run.outcome == SkillOutcome.RUNNING
    assert progressed.outcome_verification is not None
    assert progressed.outcome_verification.signal.value == "locomotion_progress"


def test_internal_obstacle_retry_can_finish_on_verified_locomotion_progress() -> None:
    now = time.monotonic_ns()
    policy = _TraversalPolicy(
        MotorAction(sequence=0, keys_down=("w",)),
        MotorAction(sequence=0),
        MotorAction(sequence=0),
    )
    executor = SkillExecutor(policy)
    executor.start(
        build_bootstrap_skill_library().get("traverse_visible_obstacle"),
        run_id="headroom-retry",
        now_ns=now,
        complete_on_locomotion_progress=True,
    )

    executor.tick(
        _traversal_board(now, luma_grid=_LUMA_A),
        sequence=1,
        now_ns=now,
    )
    executor.tick(
        _traversal_board(now + 300_000_000, luma_grid=_LUMA_B),
        sequence=2,
        now_ns=now + 300_000_000,
    )
    progressed = executor.tick(
        _traversal_board(now + 600_000_000, luma_grid=_LUMA_C),
        sequence=3,
        now_ns=now + 600_000_000,
    )

    assert progressed.run.outcome == SkillOutcome.SUCCEEDED
    assert progressed.outcome_verification is not None
    assert progressed.outcome_verification.signal == OutcomeSignal.LOCOMOTION_PROGRESS
    assert progressed.action is not None
    assert "w" in progressed.action.keys_up


def test_starting_next_skill_resets_terminal_traversal_verifier() -> None:
    now = time.monotonic_ns()
    policy = _TraversalPolicy(
        MotorAction(sequence=0, keys_down=("w",)),
        MotorAction(sequence=0),
        MotorAction(sequence=0),
    )
    executor = SkillExecutor(policy)
    spec = build_bootstrap_skill_library().get("traverse_level_ground")
    executor.start(spec, run_id="first-traversal", now_ns=now)
    executor.tick(
        _traversal_board(now, luma_grid=_LUMA_A),
        sequence=1,
        now_ns=now,
    )
    executor.tick(
        _traversal_board(now + 1_100_000_000, luma_grid=_LUMA_A),
        sequence=2,
        now_ns=now + 1_100_000_000,
    )
    stopped = executor.tick(
        _traversal_board(now + 2_200_000_000, luma_grid=_LUMA_A),
        sequence=3,
        now_ns=now + 2_200_000_000,
    )
    assert stopped.run.outcome == SkillOutcome.FAILED

    restarted_ns = now + 2_300_000_000
    executor.start(spec, run_id="second-traversal", now_ns=restarted_ns)
    restarted = executor.tick(
        _traversal_board(restarted_ns, luma_grid=_LUMA_A),
        sequence=4,
        now_ns=restarted_ns,
    )

    assert restarted.run.outcome == SkillOutcome.RUNNING
    assert executor._outcome_verifier.active_run_id == "second-traversal"
