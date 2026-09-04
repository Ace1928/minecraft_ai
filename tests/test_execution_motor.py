from __future__ import annotations

import time

import pytest

from minecraft_ai.action_levels import ActionLevel
from minecraft_ai.builtin_skills import build_bootstrap_skill_library
from minecraft_ai.execution import SkillExecutor, conditions_satisfied
from minecraft_ai.motor import BootstrapMotorPolicy, MotorIntent
from minecraft_ai.outcome_verifier import OutcomeKind, OutcomeSignal, OutcomeStatus
from minecraft_ai.perception import (
    FrameState,
    PerceptionBlackboard,
    PerceptionFact,
    ScreenRegion,
    Track,
)
from minecraft_ai.safety import MotorAction
from minecraft_ai.perception_service import BEDROCK_HOTBAR_LOG_COUNT_SOURCE
from minecraft_ai.skills import (
    SkillActionPermissions,
    SkillCondition,
    SkillFailureCode,
    SkillOutcome,
    SkillSpec,
)
from minecraft_ai.trajectory import ActionOrigin


_ROCKET_SOURCE = "learned:minestudio-rocket2:test:aux-localization:not-training-label"


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


class _MovingIntentCapturePolicy(_IntentCapturePolicy):
    def act(
        self,
        blackboard: PerceptionBlackboard,
        intent: MotorIntent,
        *,
        sequence: int,
    ) -> MotorAction:
        super().act(blackboard, intent, sequence=sequence)
        return MotorAction(sequence=sequence, keys_down=("w",), duration_ms=50)


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


def _fact(
    key: str,
    value: str | int | float | bool,
    *,
    confidence: float = 1.0,
) -> PerceptionFact:
    return PerceptionFact(
        key=key,
        value=value,
        confidence=confidence,
        observed_ns=time.monotonic_ns(),
        source="test",
        expires_after_ms=1_000_000,
    )


def _hotbar_log_fact(
    count: int,
    *,
    observed_ns: int,
    source: str = BEDROCK_HOTBAR_LOG_COUNT_SOURCE,
    expires_after_ms: int = 250,
) -> PerceptionFact:
    return PerceptionFact(
        key="inventory.hotbar.logs",
        value=count,
        confidence=0.995,
        observed_ns=observed_ns,
        source=source,
        expires_after_ms=expires_after_ms,
    )


def _reacquisition_board(
    *,
    observed_ns: int,
    centered: bool,
    reference_observed_ns: int | None = None,
) -> PerceptionBlackboard:
    region = (
        ScreenRegion(x=0.42, y=0.38, width=0.16, height=0.24)
        if centered
        else ScreenRegion(
            x=0.318410882477959,
            y=0.09819203615188599,
            width=0.15090409368276597,
            height=0.2760867178440094,
        )
    )
    facts = [
        PerceptionFact(
            key=key,
            value=value,
            confidence=confidence,
            observed_ns=observed_ns,
            source=_ROCKET_SOURCE,
            expires_after_ms=5_000,
        )
        for key, value, confidence in (
            ("target.visible", True, 0.93),
            ("target.tracking_confidence", 0.93, 1.0),
            ("target.exists_probability", 0.83, 1.0),
            ("target.kind", "dirt", 0.93),
        )
    ]
    if reference_observed_ns is not None:
        facts.append(
            PerceptionFact(
                key="target.reference_available",
                value=True,
                confidence=1.0,
                observed_ns=reference_observed_ns,
                source="operator:cross-view-reference:operator:dirt",
                expires_after_ms=250,
            )
        )
    board = PerceptionBlackboard()
    board.publish(
        FrameState(
            frame_id=1,
            captured_ns=max(observed_ns, reference_observed_ns or observed_ns),
            instance_id="bedrock:reacquisition",
            width=1280,
            height=720,
            facts=tuple(facts),
            tracks=(
                Track(
                    track_id="operator:dirt",
                    label="dirt",
                    confidence=0.93,
                    region=region,
                    first_seen_ns=observed_ns,
                    last_seen_ns=observed_ns,
                    attributes={
                        "source": "operator",
                        "tracking_source": _ROCKET_SOURCE,
                        "target_exists_probability": 0.83,
                    },
                ),
            ),
        )
    )
    return board


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


def test_open_inventory_success_requires_calibrated_inventory_overlay() -> None:
    spec = build_bootstrap_skill_library().get("open_inventory")

    assert conditions_satisfied(
        spec.success_conditions,
        _board(_fact("scene.inventory_overlay", True, confidence=0.995)),
    )
    assert not conditions_satisfied(
        spec.success_conditions,
        _board(_fact("scene.inventory_overlay", True, confidence=0.90)),
    )
    assert not conditions_satisfied(
        spec.success_conditions,
        _board(_fact("scene.inventory_overlay", False, confidence=0.995)),
    )


def test_open_inventory_emits_one_bounded_toggle_then_waits_for_proof() -> None:
    policy = _IntentCapturePolicy()
    executor = SkillExecutor(policy)
    spec = build_bootstrap_skill_library().get("open_inventory")
    world = _board(_fact("scene.inventory_overlay", False, confidence=0.995))
    executor.start(spec, run_id="open-inventory", now_ns=100)

    first = executor.tick(world, sequence=31, now_ns=200)
    waiting = executor.tick(world, sequence=32, now_ns=300)

    assert first.run.outcome == SkillOutcome.RUNNING
    assert first.action == MotorAction(
        sequence=31,
        keys_down=("e",),
        keys_up=("e",),
        duration_ms=150,
    )
    assert first.action_origin == ActionOrigin.SYNTHETIC
    assert waiting.run.outcome == SkillOutcome.RUNNING
    assert waiting.action is None
    assert policy.intent is None


def test_open_inventory_is_a_noop_when_overlay_is_already_present() -> None:
    policy = _IntentCapturePolicy()
    executor = SkillExecutor(policy)
    spec = build_bootstrap_skill_library().get("open_inventory")
    executor.start(spec, run_id="already-open", now_ns=100)

    tick = executor.tick(
        _board(_fact("scene.inventory_overlay", True, confidence=0.995)),
        sequence=9,
        now_ns=200,
    )

    assert tick.run.outcome == SkillOutcome.SUCCEEDED
    assert tick.action is not None
    assert "e" not in tick.action.keys_down
    assert policy.intent is None


def test_collect_recent_drop_is_bounded_and_disables_interactions() -> None:
    policy = _IntentCapturePolicy()
    executor = SkillExecutor(policy)
    spec = build_bootstrap_skill_library().get("collect_recent_drop")
    board = _board(_fact("collection.recent_log_break", True))
    executor.start(spec, run_id="collect-log", now_ns=100)

    running = executor.tick(board, sequence=1, now_ns=200)

    assert running.run.outcome == SkillOutcome.RUNNING
    assert policy.intent is not None
    assert policy.intent.instruction == "collect the dropped item"
    assert policy.intent.parameters == {
        "allow_attack": False,
        "allow_drop": False,
        "allow_hotbar": False,
        "allow_inventory": False,
        "allow_jump": True,
        "allow_use": False,
    }

    terminal = executor.tick(board, sequence=2, now_ns=5_000_000_101)

    assert terminal.run.outcome == SkillOutcome.FAILED
    assert terminal.run.failure_code == SkillFailureCode.RESOURCE_PICKUP_UNVERIFIED
    assert terminal.run.failure_reason == "resource.pickup_unverified"


@pytest.mark.parametrize("before,after", [(0, 1), (6, 7), (8, 9), (9, 10), (15, 16)])
def test_collect_recent_drop_requires_stable_post_action_log_increment(
    before: int, after: int
) -> None:
    started_ns = 1_000_000_000
    policy = _MovingIntentCapturePolicy()
    executor = SkillExecutor(policy)
    spec = build_bootstrap_skill_library().get("collect_recent_drop")
    board = _board(
        _fact("collection.recent_log_break", True),
        # The drop was automatically picked up before collection began. This
        # first post-start frame must not overwrite the pre-break baseline.
        _hotbar_log_fact(after, observed_ns=started_ns + 10_000_000),
    )
    executor.start(
        spec,
        run_id="collect-verified",
        now_ns=started_ns,
        collection_hotbar_log_baseline=_hotbar_log_fact(
            before, observed_ns=started_ns - 3_000_000
        ),
    )

    baseline = executor.tick(board, sequence=1, now_ns=started_ns + 20_000_000)
    assert baseline.run.outcome == SkillOutcome.RUNNING
    assert baseline.action is not None and baseline.action.keys_down == ("w",)

    board.merge_semantics(
        instance_id="bedrock:test",
        facts=(_hotbar_log_fact(after, observed_ns=started_ns + 100_000_000),),
    )
    first_positive = executor.tick(board, sequence=2, now_ns=started_ns + 110_000_000)
    repeated_same_frame = executor.tick(board, sequence=3, now_ns=started_ns + 150_000_000)
    assert first_positive.run.outcome == SkillOutcome.RUNNING
    assert repeated_same_frame.run.outcome == SkillOutcome.RUNNING

    board.merge_semantics(
        instance_id="bedrock:test",
        facts=(_hotbar_log_fact(after, observed_ns=started_ns + 360_000_000),),
    )
    terminal = executor.tick(board, sequence=4, now_ns=started_ns + 370_000_000)

    assert terminal.run.outcome == SkillOutcome.SUCCEEDED
    assert terminal.action is not None and "w" in terminal.action.keys_up
    assert terminal.outcome_verification is not None
    assert terminal.outcome_verification.kind == OutcomeKind.RESOURCE_ACQUISITION
    assert terminal.outcome_verification.status == OutcomeStatus.SUCCEEDED
    assert terminal.outcome_verification.signal == OutcomeSignal.RESOURCE_ACQUIRED
    assert terminal.outcome_verification.target_kind == "log"
    assert terminal.outcome_verification.evidence_keys == ("inventory.hotbar.logs",)
    assert executor.tick(
        board, sequence=5, now_ns=started_ns + 380_000_000
    ).outcome_verification is None


def test_collect_recent_drop_rejects_unbound_or_noncanonical_increments() -> None:
    started_ns = 2_000_000_000
    policy = _IntentCapturePolicy()
    executor = SkillExecutor(policy)
    spec = build_bootstrap_skill_library().get("collect_recent_drop")
    board = _board(
        _fact("collection.recent_log_break", True),
        _hotbar_log_fact(0, observed_ns=started_ns + 10_000_000),
    )
    executor.start(
        spec,
        run_id="collect-no-motion",
        now_ns=started_ns,
        collection_hotbar_log_baseline=_hotbar_log_fact(
            0, observed_ns=started_ns - 1_000_000
        ),
    )
    executor.tick(board, sequence=1, now_ns=started_ns + 20_000_000)

    for sequence, observed_ns in enumerate(
        (started_ns + 100_000_000, started_ns + 400_000_000),
        start=2,
    ):
        board.merge_semantics(
            instance_id="bedrock:test",
            facts=(_hotbar_log_fact(1, observed_ns=observed_ns),),
        )
        tick = executor.tick(board, sequence=sequence, now_ns=observed_ns + 10_000_000)
        assert tick.run.outcome == SkillOutcome.RUNNING

    terminal = executor.tick(board, sequence=4, now_ns=started_ns + 5_000_000_001)
    assert terminal.run.outcome == SkillOutcome.FAILED
    assert terminal.run.failure_code == SkillFailureCode.RESOURCE_PICKUP_UNVERIFIED


def test_collect_recent_drop_rejects_vlm_stale_unchanged_and_multi_item_counts() -> None:
    started_ns = 3_000_000_000
    policy = _MovingIntentCapturePolicy()
    executor = SkillExecutor(policy)
    spec = build_bootstrap_skill_library().get("collect_recent_drop")
    board = _board(
        _fact("collection.recent_log_break", True),
        _hotbar_log_fact(0, observed_ns=started_ns + 10_000_000),
    )
    executor.start(
        spec,
        run_id="collect-reject",
        now_ns=started_ns,
        collection_hotbar_log_baseline=_hotbar_log_fact(
            0, observed_ns=started_ns - 1_000_000
        ),
    )
    executor.tick(board, sequence=1, now_ns=started_ns + 20_000_000)

    rejected = (
        _hotbar_log_fact(
            1,
            observed_ns=started_ns + 100_000_000,
            source="vlm:test:inventory-query",
        ),
        _hotbar_log_fact(0, observed_ns=started_ns + 200_000_000),
        _hotbar_log_fact(2, observed_ns=started_ns + 300_000_000),
        _hotbar_log_fact(
            1,
            observed_ns=started_ns + 400_000_000,
            expires_after_ms=1,
        ),
    )
    for sequence, fact in enumerate(rejected, start=2):
        board.merge_semantics(instance_id="bedrock:test", facts=(fact,))
        tick = executor.tick(
            board,
            sequence=sequence,
            now_ns=fact.observed_ns + 10_000_000,
        )
        assert tick.run.outcome == SkillOutcome.RUNNING

    terminal = executor.tick(board, sequence=6, now_ns=started_ns + 5_000_000_001)
    assert terminal.run.outcome == SkillOutcome.FAILED
    assert terminal.run.failure_code == SkillFailureCode.RESOURCE_PICKUP_UNVERIFIED


@pytest.mark.parametrize(
    "baseline_update",
    [
        None,
        {"key": "inventory.logs"},
        {"source": "vlm:test:inventory-query"},
        {"confidence": 0.9},
        {"value": True},
        {"observed_ns": 4_010_000_000},
    ],
)
def test_collect_recent_drop_never_invents_a_missing_prebreak_baseline(
    baseline_update: dict[str, object] | None,
) -> None:
    started_ns = 4_000_000_000
    executor = SkillExecutor(_MovingIntentCapturePolicy())
    baseline = (
        None
        if baseline_update is None
        else _hotbar_log_fact(0, observed_ns=started_ns - 1_000_000).model_copy(
            update=baseline_update
        )
    )
    executor.start(
        build_bootstrap_skill_library().get("collect_recent_drop"),
        run_id="no-prebreak-evidence",
        now_ns=started_ns,
        collection_hotbar_log_baseline=baseline,
    )
    board = _board(_fact("collection.recent_log_break", True))
    for sequence, count, offset_ns in (
        (1, 0, 10_000_000), (2, 1, 100_000_000), (3, 1, 400_000_000)
    ):
        board.merge_semantics(
            instance_id="bedrock:test",
            facts=(_hotbar_log_fact(count, observed_ns=started_ns + offset_ns),),
        )
        result = executor.tick(
            board, sequence=sequence, now_ns=started_ns + offset_ns + 1_000_000
        )
        assert result.run.outcome == SkillOutcome.RUNNING
    terminal = executor.tick(board, sequence=4, now_ns=started_ns + 5_000_000_001)
    assert terminal.run.failure_code == SkillFailureCode.RESOURCE_PICKUP_UNVERIFIED


def test_close_inventory_emits_one_bounded_toggle_then_waits_for_proof() -> None:
    policy = _IntentCapturePolicy()
    executor = SkillExecutor(policy)
    spec = build_bootstrap_skill_library().get("close_open_inventory")
    inventory = _board(_fact("scene.playable", False))
    executor.start(spec, run_id="close-inventory", now_ns=100)

    first = executor.tick(inventory, sequence=41, now_ns=200)
    waiting = executor.tick(inventory, sequence=42, now_ns=300)

    assert first.run.outcome == SkillOutcome.RUNNING
    assert first.action == MotorAction(
        sequence=41,
        keys_down=("e",),
        keys_up=("e",),
        duration_ms=150,
    )
    assert first.action_origin == ActionOrigin.SYNTHETIC
    assert waiting.run.outcome == SkillOutcome.RUNNING
    assert waiting.action is None
    assert policy.intent is None


def test_close_inventory_is_a_noop_when_world_is_already_playable() -> None:
    policy = _IntentCapturePolicy()
    executor = SkillExecutor(policy)
    spec = build_bootstrap_skill_library().get("close_open_inventory")
    executor.start(spec, run_id="already-closed", now_ns=100)

    tick = executor.tick(_board(_fact("scene.playable", True)), sequence=9, now_ns=200)

    assert tick.run.outcome == SkillOutcome.SUCCEEDED
    assert tick.action is not None
    assert "e" not in tick.action.keys_down
    assert policy.intent is None


def test_skill_contract_becomes_learned_policy_instruction() -> None:
    policy = _IntentCapturePolicy()
    executor = SkillExecutor(policy)
    spec = SkillSpec(
        skill_id="mine_log",
        name="Mine log",
        description="Approach and mine a visible tree log",
        policy_ref="mine",
        action_level=ActionLevel.GROUNDED,
        policy_instruction="mine log",
        policy_condition_scale=5.5,
    )
    executor.start(
        spec,
        run_id="r1",
        parameters={"target": "oak_log", "wood": "oak"},
        now_ns=100,
    )

    tick = executor.tick(_board(), sequence=1, now_ns=200)

    assert policy.intent is not None
    assert tick.motor_intent == policy.intent
    assert tick.policy_status == {"policy_id": "capture"}
    assert tick.action_origin == ActionOrigin.POLICY
    assert policy.intent.episode_id == "r1"
    assert policy.intent.action_level == ActionLevel.GROUNDED
    assert policy.intent.instruction == "mine log"
    assert policy.intent.condition_scale == 5.5
    assert policy.intent.target_label == "oak_log"
    assert executor.parameters == {"target": "oak_log", "wood": "oak"}
    assert executor.run is not None
    assert executor.run.parameters == {"target": "oak_log", "wood": "oak"}
    assert executor.instruction == (
        "Approach and mine a visible tree log. Parameters: target=oak_log, wood=oak"
    )


def test_skill_can_initiate_from_verified_alternative_evidence() -> None:
    policy = _IntentCapturePolicy()
    executor = SkillExecutor(policy)
    spec = SkillSpec(
        skill_id="approach_grounded",
        name="Approach grounded target",
        policy_ref="approach",
        preconditions=(SkillCondition(key="target.visible", operator="truthy"),),
        initiation_alternatives=(
            (SkillCondition(key="target.reference_available", operator="truthy"),),
        ),
    )
    executor.start(spec, run_id="grounded", now_ns=100)

    tick = executor.tick(
        _board(_fact("target.reference_available", True)),
        sequence=1,
        now_ns=200,
    )

    assert tick.run.outcome == SkillOutcome.RUNNING
    assert policy.intent is not None
    assert policy.intent.mode == "approach"


def test_reacquire_rejects_stale_off_center_tracking_despite_fresh_reference() -> None:
    started_ns = 1_000_000_000
    policy = _IntentCapturePolicy()
    executor = SkillExecutor(policy)
    executor.start(
        build_bootstrap_skill_library().get("reacquire_target"),
        run_id="reacquire:stale",
        parameters={"target": "dirt"},
        now_ns=started_ns,
    )

    tick = executor.tick(
        _reacquisition_board(
            observed_ns=started_ns - 100_000_000,
            centered=False,
            reference_observed_ns=started_ns + 50_000_000,
        ),
        sequence=1,
        now_ns=started_ns + 100_000_000,
    )

    assert tick.run.outcome == SkillOutcome.RUNNING
    assert policy.intent is not None
    assert policy.intent.mode == "navigate"


def test_reacquire_rejects_fresh_rocket_track_away_from_crosshair() -> None:
    started_ns = 1_000_000_000
    policy = _IntentCapturePolicy()
    executor = SkillExecutor(policy)
    executor.start(
        build_bootstrap_skill_library().get("reacquire_target"),
        run_id="reacquire:off-center",
        parameters={"target": "dirt"},
        now_ns=started_ns,
    )

    tick = executor.tick(
        _reacquisition_board(
            observed_ns=started_ns + 50_000_000,
            centered=False,
        ),
        sequence=1,
        now_ns=started_ns + 100_000_000,
    )

    assert tick.run.outcome == SkillOutcome.RUNNING
    assert policy.intent is not None


def test_reacquire_accepts_only_fresh_centered_same_target_rocket_track() -> None:
    started_ns = 1_000_000_000
    policy = _IntentCapturePolicy()
    executor = SkillExecutor(policy)
    executor.start(
        build_bootstrap_skill_library().get("reacquire_target"),
        run_id="reacquire:centered",
        parameters={"target": "dirt"},
        now_ns=started_ns,
    )

    tick = executor.tick(
        _reacquisition_board(
            observed_ns=started_ns + 50_000_000,
            centered=True,
        ),
        sequence=1,
        now_ns=started_ns + 100_000_000,
    )

    assert tick.run.outcome == SkillOutcome.SUCCEEDED
    assert policy.intent is None


def test_skill_action_permissions_bound_learned_policy_without_replacing_it() -> None:
    policy = _IntentCapturePolicy()
    executor = SkillExecutor(policy)
    spec = SkillSpec(
        skill_id="retreat",
        name="Retreat",
        policy_ref="retreat",
        action_permissions=SkillActionPermissions(allow_attack=False, allow_use=False),
    )
    executor.start(
        spec,
        run_id="r1",
        parameters={"allow_attack": True, "allow_jump": False},
        now_ns=100,
    )

    executor.tick(_board(), sequence=1, now_ns=200)

    assert policy.intent is not None
    assert policy.intent.parameters == {
        "allow_attack": False,
        "allow_drop": True,
        "allow_hotbar": True,
        "allow_inventory": True,
        "allow_jump": False,
        "allow_use": False,
    }
    # Planner bindings remain stable, so policy constraints cannot cause the
    # runtime to cancel/restart an otherwise identical option every tick.
    assert executor.parameters == {"allow_attack": True, "allow_jump": False}
    assert executor.policy_parameters == policy.intent.parameters


def test_skill_success_releases_held_input() -> None:
    policy = BootstrapMotorPolicy()
    executor = SkillExecutor(policy)
    spec = SkillSpec(
        skill_id="attack",
        name="Attack",
        policy_ref="attack",
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
    assert done.motor_intent == running.motor_intent
    assert done.policy_status["policy_id"] == policy.policy_id
    assert done.action_origin == ActionOrigin.RESET


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
