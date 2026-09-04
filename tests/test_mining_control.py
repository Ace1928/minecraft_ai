from __future__ import annotations

import time

import pytest

from minecraft_ai.builtin_skills import build_bootstrap_skill_library
from minecraft_ai.execution import SkillExecutor, initiation_satisfied
from minecraft_ai.mining_control import MiningLeaseGuard
from minecraft_ai.motor import MotorIntent
from minecraft_ai.perception import (
    FrameState,
    PerceptionBlackboard,
    PerceptionFact,
    ScreenRegion,
    Track,
)
from minecraft_ai.perception_service import (
    BEDROCK_HOTBAR_LOG_COUNT_SOURCE,
    BootstrapFastPerception,
)
from minecraft_ai.platforms.bedrock_x11 import CapturedFrame
from minecraft_ai.runtime import _terminal_run_event, _terminal_run_memory
from minecraft_ai.safety import MotorAction
from minecraft_ai.skills import (
    SkillFailureCode,
    SkillOutcome,
    SkillRun,
    SkillSpec,
    SkillStats,
)
from minecraft_ai.trajectory import ActionOrigin


_HASH_A = "0000000000000000"
_HASH_B = "ffffffffffffffff"
_ROCKET_SOURCE = "learned:learned:minestudio-rocket2:test:aux-localization:not-training-label"


class _ScriptedPolicy:
    policy_id = "scripted-miner"

    def __init__(self, *actions: MotorAction) -> None:
        self.actions = list(actions)
        self.last_sequence = -1
        self.reset_calls = 0

    def act(
        self,
        _blackboard: PerceptionBlackboard,
        _intent: MotorIntent,
        *,
        sequence: int,
    ) -> MotorAction:
        self.last_sequence = sequence
        template = self.actions.pop(0) if self.actions else MotorAction(sequence=0)
        return template.model_copy(update={"sequence": sequence})

    def reset(self) -> MotorAction:
        self.reset_calls += 1
        self.last_sequence += 1
        return MotorAction(sequence=max(0, self.last_sequence))


def _fact(
    key: str,
    value: str | int | float | bool,
    *,
    now_ns: int,
    source: str = "vlm:test",
) -> PerceptionFact:
    return PerceptionFact(
        key=key,
        value=value,
        confidence=1.0,
        observed_ns=now_ns,
        source=source,
        expires_after_ms=30_000,
    )


def _mining_board(
    *,
    now_ns: int,
    kind: str = "oak_log",
    item: str | None = "minecraft:stick",
    track_id: str = "target:one",
    track_label: str | None = None,
    track_attributes: dict[str, str | int | float | bool] | None = None,
    track_seen_ns: int | None = None,
    track_region: ScreenRegion | None = None,
    target_source: str = "vlm:test",
    scene_source: str = "vlm:test",
    crosshair_hash: str | None = _HASH_A,
    scene_hash: str | None = _HASH_A,
    frame_hash: str = _HASH_A,
    include_kind: bool = True,
    include_mineable: bool = True,
    include_selected_slot: bool = True,
    include_visible: bool = True,
    extra_facts: tuple[PerceptionFact, ...] = (),
) -> PerceptionBlackboard:
    facts = [
        _fact("frame.dhash", frame_hash, now_ns=now_ns, source="bootstrap:test"),
        _fact("scene.playable", True, now_ns=now_ns),
    ]
    if include_visible:
        facts.append(_fact("target.visible", True, now_ns=now_ns, source=target_source))
    if scene_hash is not None:
        facts.append(
            _fact(
                "scene.observation_dhash",
                scene_hash,
                now_ns=now_ns,
                source=scene_source,
            )
        )
    if include_mineable:
        facts.append(_fact("target.mineable", True, now_ns=now_ns, source=target_source))
    if include_kind:
        facts.append(_fact("target.kind", kind, now_ns=now_ns, source=target_source))
    if include_selected_slot:
        facts.append(_fact("player.selected_slot", 0, now_ns=now_ns))
    if item is not None and include_selected_slot:
        facts.append(_fact("hotbar.slot.0.item", item, now_ns=now_ns))
    if crosshair_hash is not None:
        facts.append(
            _fact(
                "frame.crosshair_dhash",
                crosshair_hash,
                now_ns=now_ns,
                source="bootstrap:test",
            )
        )
    facts.extend(extra_facts)
    board = PerceptionBlackboard()
    board.publish(
        FrameState(
            frame_id=1,
            captured_ns=now_ns,
            instance_id="bedrock:test",
            width=1280,
            height=720,
            tracks=(
                Track(
                    track_id=track_id,
                    label=track_label or kind,
                    confidence=1.0,
                    region=track_region
                    or ScreenRegion(x=0.4, y=0.35, width=0.2, height=0.3),
                    first_seen_ns=track_seen_ns or now_ns,
                    last_seen_ns=track_seen_ns or now_ns,
                    attributes=track_attributes or {},
                ),
            ),
            facts=tuple(facts),
        )
    )
    return board


def _executor(
    policy: _ScriptedPolicy,
    *,
    now_ns: int,
    mode: str = "mine",
    skill_id: str = "mine_test",
    parameters: dict[str, str | int | float | bool] | None = None,
) -> SkillExecutor:
    executor = SkillExecutor(policy)
    executor.start(
        SkillSpec(
            skill_id=skill_id,
            name="Mine test block",
            policy_ref=mode,
            recovery_skills=("reacquire_target",),
        ),
        run_id="episode:one",
        parameters=parameters,
        now_ns=now_ns - 1,
    )
    return executor


def test_guard_never_starts_attack_when_policy_did_not() -> None:
    now = time.monotonic_ns()
    policy = _ScriptedPolicy(MotorAction(sequence=0))
    executor = _executor(policy, now_ns=now)

    tick = executor.tick(_mining_board(now_ns=now), sequence=1, now_ns=now)

    assert tick.run.outcome == SkillOutcome.RUNNING
    assert tick.action is not None
    assert "left" not in tick.action.buttons_down
    assert tick.action_origin == ActionOrigin.POLICY


def test_operator_marked_target_can_acquire_without_policy_attack() -> None:
    now = time.monotonic_ns()
    policy = _ScriptedPolicy(MotorAction(sequence=0), MotorAction(sequence=0))
    executor = _executor(
        policy,
        now_ns=now,
        skill_id="mine_visible_block",
        parameters={"target": "dirt"},
    )
    reference_only = _mining_board(
        now_ns=now,
        kind="dirt",
        item=None,
        track_attributes={"source": "operator"},
        track_seen_ns=now - 60_000_000_000,
        include_visible=False,
        include_kind=False,
        include_mineable=False,
        include_selected_slot=False,
        extra_facts=(
            _fact(
                "target.reference_available",
                True,
                now_ns=now,
                source="operator:cross-view-reference:target:one",
            ),
        ),
    )

    waiting = executor.tick(reference_only, sequence=1, now_ns=now)
    started = executor.tick(
        _mining_board(
            now_ns=now + 100_000_000,
            kind="dirt",
            item=None,
            track_attributes={"source": "operator"},
            track_seen_ns=now - 60_000_000_000,
            target_source="operator:explicit-grounding:target:one",
            scene_hash=None,
            include_mineable=False,
            include_selected_slot=False,
        ),
        sequence=2,
        now_ns=now + 100_000_000,
    )

    assert waiting.run.outcome == SkillOutcome.RUNNING
    assert waiting.action is not None and "left" not in waiting.action.buttons_down
    assert started.run.outcome == SkillOutcome.RUNNING
    assert started.action is not None and started.action.buttons_down == ("left",)
    assert started.action_origin == ActionOrigin.SYNTHETIC


def test_operator_mining_aims_once_per_new_track_then_attacks_only_when_centered() -> None:
    now = time.monotonic_ns()
    policy = _ScriptedPolicy(
        MotorAction(
            sequence=0,
            keys_down=("w",),
            buttons_down=("left",),
            mouse_dx=77,
        ),
        MotorAction(sequence=0),
        MotorAction(sequence=0),
    )
    executor = _executor(
        policy,
        now_ns=now,
        skill_id="mine_visible_block",
        parameters={"target": "dirt"},
    )
    tracked_attributes: dict[str, str | int | float | bool] = {
        "source": "operator",
        "tracking_source": _ROCKET_SOURCE,
        "target_exists_probability": 0.96,
    }

    off_center = _mining_board(
        now_ns=now,
        kind="dirt",
        item=None,
        track_attributes=tracked_attributes,
        track_seen_ns=now - 25_000_000,
        track_region=ScreenRegion(x=0.20, y=0.30, width=0.10, height=0.50),
        target_source=_ROCKET_SOURCE,
        include_selected_slot=False,
        extra_facts=(
            _fact(
                "target.reference_available",
                True,
                now_ns=now,
                source="operator:cross-view-reference:target:one",
            ),
        ),
    )
    aimed = executor.tick(off_center, sequence=1, now_ns=now)
    repeated = executor.tick(off_center, sequence=2, now_ns=now + 50_000_000)
    centered = _mining_board(
        now_ns=now + 100_000_000,
        kind="dirt",
        item=None,
        track_attributes=tracked_attributes,
        track_seen_ns=now + 100_000_000,
        target_source=_ROCKET_SOURCE,
        include_selected_slot=False,
        extra_facts=(
            _fact(
                "target.reference_available",
                True,
                now_ns=now + 100_000_000,
                source="operator:cross-view-reference:target:one",
            ),
        ),
    )
    started = executor.tick(centered, sequence=3, now_ns=now + 100_000_000)

    assert aimed.action is not None
    assert aimed.action.keys_down == ()
    assert aimed.action.keys_up == ("w",)
    assert aimed.action.buttons_down == ()
    assert aimed.action.mouse_dx == -10
    assert aimed.action.mouse_dy == 0
    assert aimed.action_origin == ActionOrigin.SYNTHETIC
    assert repeated.action is not None
    assert repeated.action.mouse_dx == repeated.action.mouse_dy == 0
    assert "left" not in repeated.action.buttons_down
    assert started.action is not None and started.action.buttons_down == ("left",)
    assert started.action.mouse_dx == started.action.mouse_dy == 0
    assert sum(
        tick.action is not None and "left" in tick.action.buttons_down
        for tick in (aimed, repeated, started)
    ) == 1


def test_operator_aim_requires_matching_fresh_reference_and_mining_skill() -> None:
    now = time.monotonic_ns()
    tracked = _mining_board(
        now_ns=now,
        kind="dirt",
        item=None,
        track_attributes={
            "source": "operator",
            "tracking_source": _ROCKET_SOURCE,
            "target_exists_probability": 0.99,
        },
        track_region=ScreenRegion(x=0.05, y=0.30, width=0.10, height=0.50),
        target_source=_ROCKET_SOURCE,
        include_selected_slot=False,
        extra_facts=(
            _fact(
                "target.reference_available",
                True,
                now_ns=now,
                source="operator:cross-view-reference:target:two",
            ),
        ),
    )
    operator_executor = _executor(
        _ScriptedPolicy(MotorAction(sequence=0)),
        now_ns=now,
        skill_id="mine_visible_block",
        parameters={"target": "dirt"},
    )
    generic_executor = _executor(
        _ScriptedPolicy(MotorAction(sequence=0)),
        now_ns=now,
    )

    mismatched = operator_executor.tick(tracked, sequence=1, now_ns=now)
    generic = generic_executor.tick(tracked, sequence=1, now_ns=now)

    assert mismatched.action is not None
    assert mismatched.action.mouse_dx == mismatched.action.mouse_dy == 0
    assert "left" not in mismatched.action.buttons_down
    assert generic.action is not None
    assert generic.action.mouse_dx == generic.action.mouse_dy == 0
    assert "left" not in generic.action.buttons_down


def test_generic_no_policy_attack_does_not_use_operator_reference() -> None:
    now = time.monotonic_ns()
    policy = _ScriptedPolicy(MotorAction(sequence=0), MotorAction(sequence=0))
    executor = _executor(
        policy,
        now_ns=now,
        skill_id="mine_visible_block",
    )
    marked = _mining_board(
        now_ns=now,
        kind="dirt",
        track_attributes={"source": "operator"},
        include_visible=False,
        include_kind=False,
        include_mineable=False,
        include_selected_slot=False,
        extra_facts=(
            _fact(
                "target.reference_available",
                True,
                now_ns=now,
                source="operator:cross-view-reference:target:one",
            ),
        ),
    )

    first = executor.tick(marked, sequence=1, now_ns=now)
    second = executor.tick(
        _mining_board(now_ns=now + 100_000_000, kind="dirt", item=None),
        sequence=2,
        now_ns=now + 100_000_000,
    )

    assert first.action is not None and "left" not in first.action.buttons_down
    assert second.action is not None and "left" not in second.action.buttons_down


@pytest.mark.parametrize(
    ("reference_source", "track_source", "reference_age_ms"),
    [
        ("operator:cross-view-reference:target:two", "operator", 0),
        ("operator:cross-view-reference:target:one", "learned:rocket", 0),
        ("operator:cross-view-reference:target:one", "operator", 600),
    ],
)
def test_invalid_operator_reference_never_authorizes_no_policy_attack(
    reference_source: str,
    track_source: str,
    reference_age_ms: int,
) -> None:
    now = time.monotonic_ns()
    policy = _ScriptedPolicy(MotorAction(sequence=0), MotorAction(sequence=0))
    executor = _executor(
        policy,
        now_ns=now,
        skill_id="mine_visible_block",
        parameters={"target": "dirt"},
    )
    invalid_reference = _mining_board(
        now_ns=now,
        kind="dirt",
        track_attributes={"source": track_source},
        include_visible=False,
        include_kind=False,
        include_mineable=False,
        include_selected_slot=False,
        extra_facts=(
            _fact(
                "target.reference_available",
                True,
                now_ns=now - reference_age_ms * 1_000_000,
                source=reference_source,
            ),
        ),
    )

    first = executor.tick(invalid_reference, sequence=1, now_ns=now)
    second = executor.tick(
        _mining_board(now_ns=now + 100_000_000, kind="dirt", item=None),
        sequence=2,
        now_ns=now + 100_000_000,
    )

    assert first.action is not None and "left" not in first.action.buttons_down
    assert second.action is not None and "left" not in second.action.buttons_down


def test_explicit_operator_reference_can_initiate_guarded_mining() -> None:
    now = time.monotonic_ns()
    board = PerceptionBlackboard()
    board.publish(
        FrameState(
            frame_id=1,
            captured_ns=now,
            instance_id="bedrock:test",
            width=1280,
            height=720,
            facts=(
                _fact(
                    "target.reference_available",
                    True,
                    now_ns=now,
                    source="operator:cross-view-reference:target:one",
                ),
            ),
        )
    )
    spec = build_bootstrap_skill_library().get("mine_visible_block")

    assert initiation_satisfied(spec, board, now_ns=now)
    assert spec.action_permissions.allow_attack is True
    assert spec.action_permissions.allow_use is False
    assert spec.action_permissions.allow_jump is False
    assert spec.action_permissions.allow_drop is False
    assert spec.action_permissions.allow_inventory is False
    assert spec.action_permissions.allow_hotbar is False


def test_unverified_reference_waits_then_synthesizes_attack_after_grounding() -> None:
    now = time.monotonic_ns()
    policy = _ScriptedPolicy(
        MotorAction(sequence=0, buttons_down=("left",)),
        MotorAction(sequence=0),
    )
    executor = _executor(policy, now_ns=now)
    reference_only = _mining_board(
        now_ns=now,
        include_visible=False,
        include_kind=False,
        include_mineable=False,
        include_selected_slot=False,
        extra_facts=(
            _fact(
                "target.reference_available",
                True,
                now_ns=now,
                source="operator:cross-view-reference:target:one",
            ),
        ),
    )

    waiting = executor.tick(reference_only, sequence=1, now_ns=now)
    started = executor.tick(
        _mining_board(now_ns=now + 100_000_000, kind="dirt", item=None),
        sequence=2,
        now_ns=now + 100_000_000,
    )

    assert waiting.run.outcome == SkillOutcome.RUNNING
    assert waiting.action is not None and "left" not in waiting.action.buttons_down
    assert started.run.outcome == SkillOutcome.RUNNING
    assert started.action is not None and "left" in started.action.buttons_down
    assert started.action_origin == ActionOrigin.SYNTHETIC


def test_attack_pulse_survives_releases_until_fresh_grounding_then_presses_once() -> None:
    now = time.monotonic_ns()
    policy = _ScriptedPolicy(
        MotorAction(sequence=0, buttons_down=("left",), mouse_dx=3),
        MotorAction(sequence=0, buttons_up=("left",)),
        MotorAction(sequence=0, buttons_up=("left",)),
        MotorAction(sequence=0),
    )
    executor = _executor(policy, now_ns=now)
    unavailable = _mining_board(
        now_ns=now,
        include_visible=False,
        include_kind=False,
        include_mineable=False,
        include_selected_slot=False,
    )

    aiming = executor.tick(unavailable, sequence=1, now_ns=now)
    first_release = executor.tick(
        _mining_board(
            now_ns=now + 100_000_000,
            include_visible=False,
            include_kind=False,
            include_mineable=False,
            include_selected_slot=False,
        ),
        sequence=2,
        now_ns=now + 100_000_000,
    )
    settling_release = executor.tick(
        _mining_board(
            now_ns=now + 800_000_000,
            include_visible=False,
            include_kind=False,
            include_mineable=False,
            include_selected_slot=False,
        ),
        sequence=3,
        now_ns=now + 800_000_000,
    )
    started = executor.tick(
        _mining_board(now_ns=now + 900_000_000, kind="dirt", item=None),
        sequence=4,
        now_ns=now + 900_000_000,
    )

    accepted = (aiming, first_release, settling_release, started)
    assert all(tick.run.outcome == SkillOutcome.RUNNING for tick in accepted)
    assert sum(
        tick.action is not None and "left" in tick.action.buttons_down
        for tick in accepted
    ) == 1
    assert all(
        tick.action is not None and "left" not in tick.action.buttons_up
        for tick in accepted
    )
    assert started.action is not None and started.action.buttons_down == ("left",)
    assert started.action_origin == ActionOrigin.SYNTHETIC


def test_pending_attack_has_typed_wall_clock_timeout_and_forced_release() -> None:
    now = time.monotonic_ns()
    policy = _ScriptedPolicy(
        MotorAction(sequence=0, buttons_down=("left",)),
        MotorAction(sequence=0),
    )
    executor = _executor(policy, now_ns=now)
    executor._mining_guard = MiningLeaseGuard(acquisition_timeout_ms=100)
    unavailable = _mining_board(
        now_ns=now,
        include_visible=False,
        include_kind=False,
        include_mineable=False,
        include_selected_slot=False,
    )

    waiting = executor.tick(unavailable, sequence=1, now_ns=now)
    stopped = executor.tick(
        _mining_board(
            now_ns=now + 101_000_000,
            include_visible=False,
            include_kind=False,
            include_mineable=False,
            include_selected_slot=False,
        ),
        sequence=2,
        now_ns=now + 101_000_000,
    )

    assert waiting.action is not None and "left" not in waiting.action.buttons_down
    assert stopped.run.failure_code == SkillFailureCode.MINING_ACQUISITION_TIMEOUT
    assert stopped.action is not None and "left" in stopped.action.buttons_up


def test_gather_rejects_non_oak_log_before_attack() -> None:
    now = time.monotonic_ns()
    policy = _ScriptedPolicy(MotorAction(sequence=0, buttons_down=("left",)))
    executor = SkillExecutor(policy)
    executor.start(
        build_bootstrap_skill_library().get("gather_nearby_wood"),
        run_id="gather-spruce-rejected",
        now_ns=now,
    )

    tick = executor.tick(
        _mining_board(
            now_ns=now,
            kind="spruce_log",
            item=None,
            extra_facts=(
                PerceptionFact(
                    key="inventory.hotbar.logs",
                    value=0,
                    confidence=0.995,
                    observed_ns=now,
                    source=BEDROCK_HOTBAR_LOG_COUNT_SOURCE,
                    expires_after_ms=250,
                ),
            ),
        ),
        sequence=1,
        now_ns=now,
    )

    assert tick.run.outcome == SkillOutcome.FAILED
    assert tick.run.failure_code == SkillFailureCode.MINING_TARGET_MISMATCH
    assert tick.action is not None
    assert "left" not in tick.action.buttons_down
    assert "left" in tick.action.buttons_up


@pytest.mark.parametrize(
    ("kind", "item", "expected_failure"),
    [
        ("stone", "minecraft:wooden_pickaxe", None),
        ("iron_ore", "minecraft:wooden_pickaxe", SkillFailureCode.MINING_WRONG_TOOL),
        ("diamond_ore", "minecraft:iron_pickaxe", None),
        ("obsidian", "minecraft:iron_pickaxe", SkillFailureCode.MINING_WRONG_TOOL),
        ("bedrock", "minecraft:diamond_pickaxe", SkillFailureCode.MINING_WRONG_TOOL),
        ("oak_log", "minecraft:stick", None),
        ("oak_log", None, None),
        ("dirt", None, None),
        ("stone", "minecraft:stick", SkillFailureCode.MINING_WRONG_TOOL),
        ("stone", None, "pending"),
    ],
)
def test_verified_target_and_tool_gate(
    kind: str,
    item: str | None,
    expected_failure: SkillFailureCode | str | None,
) -> None:
    now = time.monotonic_ns()
    policy = _ScriptedPolicy(MotorAction(sequence=0, buttons_down=("left",)))
    executor = _executor(policy, now_ns=now)

    tick = executor.tick(
        _mining_board(now_ns=now, kind=kind, item=item),
        sequence=1,
        now_ns=now,
    )

    if expected_failure == "pending":
        assert tick.run.outcome == SkillOutcome.RUNNING
        assert tick.action is not None and "left" not in tick.action.buttons_down
    elif expected_failure is None:
        assert tick.run.outcome == SkillOutcome.RUNNING
        released = executor.cancel(now_ns=now + 1).action
        assert released is not None and "left" in released.buttons_up
    else:
        assert tick.run.outcome == SkillOutcome.FAILED
        assert tick.run.failure_code == expected_failure
        assert tick.run.failure_reason == expected_failure.value
        assert tick.action is not None and "left" in tick.action.buttons_up
        assert tick.recovery_skills == ("reacquire_target",)


def test_gather_wood_rejects_verified_dirt_target() -> None:
    now = time.monotonic_ns()
    policy = _ScriptedPolicy(MotorAction(sequence=0, buttons_down=("left",)))
    executor = _executor(policy, now_ns=now, mode="gather_wood")

    tick = executor.tick(
        _mining_board(now_ns=now, kind="dirt"),
        sequence=1,
        now_ns=now,
    )

    assert tick.run.failure_code == SkillFailureCode.MINING_TARGET_MISMATCH
    assert tick.recovery_skills == ("reacquire_target",)


@pytest.mark.parametrize(
    "label",
    [
        "visible central oak trunk",
        "oak log trunk",
        "nearby tree trunk",
        "visible_oak_trunk",
        "visible_birch_trunk",
        "visible_tree_log",
    ],
)
def test_gather_wood_accepts_fresh_rocket_trunk_without_vlm_or_hotbar(
    label: str,
) -> None:
    now = time.monotonic_ns()
    policy = _ScriptedPolicy(MotorAction(sequence=0, buttons_down=("left",)))
    executor = _executor(policy, now_ns=now, mode="gather_wood")

    tick = executor.tick(
        _mining_board(
            now_ns=now,
            kind=label,
            item=None,
            track_label=label,
            track_attributes={"tracking_source": _ROCKET_SOURCE},
            target_source=_ROCKET_SOURCE,
            scene_hash=None,
            include_mineable=False,
            include_selected_slot=False,
        ),
        sequence=1,
        now_ns=now,
    )

    assert tick.run.outcome == SkillOutcome.RUNNING
    assert tick.action is not None and tick.action.buttons_down == ("left",)


def test_gather_wood_can_infer_kind_from_fresh_rocket_track() -> None:
    now = time.monotonic_ns()
    policy = _ScriptedPolicy(MotorAction(sequence=0, buttons_down=("left",)))
    executor = _executor(policy, now_ns=now, mode="gather_wood")

    tick = executor.tick(
        _mining_board(
            now_ns=now,
            kind="unused",
            item=None,
            track_label="visible central oak trunk",
            track_attributes={"tracking_source": _ROCKET_SOURCE},
            target_source=_ROCKET_SOURCE,
            scene_hash=None,
            include_kind=False,
            include_mineable=False,
            include_selected_slot=False,
        ),
        sequence=1,
        now_ns=now,
    )

    assert tick.run.outcome == SkillOutcome.RUNNING


def test_fresh_rocket_can_infer_exact_soft_block_as_hand_mineable() -> None:
    now = time.monotonic_ns()
    policy = _ScriptedPolicy(MotorAction(sequence=0, buttons_down=("left",)))
    executor = _executor(policy, now_ns=now)

    tick = executor.tick(
        _mining_board(
            now_ns=now,
            kind="dirt",
            item=None,
            track_attributes={"tracking_source": _ROCKET_SOURCE},
            target_source=_ROCKET_SOURCE,
            scene_hash=None,
            include_mineable=False,
            include_selected_slot=False,
        ),
        sequence=1,
        now_ns=now,
    )

    assert tick.run.outcome == SkillOutcome.RUNNING


def test_fresh_rocket_does_not_override_explicit_not_mineable_evidence() -> None:
    now = time.monotonic_ns()
    policy = _ScriptedPolicy(MotorAction(sequence=0, buttons_down=("left",)))
    executor = _executor(policy, now_ns=now)

    tick = executor.tick(
        _mining_board(
            now_ns=now,
            kind="dirt",
            item=None,
            track_attributes={"tracking_source": _ROCKET_SOURCE},
            target_source=_ROCKET_SOURCE,
            scene_hash=None,
            include_mineable=False,
            include_selected_slot=False,
            extra_facts=(
                _fact(
                    "target.mineable",
                    False,
                    now_ns=now,
                    source=_ROCKET_SOURCE,
                ),
            ),
        ),
        sequence=1,
        now_ns=now,
    )

    assert tick.run.failure_code == SkillFailureCode.MINING_TARGET_UNVERIFIED


def test_current_exact_operator_dirt_region_is_hand_mineable() -> None:
    now = time.monotonic_ns()
    target_source = "operator:explicit-grounding:target:one"
    policy = _ScriptedPolicy(MotorAction(sequence=0, buttons_down=("left",)))
    executor = _executor(policy, now_ns=now)

    tick = executor.tick(
        _mining_board(
            now_ns=now,
            kind="dirt",
            item=None,
            track_attributes={"source": "operator"},
            track_seen_ns=now - 60_000_000_000,
            target_source=target_source,
            scene_hash=None,
            include_mineable=False,
            include_selected_slot=False,
        ),
        sequence=1,
        now_ns=now,
    )

    assert tick.run.outcome == SkillOutcome.RUNNING


def test_current_operator_dirt_ignores_unrelated_old_mineability_fact() -> None:
    now = time.monotonic_ns()
    target_source = "operator:explicit-grounding:target:one"
    policy = _ScriptedPolicy(MotorAction(sequence=0, buttons_down=("left",)))
    executor = _executor(policy, now_ns=now)

    tick = executor.tick(
        _mining_board(
            now_ns=now,
            kind="dirt",
            item=None,
            track_attributes={"source": "operator"},
            target_source=target_source,
            scene_hash=None,
            include_mineable=False,
            include_selected_slot=False,
            extra_facts=(
                _fact(
                    "target.mineable",
                    False,
                    now_ns=now,
                    source="vlm:previous-target",
                ),
            ),
        ),
        sequence=1,
        now_ns=now,
    )

    assert tick.run.outcome == SkillOutcome.RUNNING
    assert tick.action is not None and "left" in tick.action.buttons_down


def test_operator_only_target_expires_after_bounded_continuation_grace() -> None:
    now = time.monotonic_ns()
    target_source = "operator:explicit-grounding:target:one"
    old_track_ns = now - 60_000_000_000
    policy = _ScriptedPolicy(
        MotorAction(sequence=0, buttons_down=("left",)),
        MotorAction(sequence=0, buttons_up=("left",)),
    )
    executor = _executor(policy, now_ns=now)
    executor.tick(
        _mining_board(
            now_ns=now,
            kind="oak_log",
            item=None,
            track_attributes={"source": "operator"},
            track_seen_ns=old_track_ns,
            target_source=target_source,
            scene_hash=None,
            include_mineable=False,
            include_selected_slot=False,
        ),
        sequence=1,
        now_ns=now,
    )

    stopped = executor.tick(
        _mining_board(
            now_ns=now + 1_600_000_000,
            kind="oak_log",
            item=None,
            track_attributes={"source": "operator"},
            track_seen_ns=old_track_ns,
            scene_hash=None,
            include_kind=False,
            include_mineable=False,
            include_selected_slot=False,
            include_visible=False,
            crosshair_hash=_HASH_B,
        ),
        sequence=2,
        now_ns=now + 1_600_000_000,
    )

    assert stopped.run.failure_code == SkillFailureCode.MINING_TARGET_CHANGED


@pytest.mark.parametrize("label", ["dirt beside oak trunk", "visible dirt", "oak leaves"])
def test_rocket_log_inference_rejects_ambiguous_non_log_labels(label: str) -> None:
    now = time.monotonic_ns()
    policy = _ScriptedPolicy(MotorAction(sequence=0, buttons_down=("left",)))
    executor = _executor(policy, now_ns=now, mode="gather_wood")

    tick = executor.tick(
        _mining_board(
            now_ns=now,
            kind=label,
            item=None,
            track_attributes={"tracking_source": _ROCKET_SOURCE},
            target_source=_ROCKET_SOURCE,
            scene_hash=None,
            include_mineable=False,
            include_selected_slot=False,
        ),
        sequence=1,
        now_ns=now,
    )

    assert tick.run.failure_code in {
        SkillFailureCode.MINING_TARGET_UNVERIFIED,
        SkillFailureCode.MINING_TARGET_MISMATCH,
    }


def test_vlm_log_without_mineable_or_matching_scene_waits_without_attacking() -> None:
    now = time.monotonic_ns()
    policy = _ScriptedPolicy(MotorAction(sequence=0, buttons_down=("left",)))
    executor = _executor(policy, now_ns=now, mode="gather_wood")

    tick = executor.tick(
        _mining_board(
            now_ns=now,
            kind="visible central oak trunk",
            item=None,
            scene_hash=None,
            include_mineable=False,
            include_selected_slot=False,
        ),
        sequence=1,
        now_ns=now,
    )

    assert tick.run.outcome == SkillOutcome.RUNNING
    assert tick.action is not None and "left" not in tick.action.buttons_down


def test_attack_waits_for_fresh_target_semantics_after_scene_change() -> None:
    now = time.monotonic_ns()
    policy = _ScriptedPolicy(MotorAction(sequence=0, buttons_down=("left",)))
    executor = _executor(policy, now_ns=now)

    tick = executor.tick(
        _mining_board(now_ns=now, scene_hash=_HASH_A, frame_hash=_HASH_B),
        sequence=1,
        now_ns=now,
    )

    assert tick.run.outcome == SkillOutcome.RUNNING
    assert tick.action is not None and "left" not in tick.action.buttons_down


def test_attack_waits_when_target_and_scene_come_from_different_queries() -> None:
    now = time.monotonic_ns()
    policy = _ScriptedPolicy(MotorAction(sequence=0, buttons_down=("left",)))
    executor = _executor(policy, now_ns=now)

    tick = executor.tick(
        _mining_board(
            now_ns=now,
            target_source="vlm:test:q1",
            scene_source="vlm:test:q2",
        ),
        sequence=1,
        now_ns=now,
    )

    assert tick.run.outcome == SkillOutcome.RUNNING
    assert tick.action is not None and "left" not in tick.action.buttons_down


def test_mining_rejects_cursor_semantics_at_start() -> None:
    now = time.monotonic_ns()
    policy = _ScriptedPolicy(
        MotorAction(sequence=0, buttons_down=("left",), camera_semantics="cursor")
    )
    executor = _executor(policy, now_ns=now)

    tick = executor.tick(_mining_board(now_ns=now), sequence=1, now_ns=now)

    assert tick.run.failure_code == SkillFailureCode.MINING_CAMERA_CHANGED


def test_attack_start_suppresses_press_but_allows_simultaneous_camera_motion() -> None:
    now = time.monotonic_ns()
    policy = _ScriptedPolicy(
        MotorAction(sequence=0, buttons_down=("left",), mouse_dx=2)
    )
    executor = _executor(policy, now_ns=now)

    aiming = executor.tick(_mining_board(now_ns=now), sequence=1, now_ns=now)

    assert aiming.run.outcome == SkillOutcome.RUNNING
    assert aiming.action is not None
    assert aiming.action.mouse_dx == 2
    assert "left" not in aiming.action.buttons_down
    assert aiming.action_origin == ActionOrigin.SYNTHETIC


def test_mining_rejects_intent_target_mismatch() -> None:
    now = time.monotonic_ns()
    guard = MiningLeaseGuard()

    decision = guard.inspect(
        MotorAction(sequence=1, buttons_down=("left",)),
        _mining_board(now_ns=now, kind="oak_log"),
        MotorIntent(
            skill_id="mine",
            mode="mine",
            episode_id="one",
            target_label="diamond_ore",
        ),
        now_ns=now,
    )

    assert decision.failure_code == SkillFailureCode.MINING_TARGET_MISMATCH


def test_early_policy_release_is_suppressed_only_inside_verified_lease() -> None:
    now = time.monotonic_ns()
    policy = _ScriptedPolicy(
        MotorAction(sequence=0, buttons_down=("left",)),
        MotorAction(sequence=0, buttons_up=("left",)),
    )
    executor = _executor(policy, now_ns=now)
    started = executor.tick(_mining_board(now_ns=now), sequence=1, now_ns=now)

    continued = executor.tick(
        _mining_board(now_ns=now + 100_000_000, crosshair_hash="0000000000000003"),
        sequence=2,
        now_ns=now + 100_000_000,
    )

    assert started.action is not None and started.action.buttons_down == ("left",)
    assert continued.run.outcome == SkillOutcome.RUNNING
    assert continued.action is not None and "left" not in continued.action.buttons_up
    assert continued.action_origin == ActionOrigin.SYNTHETIC
    released = executor.cancel(now_ns=now + 200_000_000)
    assert released.action is not None and "left" in released.action.buttons_up


@pytest.mark.parametrize(
    ("second_action", "board_change", "failure"),
    [
        (
            MotorAction(sequence=0),
            {"extra_facts": ()},
            SkillFailureCode.MINING_TARGET_CHANGED,
        ),
        (
            MotorAction(sequence=0),
            {"item": "minecraft:wooden_axe"},
            SkillFailureCode.MINING_TOOL_CHANGED,
        ),
    ],
)
def test_active_lease_releases_on_target_or_tool_change(
    second_action: MotorAction,
    board_change: dict[str, object],
    failure: SkillFailureCode,
) -> None:
    now = time.monotonic_ns()
    policy = _ScriptedPolicy(
        MotorAction(sequence=0, buttons_down=("left",)),
        second_action,
    )
    executor = _executor(policy, now_ns=now)
    executor.tick(_mining_board(now_ns=now), sequence=1, now_ns=now)
    if failure == SkillFailureCode.MINING_TARGET_CHANGED:
        board_change = {"track_id": "target:two"}

    stopped = executor.tick(
        _mining_board(now_ns=now + 100_000_000, **board_change),  # type: ignore[arg-type]
        sequence=2,
        now_ns=now + 100_000_000,
    )

    assert stopped.run.failure_code == failure
    assert stopped.action is not None and "left" in stopped.action.buttons_up


def test_active_lease_rejects_cursor_semantics() -> None:
    now = time.monotonic_ns()
    policy = _ScriptedPolicy(
        MotorAction(sequence=0, buttons_down=("left",)),
        MotorAction(sequence=0, buttons_up=("left",), camera_semantics="cursor"),
    )
    executor = _executor(policy, now_ns=now)
    executor.tick(_mining_board(now_ns=now), sequence=1, now_ns=now)

    stopped = executor.tick(
        _mining_board(now_ns=now + 100_000_000),
        sequence=2,
        now_ns=now + 100_000_000,
    )

    assert stopped.run.failure_code == SkillFailureCode.MINING_CAMERA_CHANGED


def test_attack_start_suppresses_press_but_allows_initial_locomotion() -> None:
    now = time.monotonic_ns()
    policy = _ScriptedPolicy(
        MotorAction(sequence=0, keys_down=("w",), buttons_down=("left",))
    )
    executor = _executor(policy, now_ns=now)

    approaching = executor.tick(_mining_board(now_ns=now), sequence=1, now_ns=now)

    assert approaching.run.outcome == SkillOutcome.RUNNING
    assert approaching.action is not None
    assert approaching.action.keys_down == ("w",)
    assert "left" not in approaching.action.buttons_down
    assert approaching.action_origin == ActionOrigin.SYNTHETIC


def test_cancelling_pending_attack_releases_actual_emitted_locomotion() -> None:
    now = time.monotonic_ns()
    policy = _ScriptedPolicy(
        MotorAction(sequence=0, keys_down=("w",), buttons_down=("left",))
    )
    executor = _executor(policy, now_ns=now)

    approaching = executor.tick(_mining_board(now_ns=now), sequence=1, now_ns=now)
    cancelled = executor.cancel(now_ns=now + 100_000_000)

    assert approaching.action is not None and approaching.action.keys_down == ("w",)
    assert "left" not in approaching.action.buttons_down
    assert cancelled.action is not None and "w" in cancelled.action.keys_up


def test_pending_attack_quiesces_motion_then_starts_from_fresh_frame() -> None:
    now = time.monotonic_ns()
    policy = _ScriptedPolicy(
        MotorAction(sequence=0, keys_down=("w",), buttons_down=("left",)),
        MotorAction(sequence=0),
        MotorAction(sequence=0),
    )
    executor = _executor(policy, now_ns=now)
    unavailable = _mining_board(
        now_ns=now,
        include_visible=False,
        include_kind=False,
        include_mineable=False,
        include_selected_slot=False,
    )

    approaching = executor.tick(unavailable, sequence=1, now_ns=now)
    settling = executor.tick(
        _mining_board(
            now_ns=now + 800_000_000,
            include_visible=False,
            include_kind=False,
            include_mineable=False,
            include_selected_slot=False,
        ),
        sequence=2,
        now_ns=now + 800_000_000,
    )
    started = executor.tick(
        _mining_board(now_ns=now + 900_000_000, kind="dirt", item=None),
        sequence=3,
        now_ns=now + 900_000_000,
    )

    assert approaching.action is not None and approaching.action.keys_down == ("w",)
    assert "left" not in approaching.action.buttons_down
    assert settling.action is not None and "w" in settling.action.keys_up
    assert "left" not in settling.action.buttons_down
    assert started.action is not None and "left" in started.action.buttons_down
    assert not started.action.keys_down
    assert started.action.mouse_dx == 0 and started.action.mouse_dy == 0


def test_wrong_tool_remains_terminal_when_attack_is_combined_with_motion() -> None:
    now = time.monotonic_ns()
    policy = _ScriptedPolicy(
        MotorAction(sequence=0, keys_down=("w",), buttons_down=("left",))
    )
    executor = _executor(policy, now_ns=now)

    stopped = executor.tick(
        _mining_board(
            now_ns=now,
            kind="iron_ore",
            item="minecraft:wooden_pickaxe",
        ),
        sequence=1,
        now_ns=now,
    )

    assert stopped.run.failure_code == SkillFailureCode.MINING_WRONG_TOOL
    assert stopped.action is not None
    assert "left" in stopped.action.buttons_up
    assert "w" in stopped.action.keys_up


def test_pending_attack_failure_releases_emitted_locomotion() -> None:
    now = time.monotonic_ns()
    policy = _ScriptedPolicy(
        MotorAction(sequence=0, keys_down=("w",), buttons_down=("left",)),
        MotorAction(sequence=0),
    )
    executor = _executor(policy, now_ns=now)
    first = executor.tick(_mining_board(now_ns=now), sequence=1, now_ns=now)
    danger = _fact("danger.immediate", True, now_ns=now + 100_000_000)

    stopped = executor.tick(
        _mining_board(now_ns=now + 100_000_000, extra_facts=(danger,)),
        sequence=2,
        now_ns=now + 100_000_000,
    )

    assert first.action is not None and first.action.keys_down == ("w",)
    assert "left" not in first.action.buttons_down
    assert stopped.run.failure_code == SkillFailureCode.MINING_UNSAFE_SCENE
    assert stopped.action is not None
    assert "w" in stopped.action.keys_up
    assert "left" in stopped.action.buttons_up


@pytest.mark.parametrize("modifier", ["ctrl", "shift"])
def test_attack_start_quiesces_locomotion_modifier_before_mining(
    modifier: str,
) -> None:
    now = time.monotonic_ns()
    policy = _ScriptedPolicy(
        MotorAction(sequence=0, keys_down=(modifier,), buttons_down=("left",)),
        MotorAction(sequence=0),
    )
    executor = _executor(policy, now_ns=now)

    approaching = executor.tick(_mining_board(now_ns=now), sequence=1, now_ns=now)
    settling = executor.tick(
        _mining_board(now_ns=now + 100_000_000),
        sequence=2,
        now_ns=now + 100_000_000,
    )

    assert approaching.run.outcome == SkillOutcome.RUNNING
    assert approaching.action is not None
    assert approaching.action.keys_down == (modifier,)
    assert "left" not in approaching.action.buttons_down
    assert settling.run.outcome == SkillOutcome.RUNNING
    assert settling.action is not None
    assert modifier in settling.action.keys_up
    assert "left" not in settling.action.buttons_down


def test_attack_start_rejects_right_button_held_from_prior_tick() -> None:
    now = time.monotonic_ns()
    policy = _ScriptedPolicy(
        MotorAction(sequence=0, buttons_down=("right",)),
        MotorAction(sequence=0, buttons_down=("left",)),
    )
    executor = _executor(policy, now_ns=now)
    executor.tick(_mining_board(now_ns=now), sequence=1, now_ns=now)

    stopped = executor.tick(
        _mining_board(now_ns=now + 100_000_000),
        sequence=2,
        now_ns=now + 100_000_000,
    )

    assert stopped.run.failure_code == SkillFailureCode.MINING_CONFLICTING_INPUT
    assert stopped.action is not None
    assert "left" in stopped.action.buttons_up
    assert "right" in stopped.action.buttons_up


def test_attack_after_releasing_right_waits_for_fresh_evidence() -> None:
    now = time.monotonic_ns()
    board = _mining_board(now_ns=now)
    policy = _ScriptedPolicy(
        MotorAction(sequence=0, buttons_down=("right",)),
        MotorAction(sequence=0, buttons_down=("left",), buttons_up=("right",)),
    )
    executor = _executor(policy, now_ns=now)
    executor.tick(board, sequence=1, now_ns=now)

    stopped = executor.tick(board, sequence=2, now_ns=now + 100_000_000)

    assert stopped.run.outcome == SkillOutcome.RUNNING
    assert stopped.action is not None
    assert "left" not in stopped.action.buttons_down
    assert "right" in stopped.action.buttons_up


@pytest.mark.parametrize("key", ["e", "q"])
def test_attack_start_rejects_non_mining_key_held_from_prior_tick(key: str) -> None:
    now = time.monotonic_ns()
    policy = _ScriptedPolicy(
        MotorAction(sequence=0, keys_down=(key,)),
        MotorAction(sequence=0, buttons_down=("left",)),
    )
    executor = _executor(policy, now_ns=now)
    executor.tick(_mining_board(now_ns=now), sequence=1, now_ns=now)

    stopped = executor.tick(
        _mining_board(now_ns=now + 100_000_000),
        sequence=2,
        now_ns=now + 100_000_000,
    )

    assert stopped.run.failure_code == SkillFailureCode.MINING_CONFLICTING_INPUT
    assert stopped.action is not None
    assert key in stopped.action.keys_up


def test_attack_start_releases_locomotion_held_from_prior_policy_tick() -> None:
    now = time.monotonic_ns()
    policy = _ScriptedPolicy(
        MotorAction(sequence=0, keys_down=("w",)),
        MotorAction(sequence=0, buttons_down=("left",)),
    )
    executor = _executor(policy, now_ns=now)
    moving = executor.tick(_mining_board(now_ns=now), sequence=1, now_ns=now)
    assert moving.run.outcome == SkillOutcome.RUNNING

    stopped = executor.tick(
        _mining_board(now_ns=now + 100_000_000),
        sequence=2,
        now_ns=now + 100_000_000,
    )

    assert stopped.run.outcome == SkillOutcome.RUNNING
    assert stopped.action is not None and "w" in stopped.action.keys_up
    assert "left" not in stopped.action.buttons_down


def test_attack_waits_for_fresh_evidence_after_hotbar_change() -> None:
    now = time.monotonic_ns()
    board = _mining_board(now_ns=now)
    policy = _ScriptedPolicy(
        MotorAction(sequence=0, keys_down=("2",)),
        MotorAction(sequence=0, keys_up=("2",), buttons_down=("left",)),
    )
    executor = _executor(policy, now_ns=now)
    executor.tick(board, sequence=1, now_ns=now)

    stopped = executor.tick(board, sequence=2, now_ns=now + 100_000_000)

    assert stopped.run.outcome == SkillOutcome.RUNNING
    assert stopped.action is not None and "left" not in stopped.action.buttons_down


def test_attack_waits_for_a_new_frame_after_stopping_then_synthesizes_press() -> None:
    now = time.monotonic_ns()
    policy = _ScriptedPolicy(
        MotorAction(sequence=0, keys_down=("w",)),
        MotorAction(sequence=0, keys_up=("w",), buttons_down=("left",)),
        MotorAction(sequence=0),
        MotorAction(sequence=0),
    )
    executor = _executor(policy, now_ns=now)
    executor.tick(_mining_board(now_ns=now), sequence=1, now_ns=now)

    stopped = executor.tick(
        _mining_board(now_ns=now + 100_000_000),
        sequence=2,
        now_ns=now + 100_000_000,
    )

    assert stopped.run.outcome == SkillOutcome.RUNNING
    assert stopped.action is not None and "w" in stopped.action.keys_up
    assert "left" not in stopped.action.buttons_down

    stale = executor.tick(
        _mining_board(now_ns=now + 100_000_000),
        sequence=3,
        now_ns=now + 150_000_000,
    )
    assert stale.action is not None and "left" not in stale.action.buttons_down

    started = executor.tick(
        _mining_board(now_ns=now + 200_000_000),
        sequence=4,
        now_ns=now + 200_000_000,
    )
    assert started.run.outcome == SkillOutcome.RUNNING
    assert started.action is not None and "left" in started.action.buttons_down
    assert started.action_origin == ActionOrigin.SYNTHETIC


def test_active_lease_quiesces_policy_locomotion_and_camera_drift() -> None:
    now = time.monotonic_ns()
    policy = _ScriptedPolicy(
        MotorAction(sequence=0, buttons_down=("left",)),
        MotorAction(
            sequence=0,
            keys_down=("a", "space"),
            buttons_up=("left",),
            mouse_dx=23,
            mouse_dy=-17,
        ),
        MotorAction(sequence=0, keys_down=("w",), mouse_dx=-31),
    )
    executor = _executor(policy, now_ns=now)
    executor.tick(_mining_board(now_ns=now), sequence=1, now_ns=now)

    continued = executor.tick(
        _mining_board(
            now_ns=now + 100_000_000,
            crosshair_hash="0000000000000003",
        ),
        sequence=2,
        now_ns=now + 100_000_000,
    )
    repeated = executor.tick(
        _mining_board(
            now_ns=now + 200_000_000,
            crosshair_hash="0000000000000007",
        ),
        sequence=3,
        now_ns=now + 200_000_000,
    )

    assert continued.run.outcome == SkillOutcome.RUNNING
    assert continued.action is not None
    assert continued.action.keys_down == ()
    assert continued.action.keys_up == ("a", "space")
    assert continued.action.mouse_dx == continued.action.mouse_dy == 0
    assert "left" not in continued.action.buttons_up
    assert continued.action_origin == ActionOrigin.SYNTHETIC
    assert repeated.run.outcome == SkillOutcome.RUNNING
    assert repeated.action is not None
    assert repeated.action.keys_down == ()
    assert repeated.action.keys_up == ("w",)
    assert repeated.action.mouse_dx == repeated.action.mouse_dy == 0
    assert executor._mining_guard.held_keys == ()


@pytest.mark.parametrize(
    ("second_action", "board_change", "failure"),
    [
        (
            MotorAction(sequence=0, keys_down=("w", "2"), mouse_dx=30),
            {},
            SkillFailureCode.MINING_TOOL_CHANGED,
        ),
        (
            MotorAction(sequence=0, keys_down=("w", "e"), mouse_dx=30),
            {},
            SkillFailureCode.MINING_CONFLICTING_INPUT,
        ),
        (
            MotorAction(sequence=0, keys_down=("w",), mouse_dx=30),
            {"track_id": "target:two"},
            SkillFailureCode.MINING_TARGET_CHANGED,
        ),
    ],
)
def test_active_lease_drift_suppression_does_not_mask_hard_interlocks(
    second_action: MotorAction,
    board_change: dict[str, object],
    failure: SkillFailureCode,
) -> None:
    now = time.monotonic_ns()
    policy = _ScriptedPolicy(
        MotorAction(sequence=0, buttons_down=("left",)),
        second_action,
    )
    executor = _executor(policy, now_ns=now)
    executor.tick(_mining_board(now_ns=now), sequence=1, now_ns=now)

    stopped = executor.tick(
        _mining_board(now_ns=now + 100_000_000, **board_change),  # type: ignore[arg-type]
        sequence=2,
        now_ns=now + 100_000_000,
    )

    assert stopped.run.failure_code == failure
    assert stopped.action is not None
    assert "left" in stopped.action.buttons_up
    assert set(second_action.keys_down).issubset(stopped.action.keys_up)


def test_active_lease_releases_before_a_hotbar_tool_change() -> None:
    now = time.monotonic_ns()
    policy = _ScriptedPolicy(
        MotorAction(sequence=0, buttons_down=("left",)),
        MotorAction(sequence=0, keys_down=("2",), buttons_up=("left",)),
    )
    executor = _executor(policy, now_ns=now)
    executor.tick(_mining_board(now_ns=now), sequence=1, now_ns=now)

    stopped = executor.tick(
        _mining_board(now_ns=now + 100_000_000),
        sequence=2,
        now_ns=now + 100_000_000,
    )

    assert stopped.run.failure_code == SkillFailureCode.MINING_TOOL_CHANGED
    assert stopped.action is not None and "left" in stopped.action.buttons_up


@pytest.mark.parametrize(
    "conflicting_action",
    [
        MotorAction(sequence=0, buttons_down=("right",), buttons_up=("left",)),
        MotorAction(
            sequence=0,
            buttons_down=("right",),
            buttons_up=("right", "left"),
        ),
        MotorAction(sequence=0, keys_down=("e",), buttons_up=("left",)),
        MotorAction(
            sequence=0,
            keys_down=("e",),
            keys_up=("e",),
            buttons_up=("left",),
        ),
        MotorAction(sequence=0, keys_down=("q",), buttons_up=("left",)),
    ],
)
def test_active_lease_releases_before_conflicting_interaction(
    conflicting_action: MotorAction,
) -> None:
    now = time.monotonic_ns()
    policy = _ScriptedPolicy(
        MotorAction(sequence=0, buttons_down=("left",)),
        conflicting_action,
    )
    executor = _executor(policy, now_ns=now)
    executor.tick(_mining_board(now_ns=now), sequence=1, now_ns=now)

    stopped = executor.tick(
        _mining_board(now_ns=now + 100_000_000),
        sequence=2,
        now_ns=now + 100_000_000,
    )

    assert stopped.run.failure_code == SkillFailureCode.MINING_CONFLICTING_INPUT
    assert stopped.action is not None and "left" in stopped.action.buttons_up
    assert set(conflicting_action.keys_down).issubset(stopped.action.keys_up)
    assert set(conflicting_action.buttons_down).issubset(stopped.action.buttons_up)
    assert executor._mining_guard.held_keys == ()
    assert executor._mining_guard.held_buttons == ()


def test_hard_block_lease_fails_closed_if_tool_evidence_disappears() -> None:
    now = time.monotonic_ns()
    policy = _ScriptedPolicy(
        MotorAction(sequence=0, buttons_down=("left",)),
        MotorAction(sequence=0, buttons_up=("left",)),
    )
    executor = _executor(policy, now_ns=now)
    executor.tick(
        _mining_board(
            now_ns=now,
            kind="stone",
            item="minecraft:wooden_pickaxe",
        ),
        sequence=1,
        now_ns=now,
    )

    stopped = executor.tick(
        _mining_board(
            now_ns=now + 100_000_000,
            kind="stone",
            item=None,
        ),
        sequence=2,
        now_ns=now + 100_000_000,
    )

    assert stopped.run.failure_code == SkillFailureCode.MINING_TOOL_UNVERIFIED
    assert stopped.action is not None and "left" in stopped.action.buttons_up


def test_hand_safe_lease_accepts_item_evidence_that_was_initially_unknown() -> None:
    now = time.monotonic_ns()
    policy = _ScriptedPolicy(
        MotorAction(sequence=0, buttons_down=("left",)),
        MotorAction(sequence=0, buttons_up=("left",)),
    )
    executor = _executor(policy, now_ns=now)
    executor.tick(
        _mining_board(
            now_ns=now,
            kind="dirt",
            item=None,
            include_selected_slot=False,
        ),
        sequence=1,
        now_ns=now,
    )

    continued = executor.tick(
        _mining_board(
            now_ns=now + 100_000_000,
            kind="dirt",
            item="minecraft:stick",
        ),
        sequence=2,
        now_ns=now + 100_000_000,
    )

    assert continued.run.outcome == SkillOutcome.RUNNING
    assert continued.action is not None and "left" not in continued.action.buttons_up


def test_target_loss_is_not_inferred_as_mining_success() -> None:
    now = time.monotonic_ns()
    policy = _ScriptedPolicy(
        MotorAction(sequence=0, buttons_down=("left",)),
        MotorAction(sequence=0, buttons_up=("left",)),
    )
    executor = _executor(policy, now_ns=now)
    executor.tick(_mining_board(now_ns=now), sequence=1, now_ns=now)
    no_longer_visible = _fact(
        "target.visible",
        False,
        now_ns=now + 100_000_000,
    )

    stopped = executor.tick(
        _mining_board(
            now_ns=now + 100_000_000,
            extra_facts=(no_longer_visible,),
        ),
        sequence=2,
        now_ns=now + 100_000_000,
    )

    assert stopped.run.outcome == SkillOutcome.FAILED
    assert stopped.run.failure_code == SkillFailureCode.MINING_TARGET_CHANGED


@pytest.mark.parametrize(
    "unsafe_fact",
    [
        ("danger.immediate", True),
        ("scene.ui_overlay", True),
        ("scene.playable", False),
    ],
)
def test_active_lease_releases_on_danger_or_ui(
    unsafe_fact: tuple[str, bool],
) -> None:
    now = time.monotonic_ns()
    policy = _ScriptedPolicy(
        MotorAction(sequence=0, buttons_down=("left",)),
        MotorAction(sequence=0),
    )
    executor = _executor(policy, now_ns=now)
    executor.tick(_mining_board(now_ns=now), sequence=1, now_ns=now)
    unsafe = _fact(unsafe_fact[0], unsafe_fact[1], now_ns=now + 100_000_000)

    stopped = executor.tick(
        _mining_board(now_ns=now + 100_000_000, extra_facts=(unsafe,)),
        sequence=2,
        now_ns=now + 100_000_000,
    )

    assert stopped.run.failure_code == SkillFailureCode.MINING_UNSAFE_SCENE
    assert stopped.action is not None and "left" in stopped.action.buttons_up


def test_static_crosshair_signal_stops_lease_by_wall_clock() -> None:
    now = time.monotonic_ns()
    policy = _ScriptedPolicy(
        MotorAction(sequence=0, buttons_down=("left",)),
        MotorAction(sequence=0, buttons_up=("left",)),
    )
    executor = _executor(policy, now_ns=now)
    executor.tick(_mining_board(now_ns=now), sequence=1, now_ns=now)
    for sequence, elapsed_ms in enumerate((400, 800), start=2):
        running = executor.tick(
            _mining_board(now_ns=now + elapsed_ms * 1_000_000),
            sequence=sequence,
            now_ns=now + elapsed_ms * 1_000_000,
        )
        assert running.run.outcome == SkillOutcome.RUNNING

    stopped = executor.tick(
        _mining_board(now_ns=now + 1_300_000_000),
        sequence=4,
        now_ns=now + 1_300_000_000,
    )

    assert stopped.run.failure_code == SkillFailureCode.MINING_VISUAL_STAGNATION
    assert stopped.action is not None and "left" in stopped.action.buttons_up


def test_missing_crosshair_signal_releases_after_bounded_grace() -> None:
    now = time.monotonic_ns()
    policy = _ScriptedPolicy(MotorAction(sequence=0, buttons_down=("left",)))
    executor = _executor(policy, now_ns=now)
    executor.tick(_mining_board(now_ns=now), sequence=1, now_ns=now)

    stopped = executor.tick(
        _mining_board(now_ns=now + 800_000_000, crosshair_hash=None),
        sequence=2,
        now_ns=now + 800_000_000,
    )

    assert stopped.run.failure_code == SkillFailureCode.MINING_VISUAL_SIGNAL_LOST
    assert stopped.action is not None and "left" in stopped.action.buttons_up


def test_one_block_wall_clock_bound_releases_even_with_visual_change() -> None:
    now = time.monotonic_ns()
    policy = _ScriptedPolicy(MotorAction(sequence=0, buttons_down=("left",)))
    executor = _executor(policy, now_ns=now)
    executor.tick(_mining_board(now_ns=now), sequence=1, now_ns=now)

    stopped = executor.tick(
        _mining_board(now_ns=now + 3_601_000_000, crosshair_hash=_HASH_B),
        sequence=2,
        now_ns=now + 3_601_000_000,
    )

    assert stopped.run.failure_code == SkillFailureCode.MINING_LEASE_EXPIRED
    assert stopped.action is not None and "left" in stopped.action.buttons_up


def test_soft_block_lease_includes_bounded_break_verification_margin() -> None:
    now = time.monotonic_ns()
    guard = MiningLeaseGuard()
    intent = MotorIntent(skill_id="mine", mode="mine", episode_id="soft:one")

    started = guard.inspect(
        MotorAction(sequence=1, buttons_down=("left",)),
        _mining_board(now_ns=now, kind="dirt", item=None),
        intent,
        now_ns=now,
    )
    before_deadline = guard.inspect(
        MotorAction(sequence=2, buttons_up=("left",)),
        _mining_board(
            now_ns=now + 2_499_000_000,
            kind="dirt",
            item=None,
            crosshair_hash=_HASH_B,
        ),
        intent,
        now_ns=now + 2_499_000_000,
    )
    at_deadline = guard.inspect(
        MotorAction(sequence=3),
        _mining_board(
            now_ns=now + 2_500_000_000,
            kind="dirt",
            item=None,
            crosshair_hash=_HASH_A,
        ),
        intent,
        now_ns=now + 2_500_000_000,
    )

    assert started.failure_code is None
    assert before_deadline.failure_code is None
    assert "left" not in before_deadline.action.buttons_up
    assert at_deadline.failure_code == SkillFailureCode.MINING_LEASE_EXPIRED
    assert at_deadline.force_release_left is True


def test_absolute_mining_bound_caps_extended_soft_block_lease() -> None:
    now = time.monotonic_ns()
    guard = MiningLeaseGuard(absolute_max_ms=2_000)
    intent = MotorIntent(skill_id="mine", mode="mine", episode_id="soft:capped")
    board = _mining_board(now_ns=now, kind="dirt", item=None)

    started = guard.inspect(
        MotorAction(sequence=1, buttons_down=("left",)),
        board,
        intent,
        now_ns=now,
    )
    expired = guard.inspect(
        MotorAction(sequence=2),
        _mining_board(
            now_ns=now + 2_000_000_000,
            kind="dirt",
            item=None,
            crosshair_hash=_HASH_B,
        ),
        intent,
        now_ns=now + 2_000_000_000,
    )

    assert started.failure_code is None
    assert expired.failure_code == SkillFailureCode.MINING_LEASE_EXPIRED
    assert expired.force_release_left is True


def test_guard_rejects_episode_change_and_reset_reports_release() -> None:
    now = time.monotonic_ns()
    guard = MiningLeaseGuard()
    action = MotorAction(sequence=1, buttons_down=("left",))
    first = guard.inspect(
        action,
        _mining_board(now_ns=now),
        MotorIntent(skill_id="mine", mode="mine", episode_id="one"),
        now_ns=now,
    )
    assert first.failure_code is None

    changed = guard.inspect(
        MotorAction(sequence=2),
        _mining_board(now_ns=now + 1),
        MotorIntent(skill_id="mine", mode="mine", episode_id="two"),
        now_ns=now + 1,
    )

    assert changed.failure_code == SkillFailureCode.MINING_EPISODE_CHANGED
    assert guard.reset() is True


def test_bootstrap_crosshair_hash_ignores_unrelated_pixels_and_tracks_center() -> None:
    width, height = 90, 80
    baseline = bytearray(
        b"".join(bytes((x, x, x, 255)) for _y in range(height) for x in range(width))
    )
    outside = bytearray(baseline)
    center = bytearray(baseline)
    for y in range(0, 20):
        for x in range(0, 20):
            outside[(y * width + x) * 4 : (y * width + x + 1) * 4] = bytes(
                (255 - x, 255 - x, 255 - x, 255)
            )
    for y in range(27, 54):
        for x in range(34, 57):
            value = 255 - x
            center[(y * width + x) * 4 : (y * width + x + 1) * 4] = bytes(
                (value, value, value, 255)
            )

    def crosshair_hash(pixels: bytes | bytearray) -> str:
        frame = CapturedFrame(
            frame_id=1,
            captured_ns=1,
            width=width,
            height=height,
            bgra=bytes(pixels),
        )
        facts = {fact.key: fact.value for fact in BootstrapFastPerception().infer(frame)}
        value = facts["frame.crosshair_dhash"]
        assert isinstance(value, str)
        return value

    def crosshair_luma(pixels: bytes | bytearray) -> str:
        frame = CapturedFrame(
            frame_id=1,
            captured_ns=1,
            width=width,
            height=height,
            bgra=bytes(pixels),
        )
        facts = {fact.key: fact.value for fact in BootstrapFastPerception().infer(frame)}
        value = facts["frame.crosshair_luma_grid"]
        assert isinstance(value, str)
        assert len(value) == 128
        return value

    assert crosshair_hash(baseline) == crosshair_hash(outside)
    assert crosshair_hash(baseline) != crosshair_hash(center)
    assert crosshair_luma(baseline) == crosshair_luma(outside)
    assert crosshair_luma(baseline) != crosshair_luma(center)


def test_typed_mining_failure_reaches_event_and_memory_metadata() -> None:
    run = SkillRun(
        run_id="mining-failure",
        skill_id="mine_visible_block",
        started_ns=100,
        ended_ns=200,
        outcome=SkillOutcome.FAILED,
        failure_reason=SkillFailureCode.MINING_WRONG_TOOL.value,
        failure_code=SkillFailureCode.MINING_WRONG_TOOL,
    )

    event = _terminal_run_event(run, observed_ns=1_000, trajectory_id=None)
    memory = _terminal_run_memory(run, SkillStats(failures=1), observed_ns=1_000, existing={})

    assert event.payload["failure_code"] == SkillFailureCode.MINING_WRONG_TOOL.value
    assert memory is not None
    assert memory.metadata["failure_code"] == SkillFailureCode.MINING_WRONG_TOOL.value
