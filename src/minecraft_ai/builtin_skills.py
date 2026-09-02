from __future__ import annotations

from .skills import SkillCondition, SkillLibrary, SkillSpec, SkillStage


def build_bootstrap_skill_library() -> SkillLibrary:
    """Small generic skill substrate used before learned policies are distilled.

    These are semantic closed-loop contracts, not fixed key sequences. The
    heuristic motor policy can execute a subset; learned policies may replace
    each `policy_ref` without changing planner-facing skill IDs.
    """
    library = SkillLibrary()
    specs = (
        SkillSpec(
            skill_id="approach_visible_target",
            name="Approach visible target",
            stage=SkillStage.TRUSTED,
            parameters=("target", "sprint"),
            preconditions=(SkillCondition(key="target.visible", operator="truthy"),),
            success_conditions=(SkillCondition(key="target.near", operator="truthy"),),
            failure_conditions=(SkillCondition(key="danger.immediate", operator="truthy"),),
            expected_effects=("near_target",),
            recovery_skills=("retreat_from_danger", "reacquire_target"),
            max_duration_ms=15_000,
            policy_ref="approach",
        ),
        SkillSpec(
            skill_id="reacquire_target",
            name="Reacquire target",
            stage=SkillStage.EXPERIMENTAL,
            parameters=("target",),
            success_conditions=(SkillCondition(key="target.visible", operator="truthy"),),
            expected_effects=("target_visible",),
            max_duration_ms=8_000,
            policy_ref="explore",
        ),
        SkillSpec(
            skill_id="mine_visible_block",
            name="Mine visible block",
            stage=SkillStage.EXPERIMENTAL,
            parameters=("target",),
            preconditions=(
                SkillCondition(key="target.visible", operator="truthy"),
                SkillCondition(key="target.mineable", operator="truthy"),
            ),
            success_conditions=(SkillCondition(key="target.broken", operator="truthy"),),
            failure_conditions=(SkillCondition(key="danger.immediate", operator="truthy"),),
            expected_effects=("block_broken", "resource_gathered"),
            recovery_skills=("reacquire_target", "retreat_from_danger"),
            max_duration_ms=20_000,
            policy_ref="mine",
        ),
        SkillSpec(
            skill_id="attack_visible_hostile",
            name="Attack visible hostile",
            stage=SkillStage.EXPERIMENTAL,
            parameters=("target",),
            preconditions=(SkillCondition(key="target.hostile_visible", operator="truthy"),),
            success_conditions=(SkillCondition(key="target.hostile_defeated", operator="truthy"),),
            failure_conditions=(SkillCondition(key="player.critical_health", operator="truthy"),),
            expected_effects=("hostile_defeated",),
            recovery_skills=("retreat_from_danger",),
            max_duration_ms=30_000,
            policy_ref="attack",
        ),
        SkillSpec(
            skill_id="retreat_from_danger",
            name="Retreat from danger",
            stage=SkillStage.TRUSTED,
            success_conditions=(SkillCondition(key="danger.immediate", operator="falsy"),),
            expected_effects=("safe_distance",),
            max_duration_ms=10_000,
            policy_ref="retreat",
        ),
        SkillSpec(
            skill_id="explore_forward",
            name="Explore forward",
            stage=SkillStage.EXPERIMENTAL,
            failure_conditions=(SkillCondition(key="danger.immediate", operator="truthy"),),
            expected_effects=("new_area_observed",),
            recovery_skills=("retreat_from_danger",),
            max_duration_ms=12_000,
            policy_ref="explore",
        ),
        SkillSpec(
            skill_id="use_target",
            name="Use target",
            stage=SkillStage.EXPERIMENTAL,
            parameters=("target",),
            preconditions=(SkillCondition(key="target.near", operator="truthy"),),
            success_conditions=(SkillCondition(key="interaction.changed", operator="truthy"),),
            expected_effects=("target_used",),
            max_duration_ms=5_000,
            policy_ref="use",
        ),
        SkillSpec(
            skill_id="place_block",
            name="Place block",
            stage=SkillStage.EXPERIMENTAL,
            parameters=("block",),
            success_conditions=(SkillCondition(key="placement.changed", operator="truthy"),),
            expected_effects=("block_placed",),
            max_duration_ms=5_000,
            policy_ref="place",
        ),
    )
    for spec in specs:
        library.register(spec)
    return library
