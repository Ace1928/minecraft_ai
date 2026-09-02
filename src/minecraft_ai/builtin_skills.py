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
            version=2,
            name="Approach visible target",
            description=(
                "Approach the visually localized target across safe terrain, keep it near the "
                "crosshair, and stop within interaction range without overshooting"
            ),
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
            version=2,
            name="Reacquire target",
            description=(
                "Look around deliberately to find the requested target again, stabilize it near "
                "the crosshair, and avoid walking into unseen hazards"
            ),
            stage=SkillStage.EXPERIMENTAL,
            parameters=("target",),
            success_conditions=(SkillCondition(key="target.visible", operator="truthy"),),
            expected_effects=("target_visible",),
            max_duration_ms=8_000,
            policy_ref="explore",
        ),
        SkillSpec(
            skill_id="mine_visible_block",
            version=2,
            name="Mine visible block",
            description=(
                "Approach the visible mineable block, aim at its center, hold attack until it "
                "breaks, and collect the dropped resource"
            ),
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
            version=2,
            name="Attack visible hostile",
            description=(
                "Defend against the visible hostile while tracking it, maintaining safe combat "
                "distance, and disengaging before critical health"
            ),
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
            version=2,
            name="Retreat from danger",
            description=(
                "Move away from the immediate hazard toward visible safe ground while keeping "
                "the camera stable enough to verify the escape route"
            ),
            stage=SkillStage.TRUSTED,
            success_conditions=(SkillCondition(key="danger.immediate", operator="falsy"),),
            expected_effects=("safe_distance",),
            max_duration_ms=10_000,
            policy_ref="retreat",
        ),
        SkillSpec(
            skill_id="explore_forward",
            version=2,
            name="Explore forward",
            description=(
                "Traverse visible open terrain to discover a genuinely new area, keep the view "
                "near the horizon, avoid drops and water, and stop when a useful resource appears"
            ),
            stage=SkillStage.EXPERIMENTAL,
            failure_conditions=(SkillCondition(key="danger.immediate", operator="truthy"),),
            expected_effects=("new_area_observed",),
            recovery_skills=("retreat_from_danger",),
            max_duration_ms=12_000,
            policy_ref="explore",
        ),
        SkillSpec(
            skill_id="use_target",
            version=2,
            name="Use target",
            description=(
                "Approach and interact once with the visible target, then wait for and verify the "
                "resulting world or GUI transition"
            ),
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
            version=2,
            name="Place block",
            description=(
                "Select the requested block, aim at a stable adjacent face, place it once, and "
                "visually verify the new block before continuing"
            ),
            stage=SkillStage.EXPERIMENTAL,
            parameters=("block",),
            success_conditions=(SkillCondition(key="placement.changed", operator="truthy"),),
            expected_effects=("block_placed",),
            max_duration_ms=5_000,
            policy_ref="place",
        ),
        SkillSpec(
            skill_id="gather_nearby_wood",
            name="Gather nearby wood",
            description=(
                "Find a nearby tree, approach a visible trunk rather than leaves, mine connected "
                "logs, collect the drops, and stop after at least three logs are visible in the "
                "hotbar or inventory"
            ),
            stage=SkillStage.EXPERIMENTAL,
            parameters=("wood_kind", "minimum_logs"),
            success_conditions=(SkillCondition(key="inventory.logs", operator="gte", value=3),),
            failure_conditions=(SkillCondition(key="danger.immediate", operator="truthy"),),
            expected_effects=("inventory.logs>=3", "wood_resource_acquired"),
            recovery_skills=("retreat_from_danger", "reacquire_target"),
            max_duration_ms=90_000,
            policy_ref="gather_wood",
        ),
        SkillSpec(
            skill_id="craft_wood_planks",
            name="Craft wood planks",
            description=(
                "Open the Bedrock inventory crafting interface, convert one available log into "
                "wood planks, move the output into inventory, and close the interface safely"
            ),
            stage=SkillStage.EXPERIMENTAL,
            preconditions=(SkillCondition(key="inventory.logs", operator="gte", value=1),),
            success_conditions=(SkillCondition(key="inventory.planks", operator="gte", value=4),),
            failure_conditions=(SkillCondition(key="danger.immediate", operator="truthy"),),
            expected_effects=("inventory.logs-=1", "inventory.planks>=4"),
            max_duration_ms=45_000,
            policy_ref="craft_planks",
        ),
        SkillSpec(
            skill_id="craft_crafting_table",
            name="Craft a crafting table",
            description=(
                "Use the Bedrock crafting interface to arrange four wood planks into a crafting "
                "table, take the output into inventory, and close the interface safely"
            ),
            stage=SkillStage.EXPERIMENTAL,
            preconditions=(SkillCondition(key="inventory.planks", operator="gte", value=4),),
            success_conditions=(
                SkillCondition(key="inventory.crafting_table", operator="gte", value=1),
            ),
            failure_conditions=(SkillCondition(key="danger.immediate", operator="truthy"),),
            expected_effects=("inventory.planks-=4", "inventory.crafting_table>=1"),
            max_duration_ms=60_000,
            policy_ref="craft_crafting_table",
        ),
        SkillSpec(
            skill_id="establish_basic_shelter",
            name="Establish a basic shelter",
            description=(
                "Choose nearby level ground and build a small enclosed, enterable survival "
                "shelter with solid walls, a roof, and a controlled doorway before night"
            ),
            stage=SkillStage.CANDIDATE,
            preconditions=(
                SkillCondition(key="inventory.build_blocks", operator="gte", value=12),
            ),
            success_conditions=(
                SkillCondition(key="environment.shelter_enclosed", operator="truthy"),
            ),
            failure_conditions=(SkillCondition(key="danger.immediate", operator="truthy"),),
            expected_effects=("safe_spawn_area", "environment.shelter_enclosed"),
            recovery_skills=("retreat_from_danger",),
            max_duration_ms=180_000,
            policy_ref="build_shelter",
        ),
    )
    for spec in specs:
        library.register(spec)
    return library
