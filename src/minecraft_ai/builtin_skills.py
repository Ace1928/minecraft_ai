from __future__ import annotations

from .skills import (
    SkillActionPermissions,
    SkillCondition,
    SkillLibrary,
    SkillSpec,
    SkillStage,
)


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
            version=5,
            name="Approach visible target",
            description=(
                "Approach the visually localized target across safe terrain, keep it near the "
                "crosshair, and stop within interaction range without overshooting"
            ),
            stage=SkillStage.TRUSTED,
            parameters=("target", "sprint"),
            preconditions=(SkillCondition(key="target.visible", operator="truthy"),),
            initiation_alternatives=(
                (SkillCondition(key="target.reference_available", operator="truthy"),),
            ),
            success_conditions=(SkillCondition(key="target.near", operator="truthy"),),
            failure_conditions=(SkillCondition(key="danger.immediate", operator="truthy"),),
            expected_effects=("near_target",),
            recovery_skills=(
                "escape_submersion",
                "retreat_from_danger",
                "reacquire_target",
            ),
            max_duration_ms=15_000,
            policy_ref="approach",
            policy_instruction="approach the visible target",
            action_permissions=SkillActionPermissions(allow_attack=False, allow_use=False),
        ),
        SkillSpec(
            skill_id="reacquire_target",
            version=6,
            name="Reacquire target",
            description=(
                "Look around deliberately to find the requested target again, stabilize it near "
                "the crosshair, and avoid walking into unseen hazards"
            ),
            stage=SkillStage.EXPERIMENTAL,
            parameters=("target",),
            success_conditions=(
                SkillCondition(
                    key="target.tracking_confidence",
                    operator="gte",
                    value=0.65,
                ),
            ),
            expected_effects=("target_visible",),
            max_duration_ms=8_000,
            policy_ref="navigate",
            policy_instruction="find the target",
            action_permissions=SkillActionPermissions(
                allow_attack=False,
                allow_use=False,
                allow_jump=False,
            ),
        ),
        SkillSpec(
            skill_id="mine_visible_block",
            version=3,
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
            recovery_skills=(
                "escape_submersion",
                "reacquire_target",
                "retreat_from_danger",
            ),
            max_duration_ms=20_000,
            policy_ref="mine",
            policy_instruction="mine the target block",
        ),
        SkillSpec(
            skill_id="attack_visible_hostile",
            version=3,
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
            recovery_skills=("escape_submersion", "retreat_from_danger"),
            max_duration_ms=30_000,
            policy_ref="attack",
            policy_instruction="fight the target",
        ),
        SkillSpec(
            skill_id="retreat_from_danger",
            version=4,
            name="Retreat from danger",
            description=(
                "Move away from the immediate hazard toward visible safe ground while keeping "
                "the camera stable enough to verify the escape route"
            ),
            stage=SkillStage.TRUSTED,
            preconditions=(SkillCondition(key="danger.immediate", operator="truthy"),),
            success_conditions=(SkillCondition(key="danger.immediate", operator="falsy"),),
            expected_effects=("safe_distance",),
            max_duration_ms=10_000,
            policy_ref="retreat",
            policy_instruction="escape danger",
            action_permissions=SkillActionPermissions(allow_attack=False, allow_use=False),
        ),
        SkillSpec(
            skill_id="escape_submersion",
            version=2,
            name="Escape submersion",
            description=(
                "Swim to the visible surface, keep moving until the air HUD disappears, and "
                "leave the water onto stable dry ground"
            ),
            stage=SkillStage.EXPERIMENTAL,
            preconditions=(
                SkillCondition(key="environment.underwater", operator="truthy"),
            ),
            success_conditions=(
                SkillCondition(key="environment.underwater", operator="falsy"),
            ),
            expected_effects=("player_resurfaced", "dry_ground_reached"),
            max_duration_ms=12_000,
            policy_ref="escape_submersion",
            policy_instruction="swim to the surface and leave the water",
            action_permissions=SkillActionPermissions(allow_attack=False, allow_use=False),
        ),
        SkillSpec(
            skill_id="explore_forward",
            version=7,
            name="Explore forward",
            description=(
                "Traverse visible open terrain to discover a genuinely new area, keep the view "
                "near the horizon, avoid drops and water, and stop when a useful resource appears"
            ),
            stage=SkillStage.EXPERIMENTAL,
            parameters=("allow_attack", "allow_use", "allow_jump"),
            failure_conditions=(SkillCondition(key="danger.immediate", operator="truthy"),),
            expected_effects=("new_area_observed",),
            recovery_skills=("escape_submersion", "retreat_from_danger"),
            max_duration_ms=12_000,
            policy_ref="explore",
            # This published-style command produced locomotion, camera control,
            # and jumps in the frozen live-frame probe. The shorter STEVE paper
            # prompt ``go explore`` collapsed to forward-only motion on the same
            # Bedrock corner frame. The checkpoint still chooses every action.
            policy_instruction="Run around and explore the Minecraft world.",
            action_permissions=SkillActionPermissions(allow_attack=False, allow_use=False),
        ),
        SkillSpec(
            skill_id="traverse_visible_obstacle",
            version=1,
            name="Traverse visible obstacle",
            description=(
                "Use learned short-horizon movement and camera control to jump over or climb "
                "out of the visible terrain obstruction, then hand control back for replanning"
            ),
            stage=SkillStage.EXPERIMENTAL,
            parameters=("allow_attack", "allow_use", "allow_jump"),
            failure_conditions=(SkillCondition(key="danger.immediate", operator="truthy"),),
            expected_effects=("obstacle_crossed", "locomotion_progress"),
            recovery_skills=("escape_submersion", "retreat_from_danger"),
            max_duration_ms=8_000,
            policy_ref="traverse_obstacle",
            # Frozen-frame evaluation on the current Bedrock corner produced
            # learned jump on 20/40 policy decisions, versus 0/40 for the
            # generic ``go explore`` latent. This remains a STEVE-1 option, not
            # an injected Space key or handcrafted obstacle reflex.
            policy_instruction="jump forward",
            action_permissions=SkillActionPermissions(allow_attack=False, allow_use=False),
        ),
        SkillSpec(
            skill_id="use_target",
            version=3,
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
            policy_instruction="use the target",
        ),
        SkillSpec(
            skill_id="activate_visible_gui_control",
            version=1,
            name="Activate visible GUI control",
            description=(
                "Use the explicitly grounded visible GUI control, verify the resulting screen "
                "transition, and avoid emitting world locomotion while a menu is open"
            ),
            stage=SkillStage.EXPERIMENTAL,
            parameters=("control",),
            success_conditions=(SkillCondition(key="scene.playable", operator="truthy"),),
            expected_effects=("gui_transition",),
            max_duration_ms=10_000,
            policy_ref="gui",
            policy_instruction="click button",
        ),
        SkillSpec(
            skill_id="respawn_after_death",
            version=1,
            name="Respawn after death",
            description=(
                "Use the learned GUI policy to activate Bedrock's visible Respawn control, "
                "then verify that a playable world frame returns before resuming any project"
            ),
            stage=SkillStage.EXPERIMENTAL,
            preconditions=(SkillCondition(key="scene.death", operator="truthy"),),
            success_conditions=(SkillCondition(key="scene.playable", operator="truthy"),),
            expected_effects=("respawned", "playable_scene_restored"),
            max_duration_ms=15_000,
            # This remains a learned UI option: the contract supplies no screen
            # coordinate, click macro, or privileged game-state action.
            policy_ref="death_gui",
            policy_instruction="respawn",
        ),
        SkillSpec(
            skill_id="close_open_inventory",
            version=2,
            name="Close open inventory",
            description=(
                "Close the currently open Bedrock inventory through STEVE-1's learned inventory "
                "action and verify that the playable world returns before resuming locomotion"
            ),
            stage=SkillStage.EXPERIMENTAL,
            success_conditions=(SkillCondition(key="scene.playable", operator="truthy"),),
            expected_effects=("inventory_closed", "playable_scene_restored"),
            max_duration_ms=10_000,
            # This mode deliberately has no ROCKET interaction ID. STEVE-1's
            # VPT action space contains the learned inventory toggle; ROCKET-2
            # is a grounded world-interaction controller and cannot emit it.
            policy_ref="close_inventory",
            policy_instruction="close inventory",
            action_permissions=SkillActionPermissions(
                allow_attack=False,
                allow_use=False,
                allow_jump=False,
            ),
        ),
        SkillSpec(
            skill_id="place_block",
            version=4,
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
            policy_instruction="place a block",
            action_permissions=SkillActionPermissions(allow_attack=False, allow_jump=False),
        ),
        SkillSpec(
            skill_id="gather_nearby_wood",
            version=3,
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
            recovery_skills=(
                "escape_submersion",
                "retreat_from_danger",
                "reacquire_target",
            ),
            max_duration_ms=90_000,
            policy_ref="gather_wood",
            policy_instruction="mine log",
            action_permissions=SkillActionPermissions(allow_use=False),
        ),
        SkillSpec(
            skill_id="craft_wood_planks",
            version=2,
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
            policy_instruction="craft planks",
        ),
        SkillSpec(
            skill_id="craft_crafting_table",
            version=2,
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
            policy_instruction="craft a crafting table",
        ),
        SkillSpec(
            skill_id="establish_basic_shelter",
            version=2,
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
            recovery_skills=("escape_submersion", "retreat_from_danger"),
            max_duration_ms=180_000,
            policy_ref="build_shelter",
            policy_instruction="build a shelter",
        ),
    )
    for spec in specs:
        library.register(spec)
    return library
