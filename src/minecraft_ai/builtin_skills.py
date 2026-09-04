from __future__ import annotations

from .action_levels import ActionLevel
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
            version=7,
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
            action_level=ActionLevel.GROUNDED,
            policy_ref="approach",
            policy_instruction="approach the visible target",
            action_permissions=SkillActionPermissions(
                allow_attack=False,
                allow_use=False,
                allow_drop=False,
                allow_inventory=False,
                allow_hotbar=False,
            ),
        ),
        SkillSpec(
            skill_id="reacquire_target",
            version=9,
            name="Reacquire target",
            description=(
                "Look around deliberately to find the requested target again, stabilize it near "
                "the crosshair, and avoid walking into unseen hazards"
            ),
            stage=SkillStage.EXPERIMENTAL,
            parameters=("target",),
            success_conditions=(
                # The executor additionally requires a coherent post-start
                # ROCKET bounding box that actually contains the crosshair.
                SkillCondition(
                    key="target.tracking_confidence",
                    operator="gte",
                    value=0.65,
                ),
            ),
            expected_effects=("target_visible", "target_centered"),
            max_duration_ms=8_000,
            action_level=ActionLevel.GROUNDED,
            policy_ref="navigate",
            policy_instruction="find the target",
            action_permissions=SkillActionPermissions(
                allow_attack=False,
                allow_use=False,
                allow_jump=False,
                allow_drop=False,
                allow_inventory=False,
                allow_hotbar=False,
            ),
        ),
        SkillSpec(
            skill_id="mine_visible_block",
            version=6,
            name="Mine visible block",
            description=(
                "Approach the visible mineable block, aim at its center, and hold attack until "
                "the bound block is visually verified broken"
            ),
            stage=SkillStage.EXPERIMENTAL,
            parameters=("target",),
            preconditions=(
                SkillCondition(key="target.visible", operator="truthy"),
                SkillCondition(key="target.mineable", operator="truthy"),
            ),
            initiation_alternatives=(
                (SkillCondition(key="target.reference_available", operator="truthy"),),
            ),
            success_conditions=(SkillCondition(key="target.broken", operator="truthy"),),
            failure_conditions=(SkillCondition(key="danger.immediate", operator="truthy"),),
            expected_effects=("block_broken",),
            recovery_skills=(
                "escape_submersion",
                "reacquire_target",
                "retreat_from_danger",
            ),
            max_duration_ms=20_000,
            action_level=ActionLevel.GROUNDED,
            policy_ref="mine",
            policy_instruction="mine the target block",
            action_permissions=SkillActionPermissions(
                allow_use=False,
                allow_jump=False,
                allow_drop=False,
                allow_inventory=False,
                allow_hotbar=False,
            ),
        ),
        SkillSpec(
            skill_id="collect_recent_drop",
            version=2,
            name="Collect recent drop",
            description=(
                "For a few seconds after a verified log break, move over the nearby dropped "
                "item without attacking, using, dropping, opening inventory, or switching slots"
            ),
            stage=SkillStage.EXPERIMENTAL,
            preconditions=(
                SkillCondition(
                    key="collection.recent_log_break",
                    operator="truthy",
                    min_confidence=0.99,
                ),
            ),
            failure_conditions=(SkillCondition(key="danger.immediate", operator="truthy"),),
            expected_effects=("resource_pickup_attempted", "resource_acquired"),
            recovery_skills=("escape_submersion", "retreat_from_danger"),
            max_duration_ms=5_000,
            action_level=ActionLevel.LATENT,
            policy_ref="navigate",
            policy_instruction="collect the dropped item",
            action_permissions=SkillActionPermissions(
                allow_attack=False,
                allow_use=False,
                allow_jump=True,
                allow_drop=False,
                allow_inventory=False,
                allow_hotbar=False,
            ),
        ),
        SkillSpec(
            skill_id="attack_visible_hostile",
            version=4,
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
            action_level=ActionLevel.GROUNDED,
            policy_ref="attack",
            policy_instruction="fight the target",
        ),
        SkillSpec(
            skill_id="retreat_from_danger",
            version=5,
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
            action_permissions=SkillActionPermissions(
                allow_attack=False,
                allow_use=False,
                allow_drop=False,
                allow_inventory=False,
                allow_hotbar=False,
            ),
        ),
        SkillSpec(
            skill_id="escape_submersion",
            version=3,
            name="Escape submersion",
            description=(
                "Swim to the visible surface, keep moving until the air HUD disappears, and "
                "leave the water onto stable dry ground"
            ),
            stage=SkillStage.EXPERIMENTAL,
            preconditions=(SkillCondition(key="environment.underwater", operator="truthy"),),
            success_conditions=(SkillCondition(key="environment.underwater", operator="falsy"),),
            expected_effects=("player_resurfaced", "dry_ground_reached"),
            max_duration_ms=12_000,
            policy_ref="escape_submersion",
            policy_instruction="swim to the surface and leave the water",
            action_permissions=SkillActionPermissions(
                allow_attack=False,
                allow_use=False,
                allow_drop=False,
                allow_inventory=False,
                allow_hotbar=False,
            ),
        ),
        SkillSpec(
            skill_id="explore_forward",
            version=9,
            name="Explore forward",
            description=(
                "Traverse visible open terrain to discover a genuinely new area, keep the view "
                "near the horizon, avoid drops and water, and stop when a useful resource appears"
            ),
            stage=SkillStage.EXPERIMENTAL,
            parameters=("allow_attack", "allow_use", "allow_jump"),
            failure_conditions=(SkillCondition(key="danger.immediate", operator="truthy"),),
            expected_effects=("new_area_observed",),
            recovery_skills=(
                "escape_submersion",
                "retreat_from_danger",
                "traverse_visible_obstacle",
            ),
            max_duration_ms=12_000,
            policy_ref="explore",
            # This published-style command produced locomotion, camera control,
            # and jumps in the frozen live-frame probe. The shorter STEVE paper
            # prompt ``go explore`` collapsed to forward-only motion on the same
            # Bedrock corner frame. The checkpoint still chooses every action.
            policy_instruction="Run around and explore the Minecraft world.",
            action_permissions=SkillActionPermissions(
                allow_attack=False,
                allow_use=False,
                allow_drop=False,
                allow_inventory=False,
                allow_hotbar=False,
            ),
        ),
        SkillSpec(
            skill_id="traverse_level_ground",
            version=3,
            name="Traverse level ground",
            description=(
                "Use the fast learned motion expert to cross a short visible lane while "
                "preserving stable world-view camera control"
            ),
            stage=SkillStage.EXPERIMENTAL,
            failure_conditions=(SkillCondition(key="danger.immediate", operator="truthy"),),
            expected_effects=("locomotion_progress", "destination_reached"),
            recovery_skills=(
                "escape_submersion",
                "retreat_from_danger",
                "traverse_visible_obstacle",
            ),
            max_duration_ms=30_000,
            action_level=ActionLevel.MOTION,
            policy_ref="traverse_level_ground",
            policy_instruction="move forward",
            action_permissions=SkillActionPermissions(
                allow_attack=False,
                allow_use=False,
                allow_jump=False,
                allow_drop=False,
                allow_inventory=False,
                allow_hotbar=False,
            ),
        ),
        SkillSpec(
            skill_id="traverse_visible_obstacle",
            version=5,
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
            # This escape needs the goal-conditioned STEVE body. The fast VPT
            # route cannot consume "jump forward" and may emit no locomotion at
            # all for this atomic option; the checkpoint still chooses every
            # native key and camera action without a scripted jump macro.
            action_level=ActionLevel.LATENT,
            policy_ref="traverse_obstacle",
            policy_instruction="jump forward",
            policy_condition_scale=6.0,
            action_permissions=SkillActionPermissions(
                allow_attack=False,
                allow_use=False,
                allow_drop=False,
                allow_inventory=False,
                allow_hotbar=False,
            ),
        ),
        SkillSpec(
            skill_id="use_target",
            version=4,
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
            action_level=ActionLevel.GROUNDED,
            policy_ref="use",
            policy_instruction="use the target",
        ),
        SkillSpec(
            skill_id="activate_visible_gui_control",
            version=3,
            name="Activate visible GUI control",
            description=(
                "Use the explicitly grounded visible GUI control, verify the resulting screen "
                "transition, and avoid emitting world locomotion while a menu is open"
            ),
            stage=SkillStage.EXPERIMENTAL,
            parameters=("control",),
            preconditions=(SkillCondition(key="scene.ui_overlay", operator="truthy"),),
            success_conditions=(SkillCondition(key="scene.playable", operator="truthy"),),
            expected_effects=("gui_transition",),
            max_duration_ms=10_000,
            action_level=ActionLevel.GUI,
            policy_ref="gui",
            policy_instruction="click button",
        ),
        SkillSpec(
            skill_id="respawn_after_death",
            version=2,
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
            action_level=ActionLevel.GUI,
            # This remains a learned UI option: the contract supplies no screen
            # coordinate, click macro, or privileged game-state action.
            policy_ref="death_gui",
            policy_instruction="respawn",
        ),
        SkillSpec(
            skill_id="open_inventory",
            version=3,
            name="Open inventory",
            description=(
                "Open the Bedrock inventory with one bounded inventory toggle and wait for "
                "the calibrated inventory overlay detector before continuing"
            ),
            stage=SkillStage.EXPERIMENTAL,
            success_conditions=(
                SkillCondition(
                    key="scene.inventory_overlay",
                    operator="truthy",
                    min_confidence=0.99,
                ),
            ),
            expected_effects=("inventory_opened", "inventory_overlay_observed"),
            max_duration_ms=10_000,
            action_level=ActionLevel.GUI,
            # Opening inventory is a fixed native control transition. The
            # learned controller does not need to rediscover the E binding.
            policy_ref="open_inventory",
            policy_instruction="open inventory",
            action_permissions=SkillActionPermissions(
                allow_attack=False,
                allow_use=False,
                allow_jump=False,
                allow_drop=False,
                allow_hotbar=False,
            ),
        ),
        SkillSpec(
            skill_id="close_open_inventory",
            version=4,
            name="Close open inventory",
            description=(
                "Close the currently open Bedrock inventory with one bounded inventory toggle "
                "and verify that the playable world returns before resuming locomotion"
            ),
            stage=SkillStage.EXPERIMENTAL,
            success_conditions=(SkillCondition(key="scene.playable", operator="truthy"),),
            expected_effects=("inventory_closed", "playable_scene_restored"),
            max_duration_ms=10_000,
            action_level=ActionLevel.GUI,
            # This recovery deliberately has no ROCKET body route. The runtime
            # emits one bounded native inventory toggle, then waits for visual
            # proof that the playable world returned.
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
            version=5,
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
            action_level=ActionLevel.GROUNDED,
            policy_ref="place",
            policy_instruction="place a block",
            action_permissions=SkillActionPermissions(allow_attack=False, allow_jump=False),
        ),
        SkillSpec(
            skill_id="gather_nearby_wood",
            version=5,
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
            action_level=ActionLevel.GROUNDED,
            policy_ref="gather_wood",
            policy_instruction="mine log",
            action_permissions=SkillActionPermissions(
                allow_use=False,
                allow_drop=False,
                allow_inventory=False,
                allow_hotbar=False,
            ),
        ),
        SkillSpec(
            skill_id="craft_wood_planks",
            version=4,
            name="Craft wood planks",
            description=(
                "Open and visually verify the Bedrock inventory, right-click one freshly "
                "grounded craftable planks recipe, require a fresh log decrease plus at least "
                "four observed planks, and close the interface safely"
            ),
            stage=SkillStage.EXPERIMENTAL,
            # The bounded controller opens the inventory first and establishes
            # a GUI-grounded baseline there; world-view inventory counts are
            # optional and must not prevent entry into the transaction.
            preconditions=(),
            success_conditions=(SkillCondition(key="inventory.planks", operator="gte", value=4),),
            failure_conditions=(SkillCondition(key="danger.immediate", operator="truthy"),),
            expected_effects=("inventory.logs-=1", "inventory.planks>=4"),
            recovery_skills=("close_open_inventory",),
            max_duration_ms=180_000,
            action_level=ActionLevel.GUI,
            policy_ref="craft_planks",
            policy_instruction="craft planks",
            action_permissions=SkillActionPermissions(
                allow_attack=False,
                allow_jump=False,
                allow_drop=False,
                allow_hotbar=False,
            ),
        ),
        SkillSpec(
            skill_id="craft_crafting_table",
            version=3,
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
            action_level=ActionLevel.GUI,
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
            preconditions=(SkillCondition(key="inventory.build_blocks", operator="gte", value=12),),
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
        SkillSpec(
            skill_id="craft_storage_units",
            version=1,
            name="Craft storage units",
            description=(
                "Use the crafting interface to convert planks into chests, take the chests "
                "into inventory, and close the interface safely"
            ),
            stage=SkillStage.EXPERIMENTAL,
            preconditions=(SkillCondition(key="inventory.planks", operator="gte", value=8),),
            success_conditions=(
                SkillCondition(key="inventory.chest", operator="gte", value=1),
            ),
            failure_conditions=(SkillCondition(key="danger.immediate", operator="truthy"),),
            expected_effects=("inventory.planks-=8", "inventory.chest>=1"),
            max_duration_ms=90_000,
            action_level=ActionLevel.GUI,
            policy_ref="craft_storage_units",
            policy_instruction="craft chests for storage",
        ),
        SkillSpec(
            skill_id="deposit_in_storage",
            version=1,
            name="Deposit materials in storage",
            description=(
                "Place a nearby storage chest if none is placed, then deposit surplus "
                "gathered materials from inventory into the chest and close the interface"
            ),
            stage=SkillStage.EXPERIMENTAL,
            preconditions=(SkillCondition(key="inventory.chest", operator="gte", value=1),),
            success_conditions=(
                SkillCondition(key="inventory.surplus_deposited", operator="truthy"),
            ),
            failure_conditions=(SkillCondition(key="danger.immediate", operator="truthy"),),
            expected_effects=("materials_stored", "inventory.surplus_deposited"),
            recovery_skills=("escape_submersion", "retreat_from_danger"),
            max_duration_ms=120_000,
            action_level=ActionLevel.GROUNDED,
            policy_ref="deposit_in_storage",
            policy_instruction="store surplus materials in a chest",
        ),
        SkillSpec(
            skill_id="build_workshop_shell",
            version=1,
            name="Build a workshop shell",
            description=(
                "Choose level ground near storage and build a sturdy enclosed workshop: "
                "solid walls, a roof, a doorway, and a chest room for stored materials"
            ),
            stage=SkillStage.CANDIDATE,
            preconditions=(SkillCondition(key="inventory.build_blocks", operator="gte", value=32),),
            success_conditions=(
                SkillCondition(key="environment.workshop_enclosed", operator="truthy"),
            ),
            failure_conditions=(SkillCondition(key="danger.immediate", operator="truthy"),),
            expected_effects=("workshop_structure", "environment.workshop_enclosed"),
            recovery_skills=("escape_submersion", "retreat_from_danger"),
            max_duration_ms=300_000,
            policy_ref="build_workshop_shell",
            policy_instruction="build a workshop",
        ),
    )
    for spec in specs:
        library.register(spec)
    return library
