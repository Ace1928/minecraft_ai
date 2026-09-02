from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class BedrockWorldContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    world_fixture_id: str
    reset_method: str
    strict_agent_observations: tuple[str, ...]
    evaluator_observations: tuple[str, ...]
    notes: str


BEDROCK_WORLD_CONTRACTS: tuple[BedrockWorldContract, ...] = (
    BedrockWorldContract(
        world_fixture_id="movement-range",
        reset_method="versioned-world-copy",
        strict_agent_observations=("pixels", "hud"),
        evaluator_observations=("start_region", "finish_region", "death_count"),
        notes="Level lane and one-block obstacle lanes with evaluator-only finish regions.",
    ),
    BedrockWorldContract(
        world_fixture_id="target-range",
        reset_method="versioned-world-copy",
        strict_agent_observations=("pixels",),
        evaluator_observations=("target_screen_error",),
        notes="Marked targets at varied scales, lighting, and initial camera offsets.",
    ),
    BedrockWorldContract(
        world_fixture_id="water-recovery",
        reset_method="versioned-world-copy",
        strict_agent_observations=("pixels", "hud"),
        evaluator_observations=("water_contact", "exit_region", "death_count"),
        notes="Bounded shallow-water recovery without hidden state exposed to the agent.",
    ),
    BedrockWorldContract(
        world_fixture_id="resource-range",
        reset_method="versioned-world-copy",
        strict_agent_observations=("pixels", "hud"),
        evaluator_observations=("block_changes", "inventory_delta"),
        notes="Visible resource targets with controlled distractors and post-hoc inventory labels.",
    ),
    BedrockWorldContract(
        world_fixture_id="gui-range",
        reset_method="versioned-world-copy",
        strict_agent_observations=("pixels", "hud", "gui"),
        evaluator_observations=("gui_transitions", "selected_slot"),
        notes="Inventory and hotbar tasks scored from evaluator-only transition labels.",
    ),
    BedrockWorldContract(
        world_fixture_id="crafting-range",
        reset_method="versioned-world-copy",
        strict_agent_observations=("pixels", "hud", "gui"),
        evaluator_observations=("inventory_delta", "recipe_output"),
        notes="Supplied inputs; result validated post-hoc rather than exposed during control.",
    ),
    BedrockWorldContract(
        world_fixture_id="smelting-range",
        reset_method="versioned-world-copy",
        strict_agent_observations=("pixels", "hud", "gui"),
        evaluator_observations=("inventory_delta", "furnace_transition"),
        notes="Supplied furnace, fuel, and raw material with evaluator-only outcome labels.",
    ),
    BedrockWorldContract(
        world_fixture_id="survival-range",
        reset_method="versioned-world-copy",
        strict_agent_observations=("pixels", "hud"),
        evaluator_observations=("health_delta", "hunger_delta", "death_count"),
        notes="Controlled hunger and survival state; agent sees only ordinary HUD evidence.",
    ),
    BedrockWorldContract(
        world_fixture_id="combat-range",
        reset_method="versioned-world-copy",
        strict_agent_observations=("pixels", "hud", "audio"),
        evaluator_observations=("entity_outcomes", "health_delta", "death_count"),
        notes="Isolated hostile trials with post-hoc entity and survival scoring.",
    ),
    BedrockWorldContract(
        world_fixture_id="building-range",
        reset_method="versioned-world-copy",
        strict_agent_observations=("pixels", "hud", "gui"),
        evaluator_observations=("block_volume", "structure_match"),
        notes="Marked build region scored after execution from controlled world state.",
    ),
    BedrockWorldContract(
        world_fixture_id="memory-range",
        reset_method="persistent-versioned-world-copy",
        strict_agent_observations=("pixels", "hud", "agent_memory"),
        evaluator_observations=("place_regions", "restart_boundary", "project_state"),
        notes="Multi-session place/project tests without coordinate observations.",
    ),
    BedrockWorldContract(
        world_fixture_id="social-range",
        reset_method="persistent-versioned-world-copy",
        strict_agent_observations=("pixels", "chat", "agent_memory"),
        evaluator_observations=("request_log", "promise_state", "world_outcome"),
        notes="Scripted player messages plus world verification; no hidden labels reach the agent.",
    ),
    BedrockWorldContract(
        world_fixture_id="persistent-survival",
        reset_method="persistent-world-snapshot",
        strict_agent_observations=("pixels", "hud", "chat", "agent_memory"),
        evaluator_observations=("inventory_delta", "place_regions", "world_changes"),
        notes="Long-horizon Bedrock survival world for compositional and autonomy evaluation.",
    ),
)


def world_contract(world_fixture_id: str) -> BedrockWorldContract:
    for contract in BEDROCK_WORLD_CONTRACTS:
        if contract.world_fixture_id == world_fixture_id:
            return contract
    raise KeyError(world_fixture_id)
