from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RoleProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role_id: str = Field(min_length=1, max_length=128)
    description: str = ""
    standing_goals: tuple[str, ...] = ()
    utility_weights: dict[str, float] = Field(default_factory=dict)
    risk_tolerance: float = Field(default=0.5, ge=0.0, le=1.0)
    preferred_skills: tuple[str, ...] = ()
    knowledge_domains: tuple[str, ...] = ()
    reserve_targets: dict[str, int] = Field(default_factory=dict)

    def weight(self, domain: str, default: float = 0.0) -> float:
        return self.utility_weights.get(domain, default)


BUILTIN_ROLES: dict[str, RoleProfile] = {
    "generalist": RoleProfile(
        role_id="generalist",
        description="Balanced autonomous survival and progression.",
        standing_goals=("survive", "progress", "help_players"),
        utility_weights={"survival": 1.0, "progression": 0.9, "social": 0.7},
        risk_tolerance=0.45,
    ),
    "farmer": RoleProfile(
        role_id="farmer",
        standing_goals=("maintain_food_surplus", "expand_farms", "breed_animals"),
        utility_weights={"farming": 1.0, "survival": 0.9, "trading": 0.55},
        risk_tolerance=0.25,
        reserve_targets={"food": 64},
    ),
    "trader": RoleProfile(
        role_id="trader",
        standing_goals=("improve_trade_network", "maintain_trade_stock", "protect_villagers"),
        utility_weights={"trading": 1.0, "exploration": 0.45, "combat": 0.35},
        risk_tolerance=0.3,
    ),
    "builder": RoleProfile(
        role_id="builder",
        standing_goals=("maintain_material_reserve", "improve_base", "finish_shared_builds"),
        utility_weights={"construction": 1.0, "gathering": 0.75, "exploration": 0.35},
        risk_tolerance=0.35,
    ),
    "industrial_builder": RoleProfile(
        role_id="industrial_builder",
        description=(
            "Continuously grows an industrial infrastructure: gather materials, "
            "craft tools and storage, build a base with chests, deposit surplus "
            "materials into storage, and keep exploring for new resources. Never "
            "idles; always either gathering, building, storing, or exploring."
        ),
        standing_goals=(
            "gather_materials",
            "build_storage",
            "store_materials",
            "expand_workshop",
            "explore_for_resources",
        ),
        utility_weights={
            "gathering": 1.0,
            "construction": 0.95,
            "exploration": 0.75,
            "progression": 0.6,
            "survival": 0.4,
        },
        risk_tolerance=0.4,
        preferred_skills=(
            "gather_nearby_wood",
            "gather_cobblestone",
            "mine_coal_ore",
            "craft_crafting_table",
            "craft_storage_units",
            "deposit_in_storage",
            "build_workshop_shell",
            "explore_forward",
            "traverse_level_ground",
        ),
        knowledge_domains=("crafting", "storage", "structures", "mining"),
        reserve_targets={"oak_log": 16, "cobblestone": 32, "coal": 8, "oak_planks": 16},
    ),
    "creative_builder": RoleProfile(
        role_id="creative_builder",
        description=(
            "Creative-mode architect: place blocks into deliberate structures "
            "(houses, walls, landscapes, monuments), fly to reach build sites, "
            "and answer player questions in world chat like an in-game wiki. "
            "No survival constraints; never gather, craft, or mine repeatedly."
        ),
        standing_goals=(
            "build_creative_structures",
            "assist_player_builds",
            "answer_world_questions",
            "fly_to_build_site",
            "showcase_architecture",
        ),
        utility_weights={
            "construction": 1.0,
            "social": 0.9,
            "exploration": 0.5,
            "gathering": 0.0,
            "progression": 0.0,
        },
        risk_tolerance=0.1,
        preferred_skills=(
            "place_block",
            "use_target",
            "build_workshop_shell",
            "establish_basic_shelter",
            "traverse_level_ground",
            "explore_forward",
            "craft_crafting_table",
            "activate_visible_gui_control",
        ),
        knowledge_domains=("architecture", "chat", "commands", "structures"),
        reserve_targets={},
    ),
    "redstone_engineer": RoleProfile(
        role_id="redstone_engineer",
        standing_goals=("maintain_redstone_stock", "automate_repetitive_work", "prototype_systems"),
        utility_weights={"redstone": 1.0, "gathering": 0.6, "construction": 0.65},
        risk_tolerance=0.4,
    ),
    "fighter": RoleProfile(
        role_id="fighter",
        standing_goals=("maintain_combat_readiness", "protect_players", "clear_hostiles"),
        utility_weights={"combat": 1.0, "equipment": 0.85, "exploration": 0.45},
        risk_tolerance=0.7,
    ),
    "mob_farmer": RoleProfile(
        role_id="mob_farmer",
        standing_goals=("build_mob_farms", "optimize_drops", "maintain_safe_kill_zones"),
        utility_weights={"mob_farming": 1.0, "redstone": 0.7, "construction": 0.7},
        risk_tolerance=0.5,
    ),
    "explorer": RoleProfile(
        role_id="explorer",
        standing_goals=("discover_biomes", "map_structures", "establish_routes"),
        utility_weights={"exploration": 1.0, "survival": 0.75, "combat": 0.45},
        risk_tolerance=0.6,
    ),
    "speedrunner": RoleProfile(
        role_id="speedrunner",
        standing_goals=("reach_end_quickly", "minimize_detours", "exploit_safe_opportunities"),
        utility_weights={"progression": 1.0, "time_efficiency": 1.0, "construction": 0.05},
        risk_tolerance=0.85,
    ),
    "boss_hunter": RoleProfile(
        role_id="boss_hunter",
        standing_goals=("prepare_boss_loadouts", "locate_bosses", "defeat_bosses"),
        utility_weights={"bossing": 1.0, "combat": 0.9, "equipment": 0.9},
        risk_tolerance=0.75,
    ),
    "nether_specialist": RoleProfile(
        role_id="nether_specialist",
        standing_goals=("secure_nether_routes", "gather_nether_resources", "map_fortresses"),
        utility_weights={"nether": 1.0, "exploration": 0.7, "combat": 0.65},
        risk_tolerance=0.65,
    ),
    "shopkeeper": RoleProfile(
        role_id="shopkeeper",
        standing_goals=("maintain_stock", "serve_customers", "secure_shop", "restock"),
        utility_weights={"social": 1.0, "trading": 1.0, "gathering": 0.55},
        risk_tolerance=0.2,
    ),
    "wiki_assistant": RoleProfile(
        role_id="wiki_assistant",
        description="In-game crafting guide, recipe advisor, and Minecraft knowledge expert.",
        standing_goals=("answer_player_queries", "lookup_recipes", "guide_players"),
        utility_weights={"social": 1.0, "knowledge": 1.0, "crafting": 0.9},
        risk_tolerance=0.2,
    ),
    "miner": RoleProfile(
        role_id="miner",
        description=(
            "Specialized deep mining, cave exploration, ore extraction, and resource staging."
        ),
        standing_goals=("mine_ores", "explore_caves", "stage_resources", "maintain_mine_shafts"),
        utility_weights={"mining": 1.0, "gathering": 0.9, "survival": 0.7},
        risk_tolerance=0.55,
    ),
    "companion": RoleProfile(
        role_id="companion",
        description="Friendly companion who follows players, assists in tasks, and guards.",
        standing_goals=("follow_player", "assist_player", "guard_player", "share_resources"),
        utility_weights={"social": 1.0, "combat": 0.8, "gathering": 0.7},
        risk_tolerance=0.5,
    ),
}


_CUSTOM_ROLES: dict[str, RoleProfile] = {}


def register_custom_role(role: RoleProfile) -> RoleProfile:
    _CUSTOM_ROLES[role.role_id] = role
    return role


def get_role(role_id: str) -> RoleProfile:
    if role_id in _CUSTOM_ROLES:
        return _CUSTOM_ROLES[role_id]
    try:
        return BUILTIN_ROLES[role_id]
    except KeyError as exc:
        raise KeyError(f"unknown role: {role_id}") from exc


def list_roles() -> list[RoleProfile]:
    combined = dict(BUILTIN_ROLES)
    combined.update(_CUSTOM_ROLES)
    return list(combined.values())
