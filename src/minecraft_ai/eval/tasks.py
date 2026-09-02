from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class BenchmarkCategory(StrEnum):
    NAVIGATION = "navigation"
    PERCEPTION = "perception_grounding"
    RESOURCE = "resource_acquisition"
    GUI = "inventory_gui"
    CRAFTING = "crafting_smelting"
    SURVIVAL = "survival"
    COMBAT = "combat"
    EXPLORATION = "exploration"
    BUILDING = "building"
    MEMORY = "memory_return"
    SOCIAL = "social_collaboration"


class BenchmarkLevel(StrEnum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"


class MetricOperator(StrEnum):
    EQ = "eq"
    GTE = "gte"
    LTE = "lte"
    TRUTHY = "truthy"


class MetricCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: str
    operator: MetricOperator
    value: float | int | bool | str | None = None


class BenchmarkTask(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    name: str
    category: BenchmarkCategory
    level: BenchmarkLevel
    instruction: str
    timeout_s: float = Field(gt=0.0, le=3600.0)
    criteria: tuple[MetricCriterion, ...]
    world_fixture_id: str
    evaluator_channel: str
    protected: bool = False
    minimum_repetitions: int = Field(default=5, ge=1, le=1000)


class BenchmarkSuite(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    suite_id: str
    version: int = Field(ge=1)
    tasks: tuple[BenchmarkTask, ...]

    def task(self, task_id: str) -> BenchmarkTask:
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        raise KeyError(task_id)


def _criterion(
    metric: str,
    operator: MetricOperator = MetricOperator.GTE,
    value: float | int | bool | str | None = 1,
) -> MetricCriterion:
    return MetricCriterion(metric=metric, operator=operator, value=value)


def _task(
    task_id: str,
    name: str,
    category: BenchmarkCategory,
    level: BenchmarkLevel,
    instruction: str,
    *criteria: MetricCriterion,
    fixture: str = "persistent-survival",
    evaluator: str = "trajectory+controlled-posthoc",
    timeout_s: float = 60.0,
    protected: bool = False,
) -> BenchmarkTask:
    return BenchmarkTask(
        task_id=task_id,
        name=name,
        category=category,
        level=level,
        instruction=instruction,
        timeout_s=timeout_s,
        criteria=criteria,
        world_fixture_id=fixture,
        evaluator_channel=evaluator,
        protected=protected,
    )


def bedrock_baseline_suite() -> BenchmarkSuite:
    """Frozen Bedrock-native M1 suite; every success criterion is machine-readable."""
    tasks = (
        _task(
            "a_turn_to_target",
            "Turn toward a visible target",
            BenchmarkCategory.PERCEPTION,
            BenchmarkLevel.A,
            "Turn the camera until the marked target is centered.",
            _criterion("action.camera_updates", value=1),
            _criterion("event.target_centered", value=1),
            fixture="target-range",
            protected=True,
        ),
        _task(
            "a_move_forward",
            "Traverse level ground",
            BenchmarkCategory.NAVIGATION,
            BenchmarkLevel.A,
            "Move forward across the marked level lane.",
            _criterion("action.forward_presses", value=1),
            _criterion("event.destination_reached", value=1),
            fixture="movement-range",
            protected=True,
        ),
        _task(
            "a_jump_obstacle",
            "Jump a one-block obstacle",
            BenchmarkCategory.NAVIGATION,
            BenchmarkLevel.A,
            "Cross the lane and jump over the one-block obstacle.",
            _criterion("action.jump_presses", value=1),
            _criterion("event.destination_reached", value=1),
            fixture="movement-range",
            protected=True,
        ),
        _task(
            "a_sprint_jump",
            "Sprint-jump traversal",
            BenchmarkCategory.NAVIGATION,
            BenchmarkLevel.A,
            "Sprint-jump to the end of the marked lane.",
            _criterion("action.sprint_jump_presses", value=1),
            _criterion("event.destination_reached", value=1),
            fixture="movement-range",
        ),
        _task(
            "a_escape_water",
            "Exit shallow water",
            BenchmarkCategory.SURVIVAL,
            BenchmarkLevel.A,
            "Swim to the visible bank and leave the water.",
            _criterion("event.water_exited", value=1),
            fixture="water-recovery",
            protected=True,
        ),
        _task(
            "b_ground_log",
            "Ground a visible log",
            BenchmarkCategory.PERCEPTION,
            BenchmarkLevel.B,
            "Locate and center the marked tree log.",
            _criterion("event.target_centered", value=1),
            fixture="resource-range",
        ),
        _task(
            "b_mine_log",
            "Mine one log",
            BenchmarkCategory.RESOURCE,
            BenchmarkLevel.B,
            "Mine the marked log until it breaks.",
            _criterion("event.block_broken", value=1),
            fixture="resource-range",
            protected=True,
        ),
        _task(
            "b_collect_drop",
            "Collect a dropped item",
            BenchmarkCategory.RESOURCE,
            BenchmarkLevel.B,
            "Collect the marked dropped resource.",
            _criterion("event.resource_obtained", value=1),
            fixture="resource-range",
        ),
        _task(
            "b_open_inventory",
            "Open and close inventory",
            BenchmarkCategory.GUI,
            BenchmarkLevel.B,
            "Open inventory, then return safely to the world.",
            _criterion("event.gui_opened", value=1),
            _criterion("event.gui_closed", value=1),
            fixture="gui-range",
            protected=True,
        ),
        _task(
            "b_select_hotbar",
            "Select requested hotbar item",
            BenchmarkCategory.GUI,
            BenchmarkLevel.B,
            "Select the requested visible hotbar slot.",
            _criterion("action.hotbar_presses", value=1),
            _criterion("event.requested_slot_selected", value=1),
            fixture="gui-range",
        ),
        _task(
            "b_craft_planks",
            "Craft wood planks",
            BenchmarkCategory.CRAFTING,
            BenchmarkLevel.B,
            "Convert one log into wood planks.",
            _criterion("event.item_crafted", value=1),
            _criterion("reward.inventory_planks_delta", value=4),
            fixture="crafting-range",
            timeout_s=90.0,
            protected=True,
        ),
        _task(
            "b_craft_table",
            "Craft a crafting table",
            BenchmarkCategory.CRAFTING,
            BenchmarkLevel.B,
            "Craft and retain one crafting table.",
            _criterion("reward.inventory_crafting_table_delta", value=1),
            fixture="crafting-range",
            timeout_s=120.0,
        ),
        _task(
            "b_smelt_iron",
            "Smelt one iron ingot",
            BenchmarkCategory.CRAFTING,
            BenchmarkLevel.B,
            "Use the supplied furnace inputs to smelt one iron ingot.",
            _criterion("event.item_smelted", value=1),
            _criterion("reward.inventory_iron_ingot_delta", value=1),
            fixture="smelting-range",
            timeout_s=180.0,
        ),
        _task(
            "b_eat_food",
            "Eat when hungry",
            BenchmarkCategory.SURVIVAL,
            BenchmarkLevel.B,
            "Eat supplied food and increase visible hunger.",
            _criterion("event.food_eaten", value=1),
            _criterion("reward.hunger_delta", value=1),
            fixture="survival-range",
            protected=True,
        ),
        _task(
            "b_defeat_zombie",
            "Defeat one zombie",
            BenchmarkCategory.COMBAT,
            BenchmarkLevel.B,
            "Defeat the isolated zombie and remain alive.",
            _criterion("event.hostile_defeated", value=1),
            _criterion("reward.alive_end", MetricOperator.EQ, True),
            fixture="combat-range",
            timeout_s=120.0,
            protected=True,
        ),
        _task(
            "b_place_block",
            "Place a requested block",
            BenchmarkCategory.BUILDING,
            BenchmarkLevel.B,
            "Place one block in the marked location.",
            _criterion("event.block_placed", value=1),
            fixture="building-range",
        ),
        _task(
            "c_gather_and_craft",
            "Gather wood and craft a table",
            BenchmarkCategory.CRAFTING,
            BenchmarkLevel.C,
            "Gather wood and craft a crafting table from an empty inventory.",
            _criterion("reward.inventory_crafting_table_delta", value=1),
            fixture="persistent-survival",
            timeout_s=600.0,
        ),
        _task(
            "c_explore_new_place",
            "Discover a new place",
            BenchmarkCategory.EXPLORATION,
            BenchmarkLevel.C,
            "Explore safely until a novel useful place is recorded.",
            _criterion("event.novel_place", value=1),
            fixture="persistent-survival",
            timeout_s=300.0,
        ),
        _task(
            "c_basic_shelter",
            "Build a basic shelter",
            BenchmarkCategory.BUILDING,
            BenchmarkLevel.C,
            "Build an enclosed enterable shelter with walls, roof, and doorway.",
            _criterion("event.structure_verified", value=1),
            fixture="building-range",
            timeout_s=900.0,
        ),
        _task(
            "c_return_to_place",
            "Return to remembered base",
            BenchmarkCategory.MEMORY,
            BenchmarkLevel.C,
            "Travel away, then return to the recorded base without coordinates.",
            _criterion("event.returned_to_place", value=1),
            fixture="memory-range",
            timeout_s=600.0,
        ),
        _task(
            "e_answer_request",
            "Acknowledge a player request",
            BenchmarkCategory.SOCIAL,
            BenchmarkLevel.E,
            "Understand, acknowledge, and retain the supplied player request.",
            _criterion("event.player_request_acknowledged", value=1),
            fixture="social-range",
        ),
        _task(
            "e_fulfill_promise",
            "Fulfill a retained promise",
            BenchmarkCategory.SOCIAL,
            BenchmarkLevel.E,
            "Complete the promised task and report only after verification.",
            _criterion("event.promise_fulfilled", value=1),
            fixture="social-range",
            timeout_s=900.0,
        ),
        _task(
            "e_resume_project",
            "Resume a project after restart",
            BenchmarkCategory.MEMORY,
            BenchmarkLevel.E,
            "After restart, recover and continue the unfinished project.",
            _criterion("event.project_resumed", value=1),
            fixture="memory-range",
            timeout_s=900.0,
        ),
        _task(
            "e_divide_shared_work",
            "Coordinate shared work",
            BenchmarkCategory.SOCIAL,
            BenchmarkLevel.E,
            "Remove the player's assigned resource task and complete the agent assignment.",
            _criterion("event.shared_plan_updated", value=1),
            _criterion("event.agent_assignment_completed", value=1),
            fixture="social-range",
            timeout_s=900.0,
        ),
    )
    return BenchmarkSuite(suite_id="bedrock-m1-baseline-v1", version=1, tasks=tasks)
