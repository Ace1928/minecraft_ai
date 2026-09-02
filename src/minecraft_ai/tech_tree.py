from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class TechAge(StrEnum):
    WOOD_AGE = "wood_age"
    STONE_AGE = "stone_age"
    IRON_AGE = "iron_age"
    DIAMOND_AGE = "diamond_age"
    AUTOMATION_AGE = "automation_age"


class Milestone(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    milestone_id: str
    age: TechAge
    name: str
    description: str
    target_node: str
    prerequisites: tuple[str, ...] = ()
    required_item: str | None = None
    required_quantity: int = 1
    skill_hint: str = "explore_forward"
    priority: float = Field(default=0.7, ge=0.0, le=1.0)


BUILTIN_MILESTONES: tuple[Milestone, ...] = (
    # Wood Age
    Milestone(
        milestone_id="gather_logs",
        age=TechAge.WOOD_AGE,
        name="Gather Wood Logs",
        description="Punch trees or harvest logs to obtain basic wood.",
        target_node="item:oak_log",
        prerequisites=(),
        required_item="minecraft:oak_log",
        required_quantity=3,
        skill_hint="mine_visible_block",
        priority=1.0,
    ),
    Milestone(
        milestone_id="craft_planks",
        age=TechAge.WOOD_AGE,
        name="Craft Wood Planks",
        description="Convert wood logs into wood planks.",
        target_node="item:oak_planks",
        prerequisites=("gather_logs",),
        required_item="minecraft:oak_planks",
        required_quantity=4,
        skill_hint="use_target",
        priority=0.95,
    ),
    Milestone(
        milestone_id="craft_crafting_table",
        age=TechAge.WOOD_AGE,
        name="Craft Crafting Table",
        description="Craft a 3x3 crafting table to unlock advanced recipes.",
        target_node="item:crafting_table",
        prerequisites=("craft_planks",),
        required_item="minecraft:crafting_table",
        required_quantity=1,
        skill_hint="place_block",
        priority=0.90,
    ),
    Milestone(
        milestone_id="craft_wooden_pickaxe",
        age=TechAge.WOOD_AGE,
        name="Craft Wooden Pickaxe",
        description="Craft a wooden pickaxe to begin mining stone.",
        target_node="item:wooden_pickaxe",
        prerequisites=("craft_crafting_table",),
        required_item="minecraft:wooden_pickaxe",
        required_quantity=1,
        skill_hint="use_target",
        priority=0.85,
    ),
    # Stone Age
    Milestone(
        milestone_id="mine_cobblestone",
        age=TechAge.STONE_AGE,
        name="Mine Cobblestone",
        description="Mine stone blocks to collect cobblestone.",
        target_node="item:cobblestone",
        prerequisites=("craft_wooden_pickaxe",),
        required_item="minecraft:cobblestone",
        required_quantity=8,
        skill_hint="mine_visible_block",
        priority=0.80,
    ),
    Milestone(
        milestone_id="craft_stone_pickaxe",
        age=TechAge.STONE_AGE,
        name="Craft Stone Pickaxe",
        description="Craft a durable stone pickaxe to mine iron ore.",
        target_node="item:stone_pickaxe",
        prerequisites=("mine_cobblestone",),
        required_item="minecraft:stone_pickaxe",
        required_quantity=1,
        skill_hint="use_target",
        priority=0.78,
    ),
    Milestone(
        milestone_id="craft_stone_sword",
        age=TechAge.STONE_AGE,
        name="Craft Stone Sword",
        description="Craft a stone sword for defense against hostile mobs.",
        target_node="item:stone_sword",
        prerequisites=("mine_cobblestone",),
        required_item="minecraft:stone_sword",
        required_quantity=1,
        skill_hint="use_target",
        priority=0.75,
    ),
    Milestone(
        milestone_id="craft_furnace",
        age=TechAge.STONE_AGE,
        name="Craft Furnace",
        description="Build a furnace to smelt ores and cook food.",
        target_node="item:furnace",
        prerequisites=("mine_cobblestone",),
        required_item="minecraft:furnace",
        required_quantity=1,
        skill_hint="place_block",
        priority=0.72,
    ),
    # Iron Age
    Milestone(
        milestone_id="mine_iron_ore",
        age=TechAge.IRON_AGE,
        name="Mine Iron Ore",
        description="Locate and mine underground iron ore deposits.",
        target_node="item:raw_iron",
        prerequisites=("craft_stone_pickaxe",),
        required_item="minecraft:raw_iron",
        required_quantity=3,
        skill_hint="mine_visible_block",
        priority=0.70,
    ),
    Milestone(
        milestone_id="smelt_iron_ingot",
        age=TechAge.IRON_AGE,
        name="Smelt Iron Ingots",
        description="Smelt raw iron in a furnace into iron ingots.",
        target_node="item:iron_ingot",
        prerequisites=("mine_iron_ore", "craft_furnace"),
        required_item="minecraft:iron_ingot",
        required_quantity=3,
        skill_hint="use_target",
        priority=0.68,
    ),
    Milestone(
        milestone_id="craft_iron_pickaxe",
        age=TechAge.IRON_AGE,
        name="Craft Iron Pickaxe",
        description="Craft an iron pickaxe to mine gold, redstone, and diamond.",
        target_node="item:iron_pickaxe",
        prerequisites=("smelt_iron_ingot",),
        required_item="minecraft:iron_pickaxe",
        required_quantity=1,
        skill_hint="use_target",
        priority=0.65,
    ),
    Milestone(
        milestone_id="craft_shield",
        age=TechAge.IRON_AGE,
        name="Craft Shield",
        description="Craft an iron shield for projectile deflection and combat protection.",
        target_node="item:shield",
        prerequisites=("smelt_iron_ingot",),
        required_item="minecraft:shield",
        required_quantity=1,
        skill_hint="use_target",
        priority=0.62,
    ),
)


@dataclass
class TechTreeTracker:
    """Tracks persistent technology progression, unlocked milestones, and current priority goal."""

    milestones: dict[str, Milestone] = field(default_factory=dict)
    completed_milestones: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not self.milestones:
            for m in BUILTIN_MILESTONES:
                self.milestones[m.milestone_id] = m

    @property
    def current_age(self) -> TechAge:
        for age in (TechAge.WOOD_AGE, TechAge.STONE_AGE, TechAge.IRON_AGE, TechAge.DIAMOND_AGE):
            age_milestones = [m for m in self.milestones.values() if m.age == age]
            if age_milestones and not all(m.milestone_id in self.completed_milestones for m in age_milestones):
                return age
        return TechAge.AUTOMATION_AGE

    def update_with_inventory(self, inventory: dict[str, int]) -> list[Milestone]:
        """Automatically complete milestones when matching items appear in inventory."""
        newly_completed: list[Milestone] = []
        for milestone in self.milestones.values():
            if milestone.milestone_id in self.completed_milestones:
                continue
            if milestone.required_item is not None:
                item_key = milestone.required_item.lower()
                clean_key = item_key.split(":")[-1]
                count = max(inventory.get(item_key, 0), inventory.get(clean_key, 0))
                if count >= milestone.required_quantity:
                    self.completed_milestones.add(milestone.milestone_id)
                    newly_completed.append(milestone)
        return newly_completed

    def next_priority_milestone(self) -> Milestone | None:
        """Find the highest-priority milestone whose prerequisites are completed."""
        ready: list[Milestone] = []
        for m in self.milestones.values():
            if m.milestone_id in self.completed_milestones:
                continue
            if all(prereq in self.completed_milestones for prereq in m.prerequisites):
                ready.append(m)

        if not ready:
            return None
        ready.sort(key=lambda m: m.priority, reverse=True)
        return ready[0]
