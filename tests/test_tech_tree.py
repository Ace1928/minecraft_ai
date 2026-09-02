from __future__ import annotations

from minecraft_ai.tech_tree import Milestone, TechAge, TechTreeTracker, BUILTIN_MILESTONES


def test_tech_tree_initialization() -> None:
    tracker = TechTreeTracker()
    assert len(tracker.milestones) == len(BUILTIN_MILESTONES)
    assert tracker.current_age == TechAge.WOOD_AGE
    
    first = tracker.next_priority_milestone()
    assert first is not None
    assert first.milestone_id == "gather_logs"


def test_tech_tree_progression_by_inventory() -> None:
    tracker = TechTreeTracker()
    
    # Simulate picking up 4 oak logs
    unlocked = tracker.update_with_inventory({"oak_log": 4})
    assert len(unlocked) == 1
    assert unlocked[0].milestone_id == "gather_logs"
    assert "gather_logs" in tracker.completed_milestones

    # Next milestone should be craft_planks
    next_m = tracker.next_priority_milestone()
    assert next_m is not None
    assert next_m.milestone_id == "craft_planks"

    # Simulate crafting planks & table
    tracker.update_with_inventory({"oak_planks": 8, "crafting_table": 1, "wooden_pickaxe": 1})
    assert "craft_planks" in tracker.completed_milestones
    assert "craft_crafting_table" in tracker.completed_milestones
    assert "craft_wooden_pickaxe" in tracker.completed_milestones

    # Age should now advance to Stone Age
    assert tracker.current_age == TechAge.STONE_AGE
    stone_m = tracker.next_priority_milestone()
    assert stone_m is not None
    assert stone_m.milestone_id == "mine_cobblestone"
