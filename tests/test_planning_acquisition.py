from __future__ import annotations

from minecraft_ai.knowledge import (
    EdgeKind,
    Edition,
    GameVersion,
    KnowledgeEdge,
    KnowledgeGraph,
    KnowledgeNode,
    NodeKind,
    Provenance,
)
from minecraft_ai.planning import DependencyPlanner, Goal
from minecraft_ai.skills import SkillLibrary


def _graph() -> KnowledgeGraph:
    version = GameVersion(edition=Edition.BEDROCK, version_id="test")
    graph = KnowledgeGraph(version)
    provenance = Provenance(
        source_type="test",
        source_id="fixture",
        version_key=version.key,
        confidence=1.0,
    )
    for name in ("planks", "bamboo", "stick", "coal", "torch"):
        graph.add_node(
            KnowledgeNode(
                node_id=f"item:minecraft:{name}",
                kind=NodeKind.ITEM,
                name=f"minecraft:{name}",
            )
        )
    graph.add_node(
        KnowledgeNode(
            node_id="process:stick",
            kind=NodeKind.PROCESS,
            name="stick recipe",
        )
    )
    graph.add_edge(
        KnowledgeEdge(
            edge_id="stick-out",
            source="process:stick",
            target="item:minecraft:stick",
            kind=EdgeKind.CRAFTS,
            quantity=4,
            provenance=provenance,
        )
    )
    graph.add_edge(
        KnowledgeEdge(
            edge_id="stick-planks",
            source="process:stick",
            target="item:minecraft:planks",
            kind=EdgeKind.ALTERNATIVE_REQUIRES,
            quantity=2,
            alternative_group="stick-source",
            provenance=provenance,
        )
    )
    graph.add_edge(
        KnowledgeEdge(
            edge_id="stick-bamboo",
            source="process:stick",
            target="item:minecraft:bamboo",
            kind=EdgeKind.ALTERNATIVE_REQUIRES,
            quantity=2,
            alternative_group="stick-source",
            provenance=provenance,
        )
    )
    graph.add_node(
        KnowledgeNode(
            node_id="process:torch",
            kind=NodeKind.PROCESS,
            name="torch recipe",
        )
    )
    graph.add_edge(
        KnowledgeEdge(
            edge_id="torch-out",
            source="process:torch",
            target="item:minecraft:torch",
            kind=EdgeKind.CRAFTS,
            quantity=4,
            provenance=provenance,
        )
    )
    for edge_id, target in (("torch-stick", "stick"), ("torch-coal", "coal")):
        graph.add_edge(
            KnowledgeEdge(
                edge_id=edge_id,
                source="process:torch",
                target=f"item:minecraft:{target}",
                kind=EdgeKind.REQUIRES,
                provenance=provenance,
            )
        )
    return graph


def test_planner_preserves_and_or_recipe_structure() -> None:
    planner = DependencyPlanner(_graph(), SkillLibrary())
    options = planner.acquisition_options("item:minecraft:torch")
    assert len(options) == 1
    torch = options[0]
    assert len(torch.requirements) == 2
    stick_slot = next(
        slot
        for slot in torch.requirements
        if any(plan.target == "item:minecraft:stick" for plan in slot)
    )
    stick_plan = stick_slot[0]
    assert len(stick_plan.requirements) == 1
    source_targets = {plan.target for plan in stick_plan.requirements[0]}
    assert source_targets == {"item:minecraft:planks", "item:minecraft:bamboo"}


def test_available_inventory_shortcuts_dependency() -> None:
    planner = DependencyPlanner(_graph(), SkillLibrary())
    plan = planner.best_acquisition_plan(
        "item:minecraft:torch",
        available={"item:minecraft:stick", "item:minecraft:coal"},
    )
    assert all(slot[0].already_available for slot in plan.requirements)
    assert plan.estimated_leaf_cost == 0.0


def test_dependency_plan_flattens_children_before_target() -> None:
    planner = DependencyPlanner(_graph(), SkillLibrary())
    plan = planner.make_dependency_plan(
        Goal(goal_id="torch", description="Make torches", target_node="item:minecraft:torch")
    )
    assert plan.validate() == []
    target = max(plan.steps.values(), key=lambda step: int(step.step_id.split("-")[1]))
    assert target.target_node == "item:minecraft:torch"
    assert target.prerequisites
