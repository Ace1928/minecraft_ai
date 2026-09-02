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
from minecraft_ai.spatial import PlaceKind, PlaceRecord, SpatialPlaceMemory
from minecraft_ai.wiki_agent import WikiQueryAgent


def _fixture_graph() -> KnowledgeGraph:
    version = GameVersion(edition=Edition.BEDROCK, version_id="1.20")
    graph = KnowledgeGraph(version)
    provenance = Provenance(
        source_type="manual_override",
        source_id="fixture",
        version_key=version.key,
        confidence=1.0,
    )
    graph.add_node(
        KnowledgeNode(node_id="item:minecraft:planks", kind=NodeKind.ITEM, name="minecraft:planks")
    )
    graph.add_node(
        KnowledgeNode(node_id="item:minecraft:stick", kind=NodeKind.ITEM, name="minecraft:stick")
    )
    graph.add_node(
        KnowledgeNode(node_id="process:stick", kind=NodeKind.PROCESS, name="stick recipe")
    )

    graph.add_edge(
        KnowledgeEdge(
            edge_id="e1",
            source="process:stick",
            target="item:minecraft:stick",
            kind=EdgeKind.CRAFTS,
            provenance=provenance,
        )
    )
    graph.add_edge(
        KnowledgeEdge(
            edge_id="e2",
            source="process:stick",
            target="item:minecraft:planks",
            kind=EdgeKind.REQUIRES,
            provenance=provenance,
        )
    )
    return graph


def test_wiki_query_agent_recipe_intent() -> None:
    graph = _fixture_graph()
    agent = WikiQueryAgent(graph=graph)

    res = agent.query("how to craft stick")
    assert res.intent == "recipe"
    assert "stick" in res.answer_text.lower()


def test_wiki_query_agent_spatial_intent() -> None:
    spatial = SpatialPlaceMemory()
    spatial.upsert(
        PlaceRecord(
            place_id="base",
            name="Alpha Base",
            kind=PlaceKind.BASE,
            x=10.0,
            y=64.0,
            z=10.0,
            discovered_ns=100,
            last_visited_ns=100,
        )
    )
    agent = WikiQueryAgent(spatial_memory=spatial)

    res = agent.query("where is base")
    assert res.intent == "spatial"
    assert "Alpha Base" in res.answer_text
