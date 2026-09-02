from __future__ import annotations

import pytest

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


def _graph() -> KnowledgeGraph:
    version = GameVersion(edition=Edition.JAVA, version_id="test-version")
    graph = KnowledgeGraph(version)
    for node in (
        KnowledgeNode(node_id="item:log", kind=NodeKind.ITEM, name="Log"),
        KnowledgeNode(node_id="item:planks", kind=NodeKind.ITEM, name="Planks"),
        KnowledgeNode(node_id="process:planks", kind=NodeKind.PROCESS, name="Craft planks"),
        KnowledgeNode(node_id="item:table", kind=NodeKind.ITEM, name="Crafting Table"),
        KnowledgeNode(node_id="process:table", kind=NodeKind.PROCESS, name="Craft table"),
    ):
        graph.add_node(node)
    prov = Provenance(
        source_type="vanilla_data",
        source_id="generated:test",
        version_key=version.key,
    )
    graph.add_edge(
        KnowledgeEdge(
            edge_id="table:requires:planks",
            source="item:table",
            target="item:planks",
            kind=EdgeKind.REQUIRES,
            quantity=4,
            provenance=prov,
        )
    )
    graph.add_edge(
        KnowledgeEdge(
            edge_id="planks:requires:log",
            source="item:planks",
            target="item:log",
            kind=EdgeKind.REQUIRES,
            provenance=prov,
        )
    )
    graph.add_edge(
        KnowledgeEdge(
            edge_id="process:table:crafts",
            source="process:table",
            target="item:table",
            kind=EdgeKind.CRAFTS,
            provenance=prov,
        )
    )
    graph.add_edge(
        KnowledgeEdge(
            edge_id="process:planks:crafts",
            source="process:planks",
            target="item:planks",
            kind=EdgeKind.CRAFTS,
            quantity=4,
            provenance=prov,
        )
    )
    return graph


def test_prerequisite_closure() -> None:
    graph = _graph()
    assert graph.prerequisite_closure("item:table") == {"item:planks", "item:log"}


def test_ways_to_obtain() -> None:
    graph = _graph()
    ways = graph.ways_to_obtain("item:table")
    assert len(ways) == 1
    assert ways[0].source == "process:table"
    assert ways[0].kind == EdgeKind.CRAFTS


def test_graph_roundtrip() -> None:
    graph = _graph()
    restored = KnowledgeGraph.from_dict(graph.to_dict())
    assert restored.version == graph.version
    assert set(restored.nodes) == set(graph.nodes)
    assert set(restored.edges) == set(graph.edges)
    assert restored.validate() == []


def test_version_mismatch_fails_loudly() -> None:
    graph = _graph()
    graph.add_node(KnowledgeNode(node_id="item:test", kind=NodeKind.ITEM, name="Test"))
    wrong = Provenance(
        source_type="vanilla_data",
        source_id="generated:wrong",
        version_key="java:other-version",
    )
    with pytest.raises(ValueError, match="version mismatch"):
        graph.add_edge(
            KnowledgeEdge(
                edge_id="bad",
                source="item:test",
                target="item:log",
                kind=EdgeKind.REQUIRES,
                provenance=wrong,
            )
        )
