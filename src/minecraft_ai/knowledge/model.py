from __future__ import annotations

from collections import defaultdict, deque
from enum import StrEnum
from typing import Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field


class Edition(StrEnum):
    JAVA = "java"
    BEDROCK = "bedrock"


class GameVersion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    edition: Edition
    version_id: str = Field(min_length=1, max_length=128)
    data_version: int | None = None
    protocol_version: int | None = None
    loader_profile: str | None = Field(default=None, max_length=128)
    source_hashes: dict[str, str] = Field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.edition.value}:{self.version_id}"


class NodeKind(StrEnum):
    ITEM = "item"
    BLOCK = "block"
    ENTITY = "entity"
    BIOME = "biome"
    STRUCTURE = "structure"
    DIMENSION = "dimension"
    ADVANCEMENT = "advancement"
    ACHIEVEMENT = "achievement"
    CAPABILITY = "capability"
    PROCESS = "process"
    TAG = "tag"
    ENCHANTMENT = "enchantment"


class EdgeKind(StrEnum):
    REQUIRES = "requires"
    ALTERNATIVE_REQUIRES = "alternative_requires"
    PRODUCES = "produces"
    DROPS = "drops"
    CRAFTS = "crafts"
    COOKS = "cooks"
    SMITHS = "smiths"
    BREWS = "brews"
    TRADES = "trades"
    BARTERS = "barters"
    LOCATED_IN = "located_in"
    REQUIRES_TOOL = "requires_tool"
    REQUIRES_DIMENSION = "requires_dimension"
    REQUIRES_STRUCTURE = "requires_structure"
    UNLOCKS = "unlocks"
    ADVANCEMENT_REQUIRES = "advancement_requires"
    ACHIEVEMENT_REQUIRES = "achievement_requires"
    MEMBER_OF = "member_of"


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_type: Literal[
        "vanilla_data",
        "minecraft_data",
        "wiki",
        "manual_override",
        "observed_world",
    ]
    source_id: str = Field(min_length=1, max_length=1024)
    version_key: str = Field(min_length=1, max_length=256)
    revision: str | None = Field(default=None, max_length=256)
    checksum: str | None = Field(default=None, max_length=256)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class KnowledgeNode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str = Field(min_length=1, max_length=512)
    kind: NodeKind
    name: str = Field(min_length=1, max_length=256)
    namespace: str = Field(default="minecraft", min_length=1, max_length=128)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class KnowledgeEdge(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    edge_id: str = Field(min_length=1, max_length=1024)
    source: str = Field(min_length=1, max_length=512)
    target: str = Field(min_length=1, max_length=512)
    kind: EdgeKind
    quantity: float = Field(default=1.0, gt=0)
    alternative_group: str | None = Field(default=None, max_length=256)
    conditions: tuple[str, ...] = ()
    provenance: Provenance


class KnowledgeGraph:
    """Small typed graph core used by importers, planning and wiki explanation.

    This class intentionally contains no Minecraft-version assumptions. Importers
    are responsible for exact-version facts and provenance.
    """

    def __init__(self, version: GameVersion) -> None:
        self.version = version
        self.nodes: dict[str, KnowledgeNode] = {}
        self.edges: dict[str, KnowledgeEdge] = {}
        self._out: dict[str, list[str]] = defaultdict(list)
        self._in: dict[str, list[str]] = defaultdict(list)

    def add_node(self, node: KnowledgeNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, edge: KnowledgeEdge) -> None:
        if edge.provenance.version_key != self.version.key:
            raise ValueError(
                f"version mismatch: graph={self.version.key} edge={edge.provenance.version_key}"
            )
        if edge.source not in self.nodes or edge.target not in self.nodes:
            raise KeyError("edge endpoints must exist before insertion")
        previous = self.edges.get(edge.edge_id)
        if previous is not None:
            self._out[previous.source].remove(edge.edge_id)
            self._in[previous.target].remove(edge.edge_id)
        self.edges[edge.edge_id] = edge
        self._out[edge.source].append(edge.edge_id)
        self._in[edge.target].append(edge.edge_id)

    def outgoing(self, node_id: str, kinds: Iterable[EdgeKind] | None = None) -> list[KnowledgeEdge]:
        allowed = None if kinds is None else frozenset(kinds)
        return [
            self.edges[edge_id]
            for edge_id in self._out.get(node_id, ())
            if allowed is None or self.edges[edge_id].kind in allowed
        ]

    def incoming(self, node_id: str, kinds: Iterable[EdgeKind] | None = None) -> list[KnowledgeEdge]:
        allowed = None if kinds is None else frozenset(kinds)
        return [
            self.edges[edge_id]
            for edge_id in self._in.get(node_id, ())
            if allowed is None or self.edges[edge_id].kind in allowed
        ]

    def requirements(self, node_id: str) -> list[KnowledgeEdge]:
        """Direct prerequisite edges emitted from a desired target/process node."""
        return self.outgoing(
            node_id,
            kinds={
                EdgeKind.REQUIRES,
                EdgeKind.ALTERNATIVE_REQUIRES,
                EdgeKind.REQUIRES_TOOL,
                EdgeKind.REQUIRES_DIMENSION,
                EdgeKind.REQUIRES_STRUCTURE,
                EdgeKind.ADVANCEMENT_REQUIRES,
                EdgeKind.ACHIEVEMENT_REQUIRES,
            },
        )

    def ways_to_obtain(self, node_id: str) -> list[KnowledgeEdge]:
        """Processes/sources with an edge into the requested node."""
        return self.incoming(
            node_id,
            kinds={
                EdgeKind.PRODUCES,
                EdgeKind.DROPS,
                EdgeKind.CRAFTS,
                EdgeKind.COOKS,
                EdgeKind.SMITHS,
                EdgeKind.BREWS,
                EdgeKind.TRADES,
                EdgeKind.BARTERS,
            },
        )

    def prerequisite_closure(self, node_id: str, *, max_depth: int = 64) -> set[str]:
        if node_id not in self.nodes:
            raise KeyError(node_id)
        result: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(node_id, 0)])
        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for edge in self.requirements(current):
                if edge.target in result:
                    continue
                result.add(edge.target)
                queue.append((edge.target, depth + 1))
        return result

    def validate(self) -> list[str]:
        errors: list[str] = []
        for edge in self.edges.values():
            if edge.source not in self.nodes:
                errors.append(f"missing source {edge.source} for {edge.edge_id}")
            if edge.target not in self.nodes:
                errors.append(f"missing target {edge.target} for {edge.edge_id}")
            if edge.provenance.version_key != self.version.key:
                errors.append(f"version mismatch on {edge.edge_id}")
        return errors

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version.model_dump(mode="json"),
            "nodes": [node.model_dump(mode="json") for node in self.nodes.values()],
            "edges": [edge.model_dump(mode="json") for edge in self.edges.values()],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "KnowledgeGraph":
        version_raw = payload.get("version")
        nodes_raw = payload.get("nodes")
        edges_raw = payload.get("edges")
        if not isinstance(version_raw, dict) or not isinstance(nodes_raw, list) or not isinstance(edges_raw, list):
            raise ValueError("invalid graph payload")
        graph = cls(GameVersion.model_validate(version_raw))
        for raw in nodes_raw:
            graph.add_node(KnowledgeNode.model_validate(raw))
        for raw in edges_raw:
            graph.add_edge(KnowledgeEdge.model_validate(raw))
        return graph
