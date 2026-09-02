from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .model import EdgeKind, KnowledgeEdge, KnowledgeGraph, NodeKind


@dataclass(frozen=True)
class RequirementChoice:
    """One required slot: the planner may satisfy any listed alternative."""

    alternatives: tuple[str, ...]
    quantity: float = 1.0
    source_edge_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class AcquisitionMethod:
    """One way to obtain a target and its AND-list of prerequisite choices."""

    target: str
    process: str
    output_quantity: float
    output_kind: EdgeKind
    requirements: tuple[RequirementChoice, ...]
    provenance_source: str


def tag_members(
    graph: KnowledgeGraph,
    tag_node_id: str,
    *,
    recursive: bool = True,
    max_depth: int = 32,
) -> tuple[str, ...]:
    """Resolve an exact-version tag to concrete leaf members."""
    node = graph.nodes.get(tag_node_id)
    if node is None:
        raise KeyError(tag_node_id)
    if node.kind != NodeKind.TAG:
        return (tag_node_id,)

    leaves: set[str] = set()
    visited: set[str] = set()
    stack: list[tuple[str, int]] = [(tag_node_id, 0)]
    while stack:
        current, depth = stack.pop()
        if current in visited or depth > max_depth:
            continue
        visited.add(current)
        incoming = graph.incoming(current, kinds={EdgeKind.MEMBER_OF})
        for edge in incoming:
            child = graph.nodes[edge.source]
            if recursive and child.kind == NodeKind.TAG:
                stack.append((child.node_id, depth + 1))
            else:
                leaves.add(child.node_id)
    return tuple(sorted(leaves))


def acquisition_methods(graph: KnowledgeGraph, target: str) -> tuple[AcquisitionMethod, ...]:
    """Return exact-version executable acquisition choices for a target node.

    Each method is an OR branch. Within a method, every RequirementChoice must
    be satisfied (AND). A RequirementChoice may contain multiple alternatives
    (OR), including all concrete members of a required item/block tag.
    """
    if target not in graph.nodes:
        raise KeyError(target)

    methods: list[AcquisitionMethod] = []
    for output in graph.ways_to_obtain(target):
        process = output.source
        raw_requirements = graph.requirements(process)
        grouped = _group_requirement_edges(raw_requirements)
        choices: list[RequirementChoice] = []
        for edges in grouped:
            alternatives: set[str] = set()
            quantity = 1.0
            source_ids: list[str] = []
            for edge in edges:
                source_ids.append(edge.edge_id)
                quantity = max(quantity, edge.quantity)
                target_node = graph.nodes[edge.target]
                if target_node.kind == NodeKind.TAG:
                    members = tag_members(graph, edge.target)
                    if members:
                        alternatives.update(members)
                    else:
                        alternatives.add(edge.target)
                else:
                    alternatives.add(edge.target)
            choices.append(
                RequirementChoice(
                    alternatives=tuple(sorted(alternatives)),
                    quantity=quantity,
                    source_edge_ids=tuple(source_ids),
                )
            )
        methods.append(
            AcquisitionMethod(
                target=target,
                process=process,
                output_quantity=output.quantity,
                output_kind=output.kind,
                requirements=tuple(choices),
                provenance_source=output.provenance.source_id,
            )
        )
    return tuple(methods)


def _group_requirement_edges(edges: list[KnowledgeEdge]) -> tuple[tuple[KnowledgeEdge, ...], ...]:
    """Preserve separate AND slots while grouping explicit OR alternatives."""
    groups: dict[str, list[KnowledgeEdge]] = defaultdict(list)
    order: list[str] = []
    for edge in edges:
        key = edge.alternative_group or edge.edge_id
        if key not in groups:
            order.append(key)
        groups[key].append(edge)
    return tuple(tuple(groups[key]) for key in order)
