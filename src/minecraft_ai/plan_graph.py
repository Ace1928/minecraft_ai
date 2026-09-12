"""Persistent typed plan graph over the existing acquisition/plan structures.

Cognition still speaks short step labels on the wire. This module turns those
labels into nodes with identity, alternatives, state and local repair so a
failed method does not merely increment a flattened string index.
"""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from .skills import SkillLibrary, SkillStage


_SENTINEL_PLAN_STEPS = frozenset(
    {
        "null",
        "none",
        "n/a",
        "na",
        "nil",
        "undefined",
        "unknown",
        "noop",
        "no-op",
        "no op",
        "-",
        "--",
        ".",
        "...",
    }
)


class PlanNodeState(StrEnum):
    READY = "ready"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    BLOCKED = "blocked"
    FAILED = "failed"
    SUSPENDED = "suspended"


class PlanNode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str
    objective: str = Field(min_length=1, max_length=120)
    skill_id: str | None = None
    parent_id: str | None = None
    alternative_ids: tuple[str, ...] = ()
    prerequisite_ids: tuple[str, ...] = ()
    state: PlanNodeState = PlanNodeState.READY
    attempts: int = Field(default=0, ge=0)
    consecutive_failures: int = Field(default=0, ge=0)
    block_reason: str | None = None
    estimated_cost: float = Field(default=1.0, ge=0.0)
    success_posterior: float | None = Field(default=None, ge=0.0, le=1.0)


class PlanGraph(BaseModel):
    """AND/OR-capable runtime plan. Sequential labels remain a projection."""

    model_config = ConfigDict(extra="forbid")

    goal_id: str | None = None
    nodes: dict[str, PlanNode] = Field(default_factory=dict)
    order: tuple[str, ...] = ()
    cursor: int = 0

    def sequential_labels(self) -> tuple[str, ...]:
        return tuple(self.nodes[node_id].objective for node_id in self.order)

    def current(self) -> PlanNode | None:
        if not (0 <= self.cursor < len(self.order)):
            return None
        return self.nodes[self.order[self.cursor]]

    def remaining_labels(self) -> tuple[str, ...]:
        return tuple(
            self.nodes[node_id].objective for node_id in self.order[self.cursor :]
        )

    def mark_running(self, skill_id: str) -> None:
        node = self.current()
        if node is None or node.skill_id != skill_id:
            return
        self.nodes[node.node_id] = node.model_copy(
            update={"state": PlanNodeState.RUNNING}
        )

    def mark_succeeded(self, skill_id: str, *, allow_unbound: bool = False) -> bool:
        """Complete a matching node, or caller-qualified legacy free-form prose."""
        node = self.current()
        if node is None or not (
            _same_skill(node, skill_id) or (allow_unbound and node.skill_id is None)
        ):
            return False
        self.nodes[node.node_id] = node.model_copy(
            update={
                "state": PlanNodeState.SUCCEEDED,
                "consecutive_failures": 0,
                "block_reason": None,
            }
        )
        self.cursor += 1
        return True

    def block_current_method(
        self,
        skill_id: str,
        *,
        reason: str,
        skills: SkillLibrary | None = None,
    ) -> bool:
        """Mark this acquisition method blocked and expose a sibling if any."""

        node = self.current()
        if node is None or not _same_skill(node, skill_id):
            return False
        attempts = node.attempts + 1
        blocked = node.model_copy(
            update={
                "state": PlanNodeState.BLOCKED,
                "attempts": attempts,
                "consecutive_failures": node.consecutive_failures + 1,
                "block_reason": reason,
            }
        )
        self.nodes[node.node_id] = blocked
        sibling = self._next_alternative(blocked, skills)
        if sibling is None:
            self.cursor += 1
            return True
        replacement = sibling.model_copy(
            update={
                "state": PlanNodeState.READY,
                "parent_id": node.parent_id or node.node_id,
            }
        )
        self.nodes[replacement.node_id] = replacement
        order = list(self.order)
        order[self.cursor] = replacement.node_id
        self.order = tuple(order)
        return True

    def _next_alternative(
        self,
        node: PlanNode,
        skills: SkillLibrary | None,
    ) -> PlanNode | None:
        for alt_id in node.alternative_ids:
            candidate = self.nodes.get(alt_id)
            if candidate is None or candidate.state in {
                PlanNodeState.BLOCKED,
                PlanNodeState.FAILED,
                PlanNodeState.SUCCEEDED,
            }:
                continue
            if candidate.node_id == node.node_id:
                continue
            return candidate
        if skills is None or node.skill_id is None:
            return None
        used = {
            self.nodes[node_id].skill_id
            for node_id in self.nodes
            if self.nodes[node_id].skill_id is not None
            and self.nodes[node_id].state in {
                PlanNodeState.BLOCKED,
                PlanNodeState.FAILED,
            }
        }
        used.add(node.skill_id)
        for skill_id in alternative_skill_ids(skills, node.skill_id):
            if skill_id in used:
                continue
            sibling_id = f"{node.node_id}:alt:{skill_id}"
            return PlanNode(
                node_id=sibling_id,
                objective=skill_id,
                skill_id=skill_id,
                parent_id=node.node_id,
                alternative_ids=node.alternative_ids,
                state=PlanNodeState.READY,
                estimated_cost=node.estimated_cost,
            )
        return None


def sanitize_plan_steps(steps: tuple[str, ...]) -> tuple[str, ...]:
    """Drop sentinel/empty labels and collapse consecutive duplicate nodes."""

    cleaned: list[str] = []
    for raw in steps:
        text = " ".join(str(raw).split())
        if not text:
            continue
        normalized = _normalized_plan_text(text)
        if not normalized or normalized in _SENTINEL_PLAN_STEPS:
            continue
        if cleaned and _normalized_plan_text(cleaned[-1]) == normalized:
            continue
        cleaned.append(text[:120])
        if len(cleaned) >= 5:
            break
    return tuple(cleaned)


def plan_graph_from_steps(
    steps: tuple[str, ...],
    *,
    goal_id: str | None,
    skills: SkillLibrary | None = None,
) -> PlanGraph:
    labels = sanitize_plan_steps(steps)
    nodes: dict[str, PlanNode] = {}
    order: list[str] = []
    for index, label in enumerate(labels):
        node_id = f"n{index:02d}"
        skill_id = bind_plan_step_skill(label, skills)
        alternatives: tuple[str, ...] = ()
        if skills is not None and skill_id is not None:
            alt_skills = alternative_skill_ids(skills, skill_id)
            alternatives = tuple(f"{node_id}:alt:{alt}" for alt in alt_skills)
            for alt_id, alt_skill in zip(alternatives, alt_skills, strict=True):
                nodes[alt_id] = PlanNode(
                    node_id=alt_id,
                    objective=alt_skill,
                    skill_id=alt_skill,
                    parent_id=node_id,
                    state=PlanNodeState.READY,
                )
        nodes[node_id] = PlanNode(
            node_id=node_id,
            objective=label,
            skill_id=skill_id,
            alternative_ids=alternatives,
            state=PlanNodeState.READY,
        )
        order.append(node_id)
    return PlanGraph(goal_id=goal_id, nodes=nodes, order=tuple(order), cursor=0)


def bind_plan_step_skill(step: str, skills: SkillLibrary | None) -> str | None:
    if skills is None:
        normalized = _normalized_plan_text(step)
        return None if not normalized else normalized.replace(" ", "_")
    for skill_id in skills.specs:
        if _normalized_plan_text(skill_id) == _normalized_plan_text(step):
            return skill_id
    return None


def alternative_skill_ids(skills: SkillLibrary, skill_id: str) -> tuple[str, ...]:
    if skill_id not in skills.specs:
        return ()
    spec = skills.get(skill_id)
    ranked: list[tuple[int, str]] = []
    effects = set(spec.expected_effects)
    for other in skills.specs.values():
        if other.skill_id == skill_id:
            continue
        if other.stage in {SkillStage.DEPRECATED, SkillStage.RETIRED}:
            continue
        if effects and effects & set(other.expected_effects):
            ranked.append((0, other.skill_id))
        elif other.skill_id in spec.recovery_skills:
            ranked.append((1, other.skill_id))
    seen: list[str] = []
    for _, candidate in sorted(ranked):
        if candidate not in seen:
            seen.append(candidate)
        if len(seen) >= 4:
            break
    return tuple(seen)


def _normalized_plan_text(value: str) -> str:
    return " ".join(value.casefold().replace("_", " ").replace("-", " ").split())


def _same_skill(node: PlanNode, skill_id: str) -> bool:
    if node.skill_id == skill_id:
        return True
    return _normalized_plan_text(node.objective) == _normalized_plan_text(skill_id)


def progression_skill_for_capabilities(
    inventory: dict[str, int],
    *,
    available_skill_ids: set[str] | frozenset[str],
    recently_failed_skill_ids: set[str] | frozenset[str] = frozenset(),
) -> str | None:
    """Next capability-backed skill from inventory evidence, not a static tree.

    Exact mechanics still decide feasibility. This only ranks the unfinished
    wood-age capability that observed counts have not yet satisfied.
    """

    logs = _inventory_count(inventory, "oak_log", "minecraft:oak_log")
    planks = _inventory_count(inventory, "oak_planks", "minecraft:oak_planks")
    table = _inventory_count(inventory, "crafting_table", "minecraft:crafting_table")
    cobble = _inventory_count(inventory, "cobblestone", "minecraft:cobblestone")
    if logs < 3:
        candidate = "gather_nearby_wood"
    elif planks < 4:
        candidate = "craft_wood_planks"
    elif table < 1:
        candidate = "craft_crafting_table"
    elif cobble < 1:
        candidate = "mine_visible_block"
    else:
        return None
    if candidate not in available_skill_ids or candidate in recently_failed_skill_ids:
        return None
    return candidate


def _inventory_count(inventory: dict[str, int], *keys: str) -> int:
    total = 0
    for key in keys:
        value = inventory.get(key, 0)
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        total += max(0, value)
    return total
