from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from .knowledge import KnowledgeGraph
from .knowledge.queries import AcquisitionMethod, acquisition_methods
from .roles import RoleProfile
from .skills import SkillLibrary, SkillStage


class GoalSource(StrEnum):
    SURVIVAL = "survival"
    PROGRESSION = "progression"
    ROLE = "role"
    PLAYER = "player"
    CUSTOM = "custom"
    OPPORTUNITY = "opportunity"


class Goal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    goal_id: str
    description: str
    target_node: str | None = None
    source: GoalSource = GoalSource.CUSTOM
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    domain: str = "general"
    deadline_ns: int | None = None
    parent_goal_id: str | None = None


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    step_id: str
    goal_id: str
    description: str
    target_node: str | None = None
    skill_id: str | None = None
    prerequisites: tuple[str, ...] = ()
    estimated_cost: float = Field(default=1.0, ge=0.0)
    risk: float = Field(default=0.0, ge=0.0, le=1.0)


class AcquisitionPlan(BaseModel):
    """One concrete OR branch for obtaining a target.

    `requirements` is an AND-list. Each nested tuple is an OR-list of valid
    alternatives for one required slot.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    target: str
    quantity: float = Field(default=1.0, gt=0.0)
    method: str | None = None
    output_quantity: float = Field(default=1.0, gt=0.0)
    requirements: tuple[tuple[AcquisitionPlan, ...], ...] = ()
    already_available: bool = False
    unresolved: bool = False
    cyclic: bool = False
    estimated_leaf_cost: float = Field(default=0.0, ge=0.0)


@dataclass
class Plan:
    goal: Goal
    steps: dict[str, PlanStep] = field(default_factory=dict)

    def add(self, step: PlanStep) -> None:
        if step.step_id in self.steps:
            raise ValueError(f"duplicate plan step: {step.step_id}")
        self.steps[step.step_id] = step

    def ready(self, completed: set[str]) -> list[PlanStep]:
        return [
            step
            for step in self.steps.values()
            if step.step_id not in completed and set(step.prerequisites).issubset(completed)
        ]

    def validate(self) -> list[str]:
        errors: list[str] = []
        ids = set(self.steps)
        for step in self.steps.values():
            missing = set(step.prerequisites) - ids
            if missing:
                errors.append(f"{step.step_id} missing prerequisites {sorted(missing)}")
        return errors


@dataclass
class GoalScorer:
    role: RoleProfile

    def score(self, goal: Goal, *, opportunity: float = 0.0, urgency: float = 0.0) -> float:
        role_weight = self.role.weight(goal.domain, 0.5)
        risk_modifier = 0.5 + self.role.risk_tolerance * 0.5
        return goal.priority * (0.5 + role_weight) + opportunity * 0.25 + urgency * risk_modifier


@dataclass
class DependencyPlanner:
    """Deterministic dependency expander used below generative cognition."""

    graph: KnowledgeGraph
    skills: SkillLibrary = field(default_factory=SkillLibrary)

    def dependency_nodes(self, target_node: str) -> set[str]:
        if target_node not in self.graph.nodes:
            raise KeyError(target_node)
        return self.graph.prerequisite_closure(target_node)

    def bind_skill(self, capability: str) -> str | None:
        candidates = [
            spec
            for spec in self.skills.specs.values()
            if capability in spec.expected_effects
            and spec.stage not in {SkillStage.DEPRECATED, SkillStage.RETIRED}
        ]
        if not candidates:
            return None
        candidates.sort(
            key=lambda spec: (
                self.skills.contextual_score(spec.skill_id),
                spec.stage in {SkillStage.TRUSTED, SkillStage.PREFERRED},
                spec.version,
            ),
            reverse=True,
        )
        return candidates[0].skill_id

    def acquisition_options(
        self,
        target_node: str,
        *,
        quantity: float = 1.0,
        available: set[str] | None = None,
        max_depth: int = 24,
    ) -> tuple[AcquisitionPlan, ...]:
        """Expand an item/process target into exact-version AND/OR alternatives."""
        if target_node not in self.graph.nodes:
            raise KeyError(target_node)
        return self._expand(
            target_node,
            quantity=max(1.0, quantity),
            available=available or set(),
            path=frozenset(),
            depth=0,
            max_depth=max_depth,
        )

    def best_acquisition_plan(
        self,
        target_node: str,
        *,
        quantity: float = 1.0,
        available: set[str] | None = None,
        max_depth: int = 24,
    ) -> AcquisitionPlan:
        options = self.acquisition_options(
            target_node,
            quantity=quantity,
            available=available,
            max_depth=max_depth,
        )
        if not options:
            return AcquisitionPlan(target=target_node, quantity=quantity, unresolved=True)
        return min(options, key=_plan_sort_key)

    def _expand(
        self,
        target: str,
        *,
        quantity: float,
        available: set[str],
        path: frozenset[str],
        depth: int,
        max_depth: int,
    ) -> tuple[AcquisitionPlan, ...]:
        if target in available:
            return (AcquisitionPlan(target=target, quantity=quantity, already_available=True),)
        if target in path:
            return (AcquisitionPlan(target=target, quantity=quantity, cyclic=True),)
        if depth >= max_depth:
            return (AcquisitionPlan(target=target, quantity=quantity, unresolved=True),)
        methods = acquisition_methods(self.graph, target)
        if not methods:
            return (
                AcquisitionPlan(
                    target=target,
                    quantity=quantity,
                    unresolved=True,
                    estimated_leaf_cost=quantity,
                ),
            )
        expanded: list[AcquisitionPlan] = []
        next_path = path | {target}
        for method in methods:
            expanded.append(
                self._expand_method(
                    method,
                    requested_quantity=quantity,
                    available=available,
                    path=next_path,
                    depth=depth + 1,
                    max_depth=max_depth,
                )
            )
        expanded.sort(key=_plan_sort_key)
        return tuple(expanded)

    def _expand_method(
        self,
        method: AcquisitionMethod,
        *,
        requested_quantity: float,
        available: set[str],
        path: frozenset[str],
        depth: int,
        max_depth: int,
    ) -> AcquisitionPlan:
        batches = max(1.0, requested_quantity / max(1.0, method.output_quantity))
        slots: list[tuple[AcquisitionPlan, ...]] = []
        leaf_cost = 0.0
        for requirement in method.requirements:
            alternatives: list[AcquisitionPlan] = []
            required_quantity = requirement.quantity * batches
            for alternative in requirement.alternatives:
                child_options = self._expand(
                    alternative,
                    quantity=required_quantity,
                    available=available,
                    path=path,
                    depth=depth,
                    max_depth=max_depth,
                )
                alternatives.extend(child_options)
            alternatives.sort(key=_plan_sort_key)
            if alternatives:
                leaf_cost += alternatives[0].estimated_leaf_cost
                slots.append(tuple(alternatives))
            else:
                unresolved = AcquisitionPlan(
                    target=f"unresolved:{method.process}",
                    quantity=required_quantity,
                    unresolved=True,
                    estimated_leaf_cost=required_quantity,
                )
                slots.append((unresolved,))
                leaf_cost += required_quantity
        return AcquisitionPlan(
            target=method.target,
            quantity=requested_quantity,
            method=method.process,
            output_quantity=method.output_quantity,
            requirements=tuple(slots),
            estimated_leaf_cost=leaf_cost,
        )

    def make_dependency_plan(self, goal: Goal) -> Plan:
        """Flatten the currently cheapest acquisition tree for the executor."""
        plan = Plan(goal=goal)
        if goal.target_node is None:
            return plan
        tree = self.best_acquisition_plan(goal.target_node)
        _flatten_acquisition_tree(tree, goal.goal_id, plan, parent_step=None, counter=[0])
        return plan


def _plan_sort_key(plan: AcquisitionPlan) -> tuple[bool, bool, bool, float, int]:
    return (
        plan.cyclic,
        plan.unresolved,
        not plan.already_available,
        plan.estimated_leaf_cost,
        len(plan.requirements),
    )


def _flatten_acquisition_tree(
    tree: AcquisitionPlan,
    goal_id: str,
    plan: Plan,
    *,
    parent_step: str | None,
    counter: list[int],
) -> str:
    child_ids: list[str] = []
    for slot in tree.requirements:
        if not slot:
            continue
        best = min(slot, key=_plan_sort_key)
        child_ids.append(
            _flatten_acquisition_tree(
                best,
                goal_id,
                plan,
                parent_step=None,
                counter=counter,
            )
        )
    step_id = f"step-{counter[0]:04d}"
    counter[0] += 1
    method_text = f" via {tree.method}" if tree.method is not None else ""
    if tree.already_available:
        description = f"Use available {tree.target} x{tree.quantity:g}"
    elif tree.unresolved:
        description = f"Acquire external/raw {tree.target} x{tree.quantity:g}"
    else:
        description = f"Obtain {tree.target} x{tree.quantity:g}{method_text}"
    prerequisites = tuple(child_ids)
    if parent_step is not None:
        prerequisites = tuple(dict.fromkeys((*prerequisites, parent_step)))
    plan.add(
        PlanStep(
            step_id=step_id,
            goal_id=goal_id,
            description=description,
            target_node=tree.target,
            prerequisites=prerequisites,
            estimated_cost=tree.estimated_leaf_cost,
        )
    )
    return step_id
