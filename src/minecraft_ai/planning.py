from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from .knowledge import KnowledgeGraph
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
    """Deterministic dependency expander used below generative cognition.

    It does not invent game facts. The knowledge graph determines prerequisite
    structure; higher cognition may choose among valid alternatives or supply
    custom project decomposition.
    """

    graph: KnowledgeGraph
    skills: SkillLibrary

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

    def make_dependency_plan(self, goal: Goal) -> Plan:
        plan = Plan(goal=goal)
        if goal.target_node is None:
            return plan
        closure = sorted(self.dependency_nodes(goal.target_node))
        previous: str | None = None
        for index, node_id in enumerate(closure):
            step_id = f"dep-{index:04d}"
            node = self.graph.nodes[node_id]
            step = PlanStep(
                step_id=step_id,
                goal_id=goal.goal_id,
                description=f"Satisfy prerequisite: {node.name}",
                target_node=node_id,
                prerequisites=() if previous is None else (previous,),
            )
            plan.add(step)
            previous = step_id
        final_id = "target"
        plan.add(
            PlanStep(
                step_id=final_id,
                goal_id=goal.goal_id,
                description=f"Achieve target {goal.target_node}",
                target_node=goal.target_node,
                prerequisites=() if previous is None else (previous,),
            )
        )
        return plan
