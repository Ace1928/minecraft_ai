from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field

from .planning import Goal, GoalScorer, GoalSource
from .roles import RoleProfile


class ProgressState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    achievements: frozenset[str] = frozenset()
    capabilities: frozenset[str] = frozenset()
    dimensions: frozenset[str] = frozenset()
    trusted_skills: frozenset[str] = frozenset()
    completed_projects: frozenset[str] = frozenset()
    technology_tiers: dict[str, int] = Field(default_factory=dict)


class CurriculumCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    goal: Goal
    prerequisite_value: float = Field(default=0.0, ge=0.0, le=1.0)
    progression_novelty: float = Field(default=0.0, ge=0.0, le=1.0)
    skill_learning_value: float = Field(default=0.0, ge=0.0, le=1.0)
    opportunity: float = Field(default=0.0, ge=0.0, le=1.0)
    urgency: float = Field(default=0.0, ge=0.0, le=1.0)
    estimated_risk: float = Field(default=0.0, ge=0.0, le=1.0)


@dataclass
class CurriculumScheduler:
    role: RoleProfile
    scorer: GoalScorer = field(init=False)

    def __post_init__(self) -> None:
        self.scorer = GoalScorer(self.role)

    def score(self, candidate: CurriculumCandidate) -> float:
        base = self.scorer.score(
            candidate.goal,
            opportunity=candidate.opportunity,
            urgency=candidate.urgency,
        )
        learning = candidate.skill_learning_value * 0.35
        progression = candidate.progression_novelty * 0.45
        prerequisite = candidate.prerequisite_value * 0.4
        risk_penalty = candidate.estimated_risk * (1.0 - self.role.risk_tolerance) * 0.8
        player_bonus = 0.6 if candidate.goal.source == GoalSource.PLAYER else 0.0
        return base + learning + progression + prerequisite + player_bonus - risk_penalty

    def choose(self, candidates: list[CurriculumCandidate]) -> CurriculumCandidate | None:
        if not candidates:
            return None
        return max(candidates, key=self.score)


def role_standing_goals(role: RoleProfile) -> list[Goal]:
    goals: list[Goal] = []
    for index, description in enumerate(role.standing_goals):
        domain = description.split("_", 1)[0]
        goals.append(
            Goal(
                goal_id=f"role:{role.role_id}:{index}:{description}",
                description=description.replace("_", " "),
                source=GoalSource.ROLE,
                priority=0.55,
                domain=domain,
            )
        )
    return goals
