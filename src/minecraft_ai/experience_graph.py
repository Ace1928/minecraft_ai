"""Contextual experience beliefs that never overwrite exact game mechanics.

The knowledge graph answers whether Minecraft permits a method. This graph
answers how often that method has worked here, for this body and context.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .skills import SkillLibrary, SkillOutcome, SkillRun, SkillSpec


class ExperienceBelief(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill_id: str
    context_key: str
    effect: str
    successes: int = Field(default=0, ge=0)
    failures: int = Field(default=0, ge=0)
    last_failure_reason: str | None = None

    @property
    def attempts(self) -> int:
        return self.successes + self.failures


class ExperienceGraph:
    """Action-effect and failure beliefs keyed by skill and context."""

    def __init__(self) -> None:
        self.beliefs: dict[tuple[str, str, str], ExperienceBelief] = {}

    def observe(self, run: SkillRun, spec: SkillSpec | None) -> None:
        if run.outcome == SkillOutcome.RUNNING:
            return
        effects: tuple[str, ...] = ("outcome",)
        if spec is not None and spec.expected_effects:
            effects = spec.expected_effects
        succeeded = run.outcome == SkillOutcome.SUCCEEDED
        for effect in effects:
            key = (run.skill_id, run.context_key, effect)
            belief = self.beliefs.setdefault(
                key,
                ExperienceBelief(
                    skill_id=run.skill_id,
                    context_key=run.context_key,
                    effect=effect,
                ),
            )
            if succeeded:
                belief.successes += 1
                belief.last_failure_reason = None
            elif run.outcome in {SkillOutcome.FAILED, SkillOutcome.TIMED_OUT}:
                belief.failures += 1
                belief.last_failure_reason = run.failure_reason

    def success_probability(
        self,
        skill_id: str,
        context_key: str,
        library: SkillLibrary,
        *,
        effect: str = "outcome",
    ) -> float:
        """Local effect belief shrunk toward the skill's global competence."""

        prior = library.hierarchical_success_probability(skill_id, context_key)
        belief = self.beliefs.get((skill_id, context_key, effect))
        if belief is None or belief.attempts == 0:
            return prior
        strength = 4.0
        return (belief.successes + strength * prior) / (belief.attempts + strength)
