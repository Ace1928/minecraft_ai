from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class SkillStage(StrEnum):
    CANDIDATE = "candidate"
    EXPERIMENTAL = "experimental"
    TRUSTED = "trusted"
    PREFERRED = "preferred"
    DEPRECATED = "deprecated"
    RETIRED = "retired"


class SkillCondition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str
    operator: str = Field(pattern="^(eq|neq|gte|lte|gt|lt|truthy|falsy|exists)$")
    value: str | int | float | bool | None = None
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class SkillActionPermissions(BaseModel):
    """Actions a learned option may emit while satisfying a skill contract.

    These permissions do not choose movement or synthesize a macro. They form a
    fail-closed boundary around the learned controller so, for example, a
    retreat option cannot turn an ambiguous visual observation into mining.
    Runtime/operator constraints may further remove permissions, but can never
    grant an action forbidden by the skill contract.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    allow_attack: bool = True
    allow_use: bool = True
    allow_jump: bool = True
    allow_drop: bool = True
    allow_inventory: bool = True
    allow_hotbar: bool = True


class SkillSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    skill_id: str = Field(min_length=1, max_length=256)
    version: int = Field(default=1, ge=1)
    name: str = Field(min_length=1, max_length=256)
    description: str = ""
    stage: SkillStage = SkillStage.CANDIDATE
    parameters: tuple[str, ...] = ()
    preconditions: tuple[SkillCondition, ...] = ()
    initiation_alternatives: tuple[tuple[SkillCondition, ...], ...] = ()
    invariants: tuple[SkillCondition, ...] = ()
    success_conditions: tuple[SkillCondition, ...] = ()
    failure_conditions: tuple[SkillCondition, ...] = ()
    expected_effects: tuple[str, ...] = ()
    recovery_skills: tuple[str, ...] = ()
    max_duration_ms: int = Field(default=30_000, ge=50, le=3_600_000)
    policy_ref: str | None = None
    policy_instruction: str | None = Field(default=None, min_length=1, max_length=256)
    policy_condition_scale: float | None = Field(default=None, ge=0.0, le=12.0)
    action_permissions: SkillActionPermissions = Field(default_factory=SkillActionPermissions)
    compatible_editions: tuple[str, ...] = ("bedrock", "java")
    compatible_versions: tuple[str, ...] = ()


class SkillOutcome(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


class SkillRun(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    skill_id: str
    started_ns: int
    ended_ns: int | None = None
    outcome: SkillOutcome = SkillOutcome.RUNNING
    context_key: str = "default"
    parameters: dict[str, str | int | float | bool] = Field(default_factory=dict)
    failure_reason: str | None = None


@dataclass
class SkillStats:
    successes: int = 0
    failures: int = 0
    timeouts: int = 0
    cancellations: int = 0
    consecutive_failures: int = 0

    @property
    def attempts(self) -> int:
        return self.successes + self.failures + self.timeouts + self.cancellations

    @property
    def success_rate(self) -> float:
        decisive = self.successes + self.failures + self.timeouts
        return self.successes / decisive if decisive else 0.0


@dataclass
class SkillLibrary:
    specs: dict[str, SkillSpec] = field(default_factory=dict)
    stats: dict[tuple[str, str], SkillStats] = field(default_factory=dict)

    def register(self, spec: SkillSpec) -> None:
        existing = self.specs.get(spec.skill_id)
        if existing is not None and spec.version <= existing.version:
            raise ValueError("skill updates must increment version")
        self.specs[spec.skill_id] = spec

    def get(self, skill_id: str) -> SkillSpec:
        try:
            return self.specs[skill_id]
        except KeyError as exc:
            raise KeyError(f"unknown skill: {skill_id}") from exc

    def record(self, run: SkillRun) -> SkillStats:
        if run.outcome == SkillOutcome.RUNNING:
            raise ValueError("cannot record an unfinished skill run")
        key = (run.skill_id, run.context_key)
        stats = self.stats.setdefault(key, SkillStats())
        if run.outcome == SkillOutcome.SUCCEEDED:
            stats.successes += 1
            stats.consecutive_failures = 0
        elif run.outcome == SkillOutcome.FAILED:
            stats.failures += 1
            stats.consecutive_failures += 1
        elif run.outcome == SkillOutcome.TIMED_OUT:
            stats.timeouts += 1
            stats.consecutive_failures += 1
        elif run.outcome == SkillOutcome.CANCELLED:
            stats.cancellations += 1
        return stats

    def contextual_score(self, skill_id: str, context_key: str = "default") -> float:
        stats = self.stats.get((skill_id, context_key))
        if stats is None:
            return 0.0
        # Beta(1,1) prior prevents one lucky success from becoming absolute confidence.
        return (stats.successes + 1.0) / (stats.successes + stats.failures + stats.timeouts + 2.0)

    def promote(self, skill_id: str, stage: SkillStage) -> SkillSpec:
        current = self.get(skill_id)
        order = {
            SkillStage.CANDIDATE: 0,
            SkillStage.EXPERIMENTAL: 1,
            SkillStage.TRUSTED: 2,
            SkillStage.PREFERRED: 3,
            SkillStage.DEPRECATED: 4,
            SkillStage.RETIRED: 5,
        }
        if order[stage] < order[current.stage] and stage not in {
            SkillStage.DEPRECATED,
            SkillStage.RETIRED,
        }:
            raise ValueError("skill lifecycle cannot move backwards")
        updated = current.model_copy(update={"stage": stage, "version": current.version + 1})
        self.specs[skill_id] = updated
        return updated
