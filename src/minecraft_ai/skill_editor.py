from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .skills import (
    SkillCondition,
    SkillLibrary,
    SkillRun,
    SkillSpec,
    SkillStage,
    SkillStats,
)
from .storage import StateDatabase


@dataclass
class SkillPromotionEvidence:
    benchmark_run_id: str
    sample_count: int
    successes: int
    context_count: int
    critical_safety_failures: int = 0
    protected_regression: bool = False

    @property
    def success_rate(self) -> float:
        return self.successes / self.sample_count if self.sample_count else 0.0


@dataclass
class SkillLifecycleManager:
    """Offline skill authoring and evidence-gated lifecycle transitions."""

    library: SkillLibrary
    db: StateDatabase | None = None

    def create_skill(
        self,
        *,
        skill_id: str,
        name: str,
        description: str = "",
        policy_ref: str | None = None,
        parameters: Sequence[str] = (),
        preconditions: Sequence[SkillCondition] = (),
        success_conditions: Sequence[SkillCondition] = (),
        failure_conditions: Sequence[SkillCondition] = (),
        recovery_skills: Sequence[str] = (),
        max_duration_ms: int = 30_000,
        stage: SkillStage = SkillStage.CANDIDATE,
    ) -> SkillSpec:
        existing = self.library.specs.get(skill_id)
        version = (existing.version + 1) if existing is not None else 1
        spec = SkillSpec(
            skill_id=skill_id,
            version=version,
            name=name,
            description=description,
            stage=stage,
            parameters=tuple(parameters),
            preconditions=tuple(preconditions),
            success_conditions=tuple(success_conditions),
            failure_conditions=tuple(failure_conditions),
            recovery_skills=tuple(recovery_skills),
            max_duration_ms=max_duration_ms,
            policy_ref=policy_ref or skill_id,
        )
        self.library.register(spec)
        if self.db is not None:
            self.db.save_skill(spec)
        return spec

    def edit_skill(
        self,
        skill_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        policy_ref: str | None = None,
        preconditions: Sequence[SkillCondition] | None = None,
        success_conditions: Sequence[SkillCondition] | None = None,
        failure_conditions: Sequence[SkillCondition] | None = None,
        recovery_skills: Sequence[str] | None = None,
        max_duration_ms: int | None = None,
    ) -> SkillSpec:
        current = self.library.get(skill_id)
        updated = current.model_copy(
            update={
                "version": current.version + 1,
                "name": current.name if name is None else name,
                "description": current.description if description is None else description,
                "policy_ref": current.policy_ref if policy_ref is None else policy_ref,
                "preconditions": (
                    current.preconditions if preconditions is None else tuple(preconditions)
                ),
                "success_conditions": (
                    current.success_conditions
                    if success_conditions is None
                    else tuple(success_conditions)
                ),
                "failure_conditions": (
                    current.failure_conditions
                    if failure_conditions is None
                    else tuple(failure_conditions)
                ),
                "recovery_skills": (
                    current.recovery_skills if recovery_skills is None else tuple(recovery_skills)
                ),
                "max_duration_ms": (
                    current.max_duration_ms if max_duration_ms is None else max_duration_ms
                ),
            }
        )
        self.library.register(updated)
        if self.db is not None:
            self.db.save_skill(updated)
        return updated

    def record(self, run: SkillRun) -> SkillStats:
        stats = self.library.record(run)
        if self.db is not None:
            self.db.save_skill_stats(run.skill_id, run.context_key, stats)
        return stats

    def evaluate_and_promote(
        self,
        skill_id: str,
        evidence: SkillPromotionEvidence,
    ) -> SkillSpec | None:
        """Promote only from an explicit held-out benchmark evidence record."""
        spec = self.library.get(skill_id)
        if spec.stage in {SkillStage.PREFERRED, SkillStage.RETIRED}:
            return None
        if evidence.critical_safety_failures or evidence.protected_regression:
            return None
        requirements = {
            SkillStage.CANDIDATE: (SkillStage.EXPERIMENTAL, 20, 3, 0.70),
            SkillStage.EXPERIMENTAL: (SkillStage.TRUSTED, 50, 5, 0.80),
            SkillStage.TRUSTED: (SkillStage.PREFERRED, 100, 8, 0.90),
        }
        requirement = requirements.get(spec.stage)
        if requirement is None:
            return None
        next_stage, minimum_samples, minimum_contexts, minimum_rate = requirement
        if (
            evidence.sample_count < minimum_samples
            or evidence.context_count < minimum_contexts
            or evidence.success_rate < minimum_rate
        ):
            return None
        promoted = self.library.promote(skill_id, next_stage)
        if self.db is not None:
            self.db.save_skill(promoted)
        return promoted

    def draft_recovery_candidate(
        self,
        skill_id: str | SkillSpec,
        failure_reason: str = "consecutive_runtime_failures",
        fallback_skills: tuple[str, ...] = (),
    ) -> SkillSpec:
        """Draft an unpromoted recovery candidate for later held-out evaluation."""
        target_id = skill_id.skill_id if isinstance(skill_id, SkillSpec) else skill_id
        parent = self.library.get(target_id)
        new_id = f"{target_id}_recovery_candidate"
        new_name = f"{parent.name} recovery candidate"

        extra_condition = SkillCondition(key=failure_reason.split(":")[-1], operator="truthy")
        new_failures = parent.failure_conditions + (extra_condition,)

        recovery = (
            tuple(fallback_skills) if fallback_skills else (parent.recovery_skills or ("retreat",))
        )
        return self.create_skill(
            skill_id=new_id,
            name=new_name,
            description=f"Unpromoted recovery candidate for {target_id}: {failure_reason}",
            policy_ref=parent.policy_ref,
            parameters=parent.parameters,
            preconditions=parent.preconditions,
            success_conditions=parent.success_conditions,
            failure_conditions=new_failures,
            recovery_skills=recovery,
            max_duration_ms=parent.max_duration_ms,
            stage=SkillStage.CANDIDATE,
        )
