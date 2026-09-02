from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .skills import (
    SkillCondition,
    SkillLibrary,
    SkillOutcome,
    SkillRun,
    SkillSpec,
    SkillStage,
    SkillStats,
)
from .storage import StateDatabase


@dataclass
class SkillEditor:
    """Continual skill growth, self-editing, and dynamic promotion pipeline."""

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
        spec = SkillSpec(
            skill_id=skill_id,
            version=1,
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
                    current.recovery_skills
                    if recovery_skills is None
                    else tuple(recovery_skills)
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

    def record_and_evaluate(self, run: SkillRun) -> tuple[SkillStats, SkillSpec | None]:
        stats = self.library.record(run)
        if self.db is not None:
            self.db.save_skill_stats(run.skill_id, run.context_key, stats)

        promoted_spec = self._try_auto_promote(run.skill_id)
        return stats, promoted_spec

    def _try_auto_promote(self, skill_id: str) -> SkillSpec | None:
        spec = self.library.get(skill_id)
        if spec.stage in {SkillStage.PREFERRED, SkillStage.RETIRED}:
            return None

        # Aggregate stats across context keys
        total_successes = sum(
            st.successes for key, st in self.library.stats.items() if key[0] == skill_id
        )
        total_failures = sum(
            st.failures + st.timeouts for key, st in self.library.stats.items() if key[0] == skill_id
        )
        attempts = total_successes + total_failures

        if attempts < 3:
            return None

        success_rate = total_successes / attempts

        next_stage: SkillStage | None = None
        if spec.stage == SkillStage.CANDIDATE and success_rate >= 0.6:
            next_stage = SkillStage.EXPERIMENTAL
        elif spec.stage == SkillStage.EXPERIMENTAL and attempts >= 5 and success_rate >= 0.75:
            next_stage = SkillStage.TRUSTED
        elif spec.stage == SkillStage.TRUSTED and attempts >= 10 and success_rate >= 0.85:
            next_stage = SkillStage.PREFERRED
        elif success_rate < 0.2 and attempts >= 5:
            next_stage = SkillStage.DEPRECATED

        if next_stage is not None and next_stage != spec.stage:
            promoted = self.library.promote(skill_id, next_stage)
            if self.db is not None:
                self.db.save_skill(promoted)
            return promoted

        return None

    def evaluate_and_promote(self, skill_id: str) -> SkillSpec | None:
        """Public endpoint to evaluate stats and auto-promote skill stage."""
        return self._try_auto_promote(skill_id)

    def synthesize_recovery_variant(
        self,
        skill_id: str | SkillSpec,
        failure_reason: str = "consecutive_runtime_failures",
        reason: str | None = None,
        fallback_skills: tuple[str, ...] = (),
    ) -> SkillSpec:
        """Create a self-healed, adapted skill variant with additional failure recovery logic."""
        target_id = skill_id.skill_id if isinstance(skill_id, SkillSpec) else skill_id
        effective_reason = reason or failure_reason
        parent = self.library.get(target_id)
        new_id = f"{target_id}_adapted"
        new_name = f"{parent.name} (Adapted)"

        # Append extra failure condition based on reason
        extra_condition = SkillCondition(key=effective_reason.split(":")[-1], operator="truthy")
        new_failures = parent.failure_conditions + (extra_condition,)

        recovery = tuple(fallback_skills) if fallback_skills else (parent.recovery_skills or ("retreat",))
        return self.create_skill(
            skill_id=new_id,
            name=new_name,
            description=f"Self-healed variant of {target_id} adapted for {effective_reason}",
            policy_ref=parent.policy_ref,
            parameters=parent.parameters,
            preconditions=parent.preconditions,
            success_conditions=parent.success_conditions,
            failure_conditions=new_failures,
            recovery_skills=recovery,
            max_duration_ms=parent.max_duration_ms,
            stage=SkillStage.EXPERIMENTAL,
        )
