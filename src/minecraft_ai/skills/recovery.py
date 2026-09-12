"""Outcome-conditioned recovery selection, separate from emergency handling."""

from __future__ import annotations

from minecraft_ai.execution import initiation_satisfied
from minecraft_ai.perception import PerceptionBlackboard
from minecraft_ai.skills.library import SkillLibrary, SkillSpec, SkillStage


def select_learned_recovery(
    skills: SkillLibrary,
    recovery_ids: tuple[str, ...],
    blackboard: PerceptionBlackboard,
    *,
    context_key: str,
    max_context_failures: int = 2,
) -> SkillSpec | None:
    """Rank currently feasible options using persisted contextual evidence.

    Declared order is only the cold-start/tie fallback. An exhausted recovery
    set returns control to cognition instead of recursively retrying a known
    ineffective option. This selector is NOT used to reorder emergency actions.
    """
    if max_context_failures < 1:
        raise ValueError("max_context_failures must be positive")
    candidates = []
    for recovery_id in dict.fromkeys(recovery_ids):
        candidate = skills.specs.get(recovery_id)
        if candidate is None or candidate.stage in {SkillStage.DEPRECATED, SkillStage.RETIRED}:
            continue
        if skills.contextual_failure_streak(recovery_id, context_key) >= max_context_failures:
            continue
        if initiation_satisfied(candidate, blackboard):
            candidates.append(candidate)
    return max(
        candidates,
        key=lambda candidate: skills.hierarchical_success_probability(
            candidate.skill_id,
            context_key,
        ),
        default=None,
    )
