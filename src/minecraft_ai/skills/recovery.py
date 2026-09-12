"""Outcome-conditioned recovery selection, separate from emergency handling."""

from __future__ import annotations

import math

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
    exploration_strength: float = 0.35,
) -> SkillSpec | None:
    """Balance contextual success evidence with trying under-observed options.

    The uncertainty bonus uses only decisive local outcomes, never cancellations
    or controller starvation. It is a selection score, not a calibrated success
    probability. Zero strength restores greedy ranking. Declared order breaks
    cold-start ties. Exhausted sets return to cognition; emergency actions are
    never reordered, and exploration cannot bypass feasibility/failure gates.
    """
    if type(max_context_failures) is not int or max_context_failures < 1:
        raise ValueError("max_context_failures must be a positive integer")
    if not math.isfinite(exploration_strength) or exploration_strength < 0:
        raise ValueError("exploration_strength must be finite and nonnegative")
    candidates = []
    for recovery_id in dict.fromkeys(recovery_ids):
        candidate = skills.specs.get(recovery_id)
        if candidate is None or candidate.stage in {SkillStage.DEPRECATED, SkillStage.RETIRED}:
            continue
        if skills.contextual_failure_streak(recovery_id, context_key) >= max_context_failures:
            continue
        if initiation_satisfied(candidate, blackboard):
            candidates.append(candidate)

    attempts = {}
    for candidate in candidates:
        stats = skills.stats.get((candidate.skill_id, context_key))
        attempts[candidate.skill_id] = 0 if stats is None else stats.decisive_attempts
    log_total = math.log(2.0 + sum(attempts.values()))

    def score(candidate: SkillSpec) -> float:
        mean = skills.hierarchical_success_probability(candidate.skill_id, context_key)
        uncertainty = math.sqrt(log_total / (1.0 + attempts[candidate.skill_id]))
        return mean + exploration_strength * uncertainty

    return max(candidates, key=score, default=None)
