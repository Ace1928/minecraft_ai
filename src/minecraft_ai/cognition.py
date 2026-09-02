from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .memory import MemoryRecord
from .models import LanguageModel, ModelMessage
from .perception import PerceptionBlackboard
from .planning import Goal
from .roles import RoleProfile
from .skills import SkillLibrary
from .social import OperatorMessage, Promise, PromiseStatus
from .wiki import WikiEvidence


class CognitionDecision(BaseModel):
    """High-level output. It deliberately contains no key/mouse commands."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reasoning_summary: str = ""
    chosen_goal_id: str | None = None
    skill_id: str | None = None
    skill_parameters: dict[str, str | int | float | bool] = Field(default_factory=dict)
    say: str | None = None
    request_replan: bool = False
    ask_perception: tuple[str, ...] = ()
    research_query: str | None = None


@dataclass
class CognitionContext:
    role: RoleProfile
    goals: tuple[Goal, ...]
    memories: tuple[MemoryRecord, ...]
    promises: tuple[Promise, ...]
    wiki: tuple[WikiEvidence, ...]
    operator_messages: tuple[OperatorMessage, ...] = ()


class BootstrapCognitionPolicy:
    """Small deterministic fallback for smoke tests and model outages.

    This is intentionally a reactive priority policy. It is not the strategic
    executive and does not own progression knowledge.
    """

    def __init__(self, skills: SkillLibrary) -> None:
        self.skills = skills
        self._last_speech_time = 0.0

    def decide(
        self,
        blackboard: PerceptionBlackboard,
        context: CognitionContext,
    ) -> CognitionDecision:
        danger = blackboard.fact("danger.immediate")
        hostile = blackboard.fact("target.hostile_visible")
        target_vis = blackboard.fact("target.visible")
        target_mineable = blackboard.fact("target.mineable")
        now_s = time.time()

        if danger and bool(danger.value):
            say = self._speech(now_s, "Hazard detected; backing off.", interval_s=15.0)
            return CognitionDecision(
                reasoning_summary="Bootstrap fallback observed an immediate hazard.",
                chosen_goal_id="survive",
                skill_id="retreat_from_danger",
                say=say,
            )

        if hostile and bool(hostile.value):
            return CognitionDecision(
                reasoning_summary="Bootstrap fallback observed a visible hostile.",
                chosen_goal_id="survive",
                skill_id="attack_visible_hostile",
                say=self._speech(now_s, "Hostile spotted; defending.", interval_s=15.0),
            )

        for promise in context.promises:
            if promise.status in {PromiseStatus.PENDING, PromiseStatus.ACTIVE}:
                return CognitionDecision(
                    reasoning_summary=f"Bootstrap fallback retained promise: {promise.summary}",
                    chosen_goal_id=f"promise:{promise.promise_id}",
                    skill_id="explore_forward",
                    say=self._speech(now_s, f"Continuing: {promise.summary}", interval_s=20.0),
                )

        if context.operator_messages:
            message = context.operator_messages[0]
            return CognitionDecision(
                reasoning_summary=(
                    f"Bootstrap fallback retained operator {message.kind.value}: {message.text}"
                ),
                chosen_goal_id=f"operator:{message.message_id}",
                skill_id="explore_forward",
                say=self._speech(now_s, f"Received: {message.text[:160]}", interval_s=10.0),
            )

        goal_id = context.goals[0].goal_id if context.goals else "explore"
        if target_vis and bool(target_vis.value):
            if target_mineable and bool(target_mineable.value):
                return CognitionDecision(
                    reasoning_summary="Bootstrap fallback observed a mineable target.",
                    chosen_goal_id=goal_id,
                    skill_id="mine_visible_block",
                )
            return CognitionDecision(
                reasoning_summary="Bootstrap fallback observed a target.",
                chosen_goal_id=goal_id,
                skill_id="approach_visible_target",
            )

        return CognitionDecision(
            reasoning_summary="Bootstrap fallback is collecting exploratory observations.",
            chosen_goal_id=goal_id,
            skill_id="explore_forward",
            ask_perception=("target.visible", "danger.immediate"),
        )

    def _speech(self, now_s: float, text: str, *, interval_s: float) -> str | None:
        if now_s - self._last_speech_time <= interval_s:
            return None
        self._last_speech_time = now_s
        return text


@dataclass
class HighLevelController:
    model: LanguageModel
    skills: SkillLibrary
    _bootstrap: BootstrapCognitionPolicy = field(init=False)

    def __post_init__(self) -> None:
        self._bootstrap = BootstrapCognitionPolicy(self.skills)

    def decide(
        self,
        blackboard: PerceptionBlackboard,
        context: CognitionContext,
    ) -> CognitionDecision:
        try:
            latest = blackboard.latest()
            facts = {
                key: fact.model_dump(mode="json")
                for key, fact in blackboard.fresh_facts(min_confidence=0.35).items()
            }
            payload: dict[str, Any] = {
                "role": context.role.model_dump(mode="json"),
                "goals": [goal.model_dump(mode="json") for goal in context.goals],
                "memories": [memory.model_dump(mode="json") for memory in context.memories],
                "promises": [promise.model_dump(mode="json") for promise in context.promises],
                "operator_messages": [
                    message.model_dump(mode="json") for message in context.operator_messages
                ],
                "wiki_evidence": [item.model_dump(mode="json") for item in context.wiki],
                "frame": None if latest is None else latest.model_dump(mode="json"),
                "fresh_facts": facts,
                "skills": [
                    {
                        "skill_id": skill.skill_id,
                        "name": skill.name,
                        "description": skill.description,
                        "stage": skill.stage.value,
                        "parameters": list(skill.parameters),
                        "preconditions": [
                            condition.model_dump(mode="json")
                            for condition in skill.preconditions
                        ],
                        "success_conditions": [
                            condition.model_dump(mode="json")
                            for condition in skill.success_conditions
                        ],
                        "expected_effects": list(skill.expected_effects),
                        "measured_competence": self.skills.contextual_score(skill.skill_id),
                    }
                    for skill in self.skills.specs.values()
                ],
            }
            messages = (
                ModelMessage(
                    role="system",
                    content=(
                        "You are the high-level controller of a Minecraft agent. Decide goals, "
                        "social responses, research/perception requests, and which available "
                        "closed-loop skill to execute. Return one JSON object matching: "
                        "reasoning_summary:string, chosen_goal_id:string|null, "
                        "skill_id:string|null, "
                        "skill_parameters:object, say:string|null, request_replan:boolean, "
                        "ask_perception:string[], research_query:string|null. "
                        "Use only listed skill ids. Set say only when directly replying to a "
                        "fresh operator/player message or when urgent social communication is "
                        "needed; ordinary private reasoning must not open in-game chat. Treat "
                        "fresh_facts as the only authoritative observed game state. Prefer the "
                        "most concrete feasible option that advances the chosen goal and has a "
                        "verifiable success condition; do not select generic exploration when a "
                        "visible resource or feasible progression option is available. Never "
                        "claim an item, outcome, or completion that has not been observed."
                    ),
                ),
                ModelMessage(role="user", content=json.dumps(payload, separators=(",", ":"))),
            )
            structured = getattr(self.model, "complete_structured", None)
            if callable(structured):
                response = structured(
                    messages,
                    name="cognition_decision",
                    schema=CognitionDecision.model_json_schema(),
                )
            else:
                response = self.model.complete(messages)
            decision = _parse_decision(response.text)
            if decision.skill_id is not None and decision.skill_id not in self.skills.specs:
                return self._bootstrap.decide(blackboard, context)
            return decision
        except Exception:
            return self._bootstrap.decide(blackboard, context)


def _parse_decision(text: str) -> CognitionDecision:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    try:
        raw = json.loads(candidate)
    except ValueError as exc:
        raise RuntimeError("high-level model did not return valid JSON") from exc
    return CognitionDecision.model_validate(raw)
