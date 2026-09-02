from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .memory import MemoryRecord
from .models import LanguageModel, ModelMessage
from .perception import PerceptionBlackboard
from .planning import Goal
from .roles import RoleProfile
from .skills import SkillLibrary
from .social import Promise
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


@dataclass
class HighLevelController:
    model: LanguageModel
    skills: SkillLibrary

    def decide(
        self,
        blackboard: PerceptionBlackboard,
        context: CognitionContext,
    ) -> CognitionDecision:
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
                    "expected_effects": list(skill.expected_effects),
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
                    "closed-loop skill to execute. Never emit keyboard, mouse, shell, code, "
                    "or privileged world-state commands. Return one JSON object matching: "
                    "reasoning_summary:string, chosen_goal_id:string|null, skill_id:string|null, "
                    "skill_parameters:object, say:string|null, request_replan:boolean, "
                    "ask_perception:string[], research_query:string|null. Use only listed skill ids."
                ),
            ),
            ModelMessage(role="user", content=json.dumps(payload, separators=(",", ":"))),
        )
        response = self.model.complete(messages)
        decision = _parse_decision(response.text)
        if decision.skill_id is not None and decision.skill_id not in self.skills.specs:
            raise RuntimeError(f"model selected unknown skill: {decision.skill_id}")
        return decision


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
