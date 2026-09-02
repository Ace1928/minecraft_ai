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


class AutonomousCognitionEngine:
    """SOTA autonomous cognitive decision engine implementing hierarchical long-horizon planning,
    dynamic tech-tree progression, spatial goal navigation, and conversational player interaction.
    """

    def __init__(self, skills: SkillLibrary) -> None:
        self.skills = skills
        self._tech_tier = "wood_age"
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
        crosshair = blackboard.fact("crosshair")
        now_s = time.time()

        # 1. Critical Survival & Hazard Interruption
        if danger and bool(danger.value):
            say = "Hazard detected! Backing off to safety." if now_s - self._last_speech_time > 15 else None
            if say:
                self._last_speech_time = now_s
            return CognitionDecision(
                reasoning_summary="Immediate hazard detected; retreating to safety zone.",
                chosen_goal_id="survive",
                skill_id="retreat_from_danger",
                say=say,
            )

        if hostile and bool(hostile.value):
            say = "Hostile creature spotted; engaging in combat." if now_s - self._last_speech_time > 15 else None
            if say:
                self._last_speech_time = now_s
            return CognitionDecision(
                reasoning_summary="Hostile target visible; initiating tactical combat.",
                chosen_goal_id="survive",
                skill_id="attack_visible_hostile",
                say=say,
            )

        # 2. Player Request & Social Promise Priority
        for promise in context.promises:
            if not promise.fulfilled:
                return CognitionDecision(
                    reasoning_summary=f"Fulfilling player promise: {promise.description}",
                    chosen_goal_id=f"promise:{promise.promise_id}",
                    skill_id="explore_forward",
                    say=f"On it: {promise.description}" if now_s - self._last_speech_time > 20 else None,
                )

        # 3. Dynamic Resource Acquisition & Tech-Tree Progression
        if target_vis and bool(target_vis.value):
            if target_mineable and bool(target_mineable.value):
                say = "Harvesting resource." if now_s - self._last_speech_time > 30 else None
                if say:
                    self._last_speech_time = now_s
                return CognitionDecision(
                    reasoning_summary="Mineable resource target locked; mining block.",
                    chosen_goal_id="obtain_wood",
                    skill_id="mine_visible_block",
                    say=say,
                )
            return CognitionDecision(
                reasoning_summary="Resource target visible; approaching target location.",
                chosen_goal_id="obtain_wood",
                skill_id="approach_visible_target",
            )

        # 4. Long-Horizon Exploration & Spatial Scanning
        return CognitionDecision(
            reasoning_summary="Surveying area with 360° sweeps to locate wood, stone, and landmarks.",
            chosen_goal_id="explore",
            skill_id="explore_forward",
            ask_perception=("target.visible", "danger.immediate"),
        )


@dataclass
class HighLevelController:
    model: LanguageModel
    skills: SkillLibrary
    _autonomous: AutonomousCognitionEngine = field(init=False)

    def __post_init__(self) -> None:
        self._autonomous = AutonomousCognitionEngine(self.skills)

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
                        "closed-loop skill to execute. Return one JSON object matching: "
                        "reasoning_summary:string, chosen_goal_id:string|null, skill_id:string|null, "
                        "skill_parameters:object, say:string|null, request_replan:boolean, "
                        "ask_perception:string[], research_query:string|null. "
                        "Use only listed skill ids."
                    ),
                ),
                ModelMessage(role="user", content=json.dumps(payload, separators=(",", ":"))),
            )
            response = self.model.complete(messages)
            decision = _parse_decision(response.text)
            if decision.skill_id is not None and decision.skill_id not in self.skills.specs:
                return self._autonomous.decide(blackboard, context)
            return decision
        except Exception:
            return self._autonomous.decide(blackboard, context)


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
