from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from minecraft_ai.memory import MemoryRecord
from minecraft_ai.planning import Goal
from minecraft_ai.roles import RoleProfile
from minecraft_ai.skills import SkillRun
from minecraft_ai.social import (
    OperatorMessage,
    Promise,
)
from minecraft_ai.wiki import WikiEvidence

_WOOD_INVENTORY_AUDIT_SKILLS = frozenset({"craft_wood_planks", "open_inventory"})

@dataclass(frozen=True, slots=True)
class DecisionModelOrigin:
    request_id: str
    attempt_id: str
    source_decision_sha256: str

class CognitionDecision(BaseModel):
    """High-level output with explicit, non-interchangeable communication channels.

    ``say`` is rendered in the operator console. ``game_chat`` is a request for
    the runtime to type into Bedrock and therefore remains subject to a separate
    observed-message/authority gate. Neither field is private reasoning.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    reasoning_summary: str = ""
    chosen_goal_id: str | None = None
    skill_id: str | None = None
    skill_parameters: dict[str, str | int | float | bool] = Field(default_factory=dict)
    say: str | None = None
    game_chat: str | None = None
    request_replan: bool = False
    ask_perception: tuple[str, ...] = ()
    research_query: str | None = None
    instruction: str | None = Field(
        default=None,
        max_length=280,
        description="Concrete direction handed to the visuomotor policy as its goal condition.",
    )
    plan_steps: tuple[str, ...] = Field(
        default=(),
        max_length=5,
        description="Short sequential next-actions the agent intends to pursue.",
    )

    _model_origin: DecisionModelOrigin | None = PrivateAttr(default=None)

    @property
    def model_origin(self) -> DecisionModelOrigin | None:
        """Selected parsed model attempt before any foundation authority rewrite."""
        return self._model_origin

def cognition_decision_sha256(decision: CognitionDecision) -> str:
    """Digest public decision content, excluding private request bookkeeping."""
    canonical = json.dumps(
        decision.model_dump(mode="json"), sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()

def _without_model_origin(decision: CognitionDecision) -> CognitionDecision:
    result = decision.model_copy()
    result._model_origin = None
    return result

class _CognitionWireDecision(BaseModel):
    """Lossless, token-efficient transport form for local structured decoders."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    r: str = Field(default="", max_length=120)
    g: str | None = None
    s: str | None = None
    p: dict[str, str | int | float | bool] = Field(default_factory=dict)
    o: str | None = Field(default=None, max_length=160)
    c: str | None = Field(default=None, max_length=160)
    x: bool = False
    q: tuple[str, ...] = Field(default=(), max_length=2)
    w: str | None = Field(default=None, max_length=160)
    d: str | None = Field(
        default=None,
        max_length=280,
        description="Specific one-line direction for the current skill (goal condition).",
    )
    n: tuple[str, ...] = Field(
        default=(),
        max_length=5,
        description="Sequential plan: up to 5 short next-steps.",
    )

    def expand(self) -> CognitionDecision:
        return CognitionDecision(
            reasoning_summary=self.r,
            chosen_goal_id=self.g,
            skill_id=self.s,
            skill_parameters=self.p,
            say=self.o,
            game_chat=self.c,
            request_replan=self.x,
            ask_perception=self.q,
            research_query=self.w,
            instruction=self.d,
            plan_steps=self.n,
        )

@dataclass
class CognitionContext:
    role: RoleProfile
    goals: tuple[Goal, ...]
    memories: tuple[MemoryRecord, ...]
    promises: tuple[Promise, ...]
    wiki: tuple[WikiEvidence, ...]
    operator_messages: tuple[OperatorMessage, ...] = ()
    recent_skill_runs: tuple[SkillRun, ...] = ()
    current_plan: tuple[str, ...] = ()
    plan_goal_id: str | None = None
    plan_index: int = 0
    plan_started_ns: int = 0
    planks_retry_requires_wood: bool = False


@dataclass
class HighLevelMetrics:
    calls: int = 0
    repairs: int = 0
    failures: int = 0
    retry_repairs: int = 0
    json_repairs: int = 0
    json_repair_failures: int = 0
    last_latency_ms: float = 0.0
    last_error: str | None = None
    last_model: str | None = None

@dataclass(frozen=True)
class _DecisionRepairBounds:
    """Small authority capsule for one learned structured-output repair."""

    allowed_skills: tuple[tuple[str, tuple[str, ...]], ...]
    authority_goal_id: str | None = None
    required_action_constraints: tuple[tuple[str, bool], ...] = ()
    requested_skill_ids: tuple[str, ...] = ()

    def prompt_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "allowed_skills": [
                {"s": skill_id, "p": parameters} for skill_id, parameters in self.allowed_skills
            ],
            "required_action_constraints": dict(self.required_action_constraints),
        }
        if self.authority_goal_id is not None:
            payload["authority_goal_id"] = self.authority_goal_id
        if self.requested_skill_ids:
            payload["skill_required"] = False
            payload["abstention_requires_perception_or_replan"] = True
            payload["requested_skill_ids"] = self.requested_skill_ids
        return payload

