from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .execution import initiation_satisfied
from .memory import MemoryRecord
from .models import LanguageModel, ModelMessage, ModelResponse
from .perception import PerceptionBlackboard
from .planning import Goal
from .roles import RoleProfile
from .skills import SkillLibrary, SkillOutcome, SkillRun
from .social import (
    OperatorMessage,
    OperatorMessageKind,
    OperatorMessageStatus,
    Promise,
    PromiseStatus,
)
from .wiki import WikiEvidence


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

    def prompt_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "allowed_skills": [
                {"s": skill_id, "p": parameters} for skill_id, parameters in self.allowed_skills
            ],
            "required_action_constraints": dict(self.required_action_constraints),
        }
        if self.authority_goal_id is not None:
            payload["authority_goal_id"] = self.authority_goal_id
        return payload


_JSON_REPAIR_SYSTEM = (
    "You are a bounded JSON normalizer, not a planner. Repair exactly one malformed or "
    "truncated Minecraft cognition response into the strict wire schema supplied by the API. "
    "Wire keys are r=brief summary, g=goal id, s=skill id or null, p=parameter object, "
    "o=operator reply, c=authorized game chat, x=replan, q=at most two perception questions, "
    "w=research query. Always emit p. Preserve only explicit complete values from "
    "rejected_output and authority_bounds. Never invent observations, inventory, outcomes, "
    "goals, chat, or parameters. s must be null or an allowed skill. Preserve authority_goal_id "
    "and required_action_constraints exactly when present. If the action cannot be recovered "
    "without invention, emit s=null, p containing only required constraints, and x=true. "
    "Return JSON only."
)

_SEMANTIC_REPAIR_SYSTEM = (
    "Make exactly one bounded correction to a rejected Minecraft cognition decision. The user "
    "JSON below is the complete repair context; do not assume or invent omitted world state. "
    "Return only the strict compact wire JSON. Select s only from authority_bounds.allowed_skills "
    "and use only its listed parameter names. Preserve authority_goal_id and every false "
    "required_action_constraint exactly. If no allowed option is justified, return s=null, "
    "p containing the required constraints, x=true, and ask for at most two needed perceptions."
)

_MAX_REJECTED_OUTPUT_CHARS = 2_048
_MAX_REPAIR_REASON_CHARS = 640


def _operator_prompt_payload(message: OperatorMessage) -> dict[str, object]:
    """Expose operator-authored content without feeding model replies back as commands."""
    return {
        "message_id": message.message_id,
        "created_ns": message.created_ns,
        "author": message.author,
        "text": message.text,
        "kind": message.kind.value,
        "priority": message.priority,
        "status": message.status.value,
    }


def _explicit_action_constraints(text: str) -> dict[str, bool]:
    """Translate literal operator prohibitions into the motor option contract.

    This is authority enforcement, not a gameplay policy: it never selects an
    action and only masks an actuator the operator explicitly prohibited. The
    strategic model still chooses the skill and every remaining learned action.
    """
    constraints: dict[str, bool] = {}
    normalized = text.casefold()
    for match in re.finditer(
        r"\b(?:do\s+not|don't|never|without)\b(?P<scope>[^.!?;]{0,160})",
        normalized,
    ):
        scope = match.group("scope")
        if re.search(r"\b(?:attack|attacking|hit|hitting|fight|fighting)\b", scope):
            constraints["allow_attack"] = False
        if re.search(r"\b(?:use|using|interact|interacting)\b", scope):
            constraints["allow_use"] = False
        if re.search(r"\b(?:jump|jumping)\b", scope):
            constraints["allow_jump"] = False
    return constraints


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
    metrics: HighLevelMetrics = field(default_factory=HighLevelMetrics, init=False)

    def __post_init__(self) -> None:
        self._bootstrap = BootstrapCognitionPolicy(self.skills)

    def decide(
        self,
        blackboard: PerceptionBlackboard,
        context: CognitionContext,
    ) -> CognitionDecision:
        try:
            latest = blackboard.latest()
            feasible_skill_payloads = self._feasible_skill_payloads(blackboard)
            repair_bounds = self._decision_repair_bounds(blackboard, context)
            facts = {
                key: {
                    "value": fact.value,
                    "confidence": round(fact.confidence, 3),
                    "source": fact.source,
                }
                for key, fact in blackboard.fresh_facts(min_confidence=0.35).items()
            }
            payload: dict[str, Any] = {
                "role": {
                    "id": context.role.role_id,
                    "goals": context.role.standing_goals,
                    "weights": context.role.utility_weights,
                    "risk": context.role.risk_tolerance,
                },
                "goals": [
                    {
                        "id": goal.goal_id,
                        "description": goal.description,
                        "target": goal.target_node,
                        "source": goal.source.value,
                        "priority": goal.priority,
                        "domain": goal.domain,
                    }
                    for goal in context.goals
                ],
                "memories": [
                    {
                        "id": memory.memory_id,
                        "kind": memory.kind.value,
                        "text": memory.text,
                        "confidence": memory.confidence,
                        "importance": memory.importance,
                        "goals": memory.goal_tags,
                        "entities": memory.entity_tags,
                        "place": memory.location_key,
                    }
                    for memory in context.memories
                ],
                "promises": [
                    {
                        "id": promise.promise_id,
                        "player": promise.player,
                        "summary": promise.summary,
                        "status": promise.status.value,
                        "goal": promise.goal_id,
                        "project": promise.project_id,
                    }
                    for promise in context.promises
                ],
                "operator_messages": [
                    _operator_prompt_payload(message) for message in context.operator_messages
                ],
                "active_operator_message": None
                if not context.operator_messages
                else _operator_prompt_payload(context.operator_messages[0]),
                "wiki_evidence": [item.model_dump(mode="json") for item in context.wiki],
                "recent_skill_runs": [
                    {
                        "skill": run.skill_id,
                        "outcome": run.outcome.value,
                        "context": run.context_key,
                        "failure": run.failure_reason,
                    }
                    for run in context.recent_skill_runs
                ],
                "frame": None
                if latest is None
                else {
                    "id": latest.frame_id,
                    "instance": latest.instance_id,
                    "size": (latest.width, latest.height),
                    "tracks": [
                        {
                            "id": track.track_id,
                            "label": track.label,
                            "confidence": round(track.confidence, 3),
                            "region": track.region.model_dump(mode="json"),
                            "attributes": track.attributes,
                        }
                        for track in latest.tracks
                    ],
                },
                "fresh_facts": facts,
                "skills": feasible_skill_payloads,
            }
            messages = (
                ModelMessage(
                    role="system",
                    content=(
                        "You control a Minecraft agent through verified closed-loop skills. "
                        "Return one compact JSON object with wire keys: r=summary under 12 words, "
                        "g=goal id, s=skill id or null, p=parameters, o=operator reply, "
                        "c=authorized in-game chat, x=replan, q=at most two perception questions, "
                        "w=research query. Omit null, empty, and false keys; p is required. "
                        "fresh_facts is the only authoritative observed game state. skills "
                        "contains only currently executable options: use only a listed skill_id, "
                        "prefer concrete progression with verifiable success evidence, and never "
                        "claim unobserved inventory, outcomes, or completion. Do not explore when "
                        "a more concrete feasible resource/progression skill exists. "
                        "active_operator_message has highest authority and must be addressed "
                        "before any conflicting standing goal. For an instruction or correction, "
                        "use g='operator:'+message_id until superseded. Set o only to "
                        "reply to that operator; it never types in game. Set c only with an "
                        "authoritative fresh player-message or game-chat authorization fact. "
                        "Keep private reasoning in r. Encode explicit operator prohibitions as "
                        "allow_attack:false, allow_use:false, or allow_jump:false in p. "
                        "Treat recent_skill_runs and evaluation as empirical evidence. Avoid a "
                        "skill after two consecutive failures; choose another listed skill or "
                        "return s null with x true and request needed perception. Every q item "
                        "must be a literal missing fresh_facts key, never a prose question. "
                        "A fresh operator correction permits one evidence-producing retry."
                    ),
                ),
                ModelMessage(role="user", content=json.dumps(payload, separators=(",", ":"))),
                *(
                    ()
                    if not context.operator_messages
                    else (
                        ModelMessage(
                            role="user",
                            content=(
                                "ACTIVE OPERATOR DIRECTIVE (highest authority; follow this "
                                "literal current request and do not substitute an older task): "
                                + json.dumps(
                                    _operator_prompt_payload(context.operator_messages[0]),
                                    separators=(",", ":"),
                                )
                            ),
                        ),
                    )
                ),
            )
            decision = self._complete(messages, repair_bounds=repair_bounds)
            decision = self._apply_decision_authority(decision, blackboard, context)
            if decision.skill_id is not None and decision.skill_id not in self.skills.specs:
                return self._repair_infeasible(
                    decision,
                    blackboard,
                    context,
                    reason=f"unknown skill id {decision.skill_id!r}",
                )
            if decision.skill_id is not None:
                selected = self.skills.get(decision.skill_id)
                if not initiation_satisfied(selected, blackboard):
                    missing = tuple(
                        condition.key
                        for group in (
                            selected.preconditions,
                            *selected.initiation_alternatives,
                        )
                        for condition in group
                    )
                    return self._repair_infeasible(
                        decision,
                        blackboard,
                        context,
                        reason=(
                            f"option {selected.skill_id!r} is infeasible because these fresh "
                            f"preconditions are missing: {', '.join(missing)}"
                        ),
                        missing=missing,
                    )
                blocked_run = self._blocking_skill_run(decision, context)
                if blocked_run is not None:
                    return self._repair_repeated_failure(
                        decision,
                        blackboard,
                        context,
                        blocked_run,
                    )
            self.metrics.last_error = None
            return decision
        except Exception as exc:
            self.metrics.failures += 1
            self.metrics.last_error = f"{type(exc).__name__}: {exc}"
            return CognitionDecision(
                reasoning_summary=(
                    "Strategic model unavailable; remaining safely idle until a valid "
                    "structured decision is available."
                ),
                request_replan=True,
            )

    def _apply_decision_authority(
        self,
        decision: CognitionDecision,
        blackboard: PerceptionBlackboard,
        context: CognitionContext,
    ) -> CognitionDecision:
        decision = self._scope_operator_decision(decision, blackboard, context)
        operator_goal_ids = {
            f"operator:{message.message_id}" for message in context.operator_messages
        }
        if decision.say is not None and decision.chosen_goal_id not in operator_goal_ids:
            return decision.model_copy(update={"say": None})
        return decision

    def _scope_operator_decision(
        self,
        decision: CognitionDecision,
        blackboard: PerceptionBlackboard,
        context: CognitionContext,
    ) -> CognitionDecision:
        active = next(
            (
                message
                for message in context.operator_messages
                if message.kind in {OperatorMessageKind.INSTRUCTION, OperatorMessageKind.CORRECTION}
            ),
            None,
        )
        danger = blackboard.fact("danger.immediate", min_confidence=0.7)
        if danger is not None and bool(danger.value):
            active = None
        if active is None:
            return decision
        danger = blackboard.fact("danger.immediate", min_confidence=0.7)
        if danger is not None and bool(danger.value):
            return decision
        parameters = dict(decision.skill_parameters)
        if decision.skill_id is not None and decision.skill_id in self.skills.specs:
            selected_skill = self.skills.get(decision.skill_id)
            if "target" in selected_skill.parameters:
                latest = blackboard.latest()
                operator_tracks = (
                    ()
                    if latest is None
                    else tuple(
                        track
                        for track in latest.tracks
                        if track.attributes.get("source") == "operator"
                    )
                )
                if operator_tracks:
                    target = max(operator_tracks, key=lambda track: track.last_seen_ns)
                    parameters["target"] = target.label
        parameters.update(_explicit_action_constraints(active.text))
        return decision.model_copy(
            update={
                "chosen_goal_id": f"operator:{active.message_id}",
                "skill_parameters": parameters,
            }
        )

    def status(self) -> dict[str, object]:
        return {
            "model_id": self.model.model_id,
            "calls": self.metrics.calls,
            "repairs": self.metrics.repairs,
            "failures": self.metrics.failures,
            "retry_repairs": self.metrics.retry_repairs,
            "json_repairs": self.metrics.json_repairs,
            "json_repair_failures": self.metrics.json_repair_failures,
            "last_latency_ms": round(self.metrics.last_latency_ms, 3),
            "last_error": self.metrics.last_error,
            "last_model": self.metrics.last_model,
        }

    def _feasible_skill_payloads(
        self,
        blackboard: PerceptionBlackboard,
    ) -> list[dict[str, object]]:
        payloads: list[dict[str, object]] = []
        for skill in self.skills.specs.values():
            if not initiation_satisfied(skill, blackboard):
                continue
            stats = self.skills.stats.get((skill.skill_id, "default"))
            payloads.append(
                {
                    "skill_id": skill.skill_id,
                    "name": skill.name,
                    "description": skill.description,
                    "stage": skill.stage.value,
                    "parameters": list(skill.parameters),
                    "success_evidence": [
                        {
                            "fact": condition.key,
                            "op": condition.operator,
                            "value": condition.value,
                            "confidence": condition.min_confidence,
                        }
                        for condition in skill.success_conditions
                    ],
                    "effects": list(skill.expected_effects),
                    "competence": round(self.skills.contextual_score(skill.skill_id), 3),
                    "evaluation": {
                        "attempts": 0 if stats is None else stats.attempts,
                        "successes": 0 if stats is None else stats.successes,
                        "failures": 0 if stats is None else stats.failures,
                        "timeouts": 0 if stats is None else stats.timeouts,
                        "consecutive_failures": (
                            0 if stats is None else stats.consecutive_failures
                        ),
                    },
                }
            )
        return payloads

    def _decision_repair_bounds(
        self,
        blackboard: PerceptionBlackboard,
        context: CognitionContext,
        *,
        allowed_skill_ids: set[str] | None = None,
    ) -> _DecisionRepairBounds:
        active = next(
            (
                message
                for message in context.operator_messages
                if message.kind in {OperatorMessageKind.INSTRUCTION, OperatorMessageKind.CORRECTION}
            ),
            None,
        )
        allowed_skills = tuple(
            (skill.skill_id, tuple(skill.parameters))
            for skill in sorted(
                self.skills.specs.values(),
                key=lambda candidate: candidate.skill_id,
            )
            if initiation_satisfied(skill, blackboard)
            and (allowed_skill_ids is None or skill.skill_id in allowed_skill_ids)
        )
        constraints = (
            () if active is None else tuple(_explicit_action_constraints(active.text).items())
        )
        return _DecisionRepairBounds(
            allowed_skills=allowed_skills,
            authority_goal_id=None if active is None else f"operator:{active.message_id}",
            required_action_constraints=constraints,
        )

    def _blocking_skill_run(
        self,
        decision: CognitionDecision,
        context: CognitionContext,
    ) -> SkillRun | None:
        if decision.skill_id is None or not context.recent_skill_runs:
            return None
        if context.operator_messages and context.operator_messages[0].status in {
            OperatorMessageStatus.QUEUED,
            OperatorMessageStatus.DELIVERED,
        }:
            # A fresh, explicit operator retry gets one evidence-producing attempt.
            return None
        for recent in context.recent_skill_runs:
            if recent.skill_id != decision.skill_id or recent.outcome not in {
                SkillOutcome.FAILED,
                SkillOutcome.TIMED_OUT,
            }:
                continue
            stats = self.skills.stats.get((decision.skill_id, recent.context_key))
            if stats is not None and stats.consecutive_failures >= 2:
                return recent
        return None

    def _recently_blocked_skill_ids(self, context: CognitionContext) -> set[str]:
        blocked: set[str] = set()
        for run in context.recent_skill_runs:
            if run.outcome not in {SkillOutcome.FAILED, SkillOutcome.TIMED_OUT}:
                continue
            stats = self.skills.stats.get((run.skill_id, run.context_key))
            if stats is not None and stats.consecutive_failures >= 2:
                blocked.add(run.skill_id)
        return blocked

    def _repair_repeated_failure(
        self,
        decision: CognitionDecision,
        blackboard: PerceptionBlackboard,
        context: CognitionContext,
        blocked_run: SkillRun,
    ) -> CognitionDecision:
        failed_skill = decision.skill_id
        assert failed_skill is not None
        blocked_skill_ids = self._recently_blocked_skill_ids(context)
        feasible = sorted(
            skill.skill_id
            for skill in self.skills.specs.values()
            if skill.skill_id not in blocked_skill_ids and initiation_satisfied(skill, blackboard)
        )
        self.metrics.repairs += 1
        self.metrics.retry_repairs += 1
        repair_bounds = self._decision_repair_bounds(
            blackboard,
            context,
            allowed_skill_ids=set(feasible),
        )
        repair_messages = _semantic_repair_messages(
            decision,
            repair_bounds,
            repair_kind="repeated_execution_failure",
            reason=(
                f"{failed_skill!r} ended as {blocked_run.outcome.value!r} with reason "
                f"{blocked_run.failure_reason!r} and has repeated consecutive failures"
            ),
            blocked_skill_ids=tuple(sorted(blocked_skill_ids)),
        )
        repaired = self._complete(repair_messages, repair_bounds=repair_bounds)
        repaired = self._apply_decision_authority(repaired, blackboard, context)
        if repaired.skill_id is None and repaired.request_replan:
            self.metrics.last_error = None
            return _enforce_repair_bounds(repaired, repair_bounds)
        if repaired.skill_id in feasible:
            self.metrics.last_error = None
            return _enforce_repair_bounds(repaired, repair_bounds)
        self.metrics.last_error = f"repeated-option-blocked:{failed_skill}"
        return decision.model_copy(
            update={
                "reasoning_summary": (
                    f"Blocked repeated {failed_skill} after empirical timeout/failure evidence."
                ),
                "skill_id": None,
                "skill_parameters": dict(repair_bounds.required_action_constraints),
                "request_replan": True,
                "ask_perception": tuple(
                    dict.fromkeys((*decision.ask_perception, "obstacle.ahead"))
                ),
            }
        )

    def _complete(
        self,
        messages: tuple[ModelMessage, ...],
        *,
        repair_bounds: _DecisionRepairBounds,
    ) -> CognitionDecision:
        response = self._request_model(messages, name="cognition_decision")
        try:
            return _parse_decision(response.text)
        except (RuntimeError, ValidationError):
            self.metrics.repairs += 1
            self.metrics.json_repairs += 1
            repair_messages = _json_repair_messages(response.text, repair_bounds)
            repaired_response = self._request_model(
                repair_messages,
                name="cognition_decision_json_repair",
            )
            try:
                repaired = _parse_decision(repaired_response.text)
            except (RuntimeError, ValidationError) as repair_exc:
                self.metrics.json_repair_failures += 1
                raise RuntimeError(
                    "high-level model returned invalid structured output after one bounded repair"
                ) from repair_exc
            return _enforce_repair_bounds(repaired, repair_bounds)

    def _request_model(
        self,
        messages: tuple[ModelMessage, ...],
        *,
        name: str,
    ) -> ModelResponse:
        structured = getattr(self.model, "complete_structured", None)
        if callable(structured):
            response = cast(
                ModelResponse,
                structured(
                    messages,
                    name=name,
                    schema=_CognitionWireDecision.model_json_schema(),
                ),
            )
        else:
            response = self.model.complete(messages)
        self.metrics.calls += 1
        self.metrics.last_latency_ms = response.latency_ms
        self.metrics.last_model = response.model
        return response

    def _repair_infeasible(
        self,
        decision: CognitionDecision,
        blackboard: PerceptionBlackboard,
        context: CognitionContext,
        *,
        reason: str,
        missing: tuple[str, ...] = (),
    ) -> CognitionDecision:
        feasible = sorted(
            skill.skill_id
            for skill in self.skills.specs.values()
            if initiation_satisfied(skill, blackboard)
        )
        self.metrics.repairs += 1
        repair_bounds = self._decision_repair_bounds(
            blackboard,
            context,
            allowed_skill_ids=set(feasible),
        )
        repair_messages = _semantic_repair_messages(
            decision,
            repair_bounds,
            repair_kind="infeasible_option",
            reason=reason,
            missing_facts=missing,
        )
        repaired = self._complete(repair_messages, repair_bounds=repair_bounds)
        repaired = self._apply_decision_authority(repaired, blackboard, context)
        if (
            repaired.skill_id is not None
            and repaired.skill_id in feasible
            and repaired.skill_id in self.skills.specs
        ):
            self.metrics.last_error = None
            return _enforce_repair_bounds(repaired, repair_bounds)
        self.metrics.last_error = f"infeasible-decision: {reason}"
        return decision.model_copy(
            update={
                "reasoning_summary": f"Blocked infeasible decision; {reason}",
                "skill_id": None,
                "skill_parameters": dict(repair_bounds.required_action_constraints),
                "request_replan": True,
                "ask_perception": tuple(dict.fromkeys((*decision.ask_perception, *missing))),
            }
        )


def _compact_wire_payload(decision: CognitionDecision) -> dict[str, object]:
    payload: dict[str, object] = {
        "r": decision.reasoning_summary[:120],
        "p": decision.skill_parameters,
    }
    optional_values: tuple[tuple[str, object | None], ...] = (
        ("g", decision.chosen_goal_id),
        ("s", decision.skill_id),
        ("o", None if decision.say is None else decision.say[:160]),
        ("c", None if decision.game_chat is None else decision.game_chat[:160]),
        ("w", None if decision.research_query is None else decision.research_query[:160]),
    )
    for key, value in optional_values:
        if value is not None:
            payload[key] = value
    if decision.request_replan:
        payload["x"] = True
    if decision.ask_perception:
        payload["q"] = tuple(question[:160] for question in decision.ask_perception[:2])
    return payload


def _json_repair_messages(
    rejected_output: str,
    bounds: _DecisionRepairBounds,
) -> tuple[ModelMessage, ...]:
    payload = {
        "rejected_output": rejected_output[:_MAX_REJECTED_OUTPUT_CHARS],
        "authority_bounds": bounds.prompt_payload(),
        "safe_fallback": {
            "s": None,
            "p": dict(bounds.required_action_constraints),
            "x": True,
        },
    }
    return (
        ModelMessage(role="system", content=_JSON_REPAIR_SYSTEM),
        ModelMessage(role="user", content=json.dumps(payload, separators=(",", ":"))),
    )


def _semantic_repair_messages(
    decision: CognitionDecision,
    bounds: _DecisionRepairBounds,
    *,
    repair_kind: str,
    reason: str,
    blocked_skill_ids: tuple[str, ...] = (),
    missing_facts: tuple[str, ...] = (),
) -> tuple[ModelMessage, ...]:
    payload = {
        "repair": repair_kind,
        "reason": reason[:_MAX_REPAIR_REASON_CHARS],
        "rejected": _compact_wire_payload(decision),
        "authority_bounds": bounds.prompt_payload(),
        "blocked_skills": blocked_skill_ids,
        "missing_facts": missing_facts,
        "safe_fallback": {
            "s": None,
            "p": dict(bounds.required_action_constraints),
            "x": True,
        },
    }
    return (
        ModelMessage(role="system", content=_SEMANTIC_REPAIR_SYSTEM),
        ModelMessage(role="user", content=json.dumps(payload, separators=(",", ":"))),
    )


def _enforce_repair_bounds(
    decision: CognitionDecision,
    bounds: _DecisionRepairBounds,
) -> CognitionDecision:
    allowed_parameters = dict(bounds.allowed_skills)
    required_constraints: dict[str, str | int | float | bool] = dict(
        bounds.required_action_constraints
    )
    goal_id = bounds.authority_goal_id or decision.chosen_goal_id
    if decision.skill_id is not None and decision.skill_id not in allowed_parameters:
        return CognitionDecision(
            reasoning_summary="Repaired decision violated the allowed option bounds.",
            chosen_goal_id=goal_id,
            skill_parameters=required_constraints,
            request_replan=True,
            ask_perception=decision.ask_perception[:2],
        )
    if decision.skill_id is None:
        parameters = required_constraints
    else:
        permitted = set(allowed_parameters[decision.skill_id]) | set(required_constraints)
        parameters = {
            key: value for key, value in decision.skill_parameters.items() if key in permitted
        }
        parameters.update(required_constraints)
    return decision.model_copy(
        update={
            "chosen_goal_id": goal_id,
            "skill_parameters": parameters,
        }
    )


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
    if isinstance(raw, dict) and any(key in raw for key in ("r", "g", "s", "p")):
        return _CognitionWireDecision.model_validate(raw).expand()
    return CognitionDecision.model_validate(raw)
