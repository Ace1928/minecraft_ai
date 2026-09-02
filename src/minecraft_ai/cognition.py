from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .execution import conditions_satisfied
from .memory import MemoryRecord
from .models import LanguageModel, ModelMessage
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
    recent_skill_runs: tuple[SkillRun, ...] = ()


@dataclass
class HighLevelMetrics:
    calls: int = 0
    repairs: int = 0
    failures: int = 0
    retry_repairs: int = 0
    last_latency_ms: float = 0.0
    last_error: str | None = None
    last_model: str | None = None


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
                    _operator_prompt_payload(message) for message in context.operator_messages
                ],
                "active_operator_message": None
                if not context.operator_messages
                else _operator_prompt_payload(context.operator_messages[0]),
                "wiki_evidence": [item.model_dump(mode="json") for item in context.wiki],
                "recent_skill_runs": [
                    run.model_dump(mode="json") for run in context.recent_skill_runs
                ],
                "frame": None if latest is None else latest.model_dump(mode="json"),
                "fresh_facts": facts,
                "skills": self._feasible_skill_payloads(blackboard),
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
                        "The skills list contains only options executable from current fresh "
                        "observations. Use only listed skill ids. Set say only when directly "
                        "replying to a "
                        "fresh operator/player message or when urgent social communication is "
                        "needed; ordinary private reasoning must not open in-game chat. Treat "
                        "fresh_facts as the only authoritative observed game state. Prefer the "
                        "most concrete feasible option that advances the chosen goal and has a "
                        "verifiable success condition; do not select generic exploration when a "
                        "visible resource or feasible progression option is available. Never "
                        "claim an item, outcome, or completion that has not been observed. "
                        "operator_messages are ordered by authority: highest priority first, "
                        "then corrections before instructions, then newest first. If "
                        "active_operator_message is present, it is the current directive and "
                        "must be addressed before any conflicting standing goal. Its status "
                        "'acknowledged' means received, not completed or superseded. For a "
                        "current instruction or correction, set chosen_goal_id exactly to "
                        "'operator:' plus its message_id until a newer directive supersedes it. "
                        "When that directive explicitly prohibits attack, use, or jump, encode "
                        "the prohibition in skill_parameters as allow_attack:false, "
                        "allow_use:false, or allow_jump:false so the policy contract can enforce "
                        "it without replacing learned movement. "
                        "recent_skill_runs and each skill's evaluation counters are empirical "
                        "execution evidence. Never repeat the same failed or timed-out option "
                        "unchanged when its consecutive failures are at least two; select a "
                        "different feasible learned option, request missing perception, or "
                        "return no skill with request_replan true. "
                        "Never imply a message was handled while choosing a different goal."
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
            decision = self._complete(messages)
            decision = self._scope_operator_decision(decision, blackboard, context)
            operator_goal_ids = {
                f"operator:{message.message_id}" for message in context.operator_messages
            }
            if decision.say is not None and decision.chosen_goal_id not in operator_goal_ids:
                decision = decision.model_copy(update={"say": None})
            if decision.skill_id is not None and decision.skill_id not in self.skills.specs:
                return self._repair_infeasible(
                    messages,
                    decision,
                    blackboard,
                    reason=f"unknown skill id {decision.skill_id!r}",
                )
            if decision.skill_id is not None:
                selected = self.skills.get(decision.skill_id)
                if not conditions_satisfied(selected.preconditions, blackboard):
                    missing = tuple(
                        condition.key for condition in selected.preconditions
                    )
                    return self._repair_infeasible(
                        messages,
                        decision,
                        blackboard,
                        reason=(
                            f"option {selected.skill_id!r} is infeasible because these fresh "
                            f"preconditions are missing: {', '.join(missing)}"
                        ),
                        missing=missing,
                    )
                if self._repeated_failed_option(decision, context):
                    return self._repair_repeated_failure(
                        messages,
                        decision,
                        blackboard,
                        context,
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
                if message.kind
                in {OperatorMessageKind.INSTRUCTION, OperatorMessageKind.CORRECTION}
            ),
            None,
        )
        if active is None:
            return decision
        danger = blackboard.fact("danger.immediate", min_confidence=0.7)
        if danger is not None and bool(danger.value):
            return decision
        executable = (
            decision.skill_id is not None
            and decision.skill_id in self.skills.specs
            and conditions_satisfied(
                self.skills.get(decision.skill_id).preconditions,
                blackboard,
            )
        )
        if not executable and decision.say is None:
            return decision
        parameters = dict(decision.skill_parameters)
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
            if not conditions_satisfied(skill.preconditions, blackboard):
                continue
            stats = self.skills.stats.get((skill.skill_id, "default"))
            payloads.append(
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
                    "currently_feasible": True,
                    "measured_competence": self.skills.contextual_score(skill.skill_id),
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

    def _repeated_failed_option(
        self,
        decision: CognitionDecision,
        context: CognitionContext,
    ) -> bool:
        if decision.skill_id is None or not context.recent_skill_runs:
            return False
        if context.operator_messages and context.operator_messages[0].status in {
            OperatorMessageStatus.QUEUED,
            OperatorMessageStatus.DELIVERED,
        }:
            # A fresh, explicit operator retry gets one evidence-producing attempt.
            return False
        recent = context.recent_skill_runs[0]
        if recent.skill_id != decision.skill_id or recent.outcome not in {
            SkillOutcome.FAILED,
            SkillOutcome.TIMED_OUT,
        }:
            return False
        stats = self.skills.stats.get((decision.skill_id, recent.context_key))
        return stats is not None and stats.consecutive_failures >= 2

    def _repair_repeated_failure(
        self,
        messages: tuple[ModelMessage, ...],
        decision: CognitionDecision,
        blackboard: PerceptionBlackboard,
        context: CognitionContext,
    ) -> CognitionDecision:
        failed_skill = decision.skill_id
        assert failed_skill is not None
        feasible = sorted(
            skill.skill_id
            for skill in self.skills.specs.values()
            if skill.skill_id != failed_skill
            and conditions_satisfied(skill.preconditions, blackboard)
        )
        recent = context.recent_skill_runs[0]
        self.metrics.repairs += 1
        self.metrics.retry_repairs += 1
        repair_messages = (
            *messages,
            ModelMessage(role="assistant", content=decision.model_dump_json()),
            ModelMessage(
                role="user",
                content=(
                    f"That decision is rejected by empirical execution evidence: option "
                    f"{failed_skill!r} just ended as {recent.outcome.value!r} with reason "
                    f"{recent.failure_reason!r} and has repeated consecutive failures. Do not "
                    "restart it unchanged. Select a different feasible learned option from "
                    f"{json.dumps(feasible)}, or return skill_id null, request_replan true, "
                    "and request the perception needed to choose safely. Preserve the current "
                    "goal and explicit operator action constraints."
                ),
            ),
        )
        repaired = self._complete(repair_messages)
        repaired = self._scope_operator_decision(repaired, blackboard, context)
        if repaired.skill_id is None and repaired.request_replan:
            self.metrics.last_error = None
            return repaired
        if repaired.skill_id in feasible:
            self.metrics.last_error = None
            return repaired
        self.metrics.last_error = f"repeated-option-blocked:{failed_skill}"
        return decision.model_copy(
            update={
                "reasoning_summary": (
                    f"Blocked repeated {failed_skill} after empirical timeout/failure evidence."
                ),
                "skill_id": None,
                "skill_parameters": {},
                "request_replan": True,
                "ask_perception": tuple(
                    dict.fromkeys(
                        (*decision.ask_perception, "walkable route around the obstacle")
                    )
                ),
            }
        )

    def _complete(self, messages: tuple[ModelMessage, ...]) -> CognitionDecision:
        structured = getattr(self.model, "complete_structured", None)
        if callable(structured):
            response = structured(
                messages,
                name="cognition_decision",
                schema=CognitionDecision.model_json_schema(),
            )
        else:
            response = self.model.complete(messages)
        self.metrics.calls += 1
        self.metrics.last_latency_ms = response.latency_ms
        self.metrics.last_model = response.model
        return _parse_decision(response.text)

    def _repair_infeasible(
        self,
        messages: tuple[ModelMessage, ...],
        decision: CognitionDecision,
        blackboard: PerceptionBlackboard,
        *,
        reason: str,
        missing: tuple[str, ...] = (),
    ) -> CognitionDecision:
        feasible = sorted(
            skill.skill_id
            for skill in self.skills.specs.values()
            if conditions_satisfied(skill.preconditions, blackboard)
        )
        self.metrics.repairs += 1
        repair_messages = (
            *messages,
            ModelMessage(role="assistant", content=decision.model_dump_json()),
            ModelMessage(
                role="user",
                content=(
                    f"That decision is rejected: {reason}. Select one concrete skill_id from "
                    f"this currently feasible set: {json.dumps(feasible)}. Preserve the goal "
                    "when possible, do not invent observations, and return the same strict "
                    "decision schema."
                ),
            ),
        )
        repaired = self._complete(repair_messages)
        if (
            repaired.skill_id is not None
            and repaired.skill_id in feasible
            and repaired.skill_id in self.skills.specs
        ):
            self.metrics.last_error = None
            return repaired
        self.metrics.last_error = f"infeasible-decision: {reason}"
        return decision.model_copy(
            update={
                "reasoning_summary": f"Blocked infeasible decision; {reason}",
                "skill_id": None,
                "skill_parameters": {},
                "request_replan": True,
                "ask_perception": tuple(
                    dict.fromkeys((*decision.ask_perception, *missing))
                ),
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
    return CognitionDecision.model_validate(raw)
