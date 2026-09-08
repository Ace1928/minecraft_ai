from __future__ import annotations

import json
import re
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, cast

from pydantic import ValidationError

from minecraft_ai.execution import initiation_satisfied
from minecraft_ai.grounded_perception import resolve_grounded_output_keys
from minecraft_ai.model_requests import ModelRequestLifecycle
from minecraft_ai.models import LanguageModel, ModelMessage, ModelRequestAttempt, ModelResponse
from minecraft_ai.perception import CognitionReadView
from minecraft_ai.skills import SkillFailureCode, SkillLibrary, SkillOutcome, SkillRun
from minecraft_ai.social import (
    OperatorMessageKind,
    OperatorMessageStatus,
)


from .bootstrap import BootstrapCognitionPolicy
from .constants import _MAX_OPERATOR_FAST_PATH_INSTRUCTION_CHARS
from .prompts import (
    _cognition_decision_grammar,
    _cognition_perception_keys,
    _compact_prompt_scalar,
    _explicit_action_constraints,
    _high_level_fact_payload,
    _operator_prompt_metadata,
    _operator_prompt_payload,
    _operator_requested_skill_ids,
    _selected_goals,
    _serialize_high_level_payload,
    _urgent_safety_required,
    _wiki_prompt_payload,
    planks_retry_requires_wood,
)
from .repair import (
    _bound_skill_parameters,
    _decision_from_response,
    _enforce_repair_bounds,
    _json_repair_messages,
    _semantic_repair_messages,
)
from .types import (
    CognitionContext,
    CognitionDecision,
    HighLevelMetrics,
    _CognitionWireDecision,
    _DecisionRepairBounds,
    _WOOD_INVENTORY_AUDIT_SKILLS,
    _without_model_origin,
)


@dataclass
class HighLevelController:
    model: LanguageModel
    skills: SkillLibrary
    _bootstrap: BootstrapCognitionPolicy = field(init=False)
    metrics: HighLevelMetrics = field(default_factory=HighLevelMetrics, init=False)
    _request_context: ContextVar[ModelRequestLifecycle | None] = field(
        default_factory=lambda: ContextVar("cognition_request", default=None),
        init=False, repr=False,
    )

    def __post_init__(self) -> None:
        self._bootstrap = BootstrapCognitionPolicy(self.skills)

    def decide(
        self,
        blackboard: CognitionReadView,
        context: CognitionContext,
        *,
        request: ModelRequestLifecycle | None = None,
    ) -> CognitionDecision:
        # This entry runs inside the worker. Thread-pool submission context is
        # deliberately irrelevant, and even a legacy nested call clears it.
        token = self._request_context.set(request)
        try:
            return self._decide(blackboard, context)
        finally:
            self._request_context.reset(token)

    def _decide(
        self,
        blackboard: CognitionReadView,
        context: CognitionContext,
    ) -> CognitionDecision:
        try:
            operator_fast_path = self._operator_fast_path_decision(blackboard, context)
            if operator_fast_path is not None:
                self.metrics.last_error = None
                return operator_fast_path
            latest = blackboard.latest()
            active_operator = (
                None
                if _urgent_safety_required(blackboard) or not context.operator_messages
                else context.operator_messages[0]
            )
            planning_query = (
                active_operator.text
                if active_operator is not None
                else " ".join(
                    (
                        *(goal.description for goal in context.goals[:3]),
                        *context.current_plan[context.plan_index : context.plan_index + 2],
                    )
                )
            )
            feasible_skill_payloads = self._feasible_skill_payloads(
                blackboard,
                query_text=planning_query,
                context=context,
            )
            # The canonical visible log count is strategic state even though
            # exact gather completion is a three-event transaction rather than
            # an absolute SkillCondition.
            required_fact_keys: set[str] = {"inventory.hotbar.logs"}
            for skill_payload in feasible_skill_payloads:
                evidence_items = skill_payload.get("success_evidence")
                if not isinstance(evidence_items, list):
                    continue
                for evidence in evidence_items:
                    if isinstance(evidence, dict) and isinstance(evidence.get("fact"), str):
                        required_fact_keys.add(str(evidence["fact"]))
            repair_bounds = self._decision_repair_bounds(
                blackboard,
                context,
                allowed_skill_ids={str(payload["skill_id"]) for payload in feasible_skill_payloads},
            )
            facts = _high_level_fact_payload(
                blackboard,
                required_keys=required_fact_keys,
            )
            selected_goals = _selected_goals(
                context.goals,
                active_goal_id=context.plan_goal_id,
            )
            payload: dict[str, Any] = {
                "role": {
                    "id": context.role.role_id[:128],
                    "goals": [goal[:80] for goal in context.role.standing_goals[:8]],
                    "weights": {
                        key[:64]: value
                        for key, value in sorted(context.role.utility_weights.items())[:12]
                    },
                    "risk": context.role.risk_tolerance,
                },
                "goals": [
                    {
                        "id": goal.goal_id[:80],
                        "description": goal.description[:120],
                        "target": None if goal.target_node is None else goal.target_node[:64],
                        "source": goal.source.value,
                        "priority": goal.priority,
                        "domain": goal.domain[:32],
                    }
                    for goal in selected_goals
                ],
                "memories": [
                    {
                        "id": memory.memory_id[:64],
                        "kind": memory.kind.value,
                        "text": memory.text[:120],
                        "confidence": memory.confidence,
                        "importance": memory.importance,
                        "goals": [tag[:40] for tag in memory.goal_tags[:2]],
                        "entities": [tag[:40] for tag in memory.entity_tags[:2]],
                        "place": (
                            None if memory.location_key is None else memory.location_key[:64]
                        ),
                    }
                    for memory in context.memories[:4]
                ],
                "promises": [
                    {
                        "id": promise.promise_id[:64],
                        "player": promise.player[:40],
                        "summary": promise.summary[:120],
                        "status": promise.status.value,
                        "goal": None if promise.goal_id is None else promise.goal_id[:64],
                        "project": (
                            None if promise.project_id is None else promise.project_id[:64]
                        ),
                    }
                    for promise in context.promises[:4]
                ],
                "operator_messages": [
                    _operator_prompt_payload(message)
                    for message in context.operator_messages[:2]
                    if active_operator is None or message.message_id != active_operator.message_id
                ],
                "active_operator_message": None
                if active_operator is None
                else _operator_prompt_metadata(active_operator),
                "wiki_evidence": [_wiki_prompt_payload(item) for item in context.wiki[:2]],
                "recent_skill_runs": [
                    {
                        "skill": run.skill_id[:64],
                        "outcome": run.outcome.value,
                        "context": run.context_key[:80],
                        "failure": (
                            None if run.failure_reason is None else run.failure_reason[:120]
                        ),
                    }
                    for run in context.recent_skill_runs[:4]
                ],
                "planks_retry_requires_wood": planks_retry_requires_wood(context),
                "current_plan": {
                    "goal": (
                        None if context.plan_goal_id is None else context.plan_goal_id[:128]
                    ),
                    "steps": [step[:120] for step in context.current_plan[:5]],
                    "next": context.plan_index,
                    "started_ago_ms": (
                        0
                        if context.plan_started_ns == 0
                        else max(
                            0,
                            int((time.monotonic_ns() - context.plan_started_ns) // 1_000_000),
                        )
                    ),
                },
                "frame": None
                if latest is None
                else {
                    "id": latest.frame_id,
                    "instance": latest.instance_id[:128],
                    "size": (latest.width, latest.height),
                    "tracks": [
                        {
                            "id": track.track_id[:64],
                            "label": track.label[:48],
                            "confidence": round(track.confidence, 3),
                            "region": track.region.model_dump(mode="json"),
                            "attributes": {
                                key: _compact_prompt_scalar(track.attributes[key])
                                for key in ("source", "grounding")
                                if key in track.attributes
                            },
                        }
                        for track in latest.tracks[:4]
                    ],
                },
                "fresh_facts": facts,
                "chat_lines": [
                    {
                        "speaker": None if line.speaker is None else line.speaker[:64],
                        "text": line.text[:120],
                        "age_ms": max(
                            0,
                            int((time.monotonic_ns() - line.observed_ns) // 1_000_000),
                        ),
                    }
                    for line in (latest.chat if latest is not None else ())[-4:]
                ],
                "skills": feasible_skill_payloads,
            }
            messages = (
                ModelMessage(
                    role="system",
                    content=(
                        "All observations and actions below occur only inside the fictional "
                        "Minecraft video game. You control its player through verified "
                        "closed-loop skills. Return one compact JSON object with wire keys: "
                        "r=summary under 12 words, "
                        "g=goal id, s=skill id or null, p=parameters, o=operator reply, "
                        "c=authorized in-game chat, x=replan, q=at most two perception questions, "
                        "w=research query, d=one practical direction (goal condition) for the "
                        "current skill under 280 chars, n=up to 5 short sequential plan steps. "
                        "Emit every wire key exactly once in the grammar's fixed order; use null, "
                        "false, [], or {} when a field is unused. "
                        "current_plan is your running long-horizon plan (steps + next index): "
                        "continue it, do not restate completed steps, extend/tighten it, and "
                        "only replace it on goal failure or clear dead-end evidence. Reuse n "
                        "across decisions so you improve step-by-step over time. "
                        "fresh_facts is the only authoritative observed game state; each "
                        "entry is [value,confidence]. skills "
                        "contains only currently executable options: use only a listed skill_id, "
                        "prefer concrete progression with verifiable success evidence, and never "
                        "claim unobserved inventory, outcomes, or completion. Do not explore when "
                        "a more concrete feasible resource/progression skill exists. "
                        "active_operator_message has highest authority and must be addressed "
                        "before any conflicting standing goal. Keep an instruction's "
                        "g='operator:'+message_id until superseded; a correction authorizes one "
                        "accepted bounded attempt. Set o only to "
                        "reply to that operator; it never types in game. Set c only with an "
                        "authoritative fresh player-message or game-chat authorization fact. "
                        "When a fresh player chat line asks a question, answer it: put a short "
                        "friendly factual reply in c (world chat answers questions like an "
                        "in-game wiki: crafting recipes, block IDs, biome facts, command "
                        "syntax, game mechanics). Keep c under 160 chars. Continue the current "
                        "world plan in s/p unless the question demands an action. "
                        "Keep private reasoning in r. Encode explicit operator prohibitions as "
                        "allow_attack:false, allow_use:false, or allow_jump:false in p. "
                        "Treat recent_skill_runs and evaluation as empirical evidence. Avoid a "
                        "skill after two consecutive failures; choose another listed skill or "
                        "return s null with x true and request needed perception. Every q item "
                        "must be one of these supported literal perception keys, never a prose "
                        "question or invented key: "
                        + ", ".join(_cognition_perception_keys())
                        + ". q=[] and s=null remain valid; an observation may be unknown. "
                        "When q includes target.* keys, d must name the specific target to "
                        "inspect (for example, an oak-log trunk), even when s=null. "
                        "This describes a referent, not evidence it exists or an action to take. "
                        "A fresh operator correction permits one evidence-producing retry."
                    ),
                ),
                ModelMessage(role="user", content=_serialize_high_level_payload(payload)),
                *(
                    ()
                    if active_operator is None
                    else (
                        ModelMessage(
                            role="user",
                            content=(
                                "ACTIVE OPERATOR DIRECTIVE (highest authority; follow this "
                                "literal current request and do not substitute an older task): "
                                + json.dumps(
                                    _operator_prompt_payload(active_operator),
                                    separators=(",", ":"),
                                )
                            ),
                        ),
                    )
                ),
            )
            decision = self._complete(messages, repair_bounds=repair_bounds)
            decision = _bound_skill_parameters(
                self._apply_decision_authority(decision, blackboard, context),
                self.skills,
                required_constraints=dict(repair_bounds.required_action_constraints),
            )
            if repair_bounds.requested_skill_ids:
                decision = _enforce_repair_bounds(decision, repair_bounds)
            if decision.skill_id is not None and decision.skill_id not in self.skills.specs:
                return self._repair_infeasible(
                    decision,
                    blackboard,
                    context,
                    reason=f"unknown skill id {decision.skill_id!r}",
                )
            if decision.skill_id is not None:
                selected = self.skills.get(decision.skill_id)
                if selected.skill_id in _WOOD_INVENTORY_AUDIT_SKILLS and planks_retry_requires_wood(
                    context
                ):
                    return self._repair_infeasible(
                        decision,
                        blackboard,
                        context,
                        reason=(
                            "wood inventory audit needs positive log evidence after inventory "
                            "had none"
                        ),
                    )
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
            if (
                decision.skill_id is None
                and (decision.request_replan or decision.ask_perception)
                and not any(
                    resolve_grounded_output_keys((), question)
                    for question in decision.ask_perception
                )
                and decision.research_query is None
            ):
                # A valid top-level abstention bypasses the repair path. It
                # still needs new evidence before another slow model call;
                # empty or entirely invented fact keys otherwise repeat forever.
                # Runtime treats any question as a replan, even when the model
                # omitted x=true. Apply the same rule before resolving its keys.
                candidates = repair_bounds.requested_skill_ids or tuple(
                    run.skill_id
                    for run in context.recent_skill_runs
                    if run.outcome in {SkillOutcome.FAILED, SkillOutcome.TIMED_OUT}
                    and run.failure_code != SkillFailureCode.CONTROLLER_STARVATION
                )
                if candidates:
                    decision = decision.model_copy(update={
                        "ask_perception": self._prerequisite_perception_keys(candidates[0]),
                        "request_replan": True,
                    })
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

    def _operator_fast_path_decision(
        self,
        blackboard: CognitionReadView,
        context: CognitionContext,
    ) -> CognitionDecision | None:
        """Execute one literal, unambiguous operator option without model latency."""
        if _urgent_safety_required(blackboard):
            return None
        active = next(
            (
                message
                for message in context.operator_messages
                if message.kind
                in {OperatorMessageKind.INSTRUCTION, OperatorMessageKind.CORRECTION}
            ),
            None,
        )
        if active is None or active.status not in {
            OperatorMessageStatus.QUEUED,
            OperatorMessageStatus.DELIVERED,
        }:
            return None
        if len(active.text) > _MAX_OPERATOR_FAST_PATH_INSTRUCTION_CHARS:
            return None

        feasible_skill_ids = tuple(
            skill_id
            for skill_id in _operator_requested_skill_ids(active.text)
            if skill_id in self.skills.specs
            and initiation_satisfied(self.skills.get(skill_id), blackboard)
        )
        if len(feasible_skill_ids) != 1:
            return None

        parameters: dict[str, str | int | float | bool] = dict(
            _explicit_action_constraints(active.text)
        )
        decision = CognitionDecision(
            reasoning_summary="Following an explicit operator instruction.",
            chosen_goal_id=f"operator:{active.message_id}",
            skill_id=feasible_skill_ids[0],
            skill_parameters=parameters,
            say="Starting that now.",
            instruction=active.text,
        )
        return self._scope_operator_decision(decision, blackboard, context)

    def _apply_decision_authority(
        self,
        decision: CognitionDecision,
        blackboard: CognitionReadView,
        context: CognitionContext,
    ) -> CognitionDecision:
        decision = self._scope_operator_decision(decision, blackboard, context)
        if (
            decision.skill_id is not None
            and decision.ask_perception
            and not any(
                resolve_grounded_output_keys((), question)
                for question in decision.ask_perception
            )
        ):
            # Runtime defers skill+question decisions too. An invented question
            # must not strand that selected skill outside the null fallback.
            decision = decision.model_copy(update={
                "ask_perception": self._prerequisite_perception_keys(decision.skill_id),
            })
        operator_goal_ids = {
            f"operator:{message.message_id}" for message in context.operator_messages
        }
        if decision.say is not None and decision.chosen_goal_id not in operator_goal_ids:
            return decision.model_copy(update={"say": None})
        return decision

    def _scope_operator_decision(
        self,
        decision: CognitionDecision,
        blackboard: CognitionReadView,
        context: CognitionContext,
    ) -> CognitionDecision:
        if _urgent_safety_required(blackboard):
            operator_goal_ids = {
                f"operator:{message.message_id}" for message in context.operator_messages
            }
            updates: dict[str, object] = {"say": None}
            if context.operator_messages:
                updates["request_replan"] = True
            if decision.chosen_goal_id in operator_goal_ids:
                updates["chosen_goal_id"] = None
            return decision.model_copy(update=updates)
        active = next(
            (
                message
                for message in context.operator_messages
                if message.kind in {OperatorMessageKind.INSTRUCTION, OperatorMessageKind.CORRECTION}
            ),
            None,
        )
        if active is None:
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
        blackboard: CognitionReadView,
        *,
        query_text: str = "",
        context: CognitionContext | None = None,
    ) -> list[dict[str, object]]:
        ranked: list[tuple[int, float, float, str, dict[str, object]]] = []
        stop_words = {
            "and",
            "are",
            "for",
            "from",
            "into",
            "that",
            "the",
            "then",
            "this",
            "through",
            "with",
        }

        def planning_tokens(text: str) -> set[str]:
            tokens: set[str] = set()
            for token in re.findall(r"[a-z0-9]+", text.casefold()):
                if len(token) < 3 or token in stop_words:
                    continue
                tokens.add(token)
                if len(token) > 4 and token.endswith("s"):
                    tokens.add(token[:-1])
                if len(token) > 5 and token.endswith("ing"):
                    tokens.update((token[:-3], token[:-3] + "e"))
            return tokens

        query_tokens = planning_tokens(query_text)
        requested_skill_ids = set(_operator_requested_skill_ids(query_text))
        safety_skills = {
            "dismiss_away_overlay",
            "escape_submersion",
            "respawn_after_death",
            "retreat_from_danger",
        }
        for skill in self.skills.specs.values():
            if (
                skill.skill_id in _WOOD_INVENTORY_AUDIT_SKILLS
                and context is not None
                and planks_retry_requires_wood(context)
            ):
                continue
            if not initiation_satisfied(skill, blackboard):
                continue
            stats = self.skills.stats.get((skill.skill_id, "default"))
            identity_tokens = planning_tokens(" ".join((skill.skill_id, skill.name)))
            description_tokens = planning_tokens(skill.description)
            overlap = 4 * len(query_tokens & identity_tokens) + len(
                query_tokens & description_tokens
            )
            competence = 0.5 if stats is None else self.skills.contextual_score(skill.skill_id)
            failure_penalty = 0.0 if stats is None else min(0.45, 0.08 * stats.consecutive_failures)
            ranking_score = max(0.0, competence - failure_penalty)
            payload = {
                "skill_id": skill.skill_id,
                "description": skill.description[:120],
                "parameters": list(skill.parameters),
                "success_evidence": [
                    {
                        "fact": condition.key,
                        "op": condition.operator,
                        "value": condition.value,
                    }
                    for condition in skill.success_conditions[:3]
                ],
                "effects": list(skill.expected_effects[:3]),
                "competence": round(ranking_score, 3),
                "evaluation": (
                    None
                    if stats is None
                    else {
                        "attempts": stats.attempts,
                        "successes": stats.successes,
                        "consecutive_failures": stats.consecutive_failures,
                    }
                ),
            }
            if skill.skill_id in safety_skills:
                rank_group = 0
            elif skill.skill_id in requested_skill_ids:
                rank_group = 1
            elif overlap > 0:
                rank_group = 2
            else:
                rank_group = 3
            ranked.append(
                (
                    rank_group,
                    -float(overlap),
                    -ranking_score,
                    skill.skill_id,
                    payload,
                )
            )
        ranked.sort(key=lambda item: item[:4])
        safety = [item[4] for item in ranked if item[0] == 0]
        if safety and _urgent_safety_required(blackboard):
            return safety[:8]
        return [item[4] for item in ranked[:6]]

    def _decision_repair_bounds(
        self,
        blackboard: CognitionReadView,
        context: CognitionContext,
        *,
        allowed_skill_ids: set[str] | None = None,
    ) -> _DecisionRepairBounds:
        active = None
        if not _urgent_safety_required(blackboard):
            active = next(
                (
                    message
                    for message in context.operator_messages
                    if message.kind
                    in {OperatorMessageKind.INSTRUCTION, OperatorMessageKind.CORRECTION}
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
            and not (
                skill.skill_id in _WOOD_INVENTORY_AUDIT_SKILLS
                and planks_retry_requires_wood(context)
            )
            and (allowed_skill_ids is None or skill.skill_id in allowed_skill_ids)
        )
        constraints = (
            () if active is None else tuple(_explicit_action_constraints(active.text).items())
        )
        requested_skill_ids = () if active is None else _operator_requested_skill_ids(active.text)
        return _DecisionRepairBounds(
            allowed_skills=allowed_skills,
            authority_goal_id=None if active is None else f"operator:{active.message_id}",
            required_action_constraints=constraints,
            requested_skill_ids=requested_skill_ids,
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
            if (
                recent.skill_id == "craft_wood_planks"
                and recent.failure_reason == "crafting-no-logs-observed-in-inventory"
                and not planks_retry_requires_wood(context)
            ):
                continue  # New possession evidence repaired this particular prerequisite.
            stats = self.skills.stats.get((decision.skill_id, recent.context_key))
            if stats is not None and stats.consecutive_failures >= 2:
                return recent
        return None

    def _recently_blocked_skill_ids(self, context: CognitionContext) -> set[str]:
        blocked: set[str] = set()
        for run in context.recent_skill_runs:
            if run.outcome not in {SkillOutcome.FAILED, SkillOutcome.TIMED_OUT}:
                continue
            if (
                run.skill_id == "craft_wood_planks"
                and run.failure_reason == "crafting-no-logs-observed-in-inventory"
                and not planks_retry_requires_wood(context)
            ):
                continue
            stats = self.skills.stats.get((run.skill_id, run.context_key))
            if stats is not None and stats.consecutive_failures >= 2:
                blocked.add(run.skill_id)
        return blocked

    def _repair_repeated_failure(
        self,
        decision: CognitionDecision,
        blackboard: CognitionReadView,
        context: CognitionContext,
        blocked_run: SkillRun,
    ) -> CognitionDecision:
        failed_skill = decision.skill_id
        assert failed_skill is not None
        blocked_skill_ids = self._recently_blocked_skill_ids(context)
        feasible = sorted(
            skill.skill_id
            for skill in self.skills.specs.values()
            if skill.skill_id not in blocked_skill_ids
            and initiation_satisfied(skill, blackboard)
            and not (
                skill.skill_id in _WOOD_INVENTORY_AUDIT_SKILLS
                and planks_retry_requires_wood(context)
            )
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
            if not any(
                resolve_grounded_output_keys((), question)
                for question in repaired.ask_perception
            ):
                repaired = repaired.model_copy(
                    update={"ask_perception": self._prerequisite_perception_keys(failed_skill)}
                )
            self.metrics.last_error = None
            return _enforce_repair_bounds(repaired, repair_bounds)
        if repaired.skill_id in feasible:
            self.metrics.last_error = None
            return _enforce_repair_bounds(repaired, repair_bounds)
        self.metrics.last_error = f"repeated-option-blocked:{failed_skill}"
        return _without_model_origin(decision.model_copy(
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
        ))

    def _prerequisite_perception_keys(self, skill_id: str | None) -> tuple[str, ...]:
        """Ask for canonical visual prerequisites, never internal run witnesses."""
        spec = None if skill_id is None else self.skills.specs.get(skill_id)
        keys = tuple(
            key
            for condition in (() if spec is None else spec.preconditions)
            for key in resolve_grounded_output_keys((), condition.key)
        ) or ("obstacle.ahead",)
        return tuple(dict.fromkeys(keys))[:2]

    def _complete(
        self,
        messages: tuple[ModelMessage, ...],
        *,
        repair_bounds: _DecisionRepairBounds,
    ) -> CognitionDecision:
        response = self._request_model(
            messages,
            name="cognition_decision",
            repair_bounds=repair_bounds,
        )
        try:
            return _decision_from_response(response)
        except (RuntimeError, ValidationError):
            self.metrics.repairs += 1
            self.metrics.json_repairs += 1
            repair_messages = _json_repair_messages(response.text, repair_bounds)
            repaired_response = self._request_model(
                repair_messages,
                name="cognition_decision_json_repair",
                repair_bounds=repair_bounds,
            )
            try:
                repaired = _decision_from_response(repaired_response)
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
        repair_bounds: _DecisionRepairBounds,
    ) -> ModelResponse:
        request = self._request_context.get()
        bound = getattr(self.model, "complete_bound_constrained", None)
        constrained = getattr(self.model, "complete_constrained", None)
        structured = getattr(self.model, "complete_structured", None)
        use_bound = request is not None and callable(bound)
        schema = (
            _CognitionWireDecision.model_json_schema()
            if use_bound or callable(constrained) or callable(structured) else {}
        )
        if schema:
            schema["properties"]["q"]["items"]["enum"] = list(_cognition_perception_keys())
        grammar = (
            _cognition_decision_grammar(repair_bounds)
            if use_bound or callable(constrained) else ""
        )
        attempt_id = None if request is None else request.start_attempt(name)
        try:
            if request is not None and callable(bound):
                response = cast(
                    ModelResponse,
                    bound(
                        messages, name=name, schema=schema, grammar=grammar,
                        request=request.binding, attempt_id=attempt_id,
                    ),
                )
            elif callable(constrained):
                response = cast(
                    ModelResponse,
                    constrained(messages, name=name, schema=schema, grammar=grammar),
                )
            elif callable(structured):
                response = cast(
                    ModelResponse,
                    structured(messages, name=name, schema=schema),
                )
            else:
                response = self.model.complete(messages)
            if not isinstance(response, ModelResponse):
                raise TypeError("model adapter must return ModelResponse")
            # Adapters may reuse a response instance. Never mutate its metadata
            # or trust an origin supplied by an adapter or an earlier call.
            response = response.model_copy()
            response._request_attempt = (
                ModelRequestAttempt(request.binding.request_id, attempt_id)
                if request is not None and attempt_id is not None else None
            )
        except BaseException as error:
            if request is not None and attempt_id is not None:
                request.finish_attempt(attempt_id, type(error).__name__)
            raise
        if request is not None and attempt_id is not None:
            request.finish_attempt(attempt_id)
        self.metrics.calls += 1
        self.metrics.last_latency_ms = response.latency_ms
        self.metrics.last_model = response.model
        return response

    def _repair_infeasible(
        self,
        decision: CognitionDecision,
        blackboard: CognitionReadView,
        context: CognitionContext,
        *,
        reason: str,
        missing: tuple[str, ...] = (),
    ) -> CognitionDecision:
        feasible = sorted(
            skill.skill_id
            for skill in self.skills.specs.values()
            if initiation_satisfied(skill, blackboard)
            and not (
                skill.skill_id in _WOOD_INVENTORY_AUDIT_SKILLS
                and planks_retry_requires_wood(context)
            )
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
        return _without_model_origin(decision.model_copy(
            update={
                "reasoning_summary": f"Blocked infeasible decision; {reason}",
                "skill_id": None,
                "skill_parameters": dict(repair_bounds.required_action_constraints),
                "request_replan": True,
                "ask_perception": tuple(dict.fromkeys((*decision.ask_perception, *missing))),
            }
        ))
