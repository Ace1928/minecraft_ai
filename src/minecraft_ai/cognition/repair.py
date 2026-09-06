from __future__ import annotations

import json


from minecraft_ai.models import ModelMessage, ModelResponse
from minecraft_ai.skills import SkillLibrary


from .constants import (
    _JSON_REPAIR_SYSTEM,
    _MAX_REJECTED_OUTPUT_CHARS,
    _MAX_REPAIR_REASON_CHARS,
    _SEMANTIC_REPAIR_SYSTEM,
)
from .types import (
    CognitionDecision,
    DecisionModelOrigin,
    _CognitionWireDecision,
    _DecisionRepairBounds,
    cognition_decision_sha256,
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
    requested_skills = {
        skill_id for skill_id in bounds.requested_skill_ids if skill_id in allowed_parameters
    }
    required_constraints: dict[str, str | int | float | bool] = dict(
        bounds.required_action_constraints
    )
    goal_id = bounds.authority_goal_id or decision.chosen_goal_id
    violates_requested_skill = (
        decision.skill_id is not None
        and bool(bounds.requested_skill_ids)
        and decision.skill_id not in requested_skills
    )
    if (
        decision.skill_id is not None and decision.skill_id not in allowed_parameters
    ) or violates_requested_skill:
        return CognitionDecision(
            reasoning_summary="Decision violated the allowed option bounds.",
            chosen_goal_id=goal_id,
            skill_parameters=required_constraints,
            request_replan=True,
            ask_perception=decision.ask_perception[:2],
        )
    if decision.skill_id is None:
        parameters = required_constraints
        if bounds.requested_skill_ids:
            # An abstention is not acknowledgement that the requested action
            # was accepted. Keep the directive pending while seeking evidence.
            decision = decision.model_copy(update={"request_replan": True})
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

def _bound_skill_parameters(
    decision: CognitionDecision,
    skills: SkillLibrary,
    *,
    required_constraints: dict[str, str | int | float | bool],
) -> CognitionDecision:
    """Keep model bindings inside the selected option's parameter contract."""

    if decision.skill_id is None or decision.skill_id not in skills.specs:
        return decision
    permitted = set(skills.get(decision.skill_id).parameters) | set(required_constraints)
    parameters = {
        key: value for key, value in decision.skill_parameters.items() if key in permitted
    }
    parameters.update(required_constraints)
    return decision.model_copy(update={"skill_parameters": parameters})

def _decision_from_response(response: ModelResponse) -> CognitionDecision:
    decision = _parse_decision(response.text)
    origin = response.request_attempt
    if origin is not None:
        decision._model_origin = DecisionModelOrigin(
            origin.request_id, origin.attempt_id, cognition_decision_sha256(decision),
        )
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
    if isinstance(raw, dict) and any(key in raw for key in ("r", "g", "s", "p")):
        return _CognitionWireDecision.model_validate(raw).expand()
    return CognitionDecision.model_validate(raw)

