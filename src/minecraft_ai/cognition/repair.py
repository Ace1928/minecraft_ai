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
from .prompts import _operator_question_perception_keys
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
        ("d", None if decision.instruction is None else decision.instruction[:280]),
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
    if bounds.reply_only:
        _validate_reply_only_decision(decision, bounds)
        return decision
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


def _validate_reply_only_decision(
    decision: CognitionDecision, bounds: _DecisionRepairBounds,
) -> None:
    if (
        decision.chosen_goal_id != bounds.authority_goal_id
        or decision.skill_id is not None
        or decision.skill_parameters
        or decision.game_chat is not None
        or decision.request_replan
        or decision.research_query is not None
        or decision.instruction is not None
        or decision.plan_steps
        or not isinstance(decision.say, str)
        or not decision.say.strip()
        or len(decision.say) > 160
        or len(decision.ask_perception) > 2
        or any(key not in _operator_question_perception_keys() for key in decision.ask_perception)
    ):
        raise ValueError("operator_question_contract_failed")


def _reply_only_decision_from_response(
    response: ModelResponse, bounds: _DecisionRepairBounds,
) -> CognitionDecision:
    """Parse a compact reply-only contract without granting action authority.

    The native World has a 128-unit generation ceiling. Its compact reply-only
    wire form carries only the exact goal binding and operator text; omitted
    action fields take safe defaults. Expanded responses remain accepted only
    after every forbidden field is checked. Invalid output is not retried.
    """
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("operator_question_duplicate_field")
            result[key] = value
        return result

    try:
        raw = json.loads(response.text, object_pairs_hook=unique_object)
    except ValueError as exc:
        raise ValueError("operator_question_invalid_json") from exc
    if isinstance(raw, dict) and set(raw) in ({"g", "o"}, {"g", "o", "q"}):
        if raw.get("g") != bounds.authority_goal_id:
            raise ValueError("operator_question_goal_mismatch")
        if not isinstance(raw.get("o"), str) or not raw["o"].strip() or len(raw["o"]) > 160:
            raise ValueError("operator_question_contract_failed")
        questions = raw.get("q", [])
        if (
            type(questions) is not list or len(questions) > 2
            or any(type(key) is not str for key in questions)
            or len(set(questions)) != len(questions)
        ):
            raise ValueError("operator_question_contract_failed")
        compact = CognitionDecision(
            reasoning_summary="Answered the active operator question.",
            chosen_goal_id=raw["g"],
            say=raw["o"],
            ask_perception=tuple(questions),
        )
        _validate_reply_only_decision(compact, bounds)
        origin = response.request_attempt
        if origin is not None:
            compact._model_origin = DecisionModelOrigin(
                origin.request_id, origin.attempt_id, cognition_decision_sha256(compact),
            )
        return compact
    if not isinstance(raw, dict) or set(raw) != set("rgspocxqwdn"):
        raise ValueError("operator_question_wire_fields")
    if (
        any(raw[key] is not None for key in ("s", "c", "w", "d"))
        or raw["p"] != {}
        or raw["n"] != []
        or raw["x"] is not False
        or not isinstance(raw["q"], list)
    ):
        raise ValueError("operator_question_forbidden_fields")
    decision = _decision_from_response(response)
    _validate_reply_only_decision(decision, bounds)
    return decision

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
