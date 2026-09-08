from __future__ import annotations

import json
import re
from typing import Any


from minecraft_ai.grounded_perception import _grounded_claim_keys
from minecraft_ai.perception import CognitionReadView, EvidenceRegion
from minecraft_ai.planning import Goal, GoalSource
from minecraft_ai.social import (
    OperatorMessage,
    OperatorMessageKind,
    OperatorMessageStatus,
)
from minecraft_ai.wiki import WikiEvidence


from .constants import (
    _MAX_HIGH_LEVEL_FACT_TEXT,
    _MAX_HIGH_LEVEL_FACTS,
    _MAX_HIGH_LEVEL_GOALS,
    _MAX_HIGH_LEVEL_PAYLOAD_CHARS,
    _MAX_OPERATOR_PROMPT_TEXT_CHARS,
    _MOTOR_ONLY_FACT_KEYS,
    _MOTOR_ONLY_FACT_PREFIXES,
    _URGENT_FACT_KEYS,
)
from .types import (
    CognitionContext,
    _DecisionRepairBounds,
    _WOOD_INVENTORY_AUDIT_SKILLS,
)


def _cognition_perception_keys() -> tuple[str, ...]:
    """Use the existing grounding contract, not a second planner vocabulary."""
    return _grounded_claim_keys((), set(EvidenceRegion))


def _cognition_decision_grammar(bounds: _DecisionRepairBounds) -> str:
    """Build a compact sampler-enforced grammar for one decision boundary.

    The fixed key order and absence of optional whitespace are deliberate.
    Gemma can otherwise spend an entire small generation budget emitting legal
    whitespace inside an object.  Skill IDs and parameter names are restricted
    to the deterministic authority capsule; downstream validation still owns
    values, preconditions, and operator constraints.
    """

    def literal(value: str) -> str:
        return json.dumps(json.dumps(value, ensure_ascii=True))

    allowed_skill_ids = tuple(dict.fromkeys(skill_id for skill_id, _ in bounds.allowed_skills))
    requested_skill_ids = tuple(
        skill_id
        for skill_id in dict.fromkeys(bounds.requested_skill_ids)
        if skill_id in allowed_skill_ids
    )
    skill_ids = requested_skill_ids if bounds.requested_skill_ids else allowed_skill_ids
    # Operator authority constrains which action can run, not whether an
    # inadequately grounded action must run. Null keeps observation/replanning
    # possible even when the only requested option has repeatedly failed.
    skill_alternatives = (*tuple(literal(skill_id) for skill_id in skill_ids), '"null"')
    skill_rule = " | ".join(skill_alternatives)
    question_rule = " | ".join(literal(key) for key in _cognition_perception_keys())
    parameter_names = tuple(
        dict.fromkeys(
            (
                *(
                    parameter
                    for _skill_id, parameters in bounds.allowed_skills
                    for parameter in parameters
                ),
                *(name for name, _value in bounds.required_action_constraints),
            )
        )
    )
    parameter_rules: tuple[str, ...]
    if parameter_names:
        parameter_key_rule = " | ".join(literal(name) for name in parameter_names)
        extra_entries = min(len(parameter_names), 8) - 1
        params_rule = (
            '"{" (parameter-entry)? "}"'
            if extra_entries == 0
            else f'"{{" (parameter-entry ("," parameter-entry){{0,{extra_entries}}})? "}}"'
        )
        parameter_rules = (
            f"parameter-key ::= {parameter_key_rule}",
            'parameter-entry ::= parameter-key ":" parameter-value',
            "parameter-value ::= boolean | number | parameter-string",
        )
    else:
        params_rule = '"{}"'
        parameter_rules = ()
    if bounds.authority_goal_id is None:
        goal_rule = "nullable-id"
    else:
        goal_rule = "authority-goal"
    authority_rule = (
        ()
        if bounds.authority_goal_id is None
        else (f"authority-goal ::= {literal(bounds.authority_goal_id)}",)
    )
    return "\n".join(
        (
            'root ::= "{\\"r\\":" summary ",\\"g\\":" goal '
            '",\\"s\\":" skill ",\\"p\\":" params '
            '",\\"o\\":" nullable-medium ",\\"c\\":" nullable-medium '
            '",\\"x\\":" boolean ",\\"q\\":" questions '
            '",\\"w\\":" nullable-medium ",\\"d\\":" nullable-direction '
            '",\\"n\\":" plan "}"',
            f"goal ::= {goal_rule}",
            f"skill ::= {skill_rule}",
            f"params ::= {params_rule}",
            *parameter_rules,
            *authority_rule,
            'questions ::= "[" (perception-key ("," perception-key){0,1})? "]"',
            f"perception-key ::= {question_rule}",
            'plan ::= "[" (medium-string ("," medium-string){0,4})? "]"',
            'nullable-id ::= "null" | id-string',
            'nullable-medium ::= "null" | medium-string',
            'nullable-direction ::= "null" | direction-string',
            'boolean ::= "true" | "false"',
            'number ::= "-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?',
            'summary ::= "\\"" char{0,120} "\\""',
            'id-string ::= "\\"" char{0,200} "\\""',
            'medium-string ::= "\\"" char{0,160} "\\""',
            'direction-string ::= "\\"" char{0,280} "\\""',
            'parameter-string ::= "\\"" char{0,160} "\\""',
            'char ::= [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" (["\\\\/bfnrt] | "u" [0-9a-fA-F]{4})',
        )
    )

def _high_level_fact_payload(
    blackboard: CognitionReadView,
    *,
    required_keys: set[str] | None = None,
) -> dict[str, list[object]]:
    """Return a bounded strategic view without motor-loop fingerprints.

    Per-frame hashes, luma grids, and verbose provenance are useful to the
    deterministic verifiers but waste a small local model's context. Facts are
    already freshness/confidence filtered by the blackboard; the strategic
    model only needs the value and confidence, in that order.
    """

    priority_prefixes = (
        "danger.",
        "player.",
        "environment.",
        "scene.",
        "inventory.",
        "target.",
        "social.",
        "obstacle.",
        "terrain.",
        "interaction.",
    )

    evidence_keys = required_keys or set()

    def rank(key: str) -> tuple[int, int, str]:
        if key in _URGENT_FACT_KEYS:
            reserve_rank = 0
        elif key == "social.player_message":
            reserve_rank = 1
        elif key in evidence_keys:
            reserve_rank = 2
        elif key.startswith("target."):
            reserve_rank = 3
        else:
            reserve_rank = 4
        return (
            reserve_rank,
            next(
                (index for index, prefix in enumerate(priority_prefixes) if key.startswith(prefix)),
                len(priority_prefixes),
            ),
            key,
        )

    selected = sorted(
        (
            (key, fact)
            for key, fact in blackboard.fresh_facts(min_confidence=0.35).items()
            if not key.startswith(_MOTOR_ONLY_FACT_PREFIXES)
            and key not in _MOTOR_ONLY_FACT_KEYS
        ),
        key=lambda item: rank(item[0]),
    )[:_MAX_HIGH_LEVEL_FACTS]
    payload: dict[str, list[object]] = {}
    for key, fact in selected:
        value: object = fact.value
        if isinstance(value, str):
            value = value[:_MAX_HIGH_LEVEL_FACT_TEXT]
        payload[key] = [value, round(fact.confidence, 3)]
    return payload

def _selected_goals(
    goals: tuple[Goal, ...],
    *,
    active_goal_id: str | None,
) -> tuple[Goal, ...]:
    """Keep the active/player/custom objectives ahead of standing-role filler."""

    source_rank = {
        GoalSource.PLAYER: 0,
        GoalSource.CUSTOM: 1,
        GoalSource.OPPORTUNITY: 2,
        GoalSource.PROGRESSION: 3,
        GoalSource.SURVIVAL: 4,
        GoalSource.ROLE: 5,
    }
    ranked = sorted(
        enumerate(goals),
        key=lambda item: (
            0 if item[1].goal_id == active_goal_id else 1,
            source_rank.get(item[1].source, len(source_rank)),
            -item[1].priority,
            item[1].deadline_ns is None,
            item[1].deadline_ns or 0,
            item[0],
        ),
    )
    return tuple(goal for _index, goal in ranked[:_MAX_HIGH_LEVEL_GOALS])

def _wiki_prompt_payload(item: WikiEvidence) -> dict[str, object]:
    return {
        "title": item.title[:120],
        "extract": item.extract[:600],
        "url": None if item.url is None else item.url[:240],
        "query": item.query[:120],
        "version": item.version_key[:80],
        "confidence": round(item.confidence, 3),
    }

def _compact_prompt_scalar(
    value: str | int | float | bool,
    *,
    text_limit: int = 100,
) -> str | int | float | bool:
    return value[:text_limit] if isinstance(value, str) else value

def _serialize_high_level_payload(payload: dict[str, Any]) -> str:
    """Fit optional context under a conservative local-model character budget."""

    while True:
        encoded = json.dumps(payload, separators=(",", ":"))
        if len(encoded) <= _MAX_HIGH_LEVEL_PAYLOAD_CHARS:
            return encoded

        reduced = False
        for key, minimum in (
            ("wiki_evidence", 0),
            ("operator_messages", 0),
            ("memories", 0),
            ("promises", 0),
            ("chat_lines", 0),
            ("recent_skill_runs", 1),
            ("goals", 1),
        ):
            values = payload.get(key)
            if isinstance(values, list) and len(values) > minimum:
                values.pop()
                reduced = True
                break
        if reduced:
            continue

        frame = payload.get("frame")
        if isinstance(frame, dict):
            tracks = frame.get("tracks")
            if isinstance(tracks, list) and tracks:
                tracks.pop()
                continue

        current_plan = payload.get("current_plan")
        if isinstance(current_plan, dict):
            steps = current_plan.get("steps")
            if isinstance(steps, list) and len(steps) > 1:
                steps.pop()
                continue

        facts = payload.get("fresh_facts")
        if isinstance(facts, dict) and len(facts) > 4:
            facts.popitem()
            continue

        skills = payload.get("skills")
        if isinstance(skills, list) and len(skills) > 4:
            skills.pop()
            continue

        raise ValueError("high-level prompt cannot fit the bounded local-model context")

def _operator_prompt_payload(message: OperatorMessage) -> dict[str, object]:
    """Expose operator-authored content without feeding model replies back as commands."""
    return {
        "message_id": message.message_id[:128],
        "created_ns": message.created_ns,
        "author": message.author[:64],
        "text": message.text[:_MAX_OPERATOR_PROMPT_TEXT_CHARS],
        "kind": message.kind.value,
        "priority": message.priority,
        "status": message.status.value,
    }

def _operator_prompt_metadata(message: OperatorMessage) -> dict[str, object]:
    return {
        "message_id": message.message_id[:128],
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

def _operator_requested_skill_ids(text: str) -> tuple[str, ...]:
    """Map an affirmative, concrete directive to compatible learned options.

    The mapping is intentionally conservative.  Its result may narrow a
    sampler grammar, so unsupported, negated, and interrogative language must
    retain the safe ``null`` decision instead of forcing an unrelated action.
    """

    # Operators naturally copy the public skill IDs from telemetry. Treat
    # their snake/kebab separators like spaces so a literal ``explore_forward``
    # directive takes the same zero-latency path as "explore forward" instead
    # of falling through to a slow model call that may alter its constraints.
    normalized = " ".join(
        text.casefold().replace("_", " ").replace("-", " ").split()
    )
    # A terminal "then stop and reassess" clause describes what to do after
    # the requested skill succeeds; it must not negate the affirmative action.
    # Keep this deliberately narrow so directives such as "stop mining" still
    # fall through to deliberation instead of starting a motor skill.
    normalized = re.sub(
        r"(?:,\s*)?\bthen\s+stop(?:\s+and\s+reassess"
        r"(?:\s+(?:the\s+)?(?:opening|result|situation))?)?[.!]?$",
        "",
        normalized,
    ).rstrip(" ,.!")
    if not normalized:
        return ()
    if "?" in normalized or re.match(
        r"^(?:can|could|did|do|does|how|is|may|should|tell|what|when|where|which|"
        r"who|why|will|would)\b",
        normalized,
    ):
        return ()
    if re.search(
        r"\b(?:avoid|do\s+not|don't|never|no\s+longer|refrain|stop|without)\b",
        normalized,
    ):
        return ()

    requested: list[str] = []

    def add(skill_id: str) -> None:
        if skill_id not in requested:
            requested.append(skill_id)

    if re.search(r"\b(?:respawn|come back to life)\b", normalized):
        add("respawn_after_death")
    if re.search(r"\b(?:away|afk|jump back)\b", normalized):
        add("dismiss_away_overlay")
    if re.search(r"\b(?:close|exit|leave) (?:the )?(?:inventory|menu)\b", normalized):
        add("close_open_inventory")
    elif re.search(
        r"\b(?:open|inspect|check|audit|view) (?:the )?inventory\b",
        normalized,
    ):
        add("open_inventory")
    if re.search(r"\b(?:click|activate|press)\b.*\b(?:button|control|menu)\b", normalized):
        add("activate_visible_gui_control")
    if re.search(r"\b(?:swim|surface|escape)\b.*\b(?:water|underwater|submersion)\b", normalized):
        add("escape_submersion")
    if re.search(r"\b(?:back away|escape|flee|retreat)\b", normalized):
        add("retreat_from_danger")
    if re.search(r"\b(?:attack|fight|kill)\b", normalized):
        add("attack_visible_hostile")
    if re.search(
        r"\b(?:gather|collect|chop|harvest|mine)\b.*\b(?:log|logs|tree|wood)\b", normalized
    ):
        add("gather_nearby_wood")
    elif re.search(r"\b(?:break|dig|mine)\b", normalized):
        add("mine_visible_block")
    if re.search(r"\bcraft\b.*\b(?:plank|planks)\b", normalized):
        add("craft_wood_planks")
    if re.search(r"\bcraft\b.*\bcrafting table\b", normalized):
        add("craft_crafting_table")
    if re.search(r"\bcraft\b.*\b(?:chest|chests|storage)\b", normalized):
        add("craft_storage_units")
    if re.search(
        r"\b(?:deposit|store)\b.*\b(?:chest|inventory|material|materials|storage)\b", normalized
    ):
        add("deposit_in_storage")
    if re.search(r"\bbuild\b.*\bworkshop\b", normalized):
        add("build_workshop_shell")
    if re.search(r"\bbuild\b.*\b(?:shelter|hut|house)\b", normalized):
        add("establish_basic_shelter")
    if re.search(r"\bplace\b.*\bblock\b", normalized):
        add("place_block")
    if re.search(r"\b(?:approach|go to|head to|move toward|walk to)\b", normalized):
        add("approach_visible_target")
    if re.search(r"\b(?:find|locate|reacquire)\b.*\b(?:target|tree|block|object)\b", normalized):
        add("reacquire_target")
    if re.search(r"\b(?:jump|climb|cross)\b.*\b(?:ledge|obstacle|rise|step)\b", normalized):
        add("traverse_visible_obstacle")
    if re.search(r"\bexplore\b", normalized):
        add("explore_forward")
    if re.search(
        r"\b(?:move|run|traverse|walk)\b.*\b(?:ahead|forward|ground|terrain)\b", normalized
    ):
        add("explore_forward")
        add("traverse_level_ground")
    if re.search(r"\b(?:interact with|use)\b.*\b(?:object|target)\b", normalized):
        add("use_target")
    return tuple(requested)

def _urgent_safety_required(blackboard: CognitionReadView) -> bool:
    for key in (
        "danger.immediate",
        "environment.underwater",
        "scene.death",
        "scene.away",
    ):
        fact = blackboard.fact(key, min_confidence=0.7)
        if fact is not None and bool(fact.value):
            return True
    return False


def planks_retry_requires_wood(context: CognitionContext) -> bool:
    """A fresh explicit wood-audit command permits one attempt, not a permanent bypass."""
    if not context.planks_retry_requires_wood:
        return False
    active = next(
        (
            message
            for message in context.operator_messages
            if message.kind in {OperatorMessageKind.INSTRUCTION, OperatorMessageKind.CORRECTION}
        ),
        None,
    )
    if (
        active is not None
        and active.status in {OperatorMessageStatus.QUEUED, OperatorMessageStatus.DELIVERED}
        and _WOOD_INVENTORY_AUDIT_SKILLS.intersection(_operator_requested_skill_ids(active.text))
    ):
        return False
    return True
