"""Strict Minecraft cognition adapter for the shared ERAIS Native World owner.

The adapter translates the planner's bounded decision request into the native
World API's intentionally small chat contract. It does not own a model, fall
back to another provider, or grant an action; the existing Minecraft decision
parser and authority checks remain downstream.
"""

from __future__ import annotations

import json
import os
import re
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from .models import ModelMessage, ModelResponse, local_model_inference_lane

MODEL_ID = "erais-native-qwen3"
MAX_PROMPT_BYTES = 2048
MAX_OUTPUT_TOKENS = 128
_RUNTIME_ID = re.compile(r"[0-9a-f]{32}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1"}

_PLANNER_RULES = (
    "Minecraft planner. Only fresh_facts proves current state. Follow the literal "
    "active_operator_directive first; obey authority_bounds and safe_fallback. "
    "Use listed goal/skill IDs and parameters only. Never invent facts or permission. "
    "Game chat needs fresh player authorization and wiki_evidence. If unsafe or "
    "unknown, use s=null and admitted q observations. Return only JSON in order "
    "r,g,s,p,o,c,x,q,w,d,n: "
    '{"r":"brief","g":null,"s":null,"p":{},"o":null,"c":null,'
    '"x":false,"q":[],"w":null,"d":null,"n":[]}'
    "\nContext:"
)
_REPLY_ONLY_RULES = (
    "ERAIS World Minecraft operator reply. Use only fresh_facts as observed truth. "
    "If facts do not answer the question, say fresh evidence is unavailable. Do not "
    "invent observations or propose actions. Return exactly one JSON object with keys "
    "g, o and q. Use the exact supplied goal ID for g; keep o under 160 characters. "
    "q is an empty list or at most two admitted perception keys.\nContext:"
)
_STRUCTURED_REPLY_ONLY_RULES = (
    "Reply to the literal operator_question using fresh_facts only; missing "
    "evidence means uncertainty. No game action/chat/plan/research/direction. "
    "Return only JSON in order r,g,s,p,o,c,x,q,w,d,n; use the exact supplied g, "
    "s=null,p={},c=null,x=false,w=null,d=null,n=[]. Keep o under160 characters; "
    "q is [] or up to2 admitted read-only observations.\nContext:"
)


def _short(value: object, limit: int) -> object:
    if isinstance(value, str):
        return value[:limit]
    if type(value) in {int, float, bool} or value is None:
        return value
    if isinstance(value, list):
        return [_short(item, limit) for item in value[:2]]
    return None


def _compact_context(
    messages: tuple[ModelMessage, ...], *, authority_bounds: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], bool]:
    payload: dict[str, Any] = {}
    directive: dict[str, Any] | None = None
    auxiliary_payloads: list[dict[str, Any]] = []
    reply_only = False
    reply_only_goal_id: str | None = None
    free_text: list[str] = []
    for message in messages:
        if message.role == "system":
            reply_only = reply_only or "response grants no game action authority" in message.content
            continue
        content = message.content
        marker = (
            "ACTIVE OPERATOR DIRECTIVE (highest authority; follow this literal current "
            "request and do not substitute an older task): "
        )
        if content.startswith(marker):
            try:
                value = json.loads(content[len(marker) :])
                if type(value) is dict:
                    directive = value
            except (ValueError, TypeError):
                free_text.append(content[:500])
            continue
        try:
            value = json.loads(content)
        except (ValueError, TypeError):
            free_text.append(content[:500])
            continue
        if type(value) is dict and any(
            key in value for key in ("fresh_facts", "skills", "active_operator_message")
        ):
            payload = value
        elif type(value) is dict:
            auxiliary_payloads.append(value)
    result: dict[str, Any] = {}
    reference = payload.get("configured_recipe_reference")
    if type(reference) is dict:
        required = ("live_engine_recipe_bytes_verified", "selected_desktop_connection_verified", "gameplay_authority")
        recipe = reference.get("recipe_text")
        catalog = reference.get("catalog_sha256")
        packs = reference.get("configured_packs")
        if (
            reference.get("state") != "configured_snapshot"
            or any(reference.get(key) is not False for key in required)
            or type(recipe) is not str or not 0 < len(recipe) <= 150
            or type(catalog) is not str or re.fullmatch(r"[0-9a-f]{64}", catalog) is None
            or type(packs) is not list or not 0 < len(packs) <= 16
            or any(type(pack) is not dict or type(pack.get("version")) is not list
                   or len(pack["version"]) != 3
                   or any(type(part) is not int or not 0 <= part <= 65535
                          for part in pack["version"]) for pack in packs)
        ):
            raise ValueError("configured recipe reference does not grant verified state or actions")
        result["configured_recipe_reference"] = {
            "scope": "configured_snapshot", "engine_loaded": False,
            "selected_client_session": False, "gameplay_authority": False,
            "catalog_sha256": catalog, "recipe_text": recipe,
            "pack_versions": sorted({".".join(map(str, pack["version"])) for pack in packs}),
        }
    for key in ("active_operator_message",):
        if type(payload.get(key)) is dict:
            result[key] = {
                field: payload[key].get(field) if field == "message_id" else _short(payload[key].get(field), 110)
                for field in ("message_id", "kind", "priority", "status")
                if field in payload[key]
            }
    if directive is not None:
        result["active_operator_directive"] = {
            key: directive[key]
            for key in ("message_id", "text", "kind", "priority", "status")
            if key in directive
        }
        if directive.get("kind") == "question":
            reply_only = True
            message_id = directive.get("message_id")
            if type(message_id) is str:
                reply_only_goal_id = f"operator:{message_id}"
    active_message = payload.get("active_operator_message")
    if (
        type(active_message) is dict
        and active_message.get("kind") == "question"
        and active_message.get("status") in {"queued", "delivered"}
        and type(active_message.get("message_id")) is str
    ):
        reply_only = True
        reply_only_goal_id = f"operator:{active_message['message_id']}"
    for value in auxiliary_payloads:
        bounds = value.get("authority_bounds")
        if type(bounds) is dict:
            compact_bounds: dict[str, Any] = {}
            allowed = bounds.get("allowed_skills")
            if type(allowed) is list:
                compact_bounds["allowed_skills"] = [
                    {
                        "s": item.get("s"),
                        "p": list(item.get("p", []))
                        if type(item.get("p", [])) is list
                        else [],
                    }
                    for item in allowed
                    if type(item) is dict
                ]
            requested = bounds.get("requested_skill_ids")
            if type(requested) is list:
                compact_bounds["requested_skill_ids"] = list(requested)
            required = bounds.get("required_action_constraints")
            if type(required) is dict:
                compact_bounds["required_action_constraints"] = dict(required)
            for key in ("authority_goal_id", "reply_only", "skill_required"):
                if key in bounds:
                    compact_bounds[key] = bounds[key]
            result["authority_bounds"] = compact_bounds
            if bounds.get("reply_only") is True:
                reply_only = True
                authority_goal_id = bounds.get("authority_goal_id")
                if type(authority_goal_id) is str:
                    reply_only_goal_id = authority_goal_id
        for key in ("repair", "reason", "blocked_skills", "missing_facts"):
            if key in value:
                item = value[key]
                if type(item) is list:
                    result[key] = [_short(part, 100) for part in item[:4]]
                else:
                    result[key] = _short(item, 180)
        fallback = value.get("safe_fallback")
        if type(fallback) is dict:
            safe_fallback: dict[str, object] = {}
            for key in ("s", "p", "x"):
                if key not in fallback:
                    continue
                item = fallback[key]
                if key == "p" and type(item) is dict:
                    safe_fallback[key] = dict(item)
                else:
                    safe_fallback[key] = item
            result["safe_fallback"] = safe_fallback
        for key in ("rejected", "rejected_output"):
            if key in value:
                rejected = value[key]
                result[key] = _short(
                    rejected
                    if type(rejected) is str
                    else json.dumps(rejected, separators=(",", ":")),
                    220,
                )
    if authority_bounds is not None:
        # _complete already parsed this from the original controller contract.
        # Copy it before menu compaction; never mutate the signed format or let
        # prose/repair history replace its declared goal, skills or constraints.
        declared = json.loads(json.dumps(authority_bounds, ensure_ascii=False))
        declared["allowed_skills"] = [
            {"s": item["skill_id"], "p": item["parameters"]}
            for item in declared.get("allowed_skills", [])
        ]
        result["authority_bounds"] = declared
    facts = payload.get("fresh_facts")
    if type(facts) is dict:
        priority = (
            "danger.immediate",
            "danger.drowning",
            "danger.burning",
            "scene.death",
            "scene.playable",
            "player.health",
            "player.critical_health",
            "player.hunger",
            "social.player_message",
            "operator.game_chat_authorized",
            "environment.underwater",
            "inventory.hotbar.logs",
        )
        selected = [key for key in priority if key in facts]
        selected.extend(
            key
            for key in sorted(facts)
            if key not in selected
            and key.startswith(("target.", "obstacle.", "terrain.", "scene."))
        )
        compact_facts = {}
        for key in selected[:10]:
            compact_facts[key] = _short(facts[key], 85)
        if compact_facts:
            result["fresh_facts"] = compact_facts
    goals = payload.get("goals")
    if type(goals) is list and goals:
        goal_id = result.get("authority_bounds", {}).get("authority_goal_id")
        rows = [item for item in goals if type(item) is dict]
        declared_goals = None if authority_bounds is None else authority_bounds.get("goal_ids", [])
        if declared_goals is not None:
            rows = [item for item in rows if item.get("id") in declared_goals]
        active = directive if directive is not None else active_message
        active_goal = f"operator:{active['message_id']}" if type(active) is dict and type(active.get("message_id")) is str else None
        mandatory = [item for item in rows if item.get("id") in {goal_id, active_goal} - {None}]
        selected = mandatory + [item for item in rows if item not in mandatory][:max(0, 2-len(mandatory))]
        result["goals"] = [
            {
                key: item.get(key) if key == "id" else _short(item.get(key), 85)
                for key in ("id", "description", "source")
                if key in item
            }
            for item in selected
        ]
    current_plan = payload.get("current_plan")
    if type(current_plan) is dict:
        result["current_plan"] = {
            "goal": _short(current_plan.get("goal"), 80),
            "next": current_plan.get("next", 0),
            "steps": [_short(step, 70) for step in current_plan.get("steps", [])[:3]],
        }
    skills = payload.get("skills")
    if type(skills) is list:
        required_ids = set(result.get("authority_bounds", {}).get("requested_skill_ids", []))
        rows = [item for item in skills if type(item) is dict]
        mandatory = [item for item in rows if item.get("skill_id") in required_ids]
        selected = mandatory + [item for item in rows if item not in mandatory][:max(0, 6-len(mandatory))]
        result["skills"] = [
            {
                "skill_id": item.get("skill_id"),
                "description": _short(item.get("description"), 72),
                "parameters": list(item.get("parameters", [])),
                "competence": item.get("competence"),
            }
            for item in selected
        ]
    for source, target, limit, count in (
        ("chat_lines", "chat_lines", 100, 2),
        ("operator_messages", "operator_messages", 180, 1),
        ("wiki_evidence", "wiki_evidence", 220, 2),
        ("recent_skill_runs", "recent_skill_runs", 80, 2),
    ):
        rows = payload.get(source)
        if type(rows) is not list:
            continue
        compact_rows = []
        for row in rows[:count]:
            if type(row) is not dict:
                continue
            if source == "wiki_evidence":
                compact_rows.append(
                    {
                        "title": _short(row.get("title"), 80),
                        "extract": _short(row.get("extract"), limit),
                        "url": _short(row.get("url"), 220),
                        "version": _short(row.get("version"), 90),
                        "confidence": row.get("confidence"),
                    }
                )
            elif source == "chat_lines":
                compact_rows.append(
                    {
                        "speaker": _short(row.get("speaker"), 32),
                        "text": _short(row.get("text"), limit),
                        "age_ms": row.get("age_ms"),
                    }
                )
            elif source == "operator_messages":
                compact_rows.append(
                    {
                        key: _short(row.get(key), limit)
                        for key in ("text", "kind", "priority")
                        if key in row
                    }
                )
            else:
                compact_rows.append(
                    {
                        key: _short(row.get(key), limit)
                        for key in ("skill", "outcome", "failure")
                        if key in row
                    }
                )
        if compact_rows:
            result[target] = compact_rows
    if "planks_retry_requires_wood" in payload:
        result["planks_retry_requires_wood"] = payload["planks_retry_requires_wood"] is True
    if reply_only_goal_id is not None:
        result["reply_only_goal_id"] = reply_only_goal_id
        result["goals"] = [
            {
                "id": reply_only_goal_id,
                "description": "Answer the active operator question.",
                "source": "operator",
            }
        ]
    if free_text:
        result["repair_or_directive_context"] = free_text[-1][:350]
    if directive is not None and type(directive.get("text")) is str:
        if reply_only:
            result["operator_question"] = directive["text"]
        metadata = result.get("active_operator_message")
        if type(metadata) is dict and metadata.get("message_id") == directive.get("message_id"):
            result.pop("active_operator_message")
    return result, reply_only


def compact_planner_prompt(
    messages: tuple[ModelMessage, ...], *, max_prompt_bytes: int = MAX_PROMPT_BYTES,
    fits_prompt: Callable[[str], bool] | None = None,
    response_format: dict[str, Any] | None = None,
) -> str:
    """Fit the native World's byte and exact tokenizer bounds before inference."""
    if type(max_prompt_bytes) is not int or not 1 <= max_prompt_bytes <= MAX_PROMPT_BYTES:
        raise ValueError("native World prompt byte budget is invalid")
    if fits_prompt is not None and not callable(fits_prompt):
        raise ValueError("native World token budget check must be callable")
    context, reply_only = _compact_context(
        messages, authority_bounds=None if response_format is None else response_format["authority"],
    )
    if response_format is not None:
        # This is already parsed from original controller authority, before
        # lossy prompt compaction. Prose cannot select a different contract mode.
        reply_only = response_format["mode"] == "operator_reply"
        context.pop("reply_only_goal_id", None)
        if reply_only:
            context["reply_only_goal_id"] = response_format["authority"]["authority_goal_id"]
    if reply_only:
        goal_id = context.get("reply_only_goal_id")
        if type(goal_id) is not str:
            raise ValueError("reply-only planner context requires its exact operator goal id")
        compact = {
            "reply_only_goal_id": goal_id,
            "operator_question": context.get("operator_question", ""),
            "fresh_facts": context.get("fresh_facts", {}),
            "wiki_evidence": context.get("wiki_evidence", []),
        }
        if context.get("configured_recipe_reference") is not None:
            compact["configured_recipe_reference"] = context["configured_recipe_reference"]
        context = compact
        rules = _STRUCTURED_REPLY_ONLY_RULES if response_format is not None else _REPLY_ONLY_RULES
        prefix = rules
    else:
        prefix = _PLANNER_RULES
    facts = context.get("fresh_facts", {})
    needs_answer = bool(context.get("operator_question")) or (
        isinstance(facts, dict) and bool(facts.get("social.player_message"))
    )
    bounds = context.get("authority_bounds", {})
    required_ids = set(bounds.get("requested_skill_ids", [])) if type(bounds) is dict else set()
    authority_goal_id = bounds.get("authority_goal_id") if type(bounds) is dict else None
    mandatory_goal_ids = {authority_goal_id} - {None}
    active = context.get("active_operator_directive", context.get("active_operator_message"))
    if type(active) is dict and type(active.get("message_id")) is str:
        active_goal = f"operator:{active['message_id']}"
        if any(type(row) is dict and row.get("id") == active_goal for row in context.get("goals", [])):
            mandatory_goal_ids.add(active_goal)
    while True:
        encoded = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
        prompt = prefix + encoded
        if len(prompt.encode("utf-8")) <= max_prompt_bytes and (
            fits_prompt is None or fits_prompt(prompt) is True
        ):
            return prompt
        for key in (
            "wiki_evidence",
            "recent_skill_runs",
            "chat_lines",
            "operator_messages",
            "current_plan",
            "goals",
            "skills",
        ):
            rows = context.get(key)
            minimum_rows = 1 if key == "skills" or (key == "wiki_evidence" and needs_answer) else 0
            if isinstance(rows, list) and len(rows) > minimum_rows:
                removable = [index for index, row in enumerate(rows) if
                    not (key == "skills" and type(row) is dict and row.get("skill_id") in required_ids)
                    and not (key == "goals" and type(row) is dict and row.get("id") in mandatory_goal_ids)]
                if removable:
                    rows.pop(removable[-1])
                    break
            if isinstance(rows, dict) and rows:
                if key == "current_plan":
                    context.pop(key)
                    break
        else:
            bounds = context.get("authority_bounds")
            if isinstance(bounds, dict):
                allowed = bounds.get("allowed_skills")
                if isinstance(allowed, list) and len(allowed) > 1:
                    removable = [index for index, row in enumerate(allowed) if
                        type(row) is dict and row.get("s") not in required_ids]
                    if removable:
                        allowed.pop(removable[-1])
                        continue
            # Descriptions and competence are context, not authority. Keep all
            # remaining skill IDs/parameter names and required fallback values.
            optional = [(row, name) for row in context.get("skills", []) if type(row) is dict
                        for name in ("description", "competence") if name in row]
            if optional:
                row, name = optional[-1]
                row.pop(name)
                continue
            facts = context.get("fresh_facts")
            if isinstance(facts, dict):
                removable = [
                    key
                    for key in facts
                    if key
                    not in {
                        "danger.immediate",
                        "danger.drowning",
                        "danger.burning",
                        "scene.death",
                        "scene.playable",
                        "player.critical_health",
                        "environment.underwater",
                        "social.player_message",
                        "operator.game_chat_authorized",
                    }
                ]
                if removable:
                    facts.pop(removable[-1])
                    continue
            if len(context.get("repair_or_directive_context", "")) > 120:
                context["repair_or_directive_context"] = context["repair_or_directive_context"][
                    :120
                ]
                continue
            # An irreducible request is refused. Never fit it by deleting a
            # literal operator prohibition, action constraint or fallback value.
            raise ValueError("native World planner context exceeds its admitted request budget")


def _read_private_file(path_value: str, *, limit: int) -> bytes:
    path = Path(path_value)
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ValueError("native World file path must be absolute and canonical")
    parent = path.parent.lstat()
    metadata = path.lstat()
    if (
        not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != os.getuid()
        or stat.S_IMODE(parent.st_mode) != 0o700
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_size <= 0
        or metadata.st_size > limit
    ):
        raise ValueError("native World credential/readiness file is not private and bounded")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise ValueError("native World file changed while opening")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(limit + 1)
    finally:
        os.close(fd)
    if len(raw) != metadata.st_size or len(raw) > limit:
        raise ValueError("native World file changed while reading")
    return raw


@dataclass
class NativeWorldCognitionModel:
    """Fail-closed language adapter borrowing the active private World owner."""

    model_id: str
    base_url: str
    token_file: str
    ready_file: str
    timeout_s: float = 90.0
    max_tokens: int = 128

    def __post_init__(self) -> None:
        parsed = urlsplit(self.base_url)
        if (
            self.model_id != MODEL_ID
            or parsed.scheme != "http"
            or parsed.hostname not in _ALLOWED_HOSTS
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path.rstrip("/") != "/v1"
            or parsed.query
            or parsed.fragment
            or self.timeout_s <= 0
            or not 1 <= self.max_tokens <= MAX_OUTPUT_TOKENS
        ):
            raise ValueError("native World model identity, loopback endpoint and bounds required")
        for value in (self.token_file, self.ready_file):
            path = Path(value)
            if not path.is_absolute() or str(path) != str(path.resolve()):
                raise ValueError("native World file paths must be canonical and absolute")

    def _client(self) -> Any:
        try:
            import httpx
        except ImportError as error:  # pragma: no cover - dependency is part of the runtime extra
            raise RuntimeError("httpx is required for the ERAIS Native World adapter") from error
        return httpx.Client(timeout=self.timeout_s)

    def _owner_identity(self, client: Any, headers: dict[str, str]) -> tuple[str, dict[str, Any]]:
        ready = json.loads(_read_private_file(self.ready_file, limit=4_194_304))
        backend = ready.get("backend") if type(ready) is dict else None
        if (
            type(ready) is not dict
            or ready.get("status") != "private_ready"
            or type(ready.get("runtime_id")) is not str
            or _RUNTIME_ID.fullmatch(ready["runtime_id"]) is None
            or type(backend) is not dict
            or backend.get("model_id") != MODEL_ID
            or backend.get("fully_native") is not True
            or backend.get("source_family") != "Qwen3"
        ):
            raise RuntimeError(
                "active ERAIS World readiness does not match the native model contract"
            )
        response = client.get(self.base_url.rstrip("/") + "/models", headers=headers)
        response.raise_for_status()
        body = response.json()
        models = body.get("data") if type(body) is dict else None
        if type(models) is not list:
            raise RuntimeError("ERAIS World model registry response is malformed")
        selected = next(
            (item for item in models if type(item) is dict and item.get("id") == MODEL_ID),
            None,
        )
        identity = selected.get("erais") if type(selected) is dict else None
        if (
            type(identity) is not dict
            or identity.get("runtime_id") != ready["runtime_id"]
            or identity.get("fully_native") is not True
            or identity.get("source_family") != "Qwen3"
            or identity.get("minecraft_cognition") != backend.get("minecraft_cognition")
        ):
            raise RuntimeError(
                "ERAIS World API and private readiness receipt identify different owners"
            )
        return ready["runtime_id"], identity

    def _identity(self, client: Any, headers: dict[str, str]) -> str:
        return self._owner_identity(client, headers)[0]

    def verify_ready(self) -> str:
        """Verify the private readiness receipt and API owner without inference."""
        token = _read_private_file(self.token_file, limit=512).decode("ascii").strip()
        if not token or any(char.isspace() for char in token):
            raise RuntimeError("invalid private ERAIS World bearer token")
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        with self._client() as client:
            return self._identity(client, headers)

    def _complete(
        self, messages: tuple[ModelMessage, ...], *, response_format: dict[str, Any] | None = None,
    ) -> ModelResponse:
        capsule = None
        if response_format is not None:
            try:
                from erais.demo.minecraft_cognition_contract import (  # type: ignore[import-not-found]
                    CONTRACT, PERCEPTION_KEYS, parse_format, validate_output,
                )
                from .cognition.prompts import _cognition_perception_keys
                if set(PERCEPTION_KEYS) != set(_cognition_perception_keys()):
                    raise ValueError("Minecraft perception contract vocabulary changed")
                capsule = parse_format(response_format)
                response_format = capsule.to_format()
            except (ImportError, ValueError) as error:
                raise RuntimeError("qualified native Minecraft contract is unavailable") from error
        token = _read_private_file(self.token_file, limit=512).decode("ascii").strip()
        if not token or any(char.isspace() for char in token):
            raise RuntimeError("invalid private ERAIS World bearer token")
        started = time.perf_counter()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        with local_model_inference_lane():
            with self._client() as client:
                structured_identity = None
                if capsule is None:
                    runtime_id = self._identity(client, headers)
                else:
                    runtime_id, identity = self._owner_identity(client, headers)
                    structured_identity = identity.get("minecraft_cognition")
                    if (
                        type(structured_identity) is not dict
                        or structured_identity.get("supported") is not True
                        or structured_identity.get("contract") != CONTRACT
                        or type(structured_identity.get("tokenizer_identity")) is not str
                        or _SHA256.fullmatch(structured_identity["tokenizer_identity"]) is None
                        or type(structured_identity.get("stop_token_ids")) is not list
                        or len(structured_identity["stop_token_ids"]) != 1
                        or type(structured_identity["stop_token_ids"][0]) is not int
                        or structured_identity["stop_token_ids"][0] < 0
                    ):
                        raise RuntimeError(
                            "active World lacks qualified Minecraft structured decoding"
                        )

                def check_receipt(receipt: object, *, complete: bool) -> None:
                    assert capsule is not None and structured_identity is not None
                    expected = {
                        "contract": CONTRACT, "mode": capsule.mode,
                        "authority_sha256": capsule.authority_sha256,
                        "grammar_sha256": capsule.grammar_sha256,
                        "tokenizer_identity": structured_identity["tokenizer_identity"],
                        "runtime_id": runtime_id, "complete": complete,
                    }
                    if (
                        type(receipt) is not dict or set(receipt) != set(expected)
                        or receipt != expected or receipt.get("complete") is not complete
                    ):
                        raise RuntimeError(
                            "native Minecraft structured receipt does not match authority/owner"
                        )
                budget_checks = 0

                def fits_prompt(prompt: str) -> bool:
                    nonlocal budget_checks
                    budget_checks += 1
                    if budget_checks > 64:
                        raise RuntimeError("ERAIS World planner exceeded bounded token checks")
                    payload = {
                        "model": MODEL_ID,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": min(self.max_tokens, MAX_OUTPUT_TOKENS),
                        "stream": False,
                        "n": 1,
                    }
                    if capsule is not None:
                        payload["response_format"] = response_format
                    budget_response = client.post(
                        self.base_url.rstrip("/") + "/tokenize", headers=headers, json=payload
                    )
                    budget_response.raise_for_status()
                    budget = budget_response.json()
                    if (
                        type(budget) is not dict
                        or budget.get("object") != "erais.native-token-budget.v1"
                        or budget.get("model") != MODEL_ID
                        or budget.get("runtime_id") != runtime_id
                        or type(budget.get("prompt_tokens")) is not int
                        or budget["prompt_tokens"] <= 0
                        or budget.get("max_prompt_tokens") != 512
                        or type(budget.get("fits")) is not bool
                        or budget["fits"] != (budget["prompt_tokens"] <= 512)
                    ):
                        raise RuntimeError(
                            "ERAIS World returned an invalid or different-owner token budget"
                        )
                    if capsule is not None:
                        check_receipt(budget.get("structured"), complete=False)
                    return budget["fits"] is True

                prompt = compact_planner_prompt(
                    messages, fits_prompt=fits_prompt, response_format=response_format,
                )
                payload = {
                    "model": MODEL_ID,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": min(self.max_tokens, MAX_OUTPUT_TOKENS),
                    "stream": False,
                    "n": 1,
                }
                if capsule is not None:
                    payload["response_format"] = response_format
                response = client.post(
                    self.base_url.rstrip("/") + "/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                body = response.json()
        choices = body.get("choices") if type(body) is dict else None
        choice = choices[0] if type(choices) is list and len(choices) == 1 else None
        message = choice.get("message") if type(choice) is dict else None
        text = message.get("content") if type(message) is dict else None
        if (
            type(body) is not dict
            or body.get("model") != MODEL_ID
            or type(text) is not str
            or not text.strip()
            or "\0" in text
            or len(text.encode("utf-8")) > 524_288
            or type(choice) is not dict
            or choice.get("finish_reason") not in {"stop", "length"}
        ):
            raise RuntimeError("ERAIS Native World returned an invalid Minecraft decision response")
        if capsule is not None:
            erais = body.get("erais")
            check_receipt(erais.get("structured") if type(erais) is dict else None, complete=True)
            try:
                validate_output(
                    text, capsule, finish_reason=choice["finish_reason"], interrupted=False,
                )
            except ValueError as error:
                raise RuntimeError("native Minecraft decision was not completed") from error
        return ModelResponse(
            text=text,
            model=MODEL_ID,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def complete(self, messages: tuple[ModelMessage, ...]) -> ModelResponse:
        return self._complete(messages)

    def complete_structured(
        self, messages: tuple[ModelMessage, ...], *, name: str, schema: dict[str, Any],
    ) -> ModelResponse:
        raise RuntimeError(
            "native Minecraft requires original named authority, not an arbitrary schema"
        )

    def complete_constrained(
        self, messages: tuple[ModelMessage, ...], *, name: str,
        schema: dict[str, Any], grammar: str,
    ) -> ModelResponse:
        raise RuntimeError(
            "native Minecraft requires original named authority, not an arbitrary grammar"
        )

    def complete_minecraft_decision(
        self, messages: tuple[ModelMessage, ...], *, name: str, authority: dict[str, Any],
    ) -> ModelResponse:
        if name not in {"cognition_decision", "cognition_decision_json_repair"}:
            raise RuntimeError("unsupported native Minecraft request name")
        return self._complete(messages, response_format=authority)
