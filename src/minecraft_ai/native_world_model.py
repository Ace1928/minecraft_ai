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
_ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1"}

_PLANNER_RULES = (
    "ERAIS native World Minecraft planner. Use only fresh_facts as observed truth; "
    "follow active_operator_message first. Choose goal IDs only from goals, select "
    "skill_id only from skills, and use only its listed parameters. In repair mode "
    "obey authority_bounds exactly. Never "
    "invent an observation, item, outcome, or permission. Set game_chat only for a "
    "fresh authorized player question and "
    "answer from wiki_evidence. If evidence or a safe action is missing, use "
    "skill_id null and request a supported perception. Return exactly one JSON "
    "object, keys in order r,g,s,p,o,c,x,q,w,d,n, no markdown. Use this shape: "
    '{"r":"brief","g":null,"s":null,"p":{},"o":null,"c":null,'
    '"x":false,"q":[],"w":null,"d":null,"n":[]}'
    ". For a reply-only "
    "operator question, use skill_id null, p {}, c null, x false, n [].\nContext:"
)
_REPLY_ONLY_RULES = (
    "ERAIS World Minecraft operator reply. Use only fresh_facts as observed truth. "
    "If facts do not answer the question, say fresh evidence is unavailable. Do not "
    "invent observations or propose actions. Return exactly one JSON object with keys "
    "g and o. Use the exact supplied goal ID for g; keep o under 160 characters.\nContext:"
)


def _short(value: object, limit: int) -> object:
    if isinstance(value, str):
        return value[:limit]
    if type(value) in {int, float, bool} or value is None:
        return value
    if isinstance(value, list):
        return [_short(item, limit) for item in value[:2]]
    return None


def _compact_context(messages: tuple[ModelMessage, ...]) -> tuple[dict[str, Any], bool]:
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
    for key in ("active_operator_message",):
        if type(payload.get(key)) is dict:
            result[key] = {
                field: _short(payload[key].get(field), 110)
                for field in ("message_id", "kind", "priority", "status")
                if field in payload[key]
            }
    if directive is not None:
        result["active_operator_directive"] = {
            key: _short(directive.get(key), 320)
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
                        "s": _short(item.get("s"), 80),
                        "p": [_short(parameter, 48) for parameter in item.get("p", [])[:8]]
                        if type(item.get("p", [])) is list
                        else [],
                    }
                    for item in allowed[:8]
                    if type(item) is dict
                ]
            requested = bounds.get("requested_skill_ids")
            if type(requested) is list:
                compact_bounds["requested_skill_ids"] = [_short(item, 80) for item in requested[:8]]
            required = bounds.get("required_action_constraints")
            if type(required) is dict:
                compact_bounds["required_action_constraints"] = {
                    str(key)[:64]: _short(item, 64) for key, item in list(required.items())[:8]
                }
            for key in ("authority_goal_id", "reply_only", "skill_required"):
                if key in bounds:
                    compact_bounds[key] = _short(bounds[key], 100)
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
            safe_fallback = {}
            for key in ("s", "p", "x"):
                if key not in fallback:
                    continue
                item = fallback[key]
                if key == "p" and type(item) is dict:
                    safe_fallback[key] = {
                        str(name)[:64]: _short(value, 64) for name, value in list(item.items())[:8]
                    }
                else:
                    safe_fallback[key] = _short(item, 100)
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
    facts = payload.get("fresh_facts")
    if type(facts) is dict:
        priority = (
            "danger.immediate",
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
        result["goals"] = [
            {
                key: _short(item.get(key), 85)
                for key in ("id", "description", "source")
                if key in item
            }
            for item in goals[:2]
            if type(item) is dict
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
        result["skills"] = [
            {
                "skill_id": _short(item.get("skill_id"), 80),
                "description": _short(item.get("description"), 72),
                "parameters": [_short(param, 48) for param in item.get("parameters", [])[:8]],
                "competence": item.get("competence"),
            }
            for item in skills[:6]
            if type(item) is dict
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
        result["operator_question"] = directive["text"][:280]
    return result, reply_only


def compact_planner_prompt(
    messages: tuple[ModelMessage, ...], *, max_prompt_bytes: int = MAX_PROMPT_BYTES,
    fits_prompt: Callable[[str], bool] | None = None,
) -> str:
    """Fit the native World's byte and exact tokenizer bounds before inference."""
    if type(max_prompt_bytes) is not int or not 1 <= max_prompt_bytes <= MAX_PROMPT_BYTES:
        raise ValueError("native World prompt byte budget is invalid")
    if fits_prompt is not None and not callable(fits_prompt):
        raise ValueError("native World token budget check must be callable")
    context, reply_only = _compact_context(messages)
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
        context = compact
        prefix = f"{_REPLY_ONLY_RULES} Exact g value: {json.dumps(goal_id)}."
    else:
        prefix = _PLANNER_RULES
    facts = context.get("fresh_facts", {})
    needs_answer = bool(context.get("operator_question")) or (
        isinstance(facts, dict) and bool(facts.get("social.player_message"))
    )
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
                rows.pop()
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
                    allowed.pop()
                    continue
                if isinstance(allowed, list) and allowed and isinstance(allowed[-1], dict):
                    parameters = allowed[-1].get("p")
                    if isinstance(parameters, list) and parameters:
                        parameters.pop()
                        continue
                requested = bounds.get("requested_skill_ids")
                if isinstance(requested, list) and requested:
                    requested.pop()
                    continue
                constraints = bounds.get("required_action_constraints")
                if isinstance(constraints, dict) and constraints:
                    constraints.pop(next(reversed(constraints)))
                    continue
            fallback = context.get("safe_fallback")
            if isinstance(fallback, dict):
                parameters = fallback.get("p")
                if isinstance(parameters, dict) and parameters:
                    parameters.pop(next(reversed(parameters)))
                    continue
            facts = context.get("fresh_facts")
            if isinstance(facts, dict):
                removable = [
                    key
                    for key in facts
                    if key
                    not in {
                        "danger.immediate",
                        "scene.death",
                        "scene.playable",
                        "social.player_message",
                        "operator.game_chat_authorized",
                    }
                ]
                if removable:
                    facts.pop(removable[-1])
                    continue
            directive = context.get("active_operator_directive")
            if (
                isinstance(directive, dict) and isinstance(directive.get("text"), str)
                and len(directive["text"]) > 160
            ):
                directive["text"] = directive["text"][:160]
                continue
            if len(context.get("repair_or_directive_context", "")) > 120:
                context["repair_or_directive_context"] = context["repair_or_directive_context"][
                    :120
                ]
                continue
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

    def _identity(self, client: Any, headers: dict[str, str]) -> str:
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
        ):
            raise RuntimeError(
                "ERAIS World API and private readiness receipt identify different owners"
            )
        return ready["runtime_id"]

    def verify_ready(self) -> str:
        """Verify the private readiness receipt and API owner without inference."""
        token = _read_private_file(self.token_file, limit=512).decode("ascii").strip()
        if not token or any(char.isspace() for char in token):
            raise RuntimeError("invalid private ERAIS World bearer token")
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        with self._client() as client:
            return self._identity(client, headers)

    def _complete(self, messages: tuple[ModelMessage, ...]) -> ModelResponse:
        token = _read_private_file(self.token_file, limit=512).decode("ascii").strip()
        if not token or any(char.isspace() for char in token):
            raise RuntimeError("invalid private ERAIS World bearer token")
        started = time.perf_counter()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        with local_model_inference_lane():
            with self._client() as client:
                runtime_id = self._identity(client, headers)
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
                    return budget["fits"]

                prompt = compact_planner_prompt(messages, fits_prompt=fits_prompt)
                payload = {
                    "model": MODEL_ID,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": min(self.max_tokens, MAX_OUTPUT_TOKENS),
                    "stream": False,
                    "n": 1,
                }
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
        return ModelResponse(
            text=text,
            model=MODEL_ID,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def complete(self, messages: tuple[ModelMessage, ...]) -> ModelResponse:
        return self._complete(messages)

    def complete_structured(self, messages, *, name, schema) -> ModelResponse:
        return self._complete(messages)

    def complete_constrained(self, messages, *, name, schema, grammar) -> ModelResponse:
        # Native World owns decoding and does not accept caller-supplied grammars.
        # The existing strict decision parser and authority layer validate output.
        return self._complete(messages)
