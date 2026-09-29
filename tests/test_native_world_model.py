from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from types import SimpleNamespace

import pytest

from minecraft_ai.config import ModelConfig, RuntimeConfig
from minecraft_ai.models import ModelMessage
from minecraft_ai.native_world_model import (
    MODEL_ID,
    NativeWorldCognitionModel,
    compact_planner_prompt,
)
from minecraft_ai.pack_recipes import PackRecipeCatalog
from minecraft_ai.resident_broker_model import configured_model
from minecraft_ai.memory import MemoryStore
from minecraft_ai.perception import FrameState, PerceptionBlackboard, PerceptionFact
from minecraft_ai.runtime import AgentRuntime
from minecraft_ai.social import SocialState
from minecraft_ai.roles import get_role


def _catalog_payload():
    return {
        "schema_version": 1,
        "world": "PokemonFamily",
        "bds_version": "1.26.52.3",
        "revision": "a" * 64,
        "items": {
            "lota:red_apricorn": {
                "name": "Red Apricorn",
                "craftable": False,
                "source": {"pack": "CobbleDrock Core BP 1.3.1"},
            },
            "minecraft:copper_ingot": {
                "name": "Copper Ingot",
                "craftable": True,
                "source": {"pack": "Minecraft"},
            },
            "lota:poke_ball": {
                "name": "Poké Ball",
                "craftable": True,
                "recipes": ["lota:poke_ball_apricorn"],
                "source": {"pack": "CobbleDrock Core BP 1.3.1"},
            },
        },
        "recipes": {
            "lota:poke_ball_apricorn": {
                "id": "lota:poke_ball_apricorn",
                "kind": "shaped",
                "ingredients": [
                    {"id": "lota:red_apricorn", "count": 4},
                    {"id": "minecraft:copper_ingot", "count": 1},
                ],
                "outputs": [{"id": "lota:poke_ball", "count": 4}],
                "stations": ["crafting_table"],
                "grid": [
                    [None, {"id": "lota:red_apricorn"}, None],
                    [
                        {"id": "lota:red_apricorn"},
                        {"id": "minecraft:copper_ingot"},
                        {"id": "lota:red_apricorn"},
                    ],
                    [None, {"id": "lota:red_apricorn"}, None],
                ],
                "source": {"pack": "CobbleDrock Core BP 1.3.1"},
            }
        },
        "warnings": [],
    }


def test_native_provider_is_explicit_and_never_used_as_vlm():
    config = ModelConfig(
        enabled=True,
        provider="erais-native-world",
        model_id=MODEL_ID,
        base_url="http://127.0.0.1:8771/v1",
        native_world_token_file="/tmp/native-world/token",
        native_world_ready_file="/tmp/native-world/ready.json",
        max_tokens=256,
    )
    model = configured_model(config, purpose="cognition")
    assert isinstance(model, NativeWorldCognitionModel)
    assert model.max_tokens == 128
    with pytest.raises(ValueError, match="not a decision-grade VLM"):
        configured_model(config, purpose="vision")
    with pytest.raises(ValueError, match="private token and ready files"):
        RuntimeConfig(
            high_level=ModelConfig(
                enabled=True,
                provider="erais-native-world",
                model_id=MODEL_ID,
                base_url="http://127.0.0.1:8771/v1",
            )
        )


def test_native_model_rejects_non_loopback_or_wrong_identity():
    with pytest.raises(ValueError, match="loopback endpoint"):
        NativeWorldCognitionModel(
            model_id=MODEL_ID,
            base_url="https://example.com/v1",
            token_file="/tmp/native-world/token",
            ready_file="/tmp/native-world/ready.json",
        )
    with pytest.raises(ValueError, match="loopback endpoint"):
        NativeWorldCognitionModel(
            model_id="gemma-4-e4b-vl",
            base_url="http://127.0.0.1:8771/v1",
            token_file="/tmp/native-world/token",
            ready_file="/tmp/native-world/ready.json",
        )


def test_compacted_context_keeps_world_facts_and_bounds_request():
    payload = {
        "active_operator_message": {"message_id": "op-7", "kind": "correction"},
        "fresh_facts": {
            "scene.death": [True, 0.99],
            "scene.playable": [False, 0.99],
            "social.player_message": ["Kid: how do I craft a Poké Ball?", 0.95],
            **{f"environment.extra{i}": ["x" * 100, 0.8] for i in range(40)},
        },
        "skills": [
            {
                "skill_id": f"skill_{i}",
                "description": "d" * 80,
                "parameters": ["allow_attack", "allow_use"],
            }
            for i in range(20)
        ],
        "goals": [{"id": "goal-a", "description": "Explore the family world"}],
        "wiki_evidence": [
            {
                "title": "Poké Ball recipe",
                "extract": "Red Apricorn and Copper Ingot make Poké Balls." * 12,
                "version": "pack",
            }
        ],
    }
    messages = (
        ModelMessage(role="system", content="Return a compact JSON decision."),
        ModelMessage(role="user", content=json.dumps(payload, ensure_ascii=False)),
        ModelMessage(
            role="user",
            content=(
                "ACTIVE OPERATOR DIRECTIVE (highest authority; follow this literal "
                "current request and do not substitute an older task): "
                + json.dumps(
                    {"message_id": "op-7", "kind": "correction", "text": "Explore safely."}
                )
            ),
        ),
    )
    prompt = compact_planner_prompt(messages)
    assert len(prompt.encode("utf-8")) <= 2048
    assert "scene.death" in prompt
    assert "social.player_message" in prompt
    assert "active_operator_directive" in prompt
    assert '"skill_id":"skill_0"' in prompt


def test_compacted_repair_context_preserves_allowed_skill_and_parameter_bounds():
    repair = {
        "repair": "infeasible_option",
        "reason": "the prior choice has no visible target",
        "authority_bounds": {
            "allowed_skills": [{"s": "look_around", "p": ["allow_attack", "allow_use"]}],
            "required_action_constraints": {"allow_attack": False},
            "requested_skill_ids": ["look_around"],
        },
        "safe_fallback": {"s": None, "p": {"allow_attack": False}, "x": True},
    }
    prompt = compact_planner_prompt(
        (
            ModelMessage(role="system", content="Repair the decision."),
            ModelMessage(role="user", content=json.dumps(repair)),
        )
    )
    assert "authority_bounds" in prompt
    assert '"s":"look_around"' in prompt
    assert "allow_attack" in prompt
    assert "infeasible_option" in prompt


def test_reply_only_prompt_carries_exact_operator_goal_and_disables_actions():
    messages = (
        ModelMessage(
            role="system",
            content=(
                "Answer the active operator question; this response grants no game "
                "action authority."
            ),
        ),
        ModelMessage(
            role="user",
            content=json.dumps(
                {
                    "active_operator_message": {
                        "message_id": "question-42",
                        "kind": "question",
                        "status": "queued",
                    },
                    "fresh_facts": {},
                    "skills": [],
                }
            ),
        ),
        ModelMessage(
            role="user",
            content=(
                "ACTIVE OPERATOR DIRECTIVE (highest authority; follow this literal "
                "current request and do not substitute an older task): "
                + json.dumps(
                    {
                        "message_id": "question-42",
                        "kind": "question",
                        "text": "What can you observe right now?",
                    }
                )
            ),
        ),
    )
    prompt = compact_planner_prompt(messages)
    assert '"reply_only_goal_id":"operator:question-42"' in prompt
    assert '"operator_question":"What can you observe right now?"' in prompt
    assert 'Exact g value: "operator:question-42"' in prompt
    assert "Do not invent observations" in prompt
    assert len(prompt.encode()) <= 2048


def test_world_adapter_uses_owner_readiness_and_only_native_api_fields(monkeypatch):
    runtime_id = "b" * 32
    ready = {
        "status": "private_ready",
        "runtime_id": runtime_id,
        "backend": {"model_id": MODEL_ID, "fully_native": True, "source_family": "Qwen3"},
    }
    model_meta = {
        "data": [
            {
                "id": MODEL_ID,
                "erais": {
                    "runtime_id": runtime_id,
                    "fully_native": True,
                    "source_family": "Qwen3",
                },
            }
        ]
    }
    decision = {
        "r": "ready",
        "g": None,
        "s": None,
        "p": {},
        "o": None,
        "c": None,
        "x": True,
        "q": ["scene.playable"],
        "w": None,
        "d": None,
        "n": [],
    }
    completion = {
        "model": MODEL_ID,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(decision, separators=(",", ":")),
                },
            }
        ],
    }
    seen = {}

    class Response:
        def __init__(self, body):
            self.body = body

        def raise_for_status(self):
            return None

        def json(self):
            return self.body

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def get(self, url, **kwargs):
            seen["get"] = url
            return Response(model_meta)

        def post(self, url, **kwargs):
            seen["url"] = url
            seen["headers"] = kwargs["headers"]
            seen["payload"] = kwargs["json"]
            return Response(completion)

    model = NativeWorldCognitionModel(
        model_id=MODEL_ID,
        base_url="http://127.0.0.1:8771/v1",
        token_file="/tmp/native-world/token",
        ready_file="/tmp/native-world/ready.json",
    )
    monkeypatch.setattr(
        "minecraft_ai.native_world_model._read_private_file",
        lambda path, **_kw: (
            b"owner-token" if path == model.token_file else json.dumps(ready).encode()
        ),
    )
    monkeypatch.setattr(model, "_client", lambda: Client())
    response = model.complete_constrained(
        (
            ModelMessage(role="system", content="Use only verified facts."),
            ModelMessage(
                role="user", content='{"fresh_facts":{"scene.playable":[false,0.99]},"skills":[]}'
            ),
        ),
        name="decision",
        schema={"type": "object"},
        grammar='root ::= "{}"',
    )
    assert response.model == MODEL_ID
    assert seen["get"].endswith("/models")
    assert seen["url"].endswith("/chat/completions")
    assert seen["headers"]["Authorization"] == "Bearer owner-token"
    assert set(seen["payload"]) == {"model", "messages", "max_tokens", "stream", "n"}
    assert len(seen["payload"]["messages"]) == 1
    assert seen["payload"]["messages"][0]["role"] == "user"
    assert "scene.playable" in seen["payload"]["messages"][0]["content"]
    assert seen["payload"]["max_tokens"] == 128


def test_world_readiness_checks_exact_native_owner_without_inference(monkeypatch):
    runtime_id = "c" * 32
    model = NativeWorldCognitionModel(
        model_id=MODEL_ID,
        base_url="http://127.0.0.1:8771/v1",
        token_file="/tmp/native-world/token",
        ready_file="/tmp/native-world/ready.json",
    )
    ready = {
        "status": "private_ready",
        "runtime_id": runtime_id,
        "backend": {"model_id": MODEL_ID, "fully_native": True, "source_family": "Qwen3"},
    }
    model_meta = {
        "data": [
            {
                "id": MODEL_ID,
                "erais": {
                    "runtime_id": runtime_id,
                    "fully_native": True,
                    "source_family": "Qwen3",
                },
            }
        ]
    }
    requests = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return model_meta

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def get(self, url, **kwargs):
            requests.append((url, kwargs))
            return Response()

        def post(self, *_args, **_kwargs):
            raise AssertionError("readiness must not perform inference")

    monkeypatch.setattr(
        "minecraft_ai.native_world_model._read_private_file",
        lambda path, **_kw: (
            b"owner-token" if path == model.token_file else json.dumps(ready).encode()
        ),
    )
    monkeypatch.setattr(model, "_client", lambda: Client())

    assert model.verify_ready() == runtime_id
    assert len(requests) == 1
    assert requests[0][0].endswith("/models")
    assert requests[0][1]["headers"]["Authorization"] == "Bearer owner-token"


def test_world_adapter_rejects_malformed_owner_receipts_and_responses(monkeypatch):
    model = NativeWorldCognitionModel(
        model_id=MODEL_ID,
        base_url="http://127.0.0.1:8771/v1",
        token_file="/tmp/native-world/token",
        ready_file="/tmp/native-world/ready.json",
    )

    class Response:
        def __init__(self, body):
            self.body = body

        def raise_for_status(self):
            return None

        def json(self):
            return self.body

    class Client:
        def __init__(self, model_body):
            self.model_body = model_body

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def get(self, _url, **_kwargs):
            return Response(self.model_body)

        def post(self, _url, **_kwargs):
            return Response({"model": MODEL_ID, "choices": "malformed"})

    ready = {
        "status": "private_ready",
        "runtime_id": "b" * 32,
        "backend": {"model_id": MODEL_ID, "fully_native": True, "source_family": "Qwen3"},
    }
    monkeypatch.setattr(
        "minecraft_ai.native_world_model._read_private_file",
        lambda path, **_kw: (
            b"owner-token" if path == model.token_file else json.dumps(ready).encode()
        ),
    )
    monkeypatch.setattr(model, "_client", lambda: Client({"data": "malformed"}))
    with pytest.raises(RuntimeError, match="model registry response is malformed"):
        model.complete((ModelMessage(role="user", content="{}"),))

    model_meta = {
        "data": [
            {
                "id": MODEL_ID,
                "erais": {
                    "runtime_id": "b" * 32,
                    "fully_native": True,
                    "source_family": "Qwen3",
                },
            }
        ]
    }
    monkeypatch.setattr(model, "_client", lambda: Client(model_meta))
    with pytest.raises(RuntimeError, match="invalid Minecraft decision response"):
        model.complete((ModelMessage(role="user", content="{}"),))


def test_pokemon_recipe_answer_uses_exact_pack_item_names_and_version():
    digest = hashlib.sha256(b"catalog bytes").hexdigest()
    catalog = PackRecipeCatalog(_catalog_payload(), digest)
    answer = catalog.lookup("How do I craft a Poke Ball?", game_version="1.26.52.3")
    assert answer is not None
    assert answer.chat_reply == (
        "At a crafting table, use 4 Red Apricorn and 1 Copper Ingot to make 4 Poké Balls. "
        "Pattern · R · / R C R / · R ·."
    )
    assert answer.evidence.confidence == 1.0
    assert "pack revision aaaaaaaaaaaaaaaa" in answer.evidence.extract
    assert catalog.lookup("How do I craft a Poke Ball?", game_version="1.26.99") is None
    assert catalog.lookup("Where can I find a Poké Ball?", game_version="1.26.52.3") is None

    mixed_query = catalog.lookup(
        "How do I craft a Poké Ball with a Copper Ingot?",
        game_version="1.26.52.3",
    )
    assert mixed_query is not None
    assert mixed_query.evidence.title.startswith("Poké Ball recipe")


def test_runtime_context_uses_pinned_family_recipe_for_fresh_player_chat(monkeypatch):
    now = time.monotonic_ns()
    blackboard = PerceptionBlackboard()
    blackboard.publish(
        FrameState(
            frame_id=1,
            captured_ns=now,
            instance_id="bedrock:1.26.52.3:PokemonFamily",
            width=1280,
            height=720,
            facts=(
                PerceptionFact(
                    key="social.player_message",
                    value="Kid: How do I craft a Poké Ball?",
                    confidence=0.99,
                    observed_ns=now,
                    source="chat:family-server",
                    expires_after_ms=10_000,
                ),
            ),
        )
    )
    runtime = object.__new__(AgentRuntime)
    runtime.role = get_role("generalist")
    runtime.custom_goals = []
    runtime.memories = MemoryStore()
    runtime.social = SocialState()
    runtime.state_db = None
    runtime.blackboard = blackboard
    runtime.perception = SimpleNamespace(instance_id="bedrock:1.26.52.3:PokemonFamily")
    runtime.pack_recipe_catalog = PackRecipeCatalog(
        _catalog_payload(), hashlib.sha256(b"catalog bytes").hexdigest()
    )
    runtime._recent_skill_runs = deque(maxlen=8)
    runtime._plan_steps = ()
    runtime._plan_index = 0
    runtime._plan_goal_id = None
    runtime._plan_started_ns = 0
    runtime._headroom_inspection_memory = None
    runtime._traversal_escalation_pending = False
    monkeypatch.setattr(runtime, "_progression_goal", lambda: None)
    monkeypatch.setattr(runtime, "_active_cognition_perception_target", lambda: None)

    context = runtime._cognition_context(requires_wood=False)
    assert context.pack_recipe_reply == (
        "At a crafting table, use 4 Red Apricorn and 1 Copper Ingot to make 4 Poké Balls. "
        "Pattern · R · / R C R / · R ·."
    )
    assert context.wiki and context.wiki[0].confidence == 1.0


def test_recipe_snapshot_must_match_file_hash_and_have_no_warnings(tmp_path):
    payload = _catalog_payload()
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
    path = tmp_path / "family-recipes.json"
    path.write_bytes(raw)
    catalog = PackRecipeCatalog.load(path, hashlib.sha256(raw).hexdigest())
    assert catalog.version_id == "1.26.52.3"
    with pytest.raises(ValueError, match="digest or size"):
        PackRecipeCatalog.load(path, "0" * 64)
    payload["warnings"] = ["unparsed pack file"]
    with pytest.raises(ValueError, match="warning-bearing"):
        PackRecipeCatalog(payload, hashlib.sha256(raw).hexdigest())
