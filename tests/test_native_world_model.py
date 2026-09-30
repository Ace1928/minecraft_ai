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


def _token_budget(runtime_id, count=100):
    return {"object": "erais.native-token-budget.v1", "model": MODEL_ID,
            "runtime_id": runtime_id, "prompt_tokens": count,
            "max_prompt_tokens": 512, "fits": count <= 512,
            "retained_history_exchanges": 0}


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
            if url.endswith("/tokenize"):
                return Response(_token_budget(runtime_id))
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
    response = model.complete(
        (
            ModelMessage(role="system", content="Use only verified facts."),
            ModelMessage(
                role="user", content='{"fresh_facts":{"scene.playable":[false,0.99]},"skills":[]}'
            ),
        ),
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


def test_planner_compacts_against_exact_owner_token_budget_before_inference(monkeypatch):
    model = NativeWorldCognitionModel(MODEL_ID, "http://127.0.0.1:8771/v1",
                                     "/tmp/native-world/token", "/tmp/native-world/ready.json")
    runtime_id = "c" * 32
    inspected, generated = [], []
    class Response:
        def __init__(self, body): self.body = body
        def raise_for_status(self): pass
        def json(self): return self.body
    class Client:
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def post(self, url, **kwargs):
            payload = kwargs["json"]
            text = payload["messages"][0]["content"]
            if url.endswith("/tokenize"):
                inspected.append(text)
                return Response(_token_budget(runtime_id, len(text.encode()) // 3 + 100))
            generated.append(text)
            return Response({"model": MODEL_ID, "choices": [
                {"finish_reason": "stop", "message": {"content": '{"s":null,"x":true}'}}
            ]})
    monkeypatch.setattr(model, "_identity", lambda *_: runtime_id)
    monkeypatch.setattr(model, "_client", Client)
    monkeypatch.setattr(
        "minecraft_ai.native_world_model._read_private_file", lambda *_a, **_kw: b"token"
    )
    payload = {"fresh_facts": {"scene.death": [False, .99], "scene.playable": [True, .99],
                              **{f"terrain.optional.{i}": ["x" * 80, .8] for i in range(25)}},
               "skills": [{"skill_id": "survey_surroundings", "description": "Look around"}]}
    model.complete((ModelMessage(role="user", content=json.dumps(payload)),))
    assert 2 <= len(inspected) <= 64
    assert len(generated) == 1 and generated[0] == inspected[-1]
    assert len(generated[0].encode()) // 3 + 100 <= 512
    assert "scene.death" in generated[0] and "scene.playable" in generated[0]


def test_token_budget_owner_mismatch_never_dispatches_inference(monkeypatch):
    model = NativeWorldCognitionModel(MODEL_ID, "http://127.0.0.1:8771/v1",
                                     "/tmp/native-world/token", "/tmp/native-world/ready.json")
    calls = []
    class Response:
        def raise_for_status(self): pass
        def json(self): return _token_budget("d" * 32)
    class Client:
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def post(self, url, **_kwargs):
            calls.append(url)
            return Response()
    monkeypatch.setattr(model, "_identity", lambda *_: "c" * 32)
    monkeypatch.setattr(model, "_client", Client)
    monkeypatch.setattr(
        "minecraft_ai.native_world_model._read_private_file", lambda *_a, **_kw: b"token"
    )
    with pytest.raises(RuntimeError, match="different-owner token budget"):
        model.complete((ModelMessage(role="user", content="{}"),))
    assert len(calls) == 1 and calls[0].endswith("/tokenize")


def test_token_compaction_retains_source_for_player_question():
    source = {"title": "Poké Ball", "extract": "4 Red Apricorn + 1 Copper Ingot makes 4 balls.",
              "url": "https://minecraft.wiki/w/Copper_Ingot", "version": "pack:1.3.143",
              "confidence": 1.0}
    payload = {"fresh_facts": {"social.player_message": ["How do I make a Poké Ball?", .99],
                              **{f"terrain.optional.{i}": ["x" * 80, .8] for i in range(9)}},
               "wiki_evidence": [source, {"title": "irrelevant", "extract": "x" * 220}],
               "skills": [{"skill_id": "survey_surroundings", "description": "Look around"}]}
    prompt = compact_planner_prompt((ModelMessage(role="user", content=json.dumps(payload)),),
                                    fits_prompt=lambda text: len(text.encode()) < 1500)
    assert source["extract"] in prompt and source["url"] in prompt
    assert source["version"] in prompt and '"confidence":1.0' in prompt
    assert "How do I make a Poké Ball?" in prompt and "irrelevant" not in prompt


def test_exact_token_compaction_fails_boundedly_when_minimum_context_cannot_fit():
    inspected = []
    marker = ("ACTIVE OPERATOR DIRECTIVE (highest authority; follow this literal current "
              "request and do not substitute an older task): ")
    messages = (ModelMessage(role="user", content=marker + json.dumps({
        "message_id": "id", "text": "x" * 200, "kind": "instruction",
    })), ModelMessage(role="user", content=json.dumps({"fresh_facts": {}, "skills": []})))
    with pytest.raises(ValueError, match="admitted request budget"):
        compact_planner_prompt(messages, fits_prompt=lambda text: inspected.append(text) or False)
    assert 1 <= len(inspected) <= 3 and len(inspected) == len(set(inspected))


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
            if _url.endswith("/tokenize"):
                return Response(_token_budget("b" * 32))
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
        "At a crafting table, use 4 Red Apricorn and 1 Copper Ingot to make 4 Poke Balls. "
        "Pattern . R . / R C R / . R .."
    )
    assert "Poké Balls" in answer.evidence.extract
    assert "Pattern · R · / R C R / · R ·." in answer.evidence.extract
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


@pytest.mark.parametrize("reverse_items", [False, True])
@pytest.mark.parametrize(
    ("query", "selected_name", "expected_recipe"),
    [
        ("How do I craft a stone pickaxe?", "Stone Pickaxe", "3 Cobblestone and 2 Stick"),
        ("How do I craft stone pickaxes?", "Stone Pickaxe", "3 Cobblestone and 2 Stick"),
        ("How do I craft stone_pickaxe?", "Stone Pickaxe", "3 Cobblestone and 2 Stick"),
        ("How do I craft a copper spear?", "Copper Spear", "1 Copper and 2 Stick"),
        ("How do I make copper spears?", "Copper Spear", "1 Copper and 2 Stick"),
        (
            "How do I craft a stone pickaxe with stone?",
            "Stone Pickaxe",
            "3 Cobblestone and 2 Stick",
        ),
        ("How do I make stone from a stone pickaxe?", "Stone", "1 Cobblestone"),
        ("With a stone pickaxe, how do I craft stone?", "Stone", "1 Cobblestone"),
        ("How do I make copper for a copper spear?", "Copper", "1 Raw Copper"),
    ],
)
def test_recipe_compound_phrase_wins_only_at_same_intent_position(
    query, selected_name, expected_recipe, reverse_items,
):
    """Invented collisions exercise the accepted catalog matcher without live data."""
    payload = _catalog_payload()
    ingredients = {
        "minecraft:cobblestone": "Cobblestone",
        "minecraft:stick": "Stick",
        "minecraft:raw_copper": "Raw Copper",
    }
    for item_id, name in ingredients.items():
        payload["items"][item_id] = {"name": name, "craftable": False}
    recipes = [
        ("stone", "Stone", "furnace", [("cobblestone", 1)]),
        ("stone_pickaxe", "Stone Pickaxe", "crafting_table", [("cobblestone", 3), ("stick", 2)]),
        ("copper", "Copper", "furnace", [("raw_copper", 1)]),
        ("copper_spear", "Copper Spear", "crafting_table", [("copper", 1), ("stick", 2)]),
    ]
    for suffix, name, station, inputs in recipes:
        item_id = f"minecraft:{suffix}"
        payload["items"][item_id] = {
            "name": name, "craftable": True, "recipes": [item_id],
        }
        payload["recipes"][item_id] = {
            "ingredients": [{"id": f"minecraft:{key}", "count": count} for key, count in inputs],
            "outputs": [{"id": item_id, "count": 1}],
            "stations": [station],
        }
    if reverse_items:
        payload["items"] = dict(reversed(tuple(payload["items"].items())))
    before = json.dumps(payload, ensure_ascii=False)
    answer = PackRecipeCatalog(payload, "a" * 64).lookup(query, game_version="1.26.52.3")
    assert answer is not None
    assert answer.evidence.title == f"{selected_name} recipe — family pack"
    assert f"use {expected_recipe} to make 1 {selected_name}." in answer.chat_reply
    # A read-only reference does not modify the catalog or assert possession.
    assert json.dumps(payload, ensure_ascii=False) == before
    assert "inventory" not in type(answer.evidence).model_fields


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
        "At a crafting table, use 4 Red Apricorn and 1 Copper Ingot to make 4 Poke Balls. "
        "Pattern . R . / R C R / . R .."
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


@pytest.mark.parametrize("name", ["神奇球", "ﬁ Ball", "Ball\ncommand", "Ball\x00"])
def test_recipe_chat_refuses_unrepresentable_item_name_without_mutating_catalog(name):
    payload = _catalog_payload()
    payload["items"]["lota:poke_ball"]["name"] = name
    before = json.dumps(payload, ensure_ascii=False)
    catalog = PackRecipeCatalog(payload, "a" * 64)
    assert catalog.lookup("How do I craft a poke_ball?", game_version="1.26.52.3") is None
    assert json.dumps(payload, ensure_ascii=False) == before


def test_recipe_answer_is_bound_to_actual_catalog_not_an_assumed_pack_upgrade():
    previous = PackRecipeCatalog(_catalog_payload(), "a" * 64)
    current_payload = _catalog_payload()
    current_payload["revision"] = "b" * 64
    current_payload["recipes"]["lota:poke_ball_apricorn"]["outputs"][0]["count"] = 8
    current_payload["recipes"]["lota:poke_ball_apricorn"]["source"]["pack"] = (
        "synthetic-next-pack-not-live"
    )
    current = PackRecipeCatalog(current_payload, "c" * 64)
    old = previous.lookup("How do I craft a Poke Ball?", game_version="1.26.52.3")
    new = current.lookup("How do I craft a Poke Ball?", game_version="1.26.52.3")
    assert old is not None and new is not None
    assert "make 4 Poke Balls" in old.chat_reply
    assert "make 8 Poke Balls" in new.chat_reply
    assert old.evidence.version_key != new.evidence.version_key
    assert "synthetic-next-pack-not-live" in new.evidence.extract
    assert "bbbbbbbbbbbbbbbb" in new.evidence.extract


def test_native_world_controller_recipe_reaches_existing_leased_chat_contract(monkeypatch):
    """Synthetic transport/speaker authority, real catalog/controller/actuator contracts."""
    from minecraft_ai.builtin_skills import build_bootstrap_skill_library
    from minecraft_ai.cognition import CognitionContext, HighLevelController
    from minecraft_ai.game_chat import game_chat_authority_matches
    from minecraft_ai.supervisor import Supervisor

    class ChatBackend:
        backend_id = "synthetic-chat-contract"
        live_capable = True

        def __init__(self):
            self.lease = None
            self.messages = []
            self.held_keys = set()
            self.held_buttons = set()

        def bind_lease(self, lease):
            self.lease = lease

        def clear_lease(self):
            self.lease = None

        def release_all(self):
            self.held_keys.clear()
            self.held_buttons.clear()

        def apply(self, action):
            pytest.fail("recipe response cannot authorize a gameplay action")

        def type_chat(self, text, *, input_permitted):
            assert self.lease is not None and input_permitted()
            self.messages.append(text)

    now = time.monotonic_ns()
    board = PerceptionBlackboard()
    board.publish(FrameState(
        frame_id=1, captured_ns=now, instance_id="bedrock:1.26.52.3:synthetic-family-chat",
        width=32, height=32,
        facts=(PerceptionFact(
            key="social.player_message", value="SyntheticKid: How do I craft a Poké Ball?",
            confidence=0.99, observed_ns=now, source="synthetic-contract:not-training-label",
            expires_after_ms=30_000,
        ),),
    ))
    catalog = PackRecipeCatalog(_catalog_payload(), "a" * 64)
    answer = catalog.lookup("How do I craft a Poké Ball?", game_version="1.26.52.3")
    assert answer is not None
    runtime_id = "b" * 32
    from erais.demo.minecraft_cognition_contract import parse_format
    identity = {"runtime_id": runtime_id, "fully_native": True, "source_family": "Qwen3",
                "minecraft_cognition": {"contract": "erais.minecraft.cognition.v1",
                    "supported": True, "tokenizer_identity": "c" * 64,
                    "stop_token_ids": [151645]}}
    ready = {"status": "private_ready", "runtime_id": runtime_id,
             "backend": {"model_id": MODEL_ID, **identity}}
    calls = []

    class Response:
        def __init__(self, body):
            self.body = body

        def raise_for_status(self):
            pass

        def json(self):
            return self.body

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def get(self, url, **kwargs):
            calls.append(url)
            return Response({"data": [{"id": MODEL_ID, "erais": identity}]})

        def post(self, url, **kwargs):
            calls.append(url)
            capsule = parse_format(kwargs["json"]["response_format"])
            structured = {"contract": "erais.minecraft.cognition.v1", "mode": capsule.mode,
                "authority_sha256": capsule.authority_sha256,
                "grammar_sha256": capsule.grammar_sha256,
                "tokenizer_identity": "c" * 64, "runtime_id": runtime_id,
                "complete": not url.endswith("/tokenize")}
            if url.endswith("/tokenize"):
                return Response({**_token_budget(runtime_id), "structured": structured})
            prompt = kwargs["json"]["messages"][0]["content"]
            assert "wiki_evidence" in prompt and "4 Red Apricorn" in prompt
            assert "social.player_message" in prompt
            return Response({"model": MODEL_ID, "erais": {"structured": structured}, "choices": [{
                "finish_reason": "stop", "message": {"content": json.dumps({
                    "r": "Use the supplied pack reference", "g": None, "s": None,
                    "p": {}, "o": None, "c": None, "x": False, "q": [],
                    "w": None, "d": None, "n": [],
                })},
            }]})

    model = NativeWorldCognitionModel(
        MODEL_ID, "http://127.0.0.1:8771/v1", "/tmp/native-test/token", "/tmp/native-test/ready",
    )
    monkeypatch.setattr("minecraft_ai.native_world_model._read_private_file", lambda path, **kw: (
        b"synthetic-bearer" if path == model.token_file else json.dumps(ready).encode()
    ))
    monkeypatch.setattr(model, "_client", Client)
    context = CognitionContext(
        role=get_role("generalist"), goals=(), memories=(), promises=(), wiki=(answer.evidence,),
        pack_recipe_reply=answer.chat_reply,
    )
    controller = HighLevelController(model, build_bootstrap_skill_library())
    decision = controller.decide(board.cognition_snapshot(), context)
    assert decision.game_chat == answer.chat_reply
    assert decision.skill_id is None and decision.skill_parameters == {}
    assert decision.plan_steps == ()
    assert game_chat_authority_matches(decision, board)
    assert calls == ["http://127.0.0.1:8771/v1/models", "http://127.0.0.1:8771/v1/tokenize",
                     "http://127.0.0.1:8771/v1/chat/completions"]
    assert all(not fact.key.startswith("inventory.") for fact in board.raw_latest().facts)
    backend = ChatBackend()
    supervisor = Supervisor()
    supervisor.start()
    supervisor.replace_backend(backend)
    lease = supervisor.arm("bedrock:1.26.52.3:synthetic-family-chat")
    supervisor.activate()
    result = supervisor.send_chat(str(lease["lease_id"]), decision.game_chat)
    assert result == {"sent": True, "characters": len(answer.chat_reply)}
    assert backend.messages == [answer.chat_reply]
