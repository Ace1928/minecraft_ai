from __future__ import annotations

import io
import json
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.parse import parse_qs

import pytest

from minecraft_ai.builtin_skills import build_bootstrap_skill_library
from minecraft_ai.cognition import CognitionContext, HighLevelController
from minecraft_ai.memory import MemoryStore
from minecraft_ai.pack_recipes import PackRecipeCatalog
from minecraft_ai.perception import FrameState, PerceptionBlackboard, PerceptionFact
from minecraft_ai.roles import get_role
from minecraft_ai.world_knowledge import WorldMinecraftSearch, minecraft_public_topic
from minecraft_ai.social import OperatorMessage, OperatorMessageKind, OperatorMessageStatus
from minecraft_ai.wiki import WikiEvidence
from minecraft_ai.models import ModelResponse


@pytest.mark.parametrize("query,subject,intent", [
    ("Where do I find copper ore?", "copper ore", "locations"),
    ("How do I craft a stone pickaxe?", "stone pickaxe", "crafting recipe"),
    ("Can spiders climb?", "spider", "mechanics"),
    ("What about wood for my secret account ABC12345?", "wood", "mechanics"),
])
def test_only_known_public_subject_and_intent_can_leave_in_search(query, subject, intent) -> None:
    assert minecraft_public_topic(query, "1.26.52.3") == (
        f"Minecraft Wiki Bedrock {subject} {intent}"
    )


@pytest.mark.parametrize("query", [
    "Where do I find red apricorns near copper ore?", "How do I craft a Poké Ball with copper?",
    "How does my Pokémon evolve with food?", "How do starters use wood?",
    "Someone: Where is copper?", "Take wood", "Where is our base?", "What is my password?",
])
def test_pack_mechanics_private_context_and_instructions_are_not_vanilla_questions(query) -> None:
    assert minecraft_public_topic(query, "1.26.52.3") is None


def test_version_is_not_an_arbitrary_search_instruction() -> None:
    assert minecraft_public_topic("Where is copper?", "1.26 secret value") is None


def _token(tmp_path):
    tmp_path.chmod(0o700)
    path = tmp_path / "token"
    path.write_text("a" * 64)
    path.chmod(0o600)
    return str(path)


def _transport(monkeypatch, rows=None, *, payload=None, status=200):
    if payload is None:
        payload = json.dumps({"results": rows or [{
            "title": "Copper Ore", "url": "https://minecraft.wiki/w/Copper_Ore",
            "content": "Copper ore generates in the Overworld.",
        }]}).encode()
    response = io.BytesIO(payload)
    response.status = status
    response.fp = None
    connection = Mock()
    connection.getresponse.return_value = response
    factory = Mock(return_value=connection)
    monkeypatch.setattr("minecraft_ai.world_knowledge.http.client.HTTPConnection", factory)
    return connection, factory


def test_shared_authenticated_proxy_yields_version_scoped_general_references(monkeypatch, tmp_path):
    connection, factory = _transport(monkeypatch)
    service = WorldMinecraftSearch(_token(tmp_path))
    result = service.search("Where do I find copper ore for KidUsername?", game_version="1.26.52.3")
    assert len(result) == 1
    evidence = result[0]
    assert evidence.version_key == "bedrock:1.26.52.3:general-wiki"
    assert evidence.confidence == 0.65
    assert "does not verify pack overrides" in evidence.extract
    assert evidence.url == "https://minecraft.wiki/w/Copper_Ore"
    factory.assert_called_once()
    assert factory.call_args.args == ("127.0.0.1", 8889)
    args, kwargs = connection.request.call_args
    assert args == ("POST", "/search")
    fields = parse_qs(kwargs["body"].decode())
    assert fields["q"] == ["Minecraft Wiki Bedrock copper ore locations"]
    assert "KidUsername" not in kwargs["body"].decode()
    connection.close.assert_called_once()


@pytest.mark.parametrize("url", [
    "http://minecraft.wiki/w/Copper", "https://example.test/w/Copper",
    "https://minecraft.wiki.evil.test/Copper", "https://user:password@minecraft.wiki/w/Copper",
    "https://minecraft.wiki:444/w/Copper",
])
def test_search_cannot_promote_untrusted_sources(monkeypatch, tmp_path, url) -> None:
    _transport(monkeypatch, [{"url": url, "title": "Copper", "content": "A claim"}])
    assert WorldMinecraftSearch(_token(tmp_path)).search(
        "Where is copper?", game_version="1.26.52.3",
    ) == ()


@pytest.mark.parametrize("case", ["expired", "bad_token", "oversize", "http_failure"])
def test_failed_or_expired_lookup_produces_no_evidence(monkeypatch, tmp_path, case) -> None:
    connection, _ = _transport(
        monkeypatch, payload=b"x" * 8193 if case == "oversize" else None,
        status=503 if case == "http_failure" else 200,
    )
    token = _token(tmp_path)
    if case == "bad_token":
        from pathlib import Path
        Path(token).chmod(0o644)
    deadline = time.monotonic_ns() - 1 if case == "expired" else None
    assert WorldMinecraftSearch(token).search(
        "Where is copper?", game_version="1.26.52.3", deadline_ns=deadline,
    ) == ()
    if case in {"expired", "bad_token"}:
        connection.request.assert_not_called()


def _context():
    return CognitionContext(
        role=get_role("generalist"), goals=(), memories=MemoryStore().retrieve(), promises=(),
        wiki=(),
    )


def _blackboard(*, query="Kid: Where do I find copper ore?", age_ms=0, instance=None):
    now = time.monotonic_ns()
    board = PerceptionBlackboard()
    board.publish(FrameState(
        frame_id=1, captured_ns=now, width=1920, height=1080,
        instance_id=instance or "bedrock:1.26.52.3:x11:42",
        facts=(PerceptionFact(
            key="social.player_message", value=query, confidence=0.99,
            observed_ns=now - age_ms * 1_000_000, source="grounded:player-chat",
            expires_after_ms=30_000,
        ),),
    ))
    return board


def test_reference_query_is_worker_only_and_does_not_write_observed_facts(monkeypatch, tmp_path):
    _transport(monkeypatch)
    service = WorldMinecraftSearch(_token(tmp_path))
    controller = HighLevelController(Mock(), build_bootstrap_skill_library(), world_search=service)
    board = _blackboard()
    before = board.raw_latest()
    enriched = controller._with_player_reference(board, _context())
    assert enriched.wiki and enriched.wiki[0].query.endswith("copper ore locations")
    assert board.raw_latest() is before
    assert board.fact("wiki.copper_ore") is None
    assert controller.model.mock_calls == []


@pytest.mark.parametrize("case", ["pack", "stale", "wrong_edition", "exact_recipe", "rejected"])
def test_reference_lookup_cannot_supersede_pack_or_stale_authority(case):
    service = Mock()
    pack = Mock(spec=PackRecipeCatalog)
    pack.mentions_pack_content.return_value = case == "pack"
    controller = HighLevelController(
        Mock(), build_bootstrap_skill_library(), world_search=service, pack_recipe_catalog=pack,
    )
    board = _blackboard(
        age_ms=30_001 if case == "stale" else 0,
        instance="java:1.21" if case == "wrong_edition" else None,
    )
    context = _context()
    if case == "exact_recipe":
        context = replace(context, pack_recipe_reply="Exact current-pack recipe")
    if case == "rejected":
        request = SimpleNamespace(snapshot=lambda: SimpleNamespace(disposition="rejected"))
        controller._request_context.set(request)
    assert controller._with_player_reference(board, context) is context
    service.search.assert_not_called()


def _operator_context(*, kind=OperatorMessageKind.QUESTION,
                      status=OperatorMessageStatus.DELIVERED,
                      text="How do I craft a Poké Ball?"):
    return replace(_context(), operator_messages=(OperatorMessage(
        message_id="recipe-question", created_ns=time.monotonic_ns(), text=text,
        kind=kind, status=status,
    ),))


def test_operator_recipe_reference_reaches_reply_only_model_without_game_authority():
    evidence = WikiEvidence(
        title="Poké Ball recipe — family pack", extract="4 red apricorns and 1 copper ingot.",
        retrieved_ns=time.time_ns(), query="How do I craft a Poké Ball?",
        version_key="bedrock:1.26.52.3:pack:" + "a" * 64, confidence=1,
    )
    pack = Mock(spec=PackRecipeCatalog)
    pack.lookup.return_value = SimpleNamespace(evidence=evidence)
    service = Mock()
    model = Mock(spec=["complete_constrained"])
    model.complete_constrained.return_value = ModelResponse(
        model="synthetic-reference-contract", latency_ms=1,
        text=json.dumps({
            "g": "operator:recipe-question", "o": "4 red apricorns and 1 copper ingot.",
        }),
    )
    controller = HighLevelController(
        model, build_bootstrap_skill_library(), world_search=service, pack_recipe_catalog=pack,
    )
    board = _blackboard(query="Kid: Where is copper?")
    before = board.raw_latest()
    decision = controller.decide(board, _operator_context())
    pack.lookup.assert_called_once_with("How do I craft a Poké Ball?", game_version="1.26.52.3")
    service.search.assert_not_called()
    payload = json.loads(model.complete_constrained.call_args.args[0][1].content)
    assert payload["wiki_evidence"][0]["version"] == evidence.version_key[:80]
    assert payload["skills"] == []
    assert decision.say == "4 red apricorns and 1 copper ingot."
    assert decision.skill_id is decision.game_chat is decision.research_query is None
    assert decision.skill_parameters == {} and not decision.plan_steps
    assert board.raw_latest() is before


@pytest.mark.parametrize("case", [
    "instruction", "answered", "wrong_edition", "cancelled", "pack_unknown",
])
def test_operator_reference_cannot_replace_authority_or_invent_pack_answers(case):
    pack = Mock(spec=PackRecipeCatalog)
    pack.lookup.return_value = None
    pack.mentions_pack_content.return_value = True
    service = Mock()
    controller = HighLevelController(
        Mock(), build_bootstrap_skill_library(), world_search=service, pack_recipe_catalog=pack,
    )
    ctx = _operator_context(
        kind=(OperatorMessageKind.INSTRUCTION if case == "instruction"
              else OperatorMessageKind.QUESTION),
        status=(OperatorMessageStatus.ACKNOWLEDGED if case == "answered"
                else OperatorMessageStatus.DELIVERED),
    )
    if case == "cancelled":
        controller._request_context.set(SimpleNamespace(
            snapshot=lambda: SimpleNamespace(disposition="rejected"),
        ))
    board = _blackboard(instance="java:1.21" if case == "wrong_edition" else None)
    assert controller._with_operator_reference(board, ctx) is ctx
    service.search.assert_not_called()
    if case != "pack_unknown":
        pack.lookup.assert_not_called()


def test_operator_vanilla_question_uses_existing_private_search_filter(monkeypatch, tmp_path):
    connection, _ = _transport(monkeypatch)
    controller = HighLevelController(
        Mock(), build_bootstrap_skill_library(),
        world_search=WorldMinecraftSearch(_token(tmp_path)),
    )
    ctx = _operator_context(text="Where do I find copper ore for PrivateKidName?")
    enriched = controller._with_operator_reference(_blackboard(), ctx)
    assert enriched.wiki[0].confidence == 0.65
    body = connection.request.call_args.kwargs["body"].decode()
    assert "PrivateKidName" not in body
    assert parse_qs(body)["q"] == ["Minecraft Wiki Bedrock copper ore locations"]
    assert enriched.pack_recipe_reply is None
