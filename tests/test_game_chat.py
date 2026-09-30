"""Retained pixels and synthetic contracts only: never connect to the game."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

import minecraft_ai.perception.service as perception_module
from minecraft_ai.cognition import CognitionDecision
from minecraft_ai.game_chat import (
    bind_game_chat_authority, game_chat_authority_matches, game_chat_delivery_admitted,
)
from minecraft_ai.perception import (
    EvidenceRegion, FrameState, PerceptionBlackboard, PerceptionEvidence, PerceptionFact,
    ScreenRegion,
)
from minecraft_ai.perception_service import BEDROCK_HUD_SAFETY_SOURCE, BootstrapFastPerception
from minecraft_ai.platforms.bedrock_x11 import CapturedFrame
from minecraft_ai.runtime import AgentRuntime, RuntimeMetrics, _authorized_game_chat
import minecraft_ai.runtime as runtime_module


NOW = 1_000_000_000
FIXTURES = Path(__file__).parent / "fixtures"


def _frame(path: Path) -> CapturedFrame:
    image = Image.open(path).convert("RGBA")
    return CapturedFrame(
        frame_id=7, captured_ns=NOW, width=image.width, height=image.height,
        bgra=image.tobytes("raw", "BGRA"),
    )


def _board(frame: CapturedFrame, monkeypatch: pytest.MonkeyPatch) -> PerceptionBlackboard:
    monkeypatch.setattr(perception_module.time, "monotonic_ns", lambda: NOW)
    board = PerceptionBlackboard()
    # A synthetic player-message fact tests channel authority. It never claims
    # that this retained frame contains a player question or qualified OCR.
    player = PerceptionFact(
        key="social.player_message", value="SyntheticKid: how do I make a Poke Ball?",
        confidence=0.99, observed_ns=NOW, source="synthetic-chat-contract:not-training-label",
        expires_after_ms=30_000,
    )
    board.publish(FrameState(
        frame_id=7, captured_ns=NOW, width=frame.width, height=frame.height,
        instance_id="bedrock:1.26.52.3:retained-test",
        facts=(*BootstrapFastPerception(game_version="1.26.52.3").infer(frame), player),
    ))
    return board


def test_chat_requires_authority_and_fresh_independent_safe_world(monkeypatch):
    frame = _frame(FIXTURES / "bedrock_health/full_health_1920x1080.png")
    board = _board(frame, monkeypatch)
    decision = CognitionDecision(game_chat="At a crafting table, use 4 Red Apricorn.")
    text = _authorized_game_chat(decision, board)
    assert text is not None
    assert game_chat_delivery_admitted(text, board, frame, now_ns=NOW)
    assert not game_chat_delivery_admitted(text, PerceptionBlackboard(), frame, now_ns=NOW)
    assert not game_chat_delivery_admitted(text, board, None, now_ns=NOW)


@pytest.mark.parametrize("age", [-1, 250_000_000, 30_000_000_001])
def test_chat_authority_never_refreshes_capture_age(monkeypatch, age):
    frame = _frame(FIXTURES / "bedrock_health/full_health_1920x1080.png")
    board = _board(frame, monkeypatch)
    assert not game_chat_delivery_admitted("hello", board, frame, now_ns=NOW + age)


@pytest.mark.parametrize("kind", ["capture_time", "dimensions", "short_pixels"])
def test_chat_refuses_unbound_pixel_surface(monkeypatch, kind):
    frame = _frame(FIXTURES / "bedrock_health/full_health_1920x1080.png")
    board = _board(frame, monkeypatch)
    if kind == "capture_time":
        frame = replace(frame, captured_ns=NOW - 1)
    elif kind == "dimensions":
        frame = replace(frame, width=1280)
    else:
        frame = replace(frame, bgra=frame.bgra[:-4])
    assert not game_chat_delivery_admitted("hello", board, frame, now_ns=NOW)


@pytest.mark.parametrize("path", [
    "bedrock_health/death_1920x1080.png",
    "bedrock_health/low_health_1920x1080.jpg",
    "bedrock_health/low_health_jitter_1920x1080.jpg",
    "bedrock_menu/server_list_transfer_1920x1080.png",
    "bedrock_menu/resource_pack_1920x1080.png",
])
def test_death_critical_health_and_real_modal_block_chat_focus(monkeypatch, path):
    frame = _frame(FIXTURES / path)
    board = _board(frame, monkeypatch)
    assert _authorized_game_chat(CognitionDecision(game_chat="hello"), board) == "hello"
    assert not game_chat_delivery_admitted("hello", board, frame, now_ns=NOW)


@pytest.mark.parametrize("key", [
    "danger.immediate", "danger.drowning", "environment.underwater", "player.critical_health",
])
def test_fresh_danger_blocks_chat_even_with_playable_hud(monkeypatch, key):
    frame = _frame(FIXTURES / "bedrock_health/full_health_1920x1080.png")
    board = _board(frame, monkeypatch)
    board.merge_semantics(instance_id="bedrock:1.26.52.3:retained-test", facts=(PerceptionFact(
        key=key, value=True, confidence=0.99, observed_ns=NOW,
        source=BEDROCK_HUD_SAFETY_SOURCE, expires_after_ms=250,
    ),))
    assert not game_chat_delivery_admitted("hello", board, frame, now_ns=NOW)


@pytest.mark.parametrize("invalid", ["wrong_source", "low_confidence", "old_observation"])
def test_assumed_scene_claim_never_substitutes_for_current_hud_witness(monkeypatch, invalid):
    frame = _frame(FIXTURES / "bedrock_health/full_health_1920x1080.png")
    board = _board(frame, monkeypatch)
    invalid_fact = PerceptionFact(
        key="scene.playable", value=True,
        confidence=0.8 if invalid == "low_confidence" else 0.995,
        observed_ns=NOW - 1 if invalid == "old_observation" else NOW,
        source="assumed" if invalid == "wrong_source" else BEDROCK_HUD_SAFETY_SOURCE,
        expires_after_ms=250,
    )
    latest = board.raw_latest()
    assert latest is not None
    amended = latest.model_copy(update={
        "facts": tuple(invalid_fact if fact.key == "scene.playable" else fact
                       for fact in latest.facts),
    })
    board = PerceptionBlackboard()
    board.publish(amended)
    assert not game_chat_delivery_admitted("hello", board, frame, now_ns=NOW)


@pytest.mark.parametrize("text", ["", "x" * 257, "bad\ntext", "Poké Ball", "/kill", " /tp me"])
def test_only_printable_bounded_chat_and_never_commands(monkeypatch, text):
    frame = _frame(FIXTURES / "bedrock_health/full_health_1920x1080.png")
    assert not game_chat_delivery_admitted(text, _board(frame, monkeypatch), frame, now_ns=NOW)


@pytest.mark.parametrize("changed", ["text", "source", "evidence", "timestamp", "expired"])
def test_old_reply_cannot_borrow_another_message_authority(monkeypatch, changed):
    frame = _frame(FIXTURES / "bedrock_health/full_health_1920x1080.png")
    board = _board(frame, monkeypatch)
    decision = bind_game_chat_authority(CognitionDecision(game_chat="Exact recipe"), board)
    assert game_chat_authority_matches(decision, board)
    original = board.fact("social.player_message")
    assert original is not None
    updates = {
        "text": {"value": "OtherKid: where is copper?"},
        "source": {"source": "different-observer"},
        "evidence": {"evidence_refs": ("different-frame",)},
        "timestamp": {"observed_ns": NOW + 1},
        "expired": {"observed_ns": 1, "expires_after_ms": 1},
    }[changed]
    latest = board.raw_latest()
    assert latest is not None
    amended = latest.model_copy(update={
        "facts": tuple(original.model_copy(update=updates)
                       if fact.key == "social.player_message" else fact for fact in latest.facts),
    })
    if changed == "evidence":
        # A schema-valid reference tests identity only, not chat transcription.
        amended = amended.model_copy(update={"evidence": (PerceptionEvidence(
            evidence_id="different-frame", frame_id=frame.frame_id, captured_ns=frame.captured_ns,
            region_kind=EvidenceRegion.CHAT,
            region=ScreenRegion(x=0, y=0, width=1, height=1),
            pixel_sha256=hashlib.sha256(frame.bgra).hexdigest(),
            crop_width=frame.width, crop_height=frame.height,
        ),)})
    replacement = PerceptionBlackboard()
    replacement.publish(amended)
    if changed == "timestamp":
        monkeypatch.setattr(perception_module.time, "monotonic_ns", lambda: NOW + 1)
    assert not game_chat_authority_matches(decision, replacement)
    assert decision.model_copy(update={"say": "operator only"}).chat_authority == (
        decision.chat_authority
    )
    assert "chat_authority" not in decision.model_dump()


def test_no_fact_or_model_authored_permission_can_create_channel_authority(monkeypatch):
    frame = _frame(FIXTURES / "bedrock_health/full_health_1920x1080.png")
    board = _board(frame, monkeypatch)
    raw = CognitionDecision(game_chat="hello")
    assert not game_chat_authority_matches(raw, board)
    empty = PerceptionBlackboard()
    assert bind_game_chat_authority(raw, empty).game_chat is None


def _runtime(board, frame):
    runtime = AgentRuntime.__new__(AgentRuntime)
    runtime.blackboard = board
    runtime.perception = SimpleNamespace(last_capture=frame)
    runtime.lease_id = "synthetic-lease"
    runtime.metrics = RuntimeMetrics()
    runtime._last_player_chat_replied_ns = None
    return runtime


@pytest.mark.parametrize("result", [
    None, {}, {"sent": False, "characters": 5}, {"sent": 1, "characters": 5},
    {"sent": True, "characters": 4}, {"sent": True, "characters": True}, RuntimeError("failed"),
])
def test_delivery_requires_exact_positive_transport_confirmation(monkeypatch, result):
    frame = _frame(FIXTURES / "bedrock_health/full_health_1920x1080.png")
    board = _board(frame, monkeypatch)
    runtime = _runtime(board, frame)
    decision = bind_game_chat_authority(CognitionDecision(game_chat="hello"), board)
    calls = []
    monkeypatch.setattr(runtime, "_release_and_reconcile_inputs", lambda: calls.append("release")
                        or True)

    def send(command, **kwargs):
        calls.append(command)
        assert kwargs == {"lease_id": "synthetic-lease", "text": "hello"}
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(runtime_module, "send_command", send)
    assert runtime._deliver_game_chat(decision) is False
    assert calls == ["release", "chat"]
    assert runtime.metrics.game_chat_messages == 0
    assert runtime._last_player_chat_replied_ns is None


def test_confirmed_reply_is_delivered_once_after_inputs_are_reconciled(monkeypatch):
    frame = _frame(FIXTURES / "bedrock_health/full_health_1920x1080.png")
    board = _board(frame, monkeypatch)
    runtime = _runtime(board, frame)
    decision = bind_game_chat_authority(CognitionDecision(game_chat="hello"), board)
    calls = []
    monkeypatch.setattr(runtime, "_release_and_reconcile_inputs", lambda: calls.append("release")
                        or True)
    monkeypatch.setattr(runtime_module, "send_command", lambda command, **kwargs: (
        calls.append(command) or {"sent": True, "characters": len(kwargs["text"])}
    ))
    assert runtime._deliver_game_chat(decision)
    assert not runtime._deliver_game_chat(decision)
    assert calls == ["release", "chat"]
    assert runtime.metrics.game_chat_messages == 1


@pytest.mark.parametrize("failure", ["unknown_release", "new_message", "modal", "expired"])
def test_release_phase_cannot_refresh_authority_or_make_unsafe_chat_playable(monkeypatch, failure):
    frame = _frame(FIXTURES / "bedrock_health/full_health_1920x1080.png")
    board = _board(frame, monkeypatch)
    runtime = _runtime(board, frame)
    decision = bind_game_chat_authority(CognitionDecision(game_chat="hello"), board)

    def release():
        if failure == "new_message":
            fact = board.fact("social.player_message")
            assert fact is not None
            board.merge_semantics(instance_id="bedrock:1.26.52.3:retained-test", facts=(
                fact.model_copy(update={"value": "NewKid: new question"}),
            ))
        elif failure == "modal":
            runtime.perception.last_capture = _frame(
                FIXTURES / "bedrock_menu/server_list_transfer_1920x1080.png",
            )
        elif failure == "expired":
            monkeypatch.setattr(perception_module.time, "monotonic_ns", lambda: NOW + 250_000_000)
        return failure != "unknown_release"

    monkeypatch.setattr(runtime, "_release_and_reconcile_inputs", release)
    monkeypatch.setattr(runtime_module, "send_command", lambda *_args, **_kwargs: pytest.fail(
        "no chat input without confirmed release and fresh same-message safe surface",
    ))
    assert not runtime._deliver_game_chat(decision)
    assert runtime.metrics.game_chat_messages == 0
