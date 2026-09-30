from __future__ import annotations

import time
import hashlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from minecraft_ai.perception import (
    ActivePerceptionQuery, ChatLine, EvidenceRegion, FrameState, PerceptionBlackboard,
    PerceptionEvidence, ScreenRegion,
)
from minecraft_ai.perception.service import (
    ActiveVLMWorker, SemanticJob, SemanticObservation, _semantic_observation,
)
from minecraft_ai.platforms.bedrock_x11 import CapturedFrame
from minecraft_ai.runtime import AgentRuntime


def _report(evidence=()):
    return SimpleNamespace(
        observed_values=lambda: {}, evidence_by_key=lambda: {}, confidence_by_key=lambda: {},
        uncertainty=0.2, deterministic_summary="One cited chat line.", evidence=evidence,
        rejections=(), claims=(), model_summary="", summary_accepted=False, tracks=(),
        chat=(SimpleNamespace(
            text="How do I craft a Poké Ball?", speaker="KidPlayer", confidence=0.93,
            evidence_id="frame-1:chat",
        ),),
    )


def test_validated_chat_identity_confidence_and_citation_survive_publication() -> None:
    now = time.monotonic_ns()
    board = PerceptionBlackboard()
    board.publish(FrameState(
        frame_id=1, captured_ns=now, instance_id="bedrock:1.26.52.3:x11:42", width=2, height=2,
    ))
    frame = CapturedFrame(1, now, 2, 2, b"\0\0\0\xff" * 4)
    evidence = PerceptionEvidence(
        evidence_id="frame-1:chat", frame_id=1, captured_ns=now,
        region_kind=EvidenceRegion.CHAT,
        region=ScreenRegion(x=0, y=0, width=1, height=1),
        pixel_sha256=hashlib.sha256(frame.bgra).hexdigest(), crop_width=2, crop_height=2,
    )
    observation = _semantic_observation(_report((evidence,)))
    model = Mock(model_id="native-chat-capability")
    worker = ActiveVLMWorker(model, board, "bedrock:1.26.52.3:x11:42")
    query = ActivePerceptionQuery(
        query_id="chat-query", question="Read player chat", frame_id=1,
    )
    worker._publish(SemanticJob(query=query, frame=frame, frame_dhash="0" * 16), observation)
    published = board.latest().chat
    assert len(published) == 1
    line = published[0]
    assert line.text == "How do I craft a Poké Ball?"
    assert line.speaker == "KidPlayer"
    assert line.confidence == 0.93
    assert line.evidence_refs == ("frame-1:chat",)
    assert line.observed_ns >= frame.captured_ns


def test_legacy_text_only_observation_remains_unauthenticated_chat() -> None:
    observation = SemanticObservation(chat=("Some text",))
    assert observation.chat_speakers == ()
    assert observation.chat_confidences == ()


@pytest.mark.parametrize("metadata", [
    {"chat_speakers": ("Kid", "Other")}, {"chat_confidences": (0.8, 0.9)},
    {"chat_confidences": (-0.2,)}, {"chat_confidences": (float("nan"),)},
])
def test_misaligned_or_invalid_metadata_cannot_assign_chat_identity(metadata) -> None:
    with pytest.raises(ValidationError):
        SemanticObservation(chat=("One observed line",), **metadata)


def _chat_runtime(*lines):
    board = PerceptionBlackboard()
    board.publish(FrameState(
        frame_id=1, captured_ns=time.monotonic_ns(), instance_id="bedrock:chat",
        width=2, height=2, chat=tuple(lines),
    ))
    return SimpleNamespace(
        blackboard=board, role=SimpleNamespace(role_id="Eidos"),
        perception=SimpleNamespace(instance_id="bedrock:chat"),
        _last_player_chat_signature=None, _last_player_chat_replied_ns=None,
    )


def test_player_question_authority_preserves_observation_confidence_and_time() -> None:
    observed_ns = time.monotonic_ns() - 1_000_000
    runtime = _chat_runtime(ChatLine(
        speaker="KidPlayer", text="Where is copper?", observed_ns=observed_ns, confidence=0.83,
    ))
    AgentRuntime._publish_player_chat_facts(runtime)
    fact = runtime.blackboard.fact("social.player_message", min_confidence=0.7)
    assert fact is not None
    assert fact.confidence == 0.83
    assert fact.observed_ns == observed_ns


@pytest.mark.parametrize("overrides", [
    {"confidence": 0.69}, {"speaker": None}, {"speaker": "eIDOS"}, {"speaker": "Console"},
    {"observed_ns": 1}, {"observed_ns": time.monotonic_ns() + 60_000_000_000},
])
def test_weak_stale_future_or_self_chat_does_not_authorize_response(overrides) -> None:
    values = dict(
        speaker="KidPlayer", text="Where is copper?",
        observed_ns=time.monotonic_ns(), confidence=0.9,
    )
    values.update(overrides)
    runtime = _chat_runtime(ChatLine(**values))
    AgentRuntime._publish_player_chat_facts(runtime)
    assert runtime.blackboard.fact("social.player_message") is None
