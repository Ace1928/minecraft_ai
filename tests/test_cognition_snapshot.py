from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError

import pytest

from minecraft_ai.perception import (
    EvidenceRegion,
    FrameState,
    PerceptionBlackboard,
    PerceptionEvidence,
    PerceptionFact,
    ScreenRegion,
    Track,
)


def _board():
    board = PerceptionBlackboard()
    region = ScreenRegion(x=0, y=0, width=1, height=1)
    evidence = PerceptionEvidence(
        evidence_id="observation", frame_id=7, captured_ns=1_000_000,
        region_kind=EvidenceRegion.WORLD, region=region, pixel_sha256="a" * 64,
        crop_width=16, crop_height=9,
    )
    track = Track(
        track_id="target", label="log", confidence=0.9, region=region,
        first_seen_ns=1_000_000, last_seen_ns=1_000_000,
        attributes={"grounding": "original"}, evidence_refs=(evidence.evidence_id,),
    )
    fact = PerceptionFact(
        key="target.visible", value=True, confidence=0.9, observed_ns=1_000_000,
        source="observed", expires_after_ms=1, evidence_refs=(evidence.evidence_id,),
    )
    board.publish(FrameState(
        frame_id=7, captured_ns=1_000_000, instance_id="world", width=16, height=9,
        facts=(fact,), tracks=(track,), evidence=(evidence,),
    ))
    return board, track


def test_snapshot_retains_fixed_freshness_and_owned_observations(monkeypatch):
    board, original_track = _board()
    snapshot = board.cognition_snapshot(now_ns=2_000_000)
    original_track.attributes["grounding"] = "mutated after submission"
    board.publish(FrameState(
        frame_id=8, captured_ns=3_000_000, instance_id="world", width=16, height=9,
    ))
    board.remove_semantic_facts(("target.visible",), expected_source="observed")
    monkeypatch.setattr("minecraft_ai.perception.time.monotonic_ns", lambda: 99_000_000)

    assert snapshot.fact("target.visible").value is True
    assert snapshot.fresh_facts()["target.visible"].observed_ns == 1_000_000
    assert snapshot.latest().tracks[0].attributes["grounding"] == "original"
    assert snapshot.latest().evidence[0].pixel_sha256 == "a" * 64
    assert (snapshot.instance_id, snapshot.frame_id, snapshot.captured_ns) == (
        "world", 7, 1_000_000,
    )
    assert snapshot.raw_latest().frame_id == 7
    assert snapshot.fact("target.visible", now_ns=snapshot.snapshot_ns) is not None
    with pytest.raises(ValueError, match="clock"):
        snapshot.fact("target.visible", now_ns=99_000_000)


def test_returned_mutable_attributes_cannot_change_snapshot_or_live_board():
    board, original_track = _board()
    snapshot = board.cognition_snapshot(now_ns=1_500_000)
    snapshot.latest().tracks[0].attributes["grounding"] = "changed returned merged frame"
    snapshot.raw_latest().tracks[0].attributes["grounding"] = "changed returned raw frame"
    returned_facts = snapshot.fresh_facts()
    returned_facts.clear()
    assert snapshot.latest().tracks[0].attributes["grounding"] == "original"
    assert original_track.attributes["grounding"] == "original"
    assert snapshot.fact("target.visible") is not None
    with pytest.raises(FrozenInstanceError):
        snapshot.frame_id = 12
    assert not hasattr(snapshot, "publish")
    assert not hasattr(snapshot, "merge_semantics")


def test_source_hash_covers_canonical_semantics_and_real_identity():
    board, _ = _board()
    snapshot = board.cognition_snapshot(now_ns=1_500_000)
    payload = json.loads(snapshot.canonical_source)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert snapshot.source_sha256 == hashlib.sha256(canonical.encode()).hexdigest()
    assert payload["contract"] == "minecraft_ai.cognition_source.v1"
    assert payload["merged_frame"]["frame_id"] == 7
    assert snapshot.source_sha256 != "a" * 64  # Referenced crop hash is a separate identity.
    assert board.cognition_snapshot(now_ns=1_500_000).source_sha256 == snapshot.source_sha256

    board.merge_semantics(instance_id="world", facts=(PerceptionFact(
        key="target.visible", value=False, confidence=0.9, observed_ns=1_000_000,
        source="new-observer", expires_after_ms=1, evidence_refs=("observation",),
    ),))
    assert board.cognition_snapshot(now_ns=1_500_000).source_sha256 != snapshot.source_sha256


def test_snapshot_does_not_resurrect_expired_or_future_facts():
    board, _ = _board()
    board.merge_semantics(instance_id="world", facts=(PerceptionFact(
        key="future", value=True, confidence=1.0, observed_ns=9_000_000, source="observed",
    ),))
    snapshot = board.cognition_snapshot(now_ns=3_000_000)
    assert snapshot.fresh_facts() == {}
    assert snapshot.latest().facts == ()
    assert snapshot.fact("target.visible") is None
    assert snapshot.fact("future") is None


def test_snapshot_captures_its_clock_under_one_blackboard_lock(monkeypatch):
    board, _ = _board()
    original_lock = board._lock

    class CheckedLock:
        entries = 0
        held = False

        def __enter__(self):
            original_lock.acquire()
            self.entries += 1
            self.held = True

        def __exit__(self, *args):
            self.held = False
            original_lock.release()

    lock = CheckedLock()
    board._lock = lock

    def now():
        assert lock.held
        return 1_500_000

    monkeypatch.setattr("minecraft_ai.perception.time.monotonic_ns", now)
    snapshot = board.cognition_snapshot()
    assert lock.entries == 1
    assert not lock.held
    assert snapshot.snapshot_ns == 1_500_000
    assert snapshot.fact("target.visible") is not None


def test_snapshot_requires_bounded_metadata_and_a_valid_capture():
    with pytest.raises(ValueError, match="captured frame"):
        PerceptionBlackboard().cognition_snapshot(now_ns=1)
    board, _ = _board()
    with pytest.raises(ValueError, match="metadata bound"):
        board.cognition_snapshot(now_ns=1_500_000, max_bytes=100)
    with pytest.raises(ValueError, match="non-future"):
        board.cognition_snapshot(now_ns=1)
    with pytest.raises(ValueError, match="clock"):
        board.cognition_snapshot(now_ns=True)
    with pytest.raises(ValueError, match="byte bound"):
        board.cognition_snapshot(max_bytes=True)


def test_existing_high_level_controller_reads_snapshot_without_mutating_board(monkeypatch):
    from minecraft_ai.builtin_skills import build_bootstrap_skill_library
    from minecraft_ai.cognition import CognitionContext, HighLevelController
    from minecraft_ai.models import ModelResponse
    from minecraft_ai.roles import get_role

    class Model:
        model_id = "snapshot-test"
        messages = ()

        def complete(self, messages):
            self.messages = messages
            return ModelResponse(
                text='{"r":"observe","g":null,"s":null,"p":{},"o":null,"c":null,'
                     '"x":false,"q":[],"w":null,"d":null,"n":[]}',
                model=self.model_id, latency_ms=0,
            )

    board, _ = _board()
    snapshot = board.cognition_snapshot(now_ns=1_500_000)
    monkeypatch.setattr("minecraft_ai.perception.time.monotonic_ns", lambda: 99_000_000)
    model = Model()
    controller = HighLevelController(model, build_bootstrap_skill_library())
    context = CognitionContext(
        role=get_role("generalist"), goals=(), memories=(), promises=(), wiki=(),
    )
    controller.decide(snapshot, context)
    assert model.messages
    assert controller.metrics.failures == 0
    assert snapshot.fact("target.visible").value is True
    assert board.raw_latest().frame_id == 7
