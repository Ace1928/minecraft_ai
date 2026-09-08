"""Keep only the named, requested target bundle during one scene-owned handoff."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from minecraft_ai.cognition import CognitionContext, CognitionDecision
from minecraft_ai.execution import initiation_satisfied
from minecraft_ai.perception import EvidenceRegion, PerceptionEvidence, PerceptionFact, ScreenRegion
from minecraft_ai.roles import get_role
from test_agent_core import (
    _attach_probe_cognition,
    _publish_probe_world_frame,
    _runtime_with_waiting_cognition_perception,
)


def _handoff(monkeypatch, *, key="target.mineable", value=True, corruption=None, anchor=True):
    clock = [10_000_000_000]
    monkeypatch.setattr("minecraft_ai.runtime.time.monotonic_ns", lambda: clock[0])
    monkeypatch.setattr("minecraft_ai.runtime.operator_pause_latched", lambda: False)
    runtime = _runtime_with_waiting_cognition_perception(now_ns=clock[0])
    clock[0] += 50_000_000
    source = "vlm:test:pending-grounding"
    evidence_id = "frame-1:world"
    companion = PerceptionFact(
        key=key, value=value, source=source, observed_ns=clock[0], confidence=0.83,
        expires_after_ms=15_000, evidence_refs=(evidence_id,),
    )
    if corruption == "source":
        companion = companion.model_copy(update={"source": "vlm:test:other-query"})
    elif corruption == "timestamp":
        companion = companion.model_copy(update={"observed_ns": clock[0] - 1})
    elif corruption == "weak":
        companion = companion.model_copy(update={"confidence": 0.69})
    anchor_facts = () if anchor is None else (PerceptionFact(
        key="target.visible", value=anchor, source=source, observed_ns=clock[0],
        confidence=0.9, expires_after_ms=15_000, evidence_refs=(evidence_id,),
    ),)
    _publish_probe_world_frame(runtime, clock[0])
    runtime.blackboard.merge_semantics(
        instance_id="bedrock:test",
        facts=anchor_facts + (companion, PerceptionFact(
            key="scene.observation_dhash", value="0123456789abcdef", confidence=1.0,
            observed_ns=clock[0], source=source, expires_after_ms=120_000,
        )),
        evidence=(PerceptionEvidence(
            evidence_id=evidence_id, frame_id=1, captured_ns=clock[0],
            region_kind=EvidenceRegion.WORLD, region=ScreenRegion(x=0, y=0, width=1, height=1),
            pixel_sha256="0" * 64, crop_width=1280, crop_height=720,
        ),),
    )
    requested = ("target.visible",) if corruption == "unrequested" else ("target.visible", key)
    description = None if corruption == "unnamed" else "   " if corruption == "blank" else "oak log"
    runtime._cognition_perception_probe = replace(
        runtime._cognition_perception_probe, requested_keys=requested,
        target_description=description, settle_dhash="0123456789abcdef",
    )
    runtime.perception = SimpleNamespace(
        active_vlm=SimpleNamespace(status=lambda: {
            "completed": 1, "failures": 0, "thread_alive": True,
        }),
        semantic_available=lambda: True,
    )
    runtime._reconcile_cognition_perception_probe()
    return runtime, clock, companion


@pytest.mark.parametrize("key,value", (
    ("target.near", True), ("target.near", False), ("target.mineable", True),
    ("target.mineable", False), ("target.kind", "oak_log"),
    ("target.dx", 0.0), ("target.dy", -0.25),
))
def test_requested_target_companion_survives_slow_followup_without_provenance_change(
    monkeypatch, key, value,
):
    runtime, clock, original = _handoff(monkeypatch, key=key, value=value)
    probe = runtime._cognition_perception_probe
    assert tuple(fact.key for fact in probe.retained_facts) == ("target.visible", key)
    clock[0] += 30_000_000_000
    _publish_probe_world_frame(runtime, clock[0])
    runtime._reconcile_cognition_perception_probe()
    assert runtime._cognition_perception_probe is probe
    assert not original.fresh()
    current = runtime.blackboard.fact(key)
    assert current is not None
    assert current.model_dump(exclude={"expires_after_ms"}) == original.model_dump(
        exclude={"expires_after_ms"},
    )
    assert current.observed_ns + current.expires_after_ms * 1_000_000 <= probe.handoff_deadline_ns
    assert runtime.metrics.semantic_requests == 0  # Retention starts no new perception work.
    if key == "target.mineable" and value is False:
        assert not initiation_satisfied(
            runtime.skills.get("mine_visible_block"), runtime.blackboard,
        )


@pytest.mark.parametrize("corruption", (
    "unnamed", "blank", "unrequested", "source", "timestamp", "weak",
))
def test_target_companion_requires_explicit_matching_requested_evidence(monkeypatch, corruption):
    runtime, clock, original = _handoff(monkeypatch, corruption=corruption)
    probe = runtime._cognition_perception_probe
    assert tuple(fact.key for fact in probe.retained_facts) == ("target.visible",)
    assert runtime.blackboard.fact("target.mineable") == original
    clock[0] += 30_000_000_000
    _publish_probe_world_frame(runtime, clock[0])
    runtime._reconcile_cognition_perception_probe()
    assert runtime._cognition_perception_probe is probe
    assert runtime.blackboard.fact("target.mineable") is None


@pytest.mark.parametrize("key", ("target.broken", "inventory.logs", "danger.immediate"))
def test_success_possession_and_safety_are_not_target_companions(monkeypatch, key):
    runtime, _clock, original = _handoff(monkeypatch, key=key, value=False)
    assert tuple(fact.key for fact in runtime._cognition_perception_probe.retained_facts) == (
        "target.visible",
    )
    assert runtime.blackboard.fact(key) == original


@pytest.mark.parametrize("anchor", (None, False))
def test_mineable_alone_cannot_anchor_target_handoff(monkeypatch, anchor):
    runtime, _clock, original = _handoff(monkeypatch, anchor=anchor)
    assert runtime._cognition_perception_probe is None
    assert runtime.blackboard.fact("target.mineable") == original
    assert runtime.executor.run is None


def test_slow_mining_followup_keeps_companion_through_only_existing_action_grace(monkeypatch):
    runtime, clock, original = _handoff(monkeypatch)
    future = _attach_probe_cognition(runtime)
    clock[0] += 30_000_000_000
    _publish_probe_world_frame(runtime, clock[0])
    future.set_result(CognitionDecision(skill_id="mine_visible_block"))
    runtime._consume_cognition()
    assert runtime._cognition_perception_probe is None
    assert runtime.executor.run is not None
    assert runtime.executor.run.skill_id == "mine_visible_block"
    current = runtime.blackboard.fact("target.mineable")
    assert current is not None and current.observed_ns == original.observed_ns
    assert current.observed_ns + current.expires_after_ms * 1_000_000 <= clock[0] + 2_000_000_000
    clock[0] += 2_100_000_000
    assert runtime.blackboard.fact("target.visible") is None
    assert runtime.blackboard.fact("target.mineable") is None


@pytest.mark.parametrize("preemption", ("scene", "execution", "pause", "operator", "replacement"))
def test_preemption_revokes_companions_but_preserves_newer_fact_producer(monkeypatch, preemption):
    runtime, clock, original = _handoff(monkeypatch)
    _attach_probe_cognition(runtime)
    clock[0] += 1_000_000_000
    replacement = original.model_copy(update={
        "source": "operator:new-query", "observed_ns": clock[0], "value": False,
    })
    _publish_probe_world_frame(
        runtime, clock[0], frame_hash="ffffffffffffffff" if preemption == "scene"
        else "0123456789abcdef", facts=(replacement,) if preemption == "replacement" else (),
    )
    if preemption == "execution":
        runtime._execution_revision += 1
    elif preemption == "pause":
        monkeypatch.setattr("minecraft_ai.runtime.operator_pause_latched", lambda: True)
    if preemption == "operator":
        runtime.cognition_hz = 1
        runtime._new_queued_operator_message_waiting = lambda: True
        runtime._cognition_context = lambda: CognitionContext(
            get_role("generalist"), (), (), (), (),
        )
        runtime._stage_operator_fast_path = lambda _context: True
        runtime._start_cognition_if_due()
    else:
        runtime._reconcile_cognition_perception_probe()
    assert runtime._cognition_perception_probe is None and runtime._pending_decision is None
    assert runtime.blackboard.fact("target.visible") is None
    assert runtime.blackboard.fact("target.mineable") == (
        replacement if preemption == "replacement" else None
    )
