from __future__ import annotations

from minecraft_ai.episodes import RuntimeEventKind
from minecraft_ai.execution import ExecutionTick
from minecraft_ai.outcome_verifier import (
    OutcomeKind,
    OutcomeSignal,
    OutcomeStatus,
    OutcomeVerification,
)
from minecraft_ai.runtime import (
    _terminal_run_memory,
    _trajectory_outcome_annotations,
    _verified_outcome_event,
)
from minecraft_ai.skills import SkillFailureCode, SkillOutcome, SkillRun, SkillStats


def _block_broken() -> tuple[SkillRun, OutcomeVerification]:
    run = SkillRun(
        run_id="verified-mine",
        skill_id="mine_visible_block",
        started_ns=100,
        ended_ns=900,
        outcome=SkillOutcome.SUCCEEDED,
    )
    verification = OutcomeVerification(
        run_id=run.run_id,
        kind=OutcomeKind.MINING,
        status=OutcomeStatus.SUCCEEDED,
        signal=OutcomeSignal.BLOCK_BROKEN,
        observed_ns=850,
        confidence=0.92,
        reason="bound target was replaced after the released attack",
        evidence_keys=("frame.crosshair_dhash", "target.visible", "target.track_id"),
        target_kind="oak_log",
    )
    return run, verification


def test_block_broken_evidence_reaches_event_trajectory_and_memory() -> None:
    run, verification = _block_broken()
    tick = ExecutionTick(run=run, action=None, outcome_verification=verification)

    rewards, event_ids = _trajectory_outcome_annotations(tick)
    event = _verified_outcome_event(
        run,
        verification,
        observed_ns=1_000,
        trajectory_id="trajectory:test",
    )
    memory = _terminal_run_memory(
        run,
        SkillStats(successes=1),
        observed_ns=1_000,
        existing={},
        outcome_verification=verification,
    )

    assert rewards == {"block_broken": 0.92}
    assert event_ids == ("skill-run:verified-mine:block-broken",)
    assert event is not None
    assert event.kind == RuntimeEventKind.BLOCK_BROKEN
    assert event.event_id == event_ids[0]
    assert event.payload["evidence_keys_json"] == (
        '["frame.crosshair_dhash", "target.visible", "target.track_id"]'
    )
    assert event.payload["target_kind"] == "oak_log"
    assert memory is not None
    assert memory.metadata["verified_outcome"] == "block_broken"
    assert memory.metadata["verified_outcome_confidence"] == 0.92
    assert memory.metadata["verified_target_kind"] == "oak_log"


def test_non_terminal_or_mismatched_evidence_never_claims_block_broken() -> None:
    run, verification = _block_broken()
    mismatched = verification.model_copy(update={"run_id": "foreign-run"})
    tick = ExecutionTick(run=run, action=None, outcome_verification=mismatched)

    assert _trajectory_outcome_annotations(tick) == ({}, ())
    assert (
        _verified_outcome_event(
            run,
            mismatched,
            observed_ns=1_000,
            trajectory_id=None,
        )
        is None
    )


def test_starvation_memory_keeps_controller_cause_not_obstacle_claim() -> None:
    reason = "controller emitted no sustained locomotion within the bounded startup window"
    run = SkillRun(
        run_id="starved-controller", skill_id="traverse_visible_obstacle",
        started_ns=100, ended_ns=3_100_000_100, outcome=SkillOutcome.FAILED,
        failure_code=SkillFailureCode.CONTROLLER_STARVATION, failure_reason=reason,
    )
    verification = OutcomeVerification(
        run_id=run.run_id, kind=OutcomeKind.TRAVERSAL, status=OutcomeStatus.STALLED,
        signal=OutcomeSignal.CONTROLLER_STARVATION, observed_ns=run.ended_ns or 0,
        confidence=0.96, reason=reason,
    )

    memory = _terminal_run_memory(
        run, SkillStats(failures=1), observed_ns=4_000_000_000, existing={},
        outcome_verification=verification,
    )

    assert memory is not None
    assert reason in memory.text
    assert memory.metadata["reported_reason"] == reason
    assert memory.metadata["failure_code"] == "controller.starvation"
    assert memory.metadata["verified_outcome"] == "controller_starvation"
    assert memory.metadata["verified_outcome_evidence_json"] == "[]"
