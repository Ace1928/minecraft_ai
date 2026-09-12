from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from minecraft_ai.control.visual_progress import MiningVisualProgress
from minecraft_ai.mining_control import MiningLeaseGuard
from minecraft_ai.motor import MotorIntent
from minecraft_ai.outcome_verifier import OutcomeKind, OutcomeStatus, TemporalOutcomeVerifier
from minecraft_ai.safety import MotorAction
from minecraft_ai.skills import (
    SkillCondition,
    SkillFailureCode,
    SkillLibrary,
    SkillOutcome,
    SkillRun,
    SkillSpec,
    SkillStage,
    SkillStats,
)
from minecraft_ai.skills.recovery import select_learned_recovery
from minecraft_ai.storage import StateDatabase
from test_mining_control import _fact, _mining_board


_LUMA_A = "20" * 64
_LUMA_B = "80" * 16 + "20" * 48


def _luma_board(now: int, value: str, *, observed: int | None = None):
    return _mining_board(
        now_ns=now,
        extra_facts=(
            _fact("frame.crosshair_luma_grid", value, now_ns=now if observed is None else observed),
        ),
    )


def test_spatial_luma_progress_without_hash_change() -> None:
    tracker = MiningVisualProgress()
    now = time.monotonic_ns()
    assert not tracker.observe(_luma_board(now, _LUMA_A), now_ns=now)
    assert tracker.observe(_luma_board(now + 1, _LUMA_B), now_ns=now + 1)
    assert tracker.changes == 1


def test_uniform_lighting_is_not_spatial_progress() -> None:
    now = time.monotonic_ns()
    tracker = MiningVisualProgress()
    tracker.observe(_luma_board(now, _LUMA_A), now_ns=now)
    assert not tracker.observe(_luma_board(now + 1, "80" * 64), now_ns=now + 1)
    assert tracker.changes == 0


@pytest.mark.parametrize("offset", [-1, 0, 2])
def test_old_repeated_or_future_luma_is_not_progress(offset: int) -> None:
    now = time.monotonic_ns()
    tracker = MiningVisualProgress()
    tracker.observe(_luma_board(now, _LUMA_A), now_ns=now)
    assert not tracker.observe(_luma_board(now + 1, _LUMA_B, observed=now + offset), now_ns=now + 1)
    assert tracker.changes == 0


@pytest.mark.parametrize("value", ["z" * 128, "20", "  " * 64])
def test_malformed_luma_is_not_progress(value: str) -> None:
    now = time.monotonic_ns()
    assert not MiningVisualProgress().observe(_luma_board(now, value), now_ns=now)


def test_repeated_hash_observation_cannot_renew_signal_grace() -> None:
    now = time.monotonic_ns()
    board = _mining_board(now_ns=now)
    intent = MotorIntent(skill_id="mine", mode="mine", episode_id="e")
    guard = MiningLeaseGuard()
    assert (
        guard.inspect(
            MotorAction(sequence=1, buttons_down=("left",)), board, intent, now_ns=now
        ).failure_code
        is None
    )
    failed = guard.inspect(MotorAction(sequence=2), board, intent, now_ns=now + 800_000_000)
    assert failed.failure_code == SkillFailureCode.MINING_VISUAL_SIGNAL_LOST
    assert failed.force_release_left


def test_luma_progress_extends_soft_budget_but_never_absolute_cap() -> None:
    now = time.monotonic_ns()
    guard = MiningLeaseGuard(absolute_max_ms=5_000)
    intent = MotorIntent(skill_id="mine", mode="mine", episode_id="e")
    for index in range(11):
        tick = now + index * 500_000_000
        result = guard.inspect(
            MotorAction(sequence=index + 1, buttons_down=("left",) if not index else ()),
            _luma_board(tick, _LUMA_A if index % 2 == 0 else _LUMA_B),
            intent,
            now_ns=tick,
        )
        if index < 10:
            assert result.failure_code is None, (index, result)
        else:
            assert result.failure_code == SkillFailureCode.MINING_LEASE_EXPIRED
            assert result.force_release_left


def test_uniform_flicker_does_not_extend_mining() -> None:
    now = time.monotonic_ns()
    guard = MiningLeaseGuard()
    intent = MotorIntent(skill_id="mine", mode="mine", episode_id="e")
    for index in range(4):
        tick = now + index * 500_000_000
        result = guard.inspect(
            MotorAction(sequence=index + 1, buttons_down=("left",) if not index else ()),
            _luma_board(tick, "20" * 64 if index % 2 == 0 else "80" * 64),
            intent,
            now_ns=tick,
        )
    assert result.failure_code == SkillFailureCode.MINING_VISUAL_STAGNATION


def test_luma_progress_is_not_a_success_label() -> None:
    now = time.monotonic_ns()
    verifier = TemporalOutcomeVerifier()
    verifier.begin("run", OutcomeKind.MINING, _luma_board(now, _LUMA_A), now_ns=now)
    for index in range(8):
        tick = now + index * 500_000_000
        result = verifier.observe(
            _luma_board(tick, _LUMA_A if index % 2 == 0 else _LUMA_B),
            action=MotorAction(sequence=index + 1, buttons_down=("left",) if not index else ()),
            now_ns=tick,
        )
        assert result.status not in {OutcomeStatus.SUCCEEDED, OutcomeStatus.STALLED}
        if result.status == OutcomeStatus.PROGRESS:
            assert "frame.crosshair_luma_grid" in result.evidence_keys


def _library() -> SkillLibrary:
    library = SkillLibrary()
    for name in ("backoff", "jump", "look"):
        library.register(SkillSpec(skill_id=name, name=name))
    return library


def test_recovery_learns_from_actual_outcomes_and_exhausts_failed_choices() -> None:
    now = time.monotonic_ns()
    library = _library()
    board = _mining_board(now_ns=now)
    options = ("backoff", "jump", "look")
    chosen = select_learned_recovery(library, options, board, context_key="wall")
    assert chosen is not None and chosen.skill_id == "backoff"
    for index in range(2):
        library.record(
            SkillRun(
                run_id=str(index),
                skill_id="backoff",
                started_ns=now,
                outcome=SkillOutcome.FAILED,
                context_key="wall",
            )
        )
    library.record(
        SkillRun(
            run_id="win",
            skill_id="jump",
            started_ns=now,
            outcome=SkillOutcome.SUCCEEDED,
            context_key="wall",
        )
    )
    assert select_learned_recovery(library, options, board, context_key="wall").skill_id == "jump"
    for skill_id in options:
        library.stats[(skill_id, "wall")] = SkillStats(failures=2, consecutive_failures=2)
    assert select_learned_recovery(library, options, board, context_key="wall") is None


def test_recovery_never_uses_infeasible_retired_or_unknown_skills() -> None:
    now = time.monotonic_ns()
    library = _library()
    library.specs["backoff"] = library.get("backoff").model_copy(
        update={"stage": SkillStage.RETIRED},
    )
    library.specs["jump"] = library.get("jump").model_copy(
        update={
            "preconditions": (SkillCondition(key="unseen", operator="truthy"),),
        }
    )
    assert (
        select_learned_recovery(
            library,
            ("unknown", "backoff", "jump", "look"),
            _mining_board(now_ns=now),
            context_key="wall",
        ).skill_id
        == "look"
    )


def test_starvation_counts_in_reporting_not_competence() -> None:
    library = _library()
    for index in range(8):
        library.record(
            SkillRun(
                run_id=str(index),
                skill_id="jump",
                started_ns=1,
                outcome=SkillOutcome.FAILED,
                context_key="wall",
                failure_code=SkillFailureCode.CONTROLLER_STARVATION,
            )
        )
    stats = library.stats[("jump", "wall")]
    assert stats.failures == stats.censored_failures == 8
    assert stats.attempts == 8 and stats.decisive_attempts == stats.consecutive_failures == 0
    assert library.hierarchical_success_probability("jump", "wall") == 0.5
    assert library.contextual_score("jump", "wall") == 0.5


def test_local_evidence_not_counted_in_its_own_prior() -> None:
    library = _library()
    library.stats[("jump", "wall")] = SkillStats(successes=1)
    assert library.hierarchical_success_probability("jump", "wall") == pytest.approx(3 / 4)
    library.stats[("jump", "water")] = SkillStats(failures=8)
    assert library.hierarchical_success_probability("jump", "wall") == pytest.approx(1.1 / 2)
    assert library.hierarchical_success_probability("jump", "unseen") == pytest.approx(2 / 11)


def test_learned_stats_survive_database_restart(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite3"
    library = _library()
    with StateDatabase(path) as db:
        for spec in library.specs.values():
            db.save_skill(spec)
        db.save_skill_stats(
            "jump",
            "wall",
            SkillStats(successes=3, failures=8, censored_failures=8),
        )
    with StateDatabase(path) as db:
        restored = db.load_skills()
        assert restored.stats[("jump", "wall")].censored_failures == 8
        selected = select_learned_recovery(
            restored,
            ("backoff", "jump"),
            _mining_board(now_ns=time.monotonic_ns()),
            context_key="wall",
        )
        assert selected is not None and selected.skill_id == "jump"


def test_v7_stats_migrate_without_relabelling_historical_failures(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite3"
    with StateDatabase(path) as db:
        db.save_skill(SkillSpec(skill_id="jump", name="Jump"))
        db.save_skill_stats("jump", "wall", SkillStats(successes=2, failures=4))
    with sqlite3.connect(path) as conn:
        conn.execute("ALTER TABLE skill_stats DROP COLUMN censored_failures")
        conn.execute("UPDATE meta SET value='7' WHERE key='schema_version'")
    with StateDatabase(path) as db:
        stats = db.load_skills().stats[("jump", "wall")]
        assert stats.failures == 4 and stats.censored_failures == 0
        assert stats.decisive_attempts == 6
        version = db.connection.execute(
            "SELECT value FROM meta WHERE key='schema_version'",
        ).fetchone()
        assert version[0] == "8"
