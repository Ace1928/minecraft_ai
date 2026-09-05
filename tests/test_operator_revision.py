from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from minecraft_ai.perception import ScreenRegion, Track
from minecraft_ai.memory import MemoryKind, MemoryRecord
from minecraft_ai.planning import Goal
from minecraft_ai.skills import SkillStats
from minecraft_ai.social import OperatorMessage, OperatorMessageKind, OperatorMessageStatus
from minecraft_ai.storage import MAX_OPERATOR_REVISION, StateDatabase


def _message(identity: str = "first", **changes: object) -> OperatorMessage:
    return OperatorMessage.model_validate({
        "message_id": identity, "created_ns": 10, "text": "Go to the tree.", **changes,
    })


def _target(identity: str = "tree", **changes: object) -> Track:
    return Track.model_validate({
        "track_id": identity, "label": "oak_log", "confidence": 1.0,
        "region": ScreenRegion(x=0.2, y=0.3, width=0.1, height=0.2),
        "first_seen_ns": 10, "last_seen_ns": 10,
        "attributes": {"source": "operator"}, **changes,
    })


def _revision_value(db: StateDatabase, value: str) -> None:
    db.connection.execute(
        "INSERT INTO meta(key, value) VALUES('operator_authority_revision', ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (value,),
    )
    db.connection.commit()


def test_revision_is_durable_and_shared_by_independent_connections(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite3"
    with StateDatabase(path) as first, StateDatabase(path) as second:
        assert first.operator_revision() == second.operator_revision() == 0
        first.save_operator_message(_message())
        assert first.operator_revision() == second.operator_revision() == 1
        second.save_operator_target(_target())
        assert first.operator_revision() == second.operator_revision() == 2
        first.clear_operator_target()
        assert second.operator_revision() == 3
    with StateDatabase(path) as restored:
        assert restored.operator_revision() == 3
        assert restored.load_operator_context().messages == (_message(),)


@pytest.mark.parametrize("changes", [
    {"text": "Stop at the tree."}, {"author": "another operator"}, {"priority": 0.9},
    {"created_ns": 30}, {"kind": OperatorMessageKind.CORRECTION},
])
def test_replacing_message_content_advances_once_and_updates_sort_identity(
    tmp_path: Path, changes: dict[str, object],
) -> None:
    with StateDatabase(tmp_path / "state.sqlite3") as db:
        first = _message()
        db.save_operator_message(first)
        db.save_operator_message(_message("second", created_ns=20))
        replacement = _message(**changes)
        db.save_operator_message(replacement)
        assert db.operator_revision() == 3
        db.save_operator_message(replacement)
        assert db.operator_revision() == 3
        assert replacement in db.load_operator_messages()
        if "created_ns" in changes:
            assert db.load_operator_messages()[0] == replacement


@pytest.mark.parametrize("kind", [
    OperatorMessageKind.INSTRUCTION, OperatorMessageKind.QUESTION, OperatorMessageKind.FEEDBACK,
])
def test_delivery_and_acknowledgement_bookkeeping_do_not_advance_revision(
    tmp_path: Path, kind: OperatorMessageKind,
) -> None:
    with StateDatabase(tmp_path / "state.sqlite3") as db:
        message = _message(kind=kind)
        db.save_operator_message(message)
        db.save_operator_message(message)
        delivered = db.update_operator_message_status(
            message.message_id, OperatorMessageStatus.DELIVERED, timestamp_ns=20,
        )
        assert delivered.delivered_ns == 20
        acknowledged = db.update_operator_message_status(
            message.message_id, OperatorMessageStatus.ACKNOWLEDGED,
            timestamp_ns=30, response_text="Understood.",
        )
        db.save_operator_message(acknowledged.model_copy(update={
            "acknowledged_ns": 40, "response_text": "Updated receipt.",
        }))
        assert db.operator_revision() == 1
        assert db.load_operator_messages()[0].response_text == "Updated receipt."


def test_correction_consumption_and_archival_change_authority_once(tmp_path: Path) -> None:
    with StateDatabase(tmp_path / "state.sqlite3") as db:
        correction = _message(kind=OperatorMessageKind.CORRECTION)
        db.save_operator_message(correction)
        db.update_operator_message_status(
            correction.message_id, OperatorMessageStatus.DELIVERED, timestamp_ns=20,
        )
        assert db.operator_revision() == 1
        db.update_operator_message_status(
            correction.message_id, OperatorMessageStatus.ACKNOWLEDGED, timestamp_ns=30,
        )
        assert db.operator_revision() == 2
        db.update_operator_message_status(
            correction.message_id, OperatorMessageStatus.ACKNOWLEDGED,
            timestamp_ns=40, response_text="Already handled.",
        )
        assert db.operator_revision() == 2
        db.save_operator_message(correction)  # Explicitly restore a consumed correction.
        assert db.operator_revision() == 3
        db.update_operator_message_status(
            correction.message_id, OperatorMessageStatus.ARCHIVED, timestamp_ns=50,
        )
        assert db.operator_revision() == 4
        db.update_operator_message_status(
            correction.message_id, OperatorMessageStatus.ARCHIVED, timestamp_ns=60,
        )
        assert db.operator_revision() == 4


def test_target_replacement_clear_and_reactivation_are_effective_mutations(tmp_path: Path) -> None:
    with StateDatabase(tmp_path / "state.sqlite3") as db:
        target = _target()
        db.clear_operator_target()
        assert db.operator_revision() == 0
        db.save_operator_target(target)
        db.save_operator_target(target)
        assert db.operator_revision() == 1
        db.save_operator_target(_target(label="birch_log"))
        assert db.operator_revision() == 2
        db.save_operator_target(_target("different-tree"))
        assert db.operator_revision() == 3
        active = db.load_operator_target()
        assert active is not None and active.track_id == "different-tree"
        db.clear_operator_target()
        db.clear_operator_target()
        assert db.operator_revision() == 4
        assert db.load_operator_target() is None
        db.save_operator_target(target)
        assert db.operator_revision() == 5


def test_context_snapshot_cannot_mix_revision_with_newer_messages_or_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "state.sqlite3"
    with StateDatabase(path) as reader, StateDatabase(path) as writer:
        reader.save_operator_message(_message())
        original_load = reader.load_operator_messages

        def replace_between_snapshot_reads(
            *, statuses: set[OperatorMessageStatus] | None = None, limit: int = 100,
        ) -> tuple[OperatorMessage, ...]:
            writer.save_operator_message(_message("new", created_ns=20, text="Go elsewhere."))
            writer.save_operator_target(_target())
            return original_load(statuses=statuses, limit=limit)

        with monkeypatch.context() as patch:
            patch.setattr(reader, "load_operator_messages", replace_between_snapshot_reads)
            snapshot = reader.load_operator_context()
        assert snapshot.revision == 1
        assert snapshot.messages == (_message(),)
        assert snapshot.target is None
        assert not reader.connection.in_transaction
        current = reader.load_operator_context()
        assert current.revision == 3
        assert current.messages[0].message_id == "new"
        assert current.target == _target()
        # Returning a snapshot retains no writer lock across planner/model work.
        writer.clear_operator_target()
        assert reader.operator_revision() == 4


def test_admission_serializes_short_application_and_defers_normal_storage_commits(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.sqlite3"
    with StateDatabase(path) as planner, StateDatabase(path) as operator:
        operator.set_busy_timeout_ms(0)
        snapshot = planner.load_operator_context()
        with planner.admit_operator_revision(snapshot.revision) as current:
            assert current
            planner.save_goal(Goal(goal_id="planned", description="Go to the tree."))
            assert planner.connection.in_transaction
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                operator.save_operator_message(_message())
            assert operator.load_goals() == ()  # The decision is not partially committed.
        assert not planner.connection.in_transaction
        assert operator.load_goals()[0].goal_id == "planned"
        operator.save_operator_message(_message())
        with planner.admit_operator_revision(snapshot.revision) as current:
            assert not current
        assert planner.operator_revision() == 1


def test_failed_admission_application_rolls_back_goal_mutation_and_revision(tmp_path: Path) -> None:
    with StateDatabase(tmp_path / "state.sqlite3") as db:
        with pytest.raises(RuntimeError, match="application failed"):
            with db.admit_operator_revision(0) as current:
                assert current
                db.save_goal(Goal(goal_id="partial", description="Should not persist."))
                db.save_operator_target(_target())
                assert db.operator_revision() == 1
                raise RuntimeError("application failed")
        assert db.operator_revision() == 0
        assert db.load_goals() == ()
        assert db.load_operator_target() is None
        assert not db.connection.in_transaction


def test_failed_message_write_cannot_commit_a_revision_without_its_content(tmp_path: Path) -> None:
    with StateDatabase(tmp_path / "state.sqlite3") as db:
        db.connection.execute(
            "CREATE TRIGGER reject_operator_insert BEFORE INSERT ON operator_messages "
            "BEGIN SELECT RAISE(ABORT, 'injected write failure'); END",
        )
        db.connection.commit()
        with pytest.raises(sqlite3.IntegrityError, match="injected write failure"):
            db.save_operator_message(_message())
        assert db.operator_revision() == 0
        assert db.load_operator_messages() == ()
        assert not db.connection.in_transaction


def test_revision_exhaustion_rejects_mutation_without_wrap_or_lost_bookkeeping(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.sqlite3"
    with StateDatabase(path) as db:
        _revision_value(db, str(MAX_OPERATOR_REVISION - 1))
        db.save_operator_message(_message())
        assert db.operator_revision() == MAX_OPERATOR_REVISION
        db.update_operator_message_status(
            "first", OperatorMessageStatus.ACKNOWLEDGED, timestamp_ns=20,
        )
        with pytest.raises(OverflowError, match="exhausted"):
            db.save_operator_message(_message("rejected"))
        with pytest.raises(OverflowError, match="exhausted"):
            db.save_operator_target(_target())
        db.clear_operator_target()  # No effective change still succeeds at the limit.
        assert db.load_operator_messages()[0].status == OperatorMessageStatus.ACKNOWLEDGED
        assert len(db.load_operator_messages()) == 1
        assert db.load_operator_target() is None
    with StateDatabase(path) as restored:
        assert restored.operator_revision() == MAX_OPERATOR_REVISION


@pytest.mark.parametrize("raw", ["-1", "01", "invalid", str(MAX_OPERATOR_REVISION + 1)])
def test_invalid_persisted_revision_fails_closed_without_rewriting_state(
    tmp_path: Path, raw: str,
) -> None:
    with StateDatabase(tmp_path / "state.sqlite3") as db:
        _revision_value(db, raw)
        with pytest.raises(ValueError, match="revision"):
            db.load_operator_context()
        with pytest.raises(ValueError, match="revision"):
            db.save_operator_message(_message())
        assert db.load_operator_messages() == ()
        stored = db.connection.execute(
            "SELECT value FROM meta WHERE key='operator_authority_revision'",
        ).fetchone()[0]
        assert stored == raw


@pytest.mark.parametrize("expected", [True, -1, MAX_OPERATOR_REVISION + 1])
def test_invalid_expected_revision_is_rejected_before_admission(
    tmp_path: Path, expected: int,
) -> None:
    with StateDatabase(tmp_path / "state.sqlite3") as db:
        with pytest.raises(ValueError, match="revision"):
            with db.admit_operator_revision(expected):
                pytest.fail("invalid revision entered the admission body")
        assert not db.connection.in_transaction


def _memory() -> MemoryRecord:
    return MemoryRecord(
        memory_id="observation", kind=MemoryKind.WORKING,
        text="A tree was observed.", created_ns=1, updated_ns=1,
    )


@pytest.mark.parametrize("save_kind", ["memory", "goal", "skill_stats"])
def test_failed_ordinary_write_leaves_operator_authority_available(
    tmp_path: Path, save_kind: str,
) -> None:
    path = tmp_path / "state.sqlite3"
    with StateDatabase(path) as writer, StateDatabase(path) as runtime:
        runtime.set_busy_timeout_ms(0)
        writer.connection.execute("BEGIN IMMEDIATE")
        try:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                if save_kind == "memory":
                    runtime.save_memory(_memory())
                elif save_kind == "goal":
                    runtime.save_goal(Goal(goal_id="tree", description="Reach the tree."))
                else:
                    runtime.save_skill_stats("explore_forward", "world", SkillStats())
            assert not runtime.connection.in_transaction
        finally:
            writer.connection.rollback()
        runtime.save_operator_message(_message())
        assert runtime.load_operator_context().revision == 1
        runtime.update_operator_message_status(
            "first", OperatorMessageStatus.DELIVERED, timestamp_ns=20,
        )
        assert runtime.load_operator_context().messages[0].status == OperatorMessageStatus.DELIVERED


def test_failed_ordinary_statement_rolls_back_its_owned_transaction(tmp_path: Path) -> None:
    with StateDatabase(tmp_path / "state.sqlite3") as db:
        db.connection.execute(
            "CREATE TRIGGER reject_memory BEFORE INSERT ON memories "
            "BEGIN SELECT RAISE(ABORT, 'injected memory failure'); END",
        )
        db.connection.commit()
        with pytest.raises(sqlite3.IntegrityError, match="injected memory failure"):
            db.save_memory(_memory())
        assert not db.connection.in_transaction
        db.save_operator_target(_target())
        assert db.operator_revision() == 1


def test_failed_nested_ordinary_write_preserves_admission_transaction(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite3"
    with StateDatabase(path) as db, StateDatabase(path) as observer:
        db.connection.execute(
            "CREATE TRIGGER reject_memory BEFORE INSERT ON memories "
            "BEGIN SELECT RAISE(ABORT, 'injected memory failure'); END",
        )
        db.connection.commit()
        with db.admit_operator_revision(0) as current:
            assert current
            db.save_goal(Goal(goal_id="before", description="Keep this accepted goal."))
            with pytest.raises(sqlite3.IntegrityError, match="injected memory failure"):
                db.save_memory(_memory())
            assert db.connection.in_transaction
            assert len(db.load_goals()) == 1
            assert observer.load_goals() == ()
            db.save_operator_message(_message())
        assert not db.connection.in_transaction
        assert observer.operator_revision() == 1
        assert len(observer.load_goals()) == 1


def _trajectory_batch(db: StateDatabase, shard_id: str, *, commit: bool) -> None:
    db.save_trajectory_shard(
        shard_id=shard_id, trajectory_id="trajectory", path=f"{shard_id}.tar",
        sha256="a" * 64, first_step_index=0, last_step_index=0,
        step_count=1, bytes_count=1, commit=commit,
    )


def _prepare_trajectory(db: StateDatabase) -> None:
    db.connection.execute(
        "INSERT INTO trajectories VALUES('trajectory', 1, NULL, 'fixture', 'fixture', '{}')",
    )
    db.connection.commit()


@pytest.mark.parametrize("final_write", ["shard", "step_index"])
def test_final_committing_write_publishes_prior_explicit_batch(
    tmp_path: Path, final_write: str,
) -> None:
    path = tmp_path / "state.sqlite3"
    with StateDatabase(path) as db, StateDatabase(path) as observer:
        _prepare_trajectory(db)
        _trajectory_batch(db, "first", commit=False)
        assert db.connection.in_transaction
        assert observer.connection.execute(
            "SELECT COUNT(*) FROM trajectory_shards",
        ).fetchone()[0] == 0
        if final_write == "shard":
            _trajectory_batch(db, "second", commit=True)
        else:
            db.save_trajectory_step_index(
                trajectory_id="trajectory", step_index=0, captured_ns=1, accepted_ns=None,
                shard_id="first", sample_key="0", frame_hash="b" * 64, action_json="{}",
                action_level="world", action_origin="synthetic", policy_id=None,
                model_version=None, route_id=None, policy_action_kind=None,
                policy_request_id=None, prediction_id=None, condition_id=None,
                condition_json=None, behavior_token=None, latent_id=None,
                target_track_id=None, skill_run_id=None, skill_id=None, goal_id=None,
                plan_node_id=None, correction_of_step=None, commit=True,
            )
        assert not db.connection.in_transaction
        assert observer.connection.execute(
            "SELECT COUNT(*) FROM trajectory_shards",
        ).fetchone()[0] == (2 if final_write == "shard" else 1)
        assert observer.connection.execute(
            "SELECT COUNT(*) FROM trajectory_steps_index",
        ).fetchone()[0] == (0 if final_write == "shard" else 1)
        db.save_operator_message(_message())
        assert observer.operator_revision() == 1


def test_failed_ordinary_write_does_not_rollback_caller_owned_batch(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite3"
    with StateDatabase(path) as db, StateDatabase(path) as observer:
        _prepare_trajectory(db)
        db.connection.execute(
            "CREATE TRIGGER reject_memory BEFORE INSERT ON memories "
            "BEGIN SELECT RAISE(ABORT, 'injected memory failure'); END",
        )
        db.connection.commit()
        _trajectory_batch(db, "first", commit=False)
        with pytest.raises(sqlite3.IntegrityError, match="injected memory failure"):
            db.save_memory(_memory())
        assert db.connection.in_transaction
        assert db.connection.execute("SELECT COUNT(*) FROM trajectory_shards").fetchone()[0] == 1
        assert observer.connection.execute(
            "SELECT COUNT(*) FROM trajectory_shards",
        ).fetchone()[0] == 0
        db.connection.rollback()
        assert db.connection.execute("SELECT COUNT(*) FROM trajectory_shards").fetchone()[0] == 0
        db.save_operator_message(_message())
        assert observer.operator_revision() == 1
