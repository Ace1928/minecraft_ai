"""Pending operator intent holds only source-proven disposable wandering."""
from __future__ import annotations

from concurrent.futures import Future
from types import SimpleNamespace
import sqlite3
import time

import pytest

from minecraft_ai.cognition import CognitionDecision
from minecraft_ai.perception import FrameState, PerceptionFact
from minecraft_ai.runtime import AgentRuntime
from minecraft_ai.runtime_support.types import SkillStartSource
from minecraft_ai.skills import SkillOutcome
from minecraft_ai.social import OperatorMessage, OperatorMessageKind, OperatorMessageStatus
from minecraft_ai.storage import StateDatabase

from test_agent_core import _runtime_with_inflight_operator_cognition


INSTRUCTION = (
    "Open the inventory once, verify the inventory screen is visible, then close it. "
    "Do not move or attack during this inventory check."
)


@pytest.mark.parametrize("kind", tuple(OperatorMessageKind))
@pytest.mark.parametrize("status", tuple(OperatorMessageStatus))
def test_only_selected_unresolved_directive_holds(tmp_path, kind, status):
    with StateDatabase(tmp_path / "state.sqlite") as database:
        database.save_operator_message(OperatorMessage(
            message_id="selected", created_ns=1, text=INSTRUCTION, kind=kind, status=status,
        ))
        runtime = object.__new__(AgentRuntime)
        runtime.state_db = database
        assert runtime._unresolved_operator_directive_waiting() is (
            status in {OperatorMessageStatus.QUEUED, OperatorMessageStatus.DELIVERED}
            and kind in {OperatorMessageKind.INSTRUCTION, OperatorMessageKind.CORRECTION}
        )


def test_lower_priority_directive_cannot_override_selected_question(tmp_path):
    with StateDatabase(tmp_path / "state.sqlite") as database:
        for message_id, priority, kind in (
            ("instruction", 0.5, OperatorMessageKind.INSTRUCTION),
            ("question", 0.9, OperatorMessageKind.QUESTION),
        ):
            database.save_operator_message(OperatorMessage(
                message_id=message_id, created_ns=1, text=INSTRUCTION,
                kind=kind, priority=priority,
            ))
        runtime = object.__new__(AgentRuntime)
        runtime.state_db = database
        assert not runtime._unresolved_operator_directive_waiting()


def test_missing_database_has_no_new_authority():
    runtime = object.__new__(AgentRuntime)
    runtime.state_db = None
    assert not runtime._unresolved_operator_directive_waiting()


def test_database_error_is_not_permission_to_resume_wandering():
    def unavailable(**_kwargs):
        raise sqlite3.OperationalError("database is locked")

    runtime = object.__new__(AgentRuntime)
    runtime.state_db = SimpleNamespace(load_operator_messages=unavailable)
    with pytest.raises(sqlite3.OperationalError):
        runtime._unresolved_operator_directive_waiting()


def _marked_runtime(database, *, text=INSTRUCTION):
    runtime, future, model, terminal, actions = _runtime_with_inflight_operator_cognition(
        database, message_text=text,
    )
    # The existing fixture has emitted W. Start provenance is the only new
    # authority to cancel that disposable run; context/skill names are not enough.
    runtime._disposable_keepalive_run_id = runtime.executor.run.run_id
    runtime._record_terminal_run = lambda run, **kwargs: terminal.append((run, kwargs))
    return runtime, future, model, terminal, actions


def test_nonliteral_directive_releases_once_without_ack_plan_or_model_work(tmp_path):
    with StateDatabase(tmp_path / "state.sqlite") as database:
        runtime, future, model, terminal, actions = _marked_runtime(database)
        runtime._plan_steps = ("existing plan",)
        runtime._plan_index = 0
        assert runtime._start_cognition_if_due() is True
        assert runtime._pending_decision is future and future.running()
        assert runtime._execution_revision == 1
        assert runtime._pending_execution_revision == 0
        assert runtime._cognition_requested
        assert len(actions) == 1 and "w" in actions[0].keys_up
        assert not actions[0].keys_down and not actions[0].buttons_down
        assert terminal[0][0].outcome == SkillOutcome.CANCELLED
        assert terminal[0][1] == {"advance_plan": False}
        assert runtime._plan_steps == ("existing plan",) and runtime._plan_index == 0
        for _ in range(3):
            assert runtime._start_cognition_if_due() is None
            assert runtime._explore_keep_alive() is None
            assert not runtime._keepalive_horizon_reorient()
        assert len(terminal) == len(actions) == 1
        assert model.calls == 0
        assert database.load_operator_messages(limit=1)[0].status == OperatorMessageStatus.QUEUED


def test_literal_fast_path_runs_after_one_fresh_capture_without_model_wait(tmp_path):
    with StateDatabase(tmp_path / "state.sqlite") as database:
        runtime, stale, model, terminal, actions = _marked_runtime(
            database, text="Mine the marked dirt block.",
        )
        assert runtime._start_cognition_if_due() is True
        original = runtime.blackboard.raw_latest()
        runtime.blackboard.publish(original.model_copy(update={
            "frame_id": original.frame_id + 1, "captured_ns": time.monotonic_ns(),
        }))
        runtime._start_cognition_if_due()
        assert stale.running() and model.calls == 0
        assert runtime.executor.run.skill_id == "mine_visible_block"
        assert len(terminal) == 1 and "w" in actions[0].keys_up
        assert database.load_operator_messages(limit=1)[0].status == (
            OperatorMessageStatus.ACKNOWLEDGED
        )


@pytest.mark.parametrize("source", tuple(SkillStartSource))
@pytest.mark.parametrize("context", ("explore-keepalive", "real-plan"))
def test_start_provenance_not_skill_name_controls_disposability(tmp_path, source, context):
    with StateDatabase(tmp_path / "state.sqlite") as database:
        runtime, _, _, terminal, _ = _marked_runtime(database)
        runtime.executor.cancel()
        run = runtime._start_skill(
            runtime.skills.get("explore_forward"), source=source,
            context_key=context, run_id="new-run",
        )
        expected = source == SkillStartSource.KEEPALIVE and context == "explore-keepalive"
        assert (runtime._disposable_keepalive_run_id == run.run_id) is expected
        assert runtime._yield_keepalive_to_operator() is expected
        assert bool(terminal) is expected


@pytest.mark.parametrize("skill", (
    "open_inventory", "respawn_after_death", "traverse_visible_obstacle", "escape_submersion",
))
def test_atomic_and_real_recovery_runs_never_acquire_disposable_marker(tmp_path, skill):
    with StateDatabase(tmp_path / "state.sqlite") as database:
        runtime, _, _, terminal, _ = _marked_runtime(database)
        runtime.executor.cancel()
        run = runtime._start_skill(
            runtime.skills.get(skill), source=SkillStartSource.RECOVERY,
            context_key="explore-keepalive", run_id="protected-run",
        )
        assert runtime._disposable_keepalive_run_id is None
        assert not runtime._yield_keepalive_to_operator()
        assert runtime.executor.run is run and not terminal


def test_current_death_scene_defers_to_existing_safety_router(tmp_path):
    with StateDatabase(tmp_path / "state.sqlite") as database:
        runtime, _, _, terminal, _ = _marked_runtime(database)
        now = time.monotonic_ns()
        runtime.blackboard.publish(FrameState(
            frame_id=2, captured_ns=now, instance_id="bedrock:operator-preemption",
            width=1280, height=720, facts=(PerceptionFact(
                key="scene.death", value=True, confidence=1.0, source="test",
                observed_ns=now, expires_after_ms=10000,
            ),),
        ))
        assert not runtime._yield_keepalive_to_operator()
        assert not terminal
        runtime._route_observed_scene_recovery()
        assert runtime.executor.run.skill_id == "respawn_after_death"


def test_real_revision_change_still_rejects_old_completed_decision(tmp_path):
    with StateDatabase(tmp_path / "state.sqlite") as database:
        runtime, old_future, _, _, _ = _marked_runtime(database)
        assert runtime._start_cognition_if_due() is True
        old_future.set_result(CognitionDecision(skill_id="explore_forward"))
        runtime._consume_cognition()
        assert runtime._pending_decision is None and runtime._cognition_requested
        assert runtime.executor.run.outcome == SkillOutcome.CANCELLED
        assert database.load_operator_messages(limit=1)[0].status == OperatorMessageStatus.QUEUED


def test_fresh_submission_uses_post_release_revision_and_selected_message(tmp_path):
    with StateDatabase(tmp_path / "state.sqlite") as database:
        runtime, old_future, _, _, _ = _marked_runtime(database)
        assert runtime._start_cognition_if_due() is True
        old_future.set_result(CognitionDecision())
        runtime._consume_cognition()
        calls = []
        pending = Future()

        def submit(function, board, context):
            calls.append((function, board, context))
            return pending

        runtime._pool = SimpleNamespace(submit=submit)
        runtime._start_cognition_if_due()
        assert len(calls) == 1 and runtime._pending_decision is pending
        assert runtime._pending_execution_revision == runtime._execution_revision == 1
        assert runtime._pending_operator_message_ids == ("new-operator-correction",)
        assert database.load_operator_messages(limit=1)[0].status == OperatorMessageStatus.DELIVERED
        assert runtime._explore_keep_alive() is None
        runtime._start_cognition_if_due()
        assert len(calls) == 1


def test_release_failure_records_cancellation_but_never_starts_successor(tmp_path):
    with StateDatabase(tmp_path / "state.sqlite") as database:
        runtime, future, model, terminal, _ = _marked_runtime(database)

        def rejected(*_args, **_kwargs):
            raise RuntimeError("supervisor rejected release")

        runtime._send_motor = rejected
        with pytest.raises(RuntimeError, match="rejected release"):
            runtime._start_cognition_if_due()
        assert runtime._pending_decision is future and model.calls == 0
        assert runtime._execution_revision == 1
        assert len(terminal) == 1 and terminal[0][0].outcome == SkillOutcome.CANCELLED


def test_tick_yields_after_release_before_any_cognition_or_optional_action(tmp_path, monkeypatch):
    with StateDatabase(tmp_path / "state.sqlite") as database:
        runtime, future, model, terminal, actions = _marked_runtime(database)
        order = []
        runtime.perception = SimpleNamespace(
            capture_once=lambda: (order.append("capture") or SimpleNamespace(frame_id=2)),
            stale=lambda: False,
        )
        runtime.telemetry = SimpleNamespace(publish=lambda _: None)
        for method in (
            "_merge_operator_target", "_merge_policy_perception", "_flush_pending_skill_stats",
            "_flush_pending_learning_records", "_flush_pending_operator_status_updates",
            "_publish_player_chat_facts", "_planks_retry_requires_wood",
        ):
            monkeypatch.setattr(runtime, method, lambda: None)
        monkeypatch.setattr(runtime, "_telemetry_payload", lambda **_: {})
        monkeypatch.setattr(runtime, "_continue_after_capture", lambda: True)
        for method in (
            "_consume_cognition", "_start_cognition_if_due", "_keepalive_horizon_reorient",
        ):
            monkeypatch.setattr(runtime, method, lambda: pytest.fail("must await next capture"))
        runtime.tick()
        assert order == ["capture"]
        assert len(actions) == len(terminal) == 1
        assert runtime._pending_decision is future and model.calls == 0
