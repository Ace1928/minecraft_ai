"""Prospective terminal-refusal controls; fake provider/clock, temporary SQLite only."""
from concurrent.futures import Future
from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from minecraft_ai.builtin_skills import build_bootstrap_skill_library
from minecraft_ai.cognition import CognitionContext, CognitionDecision, HighLevelController
from minecraft_ai.cognition.types import _DecisionRepairBounds
from minecraft_ai.models import ModelMessage, PlannerRequestBudgetError
from minecraft_ai.native_world_model import compact_planner_prompt
from minecraft_ai.perception import FrameState, PerceptionBlackboard, PerceptionFact, ScreenRegion, Track
from minecraft_ai.plan_graph import PlanGraph, PlanNode
from minecraft_ai.roles import get_role
from minecraft_ai.runtime import AgentRuntime, _PLANNER_BUDGET_RESPONSE
from minecraft_ai.runtime_support.helpers import _active_operator_messages
from minecraft_ai.social import OperatorExecutionBudget, OperatorMessage, OperatorMessageKind, OperatorMessageStatus
from minecraft_ai.storage import OperatorResponseAdmissionError, StateDatabase

MARKER = (
    "ACTIVE OPERATOR DIRECTIVE (highest authority; follow this literal current "
    "request and do not substitute an older task): "
)
END = " Do not move. Do not attack. Do not use. Do not jump."
LITERAL = "Inspect the fictional scenery. " + "a" * (
    2000 - len("Inspect the fictional scenery. ") - len(END)
) + END


def directive(message):
    return (ModelMessage(role="user", content=MARKER + json.dumps(
        message.model_dump(mode="json"), separators=(",", ":"),
    )),)


def bounds(message):
    return _DecisionRepairBounds(
        allowed_skills=(), authority_goal_id=f"operator:{message.message_id}",
        required_action_constraints=() if message.kind == OperatorMessageKind.QUESTION else tuple(
            (name, False) for name in ("allow_movement", "allow_attack", "allow_use", "allow_jump")),
        reply_only=message.kind == OperatorMessageKind.QUESTION,
        literal_operator=message,
    ).native_format()


class ImmediatePool:
    """Fixture scheduling only; the production submission/consumption paths run."""
    def __init__(self, **unused):
        self.calls = 0

    def submit(self, function, *args, **kwargs):
        self.calls += 1
        result = Future()
        try:
            result.set_result(function(*args, **kwargs))
        except BaseException as error:
            result.set_exception(error)
        return result

    def shutdown(self, **unused):
        pass


class PlannerBudgetTerminalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.database = StateDatabase(Path(self.temp.name) / "fixture.sqlite")
        self.addCleanup(self.database.close)
        self.clock = time.monotonic_ns()
        self.wall = time.time_ns()
        for target, replacement in (
            ("minecraft_ai.runtime.operator_intent_lock", lambda **unused: nullcontext()),
            ("minecraft_ai.runtime.operator_pause_latched", lambda: False),
            ("minecraft_ai.runtime.emergency_stop_latched", lambda: False),
            ("minecraft_ai.runtime.time.monotonic_ns", lambda: self.clock),
            ("minecraft_ai.runtime.time.time_ns", lambda: self.wall),
        ):
            active = patch(target, replacement)
            active.start()
            self.addCleanup(active.stop)
        self.message = OperatorMessage(
            message_id="fixture-current", created_ns=self.wall, text=LITERAL,
            status=OperatorMessageStatus.DELIVERED,
        )
        self.database.save_operator_message(self.message)
        self.board = PerceptionBlackboard()
        self.publish()
        self.provider = SimpleNamespace()
        self.calls = []
        self.forward = Mock(side_effect=AssertionError("no inference permitted"))
        self.provider.complete_minecraft_decision = self.fail_budget
        self.controller = HighLevelController(self.provider, build_bootstrap_skill_library())
        with patch("minecraft_ai.runtime.SingleWorkerDaemonExecutor", ImmediatePool):
            self.runtime = AgentRuntime(
                perception=SimpleNamespace(instance_id="fictional:instance"),
                blackboard=self.board, executor=SimpleNamespace(run=None),
                skills=build_bootstrap_skill_library(), role=get_role("generalist"),
                lease_id="fixture-lease", high_level=self.controller,
                state_db=self.database,
            )
        self.context = CognitionContext(
            role=get_role("generalist"), goals=(), memories=(), promises=(), wiki=(),
            operator_messages=(self.message,),
        )
        self.runtime._start_skill = Mock(side_effect=AssertionError("no skill start permitted"))
        self.runtime._send_motor = Mock(side_effect=AssertionError("no actuator permitted"))
        self.runtime._record_terminal_run = Mock(side_effect=AssertionError("no outcome credit permitted"))

    def publish(self, instance="fictional:instance"):
        self.board.publish(FrameState(
            frame_id=1, captured_ns=self.clock, instance_id=instance,
            width=32, height=32, facts=(PerceptionFact(
                key="scene.playable", value=True, confidence=1,
                observed_ns=self.clock, source="fixture", expires_after_ms=30_000,
            ),),
        ))

    def fail_budget(self, messages, *, name, authority):
        self.calls.append((messages, name, authority))
        compact_planner_prompt(messages, fits_prompt=lambda prompt: False, response_format=authority)
        return self.forward()

    def binding(self):
        value = self.runtime._capture_planner_budget_binding(self.context, now_ns=self.clock)
        self.assertIsNotNone(value)
        return value

    def enqueue_error(self, error=None, binding=None):
        future = Future()
        future.set_exception(error or PlannerRequestBudgetError(
            operator_goal_id=f"operator:{self.message.message_id}",
        ))
        self.runtime._pending_decision = future
        self.runtime._planner_budget_bindings[future] = binding or self.binding()
        self.runtime._pending_operator_message_ids = (self.message.message_id,)
        return future

    def current_message(self):
        return next(message for message in self.database.load_operator_messages()
                    if message.message_id == self.message.message_id)

    def assert_no_task_credit(self):
        self.runtime._start_skill.assert_not_called()
        self.runtime._send_motor.assert_not_called()
        self.runtime._record_terminal_run.assert_not_called()
        self.assertEqual(self.runtime.metrics.skill_successes, 0)
        self.assertEqual(self.runtime.metrics.motor_actions, 0)
        self.forward.assert_not_called()

    def test_full_literal_and_late_prohibitions_are_not_truncated_to_fit(self):
        with self.assertRaises(PlannerRequestBudgetError) as caught:
            compact_planner_prompt(directive(self.message), fits_prompt=lambda prompt: False,
                                   response_format=bounds(self.message))
        self.assertEqual(len(self.message.text), 2000)
        self.assertTrue(self.message.text.endswith(END))
        self.assertEqual(caught.exception.operator_goal_id, "operator:fixture-current")
        self.assertNotIn(LITERAL, str(caught.exception))
        self.assertNotIn(self.message.message_id, str(caught.exception))

    def test_compaction_without_matching_original_goal_has_no_terminal_authority(self):
        fmt = bounds(self.message)
        fmt["authority"]["authority_goal_id"] = "operator:other"
        with self.assertRaises(PlannerRequestBudgetError) as caught:
            compact_planner_prompt(directive(self.message), fits_prompt=lambda prompt: False,
                                   response_format=fmt)
        self.assertIsNone(caught.exception.operator_goal_id)

    def test_compaction_without_typed_authority_has_no_terminal_authority(self):
        with self.assertRaises(PlannerRequestBudgetError) as caught:
            compact_planner_prompt(directive(self.message), fits_prompt=lambda prompt: False)
        self.assertIsNone(caught.exception.operator_goal_id)

    def test_reply_question_keeps_original_binding_before_context_projection(self):
        question = self.message.model_copy(update={"kind": OperatorMessageKind.QUESTION})
        with self.assertRaises(PlannerRequestBudgetError) as caught:
            compact_planner_prompt(directive(question), fits_prompt=lambda prompt: False,
                                   response_format=bounds(question))
        self.assertEqual(caught.exception.operator_goal_id, "operator:fixture-current")

    def test_actual_controller_propagates_only_the_dedicated_budget_error(self):
        with self.assertRaises(PlannerRequestBudgetError):
            self.controller.decide(self.board, self.context)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.controller.metrics.failures, 1)
        self.assert_no_task_credit()

    def test_generic_value_and_grammar_errors_keep_existing_idle_fallback(self):
        for error in (ValueError("unrelated provider failure"), RuntimeError("grammar disagreement")):
            with self.subTest(error=type(error).__name__):
                self.provider.complete_minecraft_decision = Mock(side_effect=error)
                decision = self.controller.decide(self.board, self.context)
                self.assertIsNone(decision.skill_id)
                self.assertTrue(decision.request_replan)
                self.assertEqual(self.current_message().status, OperatorMessageStatus.DELIVERED)

    def test_actual_submission_and_consumption_publish_one_visible_refusal(self):
        runtime = self.runtime
        runtime._yield_keepalive_to_operator = lambda: False
        runtime._cognition_due = lambda **unused: True
        runtime._cognition_context = lambda: self.context
        runtime._stage_operator_fast_path = lambda context: False
        runtime._start_cognition_if_due()
        self.assertEqual(runtime._pool.calls, 1)
        self.assertIsNotNone(runtime._pending_decision)
        runtime._consume_cognition_decision()
        result = self.current_message()
        self.assertEqual(result.status, OperatorMessageStatus.REFUSED)
        self.assertEqual(result.response_text, _PLANNER_BUDGET_RESPONSE)
        self.assertIsNone(result.acknowledged_ns)
        self.assertEqual(runtime.metrics.operator_responses, 1)
        self.assertEqual(runtime._planner_budget_bindings, {})
        self.assertIsNone(runtime._pending_planner_budget_refusal)
        self.assert_no_task_credit()

    def test_unmatched_error_does_not_refuse_the_current_operator(self):
        self.enqueue_error(PlannerRequestBudgetError(operator_goal_id="operator:other"))
        self.runtime._consume_cognition_decision()
        self.assertEqual(self.current_message().status, OperatorMessageStatus.DELIVERED)
        self.assertEqual(self.runtime.metrics.operator_responses, 0)
        self.assertEqual(self.runtime._planner_budget_bindings, {})

    def test_cancelled_future_uses_original_failure_path(self):
        future = Future()
        future.cancel()
        self.runtime._pending_decision = future
        self.runtime._planner_budget_bindings[future] = self.binding()
        self.runtime._consume_cognition_decision()
        self.assertEqual(self.current_message().status, OperatorMessageStatus.DELIVERED)
        self.assertEqual(self.runtime._planner_budget_bindings, {})
        self.assertIsNone(self.runtime._pending_planner_budget_refusal)

    def test_new_operator_revision_rejects_old_refusal(self):
        self.enqueue_error()
        self.database.save_operator_message(self.message.model_copy(update={
            "message_id": "fresh", "created_ns": self.wall + 1, "text": "New fixture request",
        }))
        self.runtime._consume_cognition_decision()
        self.assertEqual(self.current_message().status, OperatorMessageStatus.DELIVERED)
        self.assertEqual(self.runtime.metrics.operator_responses, 0)

    def test_changed_literal_under_same_id_rejects_old_refusal(self):
        self.enqueue_error()
        self.database.save_operator_message(self.message.model_copy(update={"text": "Changed fixture literal"}))
        self.runtime._consume_cognition_decision()
        self.assertEqual(self.current_message().status, OperatorMessageStatus.DELIVERED)
        self.assertIsNone(self.current_message().response_text)

    def test_existing_plan_model_instance_execution_and_deadline_are_checked(self):
        binding = self.binding()
        changes = (
            ("_execution_revision", binding.execution_revision + 1),
            ("_plan_goal_id", "unrelated-accepted-goal"),
            ("_plan_steps", ("unrelated-accepted-step",)),
        )
        for field, value in changes:
            with self.subTest(field=field):
                original = getattr(self.runtime, field)
                setattr(self.runtime, field, value)
                self.assertFalse(self.runtime._publish_planner_budget_refusal(binding))
                self.assertEqual(getattr(self.runtime, field), value)
                setattr(self.runtime, field, original)
        original = self.runtime.high_level
        self.runtime.high_level = SimpleNamespace(model=object())
        self.assertFalse(self.runtime._publish_planner_budget_refusal(binding))
        self.runtime.high_level = original
        # A blackboard owns one immutable instance identity. Model an owner
        # replacement with its own real board; never publish a foreign frame
        # into the original board, which correctly refuses that operation.
        original_board = self.runtime.blackboard
        replacement_board = PerceptionBlackboard()
        replacement_board.publish(FrameState(
            frame_id=1, captured_ns=self.clock, instance_id="fictional:replacement",
            width=32, height=32,
        ))
        self.runtime.blackboard = replacement_board
        try:
            self.assertFalse(self.runtime._publish_planner_budget_refusal(binding))
        finally:
            self.runtime.blackboard = original_board
        self.clock = binding.deadline_ns
        self.assertFalse(self.runtime._publish_planner_budget_refusal(binding))
        self.assertEqual(self.current_message().status, OperatorMessageStatus.DELIVERED)

    def test_pause_stop_emergency_and_unreleased_inputs_prevent_response(self):
        binding = self.binding()
        for flag in ("operator_pause_latched", "emergency_stop_latched"):
            with self.subTest(flag=flag), patch("minecraft_ai.runtime." + flag, return_value=True):
                self.assertFalse(self.runtime._publish_planner_budget_refusal(binding))
        self.runtime._stop.set()
        self.assertFalse(self.runtime._publish_planner_budget_refusal(binding))
        self.runtime._stop.clear()
        self.runtime._input_release_pending_ns = self.clock
        self.assertFalse(self.runtime._publish_planner_budget_refusal(binding))
        self.assertEqual(self.current_message().status, OperatorMessageStatus.DELIVERED)

    def test_bounded_original_owner_deadline_is_not_extended_for_refusal(self):
        message = self.message.model_copy(update={
            "execution_budget": OperatorExecutionBudget(timeout_ms=50, max_skills=1),
        })
        self.database.save_operator_message(message)
        self.context = replace(self.context, operator_messages=(message,))
        binding = self.binding()
        self.assertEqual(binding.deadline_ns, self.clock + 50_000_000)
        self.clock = binding.deadline_ns
        self.assertFalse(self.runtime._publish_planner_budget_refusal(binding))
        self.assertIsNone(self.current_message().response_text)

    def test_post_write_owner_loss_rolls_back_response_and_revision(self):
        binding = self.binding()
        update = self.database.update_operator_message_status
        def lose_owner(*args, **kwargs):
            result = update(*args, **kwargs)
            self.runtime._execution_revision += 1
            return result
        with patch.object(self.database, "update_operator_message_status", side_effect=lose_owner):
            with self.assertRaises(OperatorResponseAdmissionError):
                self.runtime._publish_planner_budget_refusal(binding)
        self.assertEqual(self.current_message().status, OperatorMessageStatus.DELIVERED)
        self.assertEqual(self.database.operator_revision(), binding.operator_revision)
        self.assertEqual(self.runtime.metrics.operator_responses, 0)

    def test_nested_new_intent_after_write_rolls_back_refusal(self):
        binding = self.binding()
        update = self.database.update_operator_message_status
        def new_intent(*args, **kwargs):
            result = update(*args, **kwargs)
            self.database.save_operator_message(self.message.model_copy(update={
                "message_id": "nested", "created_ns": self.wall + 1,
            }))
            return result
        with patch.object(self.database, "update_operator_message_status", side_effect=new_intent):
            with self.assertRaises(OperatorResponseAdmissionError):
                self.runtime._publish_planner_budget_refusal(binding)
        self.assertEqual(self.database.operator_revision(), binding.operator_revision)
        self.assertEqual(self.current_message().status, OperatorMessageStatus.DELIVERED)
        self.assertEqual(self.runtime.metrics.operator_responses, 0)

    def test_writer_contention_holds_repeated_planning_then_commits_once(self):
        self.enqueue_error()
        with patch.object(self.database, "admit_operator_revision", side_effect=sqlite3.OperationalError("database is locked")):
            self.runtime._consume_cognition_decision()
        self.assertIsNotNone(self.runtime._pending_planner_budget_refusal)
        self.runtime._start_cognition_if_due()
        self.assertEqual(self.runtime._pool.calls, 0)
        self.clock += 1_000_000_001
        self.assertFalse(self.runtime._flush_planner_budget_refusal())
        self.assertEqual(self.current_message().status, OperatorMessageStatus.REFUSED)
        self.assertEqual(self.runtime.metrics.operator_responses, 1)
        self.assertFalse(self.runtime._flush_planner_budget_refusal())
        self.assertEqual(self.runtime.metrics.operator_responses, 1)

    def test_intent_lock_contention_does_not_reclassify_database_failure(self):
        self.runtime._pending_planner_budget_refusal = self.binding()
        with patch("minecraft_ai.runtime.operator_intent_lock", side_effect=RuntimeError("fixture lock busy")):
            self.assertTrue(self.runtime._flush_planner_budget_refusal())
        self.clock += 1_000_000_001
        with patch.object(self.database, "admit_operator_revision", side_effect=RuntimeError("fixture invalid database revision")):
            with self.assertRaises(RuntimeError):
                self.runtime._flush_planner_budget_refusal()
        self.assertEqual(self.current_message().status, OperatorMessageStatus.DELIVERED)

    def test_waiting_refusal_revokes_after_new_request_without_model_retry(self):
        self.runtime._pending_planner_budget_refusal = self.binding()
        self.database.save_operator_message(self.message.model_copy(update={
            "message_id": "fresh", "created_ns": self.wall + 1,
        }))
        self.assertFalse(self.runtime._flush_planner_budget_refusal())
        self.assertIsNone(self.runtime._pending_planner_budget_refusal)
        self.assertEqual(self.runtime._pool.calls, 0)
        self.assertEqual(self.current_message().status, OperatorMessageStatus.DELIVERED)

    def test_refused_history_cannot_replay_after_database_restart(self):
        old = self.message.model_copy(update={
            "message_id": "old", "created_ns": self.wall - 1,
            "status": OperatorMessageStatus.ACKNOWLEDGED,
        })
        self.database.save_operator_message(old)
        self.context = replace(self.context, operator_messages=(self.message,))
        self.assertTrue(self.runtime._publish_planner_budget_refusal(self.binding()))
        self.database.close()
        reopened = StateDatabase(Path(self.temp.name) / "fixture.sqlite")
        self.addCleanup(reopened.close)
        history = reopened.load_operator_messages(statuses={
            OperatorMessageStatus.ACKNOWLEDGED, OperatorMessageStatus.REFUSED,
        })
        self.assertEqual(_active_operator_messages(history), ())
        self.database = reopened

    def test_delivery_receipts_and_ordinary_acknowledged_instruction_stay_compatible(self):
        revision = self.database.operator_revision()
        self.database.update_operator_message_status(
            self.message.message_id, OperatorMessageStatus.DELIVERED, timestamp_ns=self.wall,
        )
        self.assertEqual(self.database.operator_revision(), revision)
        message = self.database.update_operator_message_status(
            self.message.message_id, OperatorMessageStatus.ACKNOWLEDGED,
            timestamp_ns=self.wall, response_text="Fixture existing successful response",
        )
        self.assertEqual(_active_operator_messages((message,)), (message,))
        self.assertEqual(self.database.operator_revision(), revision)
        self.assertEqual(message.acknowledged_ns, self.wall)

    def test_new_request_remains_usable_after_terminal_refusal(self):
        self.assertTrue(self.runtime._publish_planner_budget_refusal(self.binding()))
        fresh = self.message.model_copy(update={
            "message_id": "fresh", "created_ns": self.wall + 1, "text": "New fixture instruction",
        })
        self.database.save_operator_message(fresh)
        self.assertEqual(_active_operator_messages(self.database.load_operator_messages()), (fresh,))
        self.assert_no_task_credit()

    def test_in_place_plan_graph_change_is_not_hidden_by_snapshot_aliasing(self):
        self.runtime._plan_graph = PlanGraph(goal_id="unchanged-goal", nodes={
            "fixture-node": PlanNode(node_id="fixture-node", objective="Existing fixture objective"),
        }, order=("fixture-node",))
        binding = self.binding()
        self.runtime._plan_graph.cursor = 1
        self.assertFalse(self.runtime._publish_planner_budget_refusal(binding))
        self.assertEqual(self.runtime._plan_graph.cursor, 1)
        self.assertEqual(self.current_message().status, OperatorMessageStatus.DELIVERED)

    def test_buffered_delivery_cannot_reopen_a_durably_refused_request(self):
        self.runtime._pending_operator_status_updates[self.message.message_id] = (
            OperatorMessageStatus.DELIVERED, self.wall, None, None,
        )
        self.assertTrue(self.runtime._publish_planner_budget_refusal(self.binding()))
        self.assertNotIn(self.message.message_id, self.runtime._pending_operator_status_updates)
        for status in (OperatorMessageStatus.QUEUED, OperatorMessageStatus.DELIVERED,
                       OperatorMessageStatus.ACKNOWLEDGED):
            with self.subTest(status=status), self.assertRaises(OperatorResponseAdmissionError):
                self.database.update_operator_message_status(
                    self.message.message_id, status, timestamp_ns=self.wall,
                )
        self.assertEqual(self.current_message().status, OperatorMessageStatus.REFUSED)
        self.assertEqual(self.current_message().response_text, _PLANNER_BUDGET_RESPONSE)

    def test_changed_operator_target_revision_rejects_old_refusal(self):
        binding = self.binding()
        self.database.save_operator_target(Track(
            track_id="fixture-new-target", label="Fictional target",
            region=ScreenRegion(x=0.25, y=0.25, width=0.1, height=0.1),
            confidence=1, first_seen_ns=self.clock, last_seen_ns=self.clock,
        ))
        self.assertFalse(self.runtime._publish_planner_budget_refusal(binding))
        self.assertEqual(self.current_message().status, OperatorMessageStatus.DELIVERED)

    def test_expired_during_intent_contention_retires_before_cooldown_or_database(self):
        binding = self.binding()
        self.runtime._pending_planner_budget_refusal = binding
        with patch("minecraft_ai.runtime.operator_intent_lock", side_effect=RuntimeError("fixture lock busy")) as lock:
            self.assertTrue(self.runtime._flush_planner_budget_refusal())
            self.assertEqual(lock.call_count, 1)
            self.clock = binding.deadline_ns
            # A future cooldown cannot conceal this known expired owner.
            self.runtime._planner_budget_retry_ns = self.clock + 1_000_000_000
            with patch.object(self.database, "admit_operator_revision", side_effect=AssertionError("no database admission")), \
                    patch.object(self.database, "load_operator_context", side_effect=AssertionError("no database read")):
                self.assertFalse(self.runtime._flush_planner_budget_refusal())
            self.assertEqual(lock.call_count, 1)
        self.assertIsNone(self.runtime._pending_planner_budget_refusal)
        self.assertEqual(self.runtime._pool.calls, 0)
        self.assertEqual(self.current_message().status, OperatorMessageStatus.DELIVERED)

    def test_replaced_model_during_cooldown_retires_without_lock_or_database(self):
        self.runtime._pending_planner_budget_refusal = self.binding()
        self.runtime._planner_budget_retry_ns = self.clock + 1_000_000_000
        self.runtime.high_level = SimpleNamespace(model=object())
        with patch("minecraft_ai.runtime.operator_intent_lock", side_effect=AssertionError("no intent acquisition")), \
                patch.object(self.database, "admit_operator_revision", side_effect=AssertionError("no database admission")), \
                patch.object(self.database, "load_operator_context", side_effect=AssertionError("no database read")):
            self.assertFalse(self.runtime._flush_planner_budget_refusal())
        self.assertIsNone(self.runtime._pending_planner_budget_refusal)
        self.assertEqual(self.runtime._pool.calls, 0)
        self.assertEqual(self.current_message().status, OperatorMessageStatus.DELIVERED)
