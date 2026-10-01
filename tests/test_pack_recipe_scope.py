"""Fictional session identity checks; no game, model or runtime assets."""

from collections import deque
from concurrent.futures import Future
from dataclasses import replace
import json
import sqlite3
import tempfile
from pathlib import Path
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from minecraft_ai.memory import MemoryStore
from minecraft_ai.pack_recipes import PackRecipeCatalog
from minecraft_ai.perception import FrameState, PerceptionBlackboard, PerceptionFact
from minecraft_ai.roles import get_role
from minecraft_ai.runtime import AgentRuntime
from minecraft_ai.social import SocialState
from minecraft_ai.pack_scope import (
    ACTIVE_RECIPE_KEY, ACTIVE_RECIPE_SOURCE, RECIPE_SCOPE_UNAVAILABLE,
    active_recipe_identity, recipe_identity_matches,
)

from test_pack_recipe_targets import target_catalog


def scope_catalog():
    payload = target_catalog()
    payload["configured_recipe_identity"] = {
        "state": "configured_only", "scope_sha256": "c" * 64,
    }
    return PackRecipeCatalog(payload, "b" * 64)


def active_fact(catalog, instance, *, session="fictional-server-1", **changes):
    """Test-only observation: this does not attest any real runtime."""
    value = {
        "schema_version": 1, "state": "observed_active", "world": "PokemonFamily",
        "bds_version": catalog.version_id, "catalog_sha256": catalog.sha256,
        "scope_sha256": catalog._payload["configured_recipe_identity"]["scope_sha256"],
        "server_session": session, "instance_id": instance,
    }
    value.update(changes)
    return PerceptionFact(
        key=ACTIVE_RECIPE_KEY, value=json.dumps(value), confidence=1,
        observed_ns=time.monotonic_ns(), source=ACTIVE_RECIPE_SOURCE,
        expires_after_ms=5000,
    )


def publish(board, identity=None, *, instance="bedrock:1.26.52.3:client-42"):
    previous = board.raw_latest()
    now = time.monotonic_ns()
    facts = (PerceptionFact(
        key="social.player_message", value="Fixture: craft blue:beacon",
        confidence=1, observed_ns=now if previous is None else previous.captured_ns,
        source="fictional-chat", expires_after_ms=30_000,
    ),) if previous is None else ()
    if identity is not None:
        facts += (identity,)
    board.publish(FrameState(
        frame_id=1 if previous is None else previous.frame_id + 1,
        captured_ns=now, instance_id=instance, width=32, height=32, facts=facts,
    ))


def controller(catalog):
    from minecraft_ai.builtin_skills import build_bootstrap_skill_library
    from minecraft_ai.cognition import HighLevelController
    from minecraft_ai.models import ModelResponse
    model = Mock(spec=["complete_constrained"])
    model.complete_constrained.return_value = ModelResponse(
        model="fictional-no-learning", latency_ms=0,
        text='{"s":null,"o":"Invented obsolete answer","c":"Invented obsolete answer"}',
    )
    return HighLevelController(
        model, build_bootstrap_skill_library(), pack_recipe_catalog=catalog,
    )


def recipe_runtime(board, catalog):
    runtime = object.__new__(AgentRuntime)
    runtime.role = get_role("generalist")
    runtime.custom_goals = []
    runtime.memories = MemoryStore()
    runtime.social = SocialState()
    runtime.state_db = None
    runtime.blackboard = board
    runtime.perception = SimpleNamespace(instance_id=board.raw_latest().instance_id)
    runtime.pack_recipe_catalog = catalog
    runtime._recent_skill_runs = deque(maxlen=8)
    runtime._plan_steps = ()
    runtime._plan_index = runtime._plan_started_ns = 0
    runtime._plan_goal_id = runtime._headroom_inspection_memory = None
    runtime._traversal_escalation_pending = False
    runtime._progression_goal = lambda: None
    runtime._active_cognition_perception_target = lambda: None
    return runtime


class RecipeScopeTests(unittest.TestCase):
    def test_configured_file_and_client_version_do_not_prove_active_world(self):
        board = PerceptionBlackboard()
        now = time.monotonic_ns()
        board.publish(FrameState(
            frame_id=1, captured_ns=now, instance_id="bedrock:1.26.52.3:client-42",
            width=32, height=32, facts=(PerceptionFact(
                key="social.player_message", value="Fixture: craft blue:beacon",
                confidence=1, observed_ns=now, source="fictional-chat",
                expires_after_ms=30_000,
            ),),
        ))
        catalog = PackRecipeCatalog(target_catalog(), "b" * 64)
        context = recipe_runtime(board, catalog)._cognition_context(requires_wood=False)
        self.assertEqual(context.wiki, ())
        self.assertIn("active", context.pack_recipe_reply)
        self.assertNotIn("Crystal", context.pack_recipe_reply)

    def test_matching_active_scope_supplies_answer_and_private_decision_binding(self):
        catalog = scope_catalog()
        board = PerceptionBlackboard()
        publish(board, active_fact(catalog, "bedrock:1.26.52.3:client-42"))
        ctx = recipe_runtime(board, catalog)._cognition_context(requires_wood=False)
        decision = controller(catalog).decide(board, ctx)
        self.assertEqual(ctx.pack_recipe_status, "verified")
        self.assertIn("Crystal", decision.game_chat)
        self.assertIsNotNone(decision.pack_recipe_identity)
        self.assertIsNone(decision.skill_id)
        self.assertNotIn("pack_recipe_identity", decision.model_dump())

    def test_identity_changes_refuse_snapshot_and_invalidate_prepared_chat(self):
        from minecraft_ai.game_chat import game_chat_authority_matches
        catalog = scope_catalog()
        changes = (
            {"world": "OtherFictionalWorld"}, {"scope_sha256": "d" * 64},
            {"catalog_sha256": "e" * 64}, {"server_session": "fictional-server-2"},
            {"state": "configured_only"}, {"bds_version": "1.26.99"},
        )
        for change in changes:
            with self.subTest(change=change):
                board = PerceptionBlackboard()
                instance = "bedrock:1.26.52.3:client-42"
                publish(board, active_fact(catalog, instance))
                runtime = recipe_runtime(board, catalog)
                ctx = runtime._cognition_context(requires_wood=False)
                decision = controller(catalog).decide(board, ctx)
                self.assertTrue(game_chat_authority_matches(decision, board))
                publish(board, active_fact(catalog, instance, **change))
                self.assertFalse(game_chat_authority_matches(decision, board))
                self.assertFalse(recipe_identity_matches(decision.pack_recipe_identity, board))
                refreshed = runtime._cognition_context(requires_wood=False)
                if "server_session" in change:
                    # Same catalog in a new session may answer a new question.
                    self.assertEqual(refreshed.pack_recipe_status, "verified")
                    self.assertIn("Crystal", refreshed.pack_recipe_reply)
                else:
                    self.assertEqual(refreshed.wiki, ())
                    self.assertEqual(refreshed.pack_recipe_reply, RECIPE_SCOPE_UNAVAILABLE)
                # An already prepared context cannot be rebound to a new session.
                self.assertEqual(
                    controller(catalog).decide(board, ctx).game_chat,
                    RECIPE_SCOPE_UNAVAILABLE,
                )

    def test_fresh_heartbeat_keeps_identity_but_expired_snapshot_does_not(self):
        catalog = scope_catalog()
        board = PerceptionBlackboard()
        instance = "bedrock:1.26.52.3:client-42"
        fact = active_fact(catalog, instance)
        publish(board, fact)
        identity = active_recipe_identity(board)
        snapshot = board.cognition_snapshot(now_ns=time.monotonic_ns())
        heartbeat = active_fact(catalog, instance)
        publish(board, heartbeat)
        self.assertTrue(recipe_identity_matches(identity, board))
        with patch("time.monotonic_ns", return_value=heartbeat.observed_ns + 5_000_000_001):
            self.assertIsNone(active_recipe_identity(snapshot))
            self.assertFalse(recipe_identity_matches(identity, board))

    def test_malformed_unverified_or_future_identity_stays_unknown(self):
        catalog = scope_catalog()
        instance = "bedrock:1.26.52.3:client-42"
        valid = active_fact(catalog, instance)
        cases = (
            valid.model_copy(update={"source": "model:guess"}),
            valid.model_copy(update={"value": "{"}),
            valid.model_copy(update={"value": "x" * 2049}),
            valid.model_copy(update={"expires_after_ms": 30_000}),
            valid.model_copy(update={"observed_ns": time.monotonic_ns() + 1_000_000_000}),
            valid.model_copy(update={"confidence": 0.5}),
            active_fact(catalog, instance, schema_version=True),
            active_fact(catalog, instance, instance_id="other-client"),
            active_fact(catalog, instance, extra="unrecognized"),
        )
        for fact in cases:
            with self.subTest(fact=fact):
                board = PerceptionBlackboard()
                publish(board, fact)
                self.assertEqual(catalog.active_scope_status(board), "unknown")
                self.assertIsNone(catalog.lookup_live("craft blue:beacon", board))

    def test_stale_decision_is_rejected_before_operator_ack_plan_or_action(self):
        catalog = scope_catalog()
        instance = "bedrock:1.26.52.3:client-42"
        board = PerceptionBlackboard()
        publish(board, active_fact(catalog, instance))
        runtime = recipe_runtime(board, catalog)
        ctx = runtime._cognition_context(requires_wood=False)
        decision = controller(catalog).decide(board, ctx)
        publish(board, active_fact(catalog, instance, session="replacement-server"))
        future = Future()
        future.set_result(decision)
        runtime._pending_decision = future
        runtime._pending_operator_message_ids = ("fictional-question",)
        runtime._pending_operator_message_kinds = {"fictional-question": "question"}
        runtime._cognition_requested = False
        runtime._reject_bound_cognition = Mock()
        runtime._preflight_bound_cognition = Mock()
        runtime._adopt_plan = Mock()
        runtime._consume_cognition_decision()
        runtime._reject_bound_cognition.assert_called_once_with(future, "pack_identity_changed")
        runtime._preflight_bound_cognition.assert_not_called()
        runtime._adopt_plan.assert_not_called()
        self.assertEqual(runtime._pending_operator_message_ids, ())
        self.assertEqual(runtime._pending_operator_message_kinds, {})
        self.assertTrue(runtime._cognition_requested)

    def test_operator_recipe_question_refuses_unknown_and_uses_verified_snapshot(self):
        from minecraft_ai.social import OperatorMessage, OperatorMessageKind, OperatorMessageStatus
        catalog = scope_catalog()
        board = PerceptionBlackboard()
        publish(board)
        ctx = recipe_runtime(board, catalog)._cognition_context(requires_wood=False)
        ctx = replace(ctx, pack_recipe_reply=None, operator_messages=(OperatorMessage(
            message_id="fixture-question", created_ns=time.monotonic_ns(),
            text="craft blue:beacon", kind=OperatorMessageKind.QUESTION,
            status=OperatorMessageStatus.DELIVERED,
        ),))
        decision = controller(catalog).decide(board, ctx)
        self.assertEqual(decision.say, RECIPE_SCOPE_UNAVAILABLE)
        self.assertIsNone(decision.game_chat)
        publish(board, active_fact(catalog, "bedrock:1.26.52.3:client-42"))
        decision = controller(catalog).decide(board, ctx)
        self.assertIn("Crystal", decision.say)
        self.assertIsNotNone(decision.pack_recipe_identity)
        self.assertIsNone(decision.game_chat)

    def test_operator_reused_player_context_answers_current_target(self):
        from minecraft_ai.social import OperatorMessage, OperatorMessageKind, OperatorMessageStatus
        catalog = scope_catalog()
        board = PerceptionBlackboard()
        publish(board, active_fact(catalog, "bedrock:1.26.52.3:client-42"))
        ctx = recipe_runtime(board, catalog)._cognition_context(requires_wood=False)
        self.assertIn("Crystal", ctx.pack_recipe_reply)
        ctx = replace(ctx, operator_messages=(OperatorMessage(
            message_id="different-target", created_ns=time.monotonic_ns(),
            text="craft amber:beacon", kind=OperatorMessageKind.QUESTION,
            status=OperatorMessageStatus.DELIVERED,
        ),))
        decision = controller(catalog).decide(board, ctx)
        self.assertEqual(decision.chosen_goal_id, "operator:different-target")
        self.assertIn("Dust", decision.say)
        self.assertNotIn("Crystal", decision.say)

    def test_player_reused_context_cannot_answer_a_different_target(self):
        catalog = scope_catalog()
        board = PerceptionBlackboard()
        publish(board, active_fact(catalog, "bedrock:1.26.52.3:client-42"))
        ctx = recipe_runtime(board, catalog)._cognition_context(requires_wood=False)
        latest = board.raw_latest()
        now = time.monotonic_ns()
        board.publish(FrameState(
            frame_id=latest.frame_id + 1, captured_ns=now,
            instance_id=latest.instance_id, width=32, height=32, facts=(PerceptionFact(
                key="social.player_message", value="Fixture: craft amber:beacon",
                confidence=1, observed_ns=now, source="fictional-chat",
                expires_after_ms=30_000,
            ),),
        ))
        decision = controller(catalog).decide(board, ctx)
        self.assertEqual(decision.game_chat, RECIPE_SCOPE_UNAVAILABLE)
        self.assertNotIn("Crystal", decision.game_chat)

    def test_live_lookup_retains_the_single_identity_it_validated(self):
        catalog = scope_catalog()
        board = PerceptionBlackboard()
        instance = "bedrock:1.26.52.3:client-42"
        valid = active_fact(catalog, instance)
        publish(board, valid)
        changing_view = Mock()
        changing_view.raw_latest.return_value = board.raw_latest()
        changing_view.fact.side_effect = (
            valid, active_fact(catalog, instance, scope_sha256="d" * 64),
        )
        answer = catalog.lookup_live("craft blue:beacon", changing_view)
        self.assertIsNotNone(answer)
        self.assertEqual(answer.identity, active_recipe_identity(board))
        self.assertEqual(changing_view.fact.call_count, 1)

    def test_delayed_operator_reply_loses_authority_on_session_change_or_expiry(self):
        from minecraft_ai.runtime import RuntimeMetrics
        from minecraft_ai.social import OperatorMessageStatus
        for expiry in (False, True):
            with self.subTest(expiry=expiry):
                catalog = scope_catalog()
                board = PerceptionBlackboard()
                instance = "bedrock:1.26.52.3:client-42"
                fact = active_fact(catalog, instance)
                publish(board, fact)
                runtime = recipe_runtime(board, catalog)
                runtime.state_db = Mock()
                runtime.state_db.update_operator_message_status.side_effect = (
                    sqlite3.OperationalError("database is locked"), None,
                )
                runtime.metrics = RuntimeMetrics()
                runtime._pending_operator_status_updates = {}
                runtime._pending_skill_stats = {}
                runtime._pending_runtime_events = {}
                runtime._pending_memories = {}
                runtime._cognition_requested = False
                self.assertFalse(runtime._persist_operator_message_status(
                    "fixture-question", OperatorMessageStatus.ACKNOWLEDGED,
                    timestamp_ns=1, response_text="Old fictional recipe",
                    recipe_identity=active_recipe_identity(board),
                ))
                self.assertIn("fixture-question", runtime._pending_operator_status_updates)
                if expiry:
                    with patch("time.monotonic_ns", return_value=fact.observed_ns + 5_000_000_001):
                        runtime._flush_pending_operator_status_updates(force=True)
                else:
                    publish(board, active_fact(catalog, instance, session="new-fictional-run"))
                    runtime._flush_pending_operator_status_updates(force=True)
                self.assertEqual(runtime.state_db.update_operator_message_status.call_count, 1)
                self.assertEqual(runtime._pending_operator_status_updates, {})
                self.assertEqual(runtime.metrics.operator_responses, 0)
                self.assertTrue(runtime._cognition_requested)

    def test_scoped_operator_response_commits_only_while_identity_matches(self):
        from minecraft_ai.runtime import RuntimeMetrics
        from minecraft_ai.social import OperatorMessage, OperatorMessageStatus
        from minecraft_ai.storage import StateDatabase
        catalog = scope_catalog()
        board = PerceptionBlackboard()
        instance = "bedrock:1.26.52.3:client-42"
        publish(board, active_fact(catalog, instance))
        runtime = recipe_runtime(board, catalog)
        runtime.metrics = RuntimeMetrics()
        runtime._pending_operator_status_updates = {}
        runtime._pending_skill_stats = {}
        runtime._pending_runtime_events = {}
        runtime._pending_memories = {}
        runtime._cognition_requested = False
        with (
            tempfile.TemporaryDirectory() as folder,
            StateDatabase(Path(folder) / "fixture.db") as db,
        ):
            runtime.state_db = db
            initial = OperatorMessage(
                message_id="fixture-question", created_ns=1, text="craft blue:beacon",
                status=OperatorMessageStatus.DELIVERED,
            )
            db.save_operator_message(initial)
            self.assertTrue(runtime._persist_operator_message_status(
                "fixture-question", OperatorMessageStatus.ACKNOWLEDGED,
                timestamp_ns=2, response_text="Current fictional recipe",
                recipe_identity=active_recipe_identity(board),
            ))
            self.assertEqual(db.load_operator_messages()[0].response_text,
                             "Current fictional recipe")
            self.assertEqual(runtime.metrics.operator_responses, 1)
            db.save_operator_message(initial)
            store = db._store_operator_message

            def change_after_store(message):
                store(message)
                publish(board, active_fact(catalog, instance, session="new-fictional-run"))

            with patch.object(db, "_store_operator_message", side_effect=change_after_store):
                self.assertFalse(runtime._persist_operator_message_status(
                    "fixture-question", OperatorMessageStatus.ACKNOWLEDGED,
                    timestamp_ns=2, response_text="Old fictional recipe",
                    recipe_identity=active_recipe_identity(board),
                ))
            message = db.load_operator_messages()[0]
            self.assertEqual(message.status, OperatorMessageStatus.DELIVERED)
            self.assertIsNone(message.response_text)
            self.assertIsNone(message.acknowledged_ns)
        self.assertEqual(runtime.metrics.operator_responses, 1)
        self.assertTrue(runtime._cognition_requested)
