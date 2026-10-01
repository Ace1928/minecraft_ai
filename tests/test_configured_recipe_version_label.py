"""Eight source-only fixture controls; not live pack or gameplay qualification."""
from dataclasses import replace
import time
import unittest
from unittest.mock import Mock, patch

from minecraft_ai.cognition import CognitionDecision
from minecraft_ai.models import ModelResponse
from minecraft_ai.perception import FrameState, PerceptionBlackboard, PerceptionFact
from minecraft_ai.social import OperatorMessageKind, OperatorMessageStatus
from test_literal_configured_recipe_integration import (
    QUESTION, LITERAL, configured_catalog, context, model_controller, operator,
)
from test_pack_recipe_scope import publish

CORE = "cobbledrock_core_bp_1_3_1"
CONTENT = "cobbledrock_content_bp_1_3_1"
RECIPE = "At a crafting table, use 2 Crystal to make 3 Beacons."


def catalog(version=159):
    value = configured_catalog()
    value._payload["sources"] = [
        {"folder": CORE, "version": [1, 3, version], "vanilla": False},
        {"folder": CONTENT, "version": [1, 3, version], "vanilla": False},
    ]
    return value


def player_view(query=QUESTION, *, stale=False):
    view = PerceptionBlackboard()
    publish_player(view, query, stale=stale)
    return view


def publish_player(view, query, *, stale=False):
    now = time.monotonic_ns()
    last = view.raw_latest()
    view.publish(FrameState(
        frame_id=1 if last is None else last.frame_id + 1,
        captured_ns=now, instance_id="bedrock:1.26.52.3:client-42",
        width=32, height=32, facts=(PerceptionFact(
            key="social.player_message", value="Fixture: " + query,
            confidence=1, source="fictional-chat",
            observed_ns=now - (31_000_000_000 if stale else 0),
            expires_after_ms=30_000,
        ),),
    ))


class ConfiguredVersionLabelTests(unittest.TestCase):
    def assert_no_action(self, result):
        self.assertIsNone(result.skill_id)
        self.assertEqual(result.skill_parameters, {})
        self.assertEqual(result.plan_steps, ())
        self.assertIsNone(result.research_query)
        self.assertIsNone(result.instruction)
        self.assertFalse(result.request_replan)
        self.assertIsNone(result.pack_recipe_identity)

    def test_exact_159_label_source_counts_and_actual_fake_model_request(self):
        view = PerceptionBlackboard()
        publish(view)
        planner = model_controller(catalog())
        # A rejected factual claim from the fake model cannot supply the label.
        planner.model.complete_constrained.return_value = ModelResponse(
            model="fictional-no-learning", latency_ms=0,
            text='{"g":"operator:q","o":"Live engine verified 9.9.9","q":[]}',
        )
        answer = planner.decide(view, context(operator()))
        self.assertEqual(answer.say,
                         "Configured CobbleDrock 1.3.159; live load unverified. " + RECIPE)
        self.assertEqual(answer.chosen_goal_id, "operator:q")
        self.assertIsNone(answer.game_chat)
        self.assert_no_action(answer)
        self.assertEqual(planner.model.complete_constrained.call_count, 1)
        self.assertTrue(answer._configured_recipe_reference)

    def test_historical_156_keeps_its_own_declared_version(self):
        view = PerceptionBlackboard()
        publish(view)
        answer = model_controller(catalog(156)).decide(view, context(operator()))
        self.assertEqual(answer.say,
                         "Configured CobbleDrock 1.3.156; live load unverified. " + RECIPE)
        self.assertNotIn("1.3.159", answer.say)
        self.assert_no_action(answer)

    def test_mixed_missing_duplicate_and_malformed_versions_cannot_claim_159(self):
        planner = model_controller(catalog())
        variants = (
            [{"folder": CORE, "version": [1, 3, 159]},
             {"folder": CONTENT, "version": [1, 3, 156]}],
            [{"folder": CORE, "version": [1, 3, 159]}],
            [{"folder": CORE, "version": [1, 3, 159]},
             {"folder": CORE, "version": [1, 3, 159]},
             {"folder": CONTENT, "version": [1, 3, 159]}],
            [{"folder": CORE, "version": [1, 3, True]},
             {"folder": CONTENT, "version": [1, 3, 159]}],
            [{"folder": [], "version": [1, 3, 159]}],
        )
        for packs in variants:
            with self.subTest(packs=packs):
                self.assertEqual(planner._configured_reference_label(
                    {"configured_packs": packs}), "Live load unverified. ")

    def test_unknown_pack_item_cannot_fall_back_to_vanilla_search(self):
        cat = catalog()
        planner = model_controller(cat)
        planner.world_search = Mock()
        planner.world_search.search.side_effect = AssertionError("no vanilla substitute")
        query = "How do I craft unknown:beacon?"
        view = player_view(query)
        ctx = context()
        self.assertIsNone(cat.lookup_configured_information(query, game_version=cat.version_id))
        self.assertIs(planner._with_player_reference(view, ctx), ctx)
        planner.world_search.search.assert_not_called()
        planner.model.complete_constrained.assert_not_called()

    def test_reply_limits_preserve_whole_counts_or_refuse_without_console_claim(self):
        cat = catalog()
        planner = model_controller(cat)
        view = player_view()
        reference = cat.lookup_configured_information(QUESTION, game_version=cat.version_id)
        label = planner._configured_reference_label(reference)
        for full_length in (155, 175):
            with self.subTest(full_length=full_length):
                recipe = RECIPE + " " + "x" * (full_length - len(label) - len(RECIPE) - 1)
                copied = {**reference, "recipe_text": recipe}
                answer = planner._configured_reference_response(
                    CognitionDecision(), view,
                    replace(context(), pack_configured_information=copied,
                            pack_configured_question=QUESTION),
                )
                if full_length == 155:
                    self.assertEqual(answer.say, label + recipe)
                    self.assertIn("use 2 Crystal to make 3 Beacons.", answer.say)
                    self.assertEqual(answer.game_chat,
                        "The configured recipe is too long for this chat reply. Live load is unverified.")
                else:
                    self.assertNotIn("Crystal", answer.say)
                    self.assertIn("exceeds this reply limit", answer.say)
                    self.assertEqual(answer.game_chat, answer.say)
                self.assertNotIn("console", answer.game_chat)
                self.assert_no_action(answer)

    def test_pending_instruction_keeps_plan_literal_and_false_permissions(self):
        cat = catalog()
        planner = model_controller(cat)
        view = player_view()
        message = operator(LITERAL, kind=OperatorMessageKind.INSTRUCTION)
        ctx = context(message)
        with patch.object(cat, "lookup_configured_information",
                          side_effect=AssertionError("instruction remains highest authority")):
            self.assertIs(planner._with_player_reference(view, ctx), ctx)
        ref = cat.lookup_configured_information(QUESTION, game_version=cat.version_id)
        decision = CognitionDecision(
            chosen_goal_id="operator:q", skill_id="survey_surroundings",
            skill_parameters={"allow_movement": False, "allow_attack": False,
                              "allow_use": False, "allow_jump": False},
            plan_steps=("Observe ahead",),
        )
        answer = planner._configured_reference_response(
            decision, view, replace(ctx, pack_configured_information=ref,
                                  pack_configured_question=QUESTION))
        self.assertIs(answer, decision)
        self.assertEqual(ctx.operator_messages[0].text, LITERAL)
        self.assertEqual(len(ctx.operator_messages[0].text), 2000)
        self.assertEqual(answer.plan_steps, ("Observe ahead",))
        self.assertFalse(any(answer.skill_parameters.values()))
        planner.model.complete_constrained.assert_not_called()

    def test_expired_or_replaced_original_player_question_suppressed_request_none(self):
        cat = catalog()
        planner = model_controller(cat)
        for query, stale in ((QUESTION, True), ("How do I craft blue:material?", False)):
            with self.subTest(query=query, stale=stale):
                view = player_view()
                refctx = planner._with_player_reference(view, context())
                self.assertEqual(refctx.pack_configured_question, QUESTION)
                self.assertIsNotNone(refctx.pack_configured_information)
                original = view.fact("social.player_message", min_confidence=0.7)
                self.assertIsNotNone(original)
                self.assertEqual(original.value, "Fixture: " + QUESTION)
                self.assertEqual(original.observed_ns, view.raw_latest().captured_ns)
                self.assertEqual(original.expires_after_ms, 30_000)
                if stale:
                    expired_ns = original.observed_ns + original.expires_after_ms * 1_000_000 + 1
                    # Expire the original observation; an older replacement is
                    # correctly discarded by the production blackboard.
                    with patch("minecraft_ai.perception.types.time.monotonic_ns",
                               return_value=expired_ns):
                        self.assertFalse(original.fresh())
                        answer = planner._configured_reference_response(
                            CognitionDecision(), view, refctx)
                else:
                    publish_player(view, query)
                    answer = planner._configured_reference_response(
                        CognitionDecision(), view, refctx)
                self.assertIsNone(planner._request_context.get())
                self.assertIsNone(answer.say)
                self.assertIsNone(answer.game_chat)
                self.assertIsNone(answer.chosen_goal_id)
                self.assert_no_action(answer)

    def test_refused_operator_message_gets_no_reply_acknowledgement_or_action(self):
        cat = catalog()
        planner = model_controller(cat)
        view = PerceptionBlackboard()
        publish(view)
        # No current player question matches the supplied reference.
        message = operator().model_copy(update={"status": OperatorMessageStatus.REFUSED})
        ctx = context(message)
        with patch.object(cat, "lookup_configured_information",
                          side_effect=AssertionError("REFUSED is terminal")):
            self.assertIs(planner._with_operator_reference(view, ctx), ctx)
        ref = cat.lookup_configured_information(QUESTION, game_version=cat.version_id)
        answer = planner._configured_reference_response(
            CognitionDecision(chosen_goal_id="operator:q"), view,
            replace(ctx, pack_configured_information=ref, pack_configured_question=QUESTION))
        self.assertIsNone(answer.say)
        self.assertIsNone(answer.game_chat)
        self.assertIsNone(answer.chosen_goal_id)
        self.assertEqual(message.status, OperatorMessageStatus.REFUSED)
        self.assert_no_action(answer)
        planner.model.complete_constrained.assert_not_called()
