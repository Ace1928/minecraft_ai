"""Prospective source regressions: fictional facts/models, no live world."""

from dataclasses import replace
import json
import time
import unittest
from unittest.mock import patch

from minecraft_ai.cognition import CognitionContext, CognitionDecision
from minecraft_ai.cognition.prompts import _operator_prompt_payload
from minecraft_ai.cognition.repair import _json_repair_messages, _semantic_repair_messages
from minecraft_ai.cognition.types import _DecisionRepairBounds
from minecraft_ai.models import ModelResponse
from minecraft_ai.pack_scope import active_recipe_identity
from minecraft_ai.perception import FrameState, PerceptionBlackboard, PerceptionFact
from minecraft_ai.roles import get_role
from minecraft_ai.social import OperatorMessage, OperatorMessageKind, OperatorMessageStatus

from test_pack_recipe_scope import active_fact, controller, publish, scope_catalog


QUESTION = "How do I craft blue:beacon?"
PROHIBITIONS = " Do not move. Do not attack. Do not use. Do not jump."
LITERAL = "Observe the scene. " + "a" * (
    2000 - len("Observe the scene. ") - len(PROHIBITIONS)
) + PROHIBITIONS
DIRECTIVE_MARKER = (
    "ACTIVE OPERATOR DIRECTIVE (highest authority; follow this literal current "
    "request and do not substitute an older task): "
)


def operator(text=QUESTION, *, kind=OperatorMessageKind.QUESTION):
    return OperatorMessage(
        message_id="q", created_ns=1, text=text, kind=kind,
        status=OperatorMessageStatus.DELIVERED,
    )


def context(*messages, **changes):
    value = CognitionContext(
        role=get_role("generalist"), goals=(), memories=(), promises=(), wiki=(),
        operator_messages=messages,
    )
    return replace(value, **changes)


def configured_catalog():
    # Existing invented namespace/secondary-output catalogue, with one fake
    # configured source descriptor. It attests no real engine or player.
    catalog = scope_catalog()
    catalog._payload["sources"] = [
        {"folder": "fictional_pack", "version": [1, 0, 0], "vanilla": False},
    ]
    return catalog


def model_controller(catalog):
    result = controller(catalog)
    result.model.complete_constrained.return_value = ModelResponse(
        model="fictional-no-learning", latency_ms=0,
        text='{"g":"operator:q","o":"Invented obsolete answer","q":[]}',
    )
    return result


class LiteralConfiguredRecipeIntegrationTests(unittest.TestCase):
    def test_full_2000_literal_and_late_prohibitions_survive_both_repairs(self):
        message = operator(LITERAL, kind=OperatorMessageKind.INSTRUCTION)
        self.assertEqual(len(message.text), 2000)
        self.assertEqual(_operator_prompt_payload(message)["text"], message.text)
        mask = tuple((name, False) for name in (
            "allow_movement", "allow_attack", "allow_use", "allow_jump",
        ))
        bounds = _DecisionRepairBounds(
            allowed_skills=(), authority_goal_id="operator:q",
            required_action_constraints=mask, literal_operator=message,
        )
        repairs = (
            _json_repair_messages("rejected model text", bounds),
            _semantic_repair_messages(
                CognitionDecision(chosen_goal_id="operator:q"), bounds,
                repair_kind="infeasible", reason="fixture failure",
            ),
        )
        for messages in repairs:
            with self.subTest(repair=messages[0].content[:40]):
                directive = messages[-1].content
                self.assertTrue(directive.startswith(DIRECTIVE_MARKER))
                payload = json.loads(directive[len(DIRECTIVE_MARKER):])
                self.assertEqual(payload["text"], LITERAL)
                self.assertTrue(payload["text"].endswith(PROHIBITIONS))
                self.assertEqual(
                    bounds.native_format()["authority"]["required_action_constraints"],
                    dict(mask),
                )
                self.assertNotIn("literal_operator", bounds.native_format()["authority"])

    def test_configured_answer_uses_source_counts_and_no_live_identity(self):
        catalog = configured_catalog()
        view = PerceptionBlackboard()
        publish(view)
        planner = model_controller(catalog)
        answer = planner.decide(view, context(operator()))
        self.assertEqual(answer.say,
                         "Live load unverified. At a crafting table, use 2 Crystal to make 3 Beacons.")
        self.assertEqual(answer.chosen_goal_id, "operator:q")
        self.assertIsNone(answer.skill_id)
        self.assertEqual(answer.skill_parameters, {})
        self.assertEqual(answer.plan_steps, ())
        self.assertIsNone(answer.research_query)
        self.assertIsNone(answer.pack_recipe_identity)
        self.assertTrue(answer._configured_recipe_reference)
        self.assertNotIn("_configured_recipe_reference", answer.model_dump())
        self.assertEqual(planner.model.complete_constrained.call_count, 1)

    def test_instruction_priority_keeps_plan_and_suppresses_player_reference(self):
        catalog = configured_catalog()
        view = PerceptionBlackboard()
        now = time.monotonic_ns()
        view.publish(FrameState(
            frame_id=1, captured_ns=now, instance_id="bedrock:1.26.52.3:client-42",
            width=32, height=32, facts=(PerceptionFact(
                key="social.player_message", value="Fixture: " + QUESTION,
                confidence=1, observed_ns=now, source="fictional-chat", expires_after_ms=30_000,
            ),),
        ))
        planner = model_controller(catalog)
        reference = catalog.lookup_configured_information(QUESTION, game_version=catalog.version_id)
        ctx = context(operator("Observe the scene; do not move.", kind=OperatorMessageKind.INSTRUCTION))
        with patch.object(catalog, "lookup_configured_information",
                          side_effect=AssertionError("reference cannot replace instruction")):
            self.assertIs(planner._with_player_reference(view, ctx), ctx)
        original = CognitionDecision(
            chosen_goal_id="operator:q", skill_id="survey_surroundings",
            skill_parameters={"allow_movement": False}, plan_steps=("Observe ahead",),
        )
        projected = planner._configured_reference_response(
            original, view,
            replace(ctx, pack_configured_information=reference, pack_configured_question=QUESTION),
        )
        self.assertIs(projected, original)
        self.assertEqual(projected.plan_steps, ("Observe ahead",))
        self.assertEqual(planner.model.complete_constrained.call_count, 0)

    def test_verified_scope_uses_exact_source_reply_without_model_or_fallback(self):
        catalog = configured_catalog()
        view = PerceptionBlackboard()
        instance = "bedrock:1.26.52.3:client-42"
        publish(view, active_fact(catalog, instance))
        expected = catalog.lookup_live(QUESTION, view)
        planner = model_controller(catalog)
        with patch.object(catalog, "lookup_configured_information",
                          side_effect=AssertionError("verified scope cannot use fallback")):
            answer = planner.decide(view, context(operator()))
        self.assertEqual(answer.say, expected.chat_reply)
        self.assertEqual(answer.pack_recipe_identity, active_recipe_identity(view))
        self.assertFalse(answer._configured_recipe_reference)
        self.assertEqual(planner.model.complete_constrained.call_count, 0)

    def test_scope_revocation_during_decide_cannot_rebind_configured_reply(self):
        catalog = configured_catalog()
        view = PerceptionBlackboard()
        instance = "bedrock:1.26.52.3:client-42"
        publish(view, active_fact(catalog, instance))
        old_identity = active_recipe_identity(view)
        planner = model_controller(catalog)

        def revoke(_board, _context):
            publish(view, active_fact(catalog, instance, catalog_sha256="d" * 64))
            return None

        with patch.object(planner, "_operator_fast_path_decision", side_effect=revoke):
            answer = planner.decide(view, context(operator()))
        self.assertIsNotNone(old_identity)
        self.assertIsNone(answer.pack_recipe_identity)
        self.assertTrue(answer._configured_recipe_reference)
        self.assertIn("Live load unverified.", answer.say)
        self.assertNotIn("_configured_recipe_reference", answer.model_dump())

    def test_model_cannot_supply_private_configured_projection_marker(self):
        with self.assertRaises(ValueError):
            CognitionDecision.model_validate({"_configured_recipe_reference": True})
        self.assertNotIn("_configured_recipe_reference", CognitionDecision.model_json_schema()["properties"])
