"""Eight prospective pure controls; fictional catalogue/scene/model, no live pack."""
from dataclasses import replace
import copy
import unittest
from unittest.mock import patch

from minecraft_ai.cognition import CognitionDecision
from minecraft_ai.pack_recipes import PackRecipeCatalog, configured_recipe_label
from minecraft_ai.perception import PerceptionBlackboard
from minecraft_ai.social import OperatorMessageKind, OperatorMessageStatus
from test_configured_recipe_information import information_payload
from test_configured_recipe_version_label import player_view
from test_literal_configured_recipe_integration import context, model_controller, operator, LITERAL
from test_pack_recipe_scope import active_fact, publish

QUERY = "How do I craft lota:poke_ball?"
LABEL = "Configured CobbleDrock 1.3.182; live load unverified. "


def payload():
    value = information_payload()
    value["sources"] = [
        {"folder": "cobbledrock_core_bp_1_3_1", "version": [1, 3, 182], "vanilla": False},
        {"folder": "cobbledrock_content_bp_1_3_1", "version": [1, 3, 182], "vanilla": False},
    ]
    value["items"].update({
        "lota:poke_ball": {"name": "Poké Ball", "craftable": True, "recipes": ["lota:ball"]},
        "lota:red_apricorn": {"name": "Red Apricorn", "craftable": False},
        "minecraft:copper_ingot": {"name": "Copper Ingot", "craftable": False},
    })
    red = {"key": "lota:red_apricorn", "id": "lota:red_apricorn", "count": 1}
    copper = {"key": "minecraft:copper_ingot", "id": "minecraft:copper_ingot", "count": 1}
    value["recipes"]["lota:ball"] = {
        "ingredients": [{**red, "count": 4}, copper],
        "outputs": [{"key": "lota:poke_ball", "id": "lota:poke_ball", "count": 4}],
        "stations": ["crafting_table"],
        "grid": [[None, red, None], [red, copper, red], [None, red, None]],
        "source": {"pack": "Fictional version182 fixture"},
    }
    return value


def catalog(value=None):
    return PackRecipeCatalog(payload() if value is None else value, "b" * 64)


class ConfiguredCompactRecipeProjectionTests(unittest.TestCase):
    def assert_no_action(self, result):
        self.assertIsNone(result.skill_id)
        self.assertEqual(result.skill_parameters, {})
        self.assertEqual(result.plan_steps, ())
        self.assertIsNone(result.research_query)
        self.assertIsNone(result.instruction)
        self.assertFalse(result.request_replan)
        self.assertIsNone(result.pack_recipe_identity)

    def test_scoped_shaped_recipe_preserves_all_counts_and_drops_only_optional_grid(self):
        cat = catalog()
        before = copy.deepcopy(cat._payload)
        ref = cat.lookup_configured_information(QUERY, game_version=cat.version_id)
        self.assertIsNotNone(ref)
        self.assertNotIn("Pattern", ref["recipe_text"])
        self.assertIn("4 Red Apricorn", ref["recipe_text"])
        self.assertIn("1 Copper Ingot", ref["recipe_text"])
        self.assertIn("4 Poke Balls", ref["recipe_text"])
        self.assertIn("crafting table", ref["recipe_text"])
        planner = model_controller(cat)
        view = player_view(QUERY)
        result = planner.decide(view, context(operator(QUERY)))
        self.assertEqual(result.say, LABEL + ref["recipe_text"])
        self.assertLessEqual(len(result.say), 150)
        self.assertEqual(result.chosen_goal_id, "operator:q")
        self.assertIsNone(result.game_chat)
        self.assert_no_action(result)
        self.assertEqual(cat._payload, before)
        planner.model.complete_constrained.assert_called_once()

    def test_default_and_verified_live_lookup_retain_original_pattern_and_reply(self):
        cat = catalog()
        plain = cat.lookup(QUERY, game_version=cat.version_id)
        self.assertIn("Pattern . R . / R C R / . R .", plain.chat_reply)
        self.assertNotIn("Configured", plain.chat_reply)
        view = PerceptionBlackboard()
        instance = "bedrock:1.26.52.3:client-42"
        publish(view, active_fact(cat, instance))
        live = cat.lookup_live(QUERY, view)
        self.assertEqual(live.chat_reply, plain.chat_reply)
        self.assertIsNotNone(live.identity)
        self.assertIsNone(plain.identity)

    def test_multiingredient_compact_form_keeps_each_whole_name_quantity_and_station(self):
        value = payload()
        recipe = value["recipes"]["blue:beacon"]
        ingredients = []
        for index, name in enumerate(("Bright Magic Shard", "Clear Crystal Rod", "Fine Silver Dust"), 2):
            key = f"blue:material{index}"
            value["items"][key] = {"name": name, "craftable": False}
            ingredients.append({"key": key, "id": key, "count": index})
        recipe["ingredients"] = ingredients
        cat = catalog(value)
        scoped_budget = 150 - len(LABEL)
        verbose = ("At a crafting table, use 2 Bright Magic Shard and 3 Clear Crystal Rod "
                   "and 4 Fine Silver Dust to make 3 Beacons.")
        compact = ("3 Beacons <- 2 Bright Magic Shard + 3 Clear Crystal Rod "
                   "+ 4 Fine Silver Dust; crafting table.")
        self.assertGreater(len(verbose), scoped_budget)
        self.assertLess(len(compact), scoped_budget)
        ref = cat.lookup_configured_information("How do I craft blue:beacon?", game_version=cat.version_id)
        self.assertEqual(ref["recipe_text"],
            '3 Beacons <- 2 Bright Magic Shard + 3 Clear Crystal Rod + 4 Fine Silver Dust; crafting table.')
        self.assertLessEqual(len(LABEL + ref["recipe_text"]), 150)
        self.assertIs(ref["gameplay_authority"], False)
        self.assertIs(ref["selected_desktop_connection_verified"], False)
        self.assertIs(ref["live_engine_recipe_bytes_verified"], False)

    def test_irreducible_long_names_refuse_without_partial_counts_or_truncated_identity(self):
        for key in ("lota:poke_ball", "lota:red_apricorn"):
            with self.subTest(key=key):
                value = payload()
                value["items"][key]["name"] = "UnabridgedName" * 20
                cat = catalog(value)
                self.assertIsNone(cat.lookup_configured_information(QUERY, game_version=cat.version_id))
        cat = catalog()
        for limit in (True, 0, -1, 151, "97"):
            with self.subTest(limit=limit):
                self.assertIsNone(cat.lookup(QUERY, game_version=cat.version_id,
                                            _configured_reply_limit=limit))

    def test_incomplete_and_malformed_recipe_payload_never_yields_an_informational_answer(self):
        variants = []
        for field, item_field, bad in (("ingredients", "count", False),
                                     ("ingredients", "count", 0),
                                     ("outputs", "count", -1),
                                     ("outputs", "key", "lota:other")):
            value = payload()
            value["recipes"]["lota:ball"][field][0][item_field] = bad
            variants.append(value)
        value = payload(); value["recipes"]["lota:ball"]["stations"] = []; variants.append(value)
        value = payload(); value["items"]["lota:red_apricorn"]["name"] = "Red\nApricorn"; variants.append(value)
        for value in variants:
            with self.subTest(value=value["recipes"]["lota:ball"]):
                self.assertIsNone(catalog(value).lookup_configured_information(QUERY, game_version="1.26.52.3"))

    def test_pack_scope_and_game_version_remain_exact_and_unknown_versions_not_relabelled(self):
        cat = catalog()
        self.assertIsNone(cat.lookup_configured_information(QUERY, game_version="1.21.0"))
        value = payload(); value["sources"][1]["version"] = [1, 3, 159]
        ref = catalog(value).lookup_configured_information(QUERY, game_version="1.26.52.3")
        self.assertEqual(configured_recipe_label(ref), "Live load unverified. ")
        self.assertNotIn("1.3.182", configured_recipe_label(ref))
        for change in ("scope", "duplicate", "bool-version"):
            with self.subTest(change=change):
                value = payload()
                if change == "scope":value["configured_recipe_identity"]["scope_sha256"] = "x" * 64
                elif change == "duplicate":value["sources"].append(copy.deepcopy(value["sources"][0]))
                else:value["sources"][0]["version"][2] = True
                self.assertIsNone(catalog(value).lookup_configured_information(QUERY, game_version="1.26.52.3"))

    def test_original_player_question_expiry_suppresses_compact_say_chat_and_goal(self):
        cat = catalog(); planner = model_controller(cat); view = player_view(QUERY)
        ctx = planner._with_player_reference(view, context())
        fact = view.fact("social.player_message", min_confidence=0.7)
        deadline = fact.observed_ns + fact.expires_after_ms * 1_000_000
        with patch("minecraft_ai.perception.types.time.monotonic_ns", return_value=deadline+1):
            result = planner._configured_reference_response(CognitionDecision(), view, ctx)
        self.assertIsNone(result.say); self.assertIsNone(result.game_chat)
        self.assertIsNone(result.chosen_goal_id); self.assert_no_action(result)
        planner.model.complete_constrained.assert_not_called()

    def test_instruction_priority_and_refused_question_remain_terminal_without_acknowledgement(self):
        cat = catalog(); planner = model_controller(cat); view = player_view(QUERY)
        message = operator(LITERAL, kind=OperatorMessageKind.INSTRUCTION)
        ctx = context(message)
        with patch.object(cat, "lookup_configured_information",
                          side_effect=AssertionError("instruction retains highest authority")):
            self.assertIs(planner._with_player_reference(view, ctx), ctx)
        ref = cat.lookup_configured_information(QUERY, game_version=cat.version_id)
        original = CognitionDecision(chosen_goal_id="operator:q", skill_id="survey_surroundings",
                                    skill_parameters={"allow_movement": False}, plan_steps=("Observe ahead",))
        projected = planner._configured_reference_response(original, view,
            replace(ctx, pack_configured_information=ref, pack_configured_question=QUERY))
        self.assertIs(projected, original);self.assertEqual(len(ctx.operator_messages[0].text), 2000)
        board = PerceptionBlackboard();publish(board)
        refused = operator(QUERY).model_copy(update={"status": OperatorMessageStatus.REFUSED})
        result = planner._configured_reference_response(CognitionDecision(chosen_goal_id="operator:q"), board,
            replace(context(refused), pack_configured_information=ref, pack_configured_question=QUERY))
        self.assertIsNone(result.say);self.assertIsNone(result.game_chat)
        self.assertIsNone(result.chosen_goal_id);self.assert_no_action(result)
        self.assertEqual(refused.status, OperatorMessageStatus.REFUSED)
        planner.model.complete_constrained.assert_not_called()
