"""Fictional pack targets: no models, server, or game input are used."""

from __future__ import annotations

import copy
import unittest

from minecraft_ai.pack_recipes import PackRecipeCatalog


def target_catalog() -> dict:
    """Build invented colliding names and a recipe with a secondary output."""
    items = {}
    recipes = {}
    for namespace, reagent in (("amber", "Dust"), ("blue", "Crystal")):
        target = f"{namespace}:beacon"
        ingredient = f"{namespace}:material"
        remainder = f"{namespace}:remainder"
        items[target] = {"name": "Beacon", "craftable": True, "recipes": [target]}
        items[ingredient] = {"name": reagent, "craftable": False}
        items[remainder] = {
            "name": "Empty Shell",
            "craftable": True,
            "recipes": [target],
        }
        recipes[target] = {
            "id": target,
            "kind": "shapeless",
            "ingredients": [{"key": ingredient, "id": ingredient, "count": 2}],
            "outputs": [
                {"key": remainder, "id": remainder, "count": 1},
                {"key": target, "id": target, "count": 3},
            ],
            "stations": ["crafting_table"],
            "grid": [],
            "source": {"pack": f"Fictional {namespace} Pack"},
        }
    return {
        "schema_version": 1,
        "world": "PokemonFamily",
        "bds_version": "1.26.52.3",
        "revision": "a" * 64,
        "items": items,
        "recipes": recipes,
        "warnings": [],
    }


class RecipeTargetTests(unittest.TestCase):
    def lookup(self, query: str, payload: dict | None = None):
        data = payload or target_catalog()
        before = copy.deepcopy(data)
        answer = PackRecipeCatalog(data, "b" * 64).lookup(
            query, game_version="1.26.52.3"
        )
        self.assertEqual(data, before)
        return answer

    def test_namespace_selects_exact_target_and_its_secondary_output(self):
        answer = self.lookup("How do I craft blue:beacon?")
        self.assertIsNotNone(answer)
        self.assertEqual(
            answer.chat_reply, "At a crafting table, use 2 Crystal to make 3 Beacons."
        )
        self.assertIn("Fictional blue Pack", answer.evidence.extract)

    def test_unknown_namespace_cannot_fall_back_to_same_display_name(self):
        self.assertIsNone(self.lookup("How do I craft absent:beacon?"))

    def test_unqualified_colliding_names_require_disambiguation(self):
        self.assertIsNone(self.lookup("How do I craft a Beacon?"))

    def test_explicit_ingredient_does_not_replace_unqualified_target(self):
        payload = target_catalog()
        payload["items"].pop("amber:beacon")
        answer = self.lookup("How do I craft a Beacon using amber:material?", payload)
        self.assertIsNotNone(answer)
        self.assertIn("2 Crystal to make 3 Beacons", answer.chat_reply)

    def test_multiple_explicit_target_choices_are_not_silently_resolved(self):
        self.assertIsNone(self.lookup("How do I craft blue:beacon or amber:beacon?"))

    def test_recipe_index_cannot_claim_an_output_that_recipe_does_not_produce(self):
        payload = target_catalog()
        payload["recipes"]["blue:beacon"]["outputs"].pop()
        self.assertIsNone(self.lookup("How do I craft blue:beacon?", payload))

    def test_variant_key_is_not_projected_to_base_item(self):
        payload = target_catalog()
        payload["items"]["blue:beacon@2"] = {
            "name": "Bright Beacon",
            "craftable": True,
            "recipes": ["blue:beacon"],
        }
        payload["recipes"]["blue:beacon"]["outputs"][1] = {
            "key": "blue:beacon@2",
            "id": "blue:beacon",
            "data": 2,
            "count": 3,
        }
        answer = self.lookup("What is the recipe for blue:beacon@2?", payload)
        self.assertIsNotNone(answer)
        self.assertIn("make 3 Bright Beacons", answer.chat_reply)
        self.assertIsNone(self.lookup("How do I craft blue:beacon?", payload))

    def test_explicit_key_before_recipe_intent_retains_its_identity(self):
        for query in (
            "What is absent:beacon's recipe?",
            "absent:beacon crafting recipe",
        ):
            with self.subTest(query=query):
                self.assertIsNone(self.lookup(query))
        answer = self.lookup("What is blue:beacon's recipe?")
        self.assertIsNotNone(answer)
        self.assertIn("2 Crystal to make 3 Beacons", answer.chat_reply)

    def test_ingredient_prelude_is_not_an_output_selector(self):
        payload = target_catalog()
        payload["items"].pop("amber:beacon")
        answer = self.lookup("With amber:material, how do I craft a Beacon?", payload)
        self.assertIsNotNone(answer)
        self.assertIn("2 Crystal to make 3 Beacons", answer.chat_reply)

    def test_unsupported_suffix_cannot_match_a_valid_identity_prefix(self):
        for key in ("blue:beacon:9", "blue:beacon@2.9", "blue:beacon@2@3"):
            with self.subTest(key=key):
                self.assertIsNone(self.lookup(f"How do I craft {key}?"))

    def test_unknown_explicit_key_suppresses_unrelated_vanilla_reference(self):
        catalog = PackRecipeCatalog(target_catalog(), "b" * 64)
        for key in ("absent:beacon", "minecraft:absent_beacon"):
            with self.subTest(key=key):
                self.assertTrue(
                    catalog.mentions_pack_content(f"How do I craft {key} with copper?")
                )

    def test_duplicate_matching_outputs_are_refused(self):
        payload = target_catalog()
        outputs = payload["recipes"]["blue:beacon"]["outputs"]
        outputs.append(copy.deepcopy(outputs[1]))
        self.assertIsNone(self.lookup("How do I craft blue:beacon?", payload))

    def test_intent_words_inside_namespaces_are_not_query_intents(self):
        for namespace in ("create", "craft", "recipe", "make"):
            with self.subTest(namespace=namespace):
                self.assertIsNone(self.lookup(f"What is {namespace}:beacon's recipe?"))

    def test_intent_words_inside_item_paths_keep_exact_target_identity(self):
        for path in ("create", "craft", "recipe", "make"):
            with self.subTest(path=path):
                payload = target_catalog()
                key = f"blue:{path}"
                payload["items"][key] = payload["items"].pop("blue:beacon")
                payload["recipes"]["blue:beacon"]["outputs"][1].update(id=key, key=key)
                answer = self.lookup(f"What is {key}'s crafting recipe?", payload)
                self.assertIsNotNone(answer)
                self.assertIn("2 Crystal to make 3 Beacons", answer.chat_reply)

    def test_ingredient_words_inside_keys_do_not_split_the_target(self):
        for namespace in ("with", "using", "from", "for"):
            for template in ("How do I craft {key}?", "{key} crafting recipe"):
                with self.subTest(namespace=namespace, template=template):
                    self.assertIsNone(
                        self.lookup(template.format(key=f"{namespace}:beacon"))
                    )
        payload = target_catalog()
        key = "for:beacon"
        payload["items"][key] = payload["items"].pop("blue:beacon")
        payload["recipes"]["blue:beacon"]["outputs"][1].update(id=key, key=key)
        self.assertIn(
            "2 Crystal", self.lookup("How do I craft for:beacon?", payload).chat_reply
        )
