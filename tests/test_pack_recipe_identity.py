"""Recipe reference identities remain exact; no model, server or input is used."""

from __future__ import annotations

import copy
import json
import unittest

from minecraft_ai.pack_recipes import PackRecipeCatalog


class RecipeIdentityTests(unittest.TestCase):
    def setUp(self):
        self.items = {
            "minecraft:planks": {"name": "Incorrect Base Planks"},
            "#minecraft:planks": {"name": "Any Planks"},
            "minecraft:stick": {"name": "Stick"},
            "minecraft:wooden_pickaxe": {"name": "Wooden Pickaxe"},
            "minecraft:potion": {"name": "Incorrect Generic Potion"},
            "minecraft:potion@awkward": {"name": "Awkward Potion"},
            "minecraft:potion@strength": {"name": "Strength Potion"},
            "minecraft:blaze_powder": {"name": "Blaze Powder"},
        }
        self.recipe = {
            "ingredients": [
                {"id": "minecraft:planks", "key": "#minecraft:planks", "tag": True,
                 "count": 3},
                {"id": "minecraft:stick", "key": "minecraft:stick", "count": 2},
            ],
            "outputs": [{"id": "minecraft:wooden_pickaxe", "count": 1}],
            "stations": ["crafting_table"],
            "grid": [[{"id": "minecraft:planks", "key": "#minecraft:planks"},
                      {"id": "minecraft:stick", "key": "minecraft:stick"}]],
        }

    def render(self):
        before = json.dumps((self.recipe, self.items), sort_keys=True)
        result = PackRecipeCatalog._render_recipe(self.recipe, self.items)
        self.assertEqual(json.dumps((self.recipe, self.items), sort_keys=True), before)
        return result

    def test_tag_ingredient_uses_exact_exported_name(self):
        self.assertEqual(self.render(),
                         "At a crafting table, use 3 Any Planks and 2 Stick to make 1 "
                         "Wooden Pickaxe. Pattern A S.")

    def test_metadata_ingredient_and_output_do_not_become_generic_items(self):
        self.recipe.update(
            ingredients=[
                {"id": "minecraft:potion", "key": "minecraft:potion@awkward", "count": 1},
                {"id": "minecraft:blaze_powder", "count": 1},
            ],
            outputs=[{"id": "minecraft:potion", "key": "minecraft:potion@strength",
                      "count": 1}],
            stations=["brewing_stand"], grid=[],
        )
        self.assertEqual(self.render(), "At a brewing stand, use 1 Awkward Potion and 1 "
                         "Blaze Powder to make 1 Strength Potion.")

    def test_grid_preserves_two_references_with_the_same_raw_id(self):
        self.items.update({"test:variant@1": {"name": "Amber"},
                           "test:variant@2": {"name": "Blue"}})
        self.recipe["ingredients"] = [
            {"id": "test:variant", "key": "test:variant@1", "count": 1},
            {"id": "test:variant", "key": "test:variant@2", "count": 1},
        ]
        self.recipe["grid"] = [copy.deepcopy(self.recipe["ingredients"])]
        self.assertIn("Pattern A B.", self.render())

    def test_missing_declared_key_cannot_fall_back_to_a_different_item(self):
        self.recipe["ingredients"][0]["key"] = "#minecraft:missing"
        self.assertIsNone(self.render())

    def test_legacy_plain_id_remains_supported(self):
        self.recipe["ingredients"][0] = {"id": "minecraft:planks", "count": 3}
        self.recipe["grid"] = []
        self.assertIn("3 Incorrect Base Planks", self.render())

    def test_identity_bearing_legacy_reference_without_key_is_refused(self):
        for field, value in (("tag", True), ("data", 1)):
            with self.subTest(field=field):
                self.recipe["ingredients"][0] = {
                    "id": "minecraft:planks", "count": 3, field: value,
                }
                self.assertIsNone(self.render())

    def test_grid_cannot_substitute_base_identity_or_drop_unknown_cells(self):
        for cell in ({"id": "minecraft:planks"}, "minecraft:planks"):
            with self.subTest(cell=cell):
                self.recipe["grid"] = [[cell]]
                self.assertIsNone(self.render())
