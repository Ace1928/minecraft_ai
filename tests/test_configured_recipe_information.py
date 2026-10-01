"""Explicit installed-file reference remains separate from active game authority."""
from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

from minecraft_ai.pack_recipes import PackRecipeCatalog
from test_pack_recipe_targets import target_catalog


def information_payload() -> dict:
    payload = target_catalog()
    payload["configured_recipe_identity"] = {
        "state": "configured_only", "scope_sha256": "c" * 64,
    }
    payload["sources"] = [
        {"folder": "vanilla", "version": [0, 0, 1], "vanilla": True},
        {"folder": "fictional_blue_pack", "version": [1, 3, 156], "vanilla": False},
    ]
    return payload


class ConfiguredRecipeInformationTests(unittest.TestCase):
    def lookup(self, payload=None, *, query="How do I craft blue:beacon?",
               version="1.26.52.3"):
        data = information_payload() if payload is None else payload
        before = copy.deepcopy(data)
        answer = PackRecipeCatalog(data, "b" * 64).lookup_configured_information(
            query, game_version=version,
        )
        self.assertEqual(data, before)
        return answer

    def test_explicit_reference_preserves_exact_quantities_and_namespace(self):
        answer = self.lookup()
        self.assertIsNotNone(answer)
        self.assertEqual(answer["recipe_text"],
                         "At a crafting table, use 2 Crystal to make 3 Beacons.")
        self.assertEqual(answer["configured_packs"], [
            {"folder": "fictional_blue_pack", "version": [1, 3, 156]},
        ])
        self.assertEqual(answer["catalog_sha256"], "b" * 64)
        self.assertEqual(answer["catalog_revision"], "a" * 64)
        self.assertEqual(answer["configured_scope_sha256"], "c" * 64)

    def test_tool_labels_scope_and_carries_no_gameplay_identity_or_action(self):
        answer = self.lookup()
        self.assertEqual(answer["tool"], "pack_recipe.configured_snapshot")
        self.assertEqual(answer["state"], "configured_snapshot")
        self.assertIn("unverified", answer["notice"])
        self.assertIn("no crafting or motor permission", answer["notice"])
        for field in ("gameplay_authority", "live_engine_recipe_bytes_verified",
                      "selected_desktop_connection_verified"):
            self.assertIs(answer[field], False)
        for field in ("identity", "instance_id", "server_session", "skill",
                      "action", "chat_reply", "press", "goal"):
            self.assertNotIn(field, answer)

    def test_reference_does_not_upgrade_unknown_live_scope(self):
        catalog = PackRecipeCatalog(information_payload(), "b" * 64)
        with patch("minecraft_ai.pack_recipes.active_recipe_identity", return_value=None):
            self.assertIsNotNone(catalog.lookup_configured_information(
                "craft blue:beacon", game_version="1.26.52.3"))
            self.assertIsNone(catalog.lookup_live("craft blue:beacon", object()))
            self.assertEqual(catalog.active_scope_status(object()), "unknown")

    def test_result_mutation_cannot_change_catalog_or_later_reference(self):
        catalog = PackRecipeCatalog(information_payload(), "b" * 64)
        first = catalog.lookup_configured_information(
            "craft blue:beacon", game_version="1.26.52.3")
        first["configured_packs"][0]["version"][2] = 999
        second = catalog.lookup_configured_information(
            "craft blue:beacon", game_version="1.26.52.3")
        self.assertEqual(second["configured_packs"][0]["version"], [1, 3, 156])

    def test_absent_configured_identity_refuses_information(self):
        payload = information_payload()
        del payload["configured_recipe_identity"]
        self.assertIsNone(self.lookup(payload))

    def test_model_like_active_state_cannot_be_used_as_configured_provenance(self):
        payload = information_payload()
        payload["configured_recipe_identity"]["state"] = "observed_active"
        self.assertIsNone(self.lookup(payload))

    def test_invalid_scope_hash_refuses_information(self):
        payload = information_payload()
        for value in (True, "c" * 63, "C" * 64, "../catalog"):
            with self.subTest(value=value):
                payload["configured_recipe_identity"]["scope_sha256"] = value
                self.assertIsNone(self.lookup(payload))

    def test_sources_missing_or_oversized_refuse(self):
        payload = information_payload()
        for value in (None, {}, [payload["sources"][0]] * 129):
            with self.subTest(value=type(value).__name__):
                payload["sources"] = value
                self.assertIsNone(self.lookup(payload))

    def test_untyped_vanilla_marker_cannot_hide_missing_pack_metadata(self):
        payload = information_payload()
        payload["sources"][1]["vanilla"] = 1
        self.assertIsNone(self.lookup(payload))

    def test_unsafe_folder_or_invalid_version_refuses(self):
        for field, value in (("folder", "../pack"), ("folder", "Pack Name"),
                             ("version", [1, 3, True]), ("version", [1, -1, 156]),
                             ("version", [1, 3, 65536]), ("version", [1, 3])):
            with self.subTest(field=field, value=value):
                payload = information_payload()
                payload["sources"][1][field] = value
                self.assertIsNone(self.lookup(payload))

    def test_ambiguous_duplicate_folder_refuses(self):
        payload = information_payload()
        payload["sources"].append(copy.deepcopy(payload["sources"][1]))
        payload["sources"][-1]["version"] = [1, 3, 157]
        self.assertIsNone(self.lookup(payload))

    def test_no_configured_pack_and_excessive_pack_count_refuse(self):
        payload = information_payload()
        payload["sources"] = [payload["sources"][0]]
        self.assertIsNone(self.lookup(payload))
        payload = information_payload()
        payload["sources"] = [
            {"folder": "pack_" + str(n), "version": [1, 0, 0], "vanilla": False}
            for n in range(17)
        ]
        self.assertIsNone(self.lookup(payload))

    def test_wrong_game_version_and_non_recipe_request_refuse(self):
        self.assertIsNone(self.lookup(version="1.21.0"))
        self.assertIsNone(self.lookup(query="walk to blue:beacon"))

    def test_unknown_or_ambiguous_target_does_not_invent_a_recipe(self):
        self.assertIsNone(self.lookup(query="craft absent:beacon"))
        self.assertIsNone(self.lookup(query="craft a Beacon"))

    def test_missing_output_is_not_reference_information(self):
        payload = information_payload()
        payload["recipes"]["blue:beacon"]["outputs"].pop()
        self.assertIsNone(self.lookup(payload))
