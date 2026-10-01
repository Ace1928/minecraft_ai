"""Optional cross-repo offline contract test; all pack/world data is invented.

Set GAME_FORGE_KIT to game_forge/minecraft/pokemon-family-server. This uses the
real producer, file pin, runtime context, controller and chat-authority code.
The model transport is a stub; this does not qualify model quality or learning.
"""

from __future__ import annotations

from collections import deque
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from minecraft_ai.pack_recipes import PackRecipeCatalog


@unittest.skipUnless(
    os.environ.get("GAME_FORGE_KIT"), "set GAME_FORGE_KIT for cross-repo test"
)
class GameForgeRecipeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        kit = Path(os.environ["GAME_FORGE_KIT"]).resolve(strict=True)
        sys.path.insert(0, str(kit))
        try:
            cls.producer = importlib.import_module("crafting")
        finally:
            sys.path.remove(str(kit))

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.runtime = self.root / "invented-runtime"
        self.write("server.properties", "level-name=PokemonFamily\n", text=True)
        self.write("pokemon-family-install.json", {"bds_version": "1.26.52.3"})
        self.write(
            "worlds/PokemonFamily/world_behavior_packs.json",
            [{"pack_id": "blue-fixture", "version": [1, 0, 0]}],
        )
        (self.runtime / "resource_packs/vanilla").mkdir(parents=True)
        for pack in ("blue", "amber"):
            self.write(
                f"behavior_packs/{pack}/manifest.json",
                {
                    "header": {
                        "uuid": f"{pack}-fixture",
                        "version": [1, 0, 0],
                        "name": f"Fictional {pack} Pack",
                    }
                },
            )
            self.write(
                f"behavior_packs/{pack}/recipes/beacon.json",
                {
                    "minecraft:recipe_shapeless": {
                        "description": {"identifier": f"{pack}:beacon"},
                        "ingredients": [{"item": f"{pack}:crystal", "count": 2}],
                        "result": [
                            {"item": f"{pack}:shell"},
                            {"item": f"{pack}:beacon", "count": 3},
                        ],
                        "tags": ["crafting_table"],
                    },
                },
            )

    def write(self, relative, value, *, text=False):
        path = self.runtime / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if text else json.dumps(value), encoding="utf-8")

    def export(self):
        # Keep registry/client-resource discovery inside the fictional fixture.
        with (
            patch.object(self.producer, "ROOT", self.root),
            patch.object(
                Path,
                "home",
                return_value=self.root,
            ),
        ):
            library = self.producer.build_library(self.runtime)
        output = self.root / "catalog.json"
        digest = self.producer.export_library(library, output)
        return PackRecipeCatalog.load(output, digest), library, output, digest

    def test_active_world_snapshot_reaches_existing_cognition_and_chat_authority(self):
        from minecraft_ai.builtin_skills import build_bootstrap_skill_library
        from minecraft_ai.cognition import HighLevelController
        from minecraft_ai.game_chat import game_chat_authority_matches
        from minecraft_ai.memory import MemoryStore
        from minecraft_ai.models import ModelResponse
        from minecraft_ai.perception import (
            FrameState,
            PerceptionBlackboard,
            PerceptionFact,
        )
        from minecraft_ai.roles import get_role
        from minecraft_ai.runtime import AgentRuntime
        from minecraft_ai.social import SocialState

        catalog, library, _output, _digest = self.export()
        self.assertNotIn("amber:beacon", library["items"])
        self.assertIsNone(
            catalog.lookup("How do I craft amber:beacon?", game_version="1.26.52.3")
        )
        now = time.monotonic_ns()
        board = PerceptionBlackboard()

        def publish(
            instance="bedrock:1.26.52.3:fictional-session", age_ms=0, target=None
        ):
            selected = board if target is None else target
            previous = selected.raw_latest()
            selected.publish(
                FrameState(
                    frame_id=1 if previous is None else previous.frame_id + 1,
                    captured_ns=time.monotonic_ns(),
                    instance_id=instance,
                    width=32,
                    height=32,
                    facts=(
                        PerceptionFact(
                            key="social.player_message",
                            value="FixturePlayer: How do I craft blue:beacon?",
                            confidence=0.99,
                            observed_ns=now - age_ms * 1_000_000,
                            source="fictional-contract",
                            expires_after_ms=30_000,
                        ),
                    ),
                )
            )

        publish()
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
        before = board.raw_latest()
        context = runtime._cognition_context(requires_wood=False)
        expected = "At a crafting table, use 2 Crystal to make 3 Beacons."
        self.assertEqual(context.pack_recipe_reply, expected)
        self.assertIn(library["revision"], context.wiki[0].version_key)
        model = Mock(spec=["complete_constrained"])
        model.complete_constrained.return_value = ModelResponse(
            model="fictional-transport-no-learning",
            latency_ms=0,
            text='{"s":null}',
        )
        controller = HighLevelController(model, build_bootstrap_skill_library())
        decision = controller.decide(board, context)
        self.assertEqual(decision.game_chat, expected)
        self.assertTrue(game_chat_authority_matches(decision, board))
        self.assertIsNone(
            decision.skill_id
        )  # Recipe evidence grants no craft/mine action.
        self.assertIsNone(decision.research_query)
        self.assertEqual(decision.skill_parameters, {})
        self.assertIs(board.raw_latest(), before)
        self.assertIsNone(board.fact("inventory.beacon"))
        self.assertEqual(model.complete_constrained.call_count, 1)
        replacement = PerceptionBlackboard()
        publish("bedrock:1.26.52.3:replacement-session", target=replacement)
        self.assertFalse(game_chat_authority_matches(decision, replacement))
        with patch("time.monotonic_ns", return_value=now + 31_000_000_000):
            self.assertFalse(game_chat_authority_matches(decision, board))
            self.assertIsNone(
                runtime._cognition_context(requires_wood=False).pack_recipe_reply
            )

    def test_pack_update_changes_answer_and_invalidates_old_file_pin(self):
        old, library, output, pin = self.export()
        raw_before = output.read_bytes()
        recipe = self.runtime / "behavior_packs/blue/recipes/beacon.json"
        document = json.loads(recipe.read_text())
        document["minecraft:recipe_shapeless"]["result"][1]["count"] = 6
        recipe.write_text(json.dumps(document))
        current, updated_library, _output, new_pin = self.export()
        self.assertNotEqual(library["revision"], updated_library["revision"])
        self.assertNotEqual(pin, new_pin)
        self.assertNotEqual(raw_before, output.read_bytes())
        with self.assertRaisesRegex(ValueError, "digest or size"):
            PackRecipeCatalog.load(output, pin)
        query = "How do I craft blue:beacon?"
        self.assertIn(
            "make 3 Beacons", old.lookup(query, game_version="1.26.52.3").chat_reply
        )
        self.assertIn(
            "make 6 Beacons", current.lookup(query, game_version="1.26.52.3").chat_reply
        )
        self.assertIsNone(current.lookup(query, game_version="1.26.99"))
