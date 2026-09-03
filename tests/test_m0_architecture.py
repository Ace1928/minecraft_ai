from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from minecraft_ai.action_levels import ActionLevel
from minecraft_ai.builtin_skills import build_bootstrap_skill_library
from minecraft_ai.cognition import BootstrapCognitionPolicy, CognitionContext
from minecraft_ai.perception import FrameState, PerceptionBlackboard
from minecraft_ai.perception_service import BootstrapFastPerception
from minecraft_ai.platforms.bedrock_x11 import CapturedFrame
from minecraft_ai.roles import get_role
from minecraft_ai.social import Promise
from minecraft_ai.spatial import (
    PlaceKind,
    PlaceRecord,
    SpatialPlaceMemory,
    TopologicalWaypointGraph,
)
from minecraft_ai.storage import SCHEMA_VERSION, StateDatabase
from minecraft_ai.supervisor import Supervisor


def test_bootstrap_rgb_perception_cannot_supply_training_labels() -> None:
    perception = BootstrapFastPerception()
    frame = CapturedFrame(
        frame_id=1,
        captured_ns=time.monotonic_ns(),
        width=32,
        height=32,
        bgra=bytes((80, 100, 120, 255)) * (32 * 32),
    )

    facts = perception.infer(frame)

    assert perception.training_label_eligible is False
    assert facts
    assert all(fact.source.startswith(("bootstrap:", "safety:")) for fact in facts)
    assert all("not-training-label" in fact.source for fact in facts)
    assert all("not-training-label" in fact.source for fact in facts)


def test_bootstrap_cognition_uses_actual_promise_schema() -> None:
    board = PerceptionBlackboard()
    board.publish(
        FrameState(
            frame_id=1,
            captured_ns=time.monotonic_ns(),
            instance_id="bedrock:test",
            width=1280,
            height=720,
        )
    )
    promise = Promise(
        promise_id="promise-1",
        player="Lloyd",
        summary="finish the workshop",
        created_ns=time.monotonic_ns(),
    )
    context = CognitionContext(
        role=get_role("generalist"),
        goals=(),
        memories=(),
        promises=(promise,),
        wiki=(),
    )

    decision = BootstrapCognitionPolicy(build_bootstrap_skill_library()).decide(board, context)

    assert decision.chosen_goal_id == "promise:promise-1"
    assert "finish the workshop" in decision.reasoning_summary


def test_topological_routing_does_not_require_exact_coordinates() -> None:
    now = time.monotonic_ns()
    memory = SpatialPlaceMemory()
    for place_id, kind in (
        ("base", PlaceKind.BASE),
        ("door", PlaceKind.WAYPOINT),
        ("village", PlaceKind.VILLAGE),
    ):
        memory.upsert(
            PlaceRecord(
                place_id=place_id,
                name=place_id,
                kind=kind,
                discovered_ns=now,
                last_visited_ns=now,
            )
        )
    graph = TopologicalWaypointGraph()
    graph.add_edge("base", "door", 1.0)
    graph.add_edge("door", "village", 10.0)

    assert graph.find_path("base", "village", memory) == ["base", "door", "village"]


def test_state_schema_migrates_v1_and_preserves_failure_streak(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO meta(key, value) VALUES('schema_version', '1');
        CREATE TABLE skill_stats (
            skill_id TEXT NOT NULL,
            context_key TEXT NOT NULL,
            successes INTEGER NOT NULL,
            failures INTEGER NOT NULL,
            timeouts INTEGER NOT NULL,
            cancellations INTEGER NOT NULL,
            PRIMARY KEY(skill_id, context_key)
        );
        CREATE TABLE spatial_places (
            place_id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            dimension TEXT NOT NULL,
            x REAL NOT NULL,
            y REAL NOT NULL,
            z REAL NOT NULL,
            payload TEXT NOT NULL
        );
        """
    )
    connection.close()

    with StateDatabase(path) as database:
        version = database.connection.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()
        columns = {
            str(row[1]) for row in database.connection.execute("PRAGMA table_info(skill_stats)")
        }

    assert version == (str(SCHEMA_VERSION),)
    assert "consecutive_failures" in columns


def test_supervisor_rejects_unbound_host_display_input_even_when_requested() -> None:
    supervisor = Supervisor()
    supervisor.start()

    with pytest.raises(RuntimeError, match="dedicated-monitor binding"):
        supervisor.attach_bedrock_x11(":0", 1, allow_host=True)


def test_progression_skills_are_goal_conditioned_contracts_not_key_scripts() -> None:
    skills = build_bootstrap_skill_library()
    gather = skills.get("gather_nearby_wood")
    approach = skills.get("approach_visible_target")
    crafting = skills.get("craft_crafting_table")
    retreat = skills.get("retreat_from_danger")
    escape = skills.get("escape_submersion")
    open_inventory = skills.get("open_inventory")
    close_inventory = skills.get("close_open_inventory")
    exploration = skills.get("explore_forward")
    level_ground = skills.get("traverse_level_ground")
    obstacle = skills.get("traverse_visible_obstacle")

    assert "visible trunk" in gather.description
    assert approach.initiation_alternatives[0][0].key == "target.reference_available"
    assert gather.success_conditions[0].key == "inventory.logs"
    assert "Bedrock crafting interface" in crafting.description
    assert crafting.success_conditions[0].key == "inventory.crafting_table"
    assert retreat.preconditions[0].key == "danger.immediate"
    assert retreat.action_permissions.allow_attack is False
    assert retreat.action_permissions.allow_use is False
    assert retreat.action_permissions.allow_jump is True
    assert exploration.action_permissions.allow_attack is False
    assert exploration.action_permissions.allow_jump is True
    assert exploration.policy_instruction == "Run around and explore the Minecraft world."
    assert level_ground.policy_ref == "traverse_level_ground"
    assert level_ground.action_level == ActionLevel.MOTION
    assert level_ground.action_permissions.allow_jump is False
    assert level_ground.action_permissions.allow_attack is False
    assert obstacle.policy_ref == "traverse_obstacle"
    assert obstacle.action_level == ActionLevel.MOTION
    assert obstacle.policy_instruction == "jump forward"
    assert obstacle.policy_condition_scale == 6.0
    assert obstacle.action_permissions.allow_jump is True
    assert obstacle.action_permissions.allow_attack is False
    assert obstacle.action_permissions.allow_use is False
    assert open_inventory.policy_ref == "open_inventory"
    assert open_inventory.action_level == ActionLevel.GUI
    assert open_inventory.policy_instruction == "open inventory"
    assert open_inventory.success_conditions[0].key == "scene.mode"
    assert open_inventory.success_conditions[0].value == "inventory"
    assert open_inventory.action_permissions.allow_inventory is True
    assert open_inventory.action_permissions.allow_attack is False
    assert close_inventory.policy_ref == "close_inventory"
    assert close_inventory.action_level == ActionLevel.GUI
    assert close_inventory.policy_instruction == "close inventory"
    assert close_inventory.success_conditions[0].key == "scene.playable"

    reacquire = skills.get("reacquire_target")
    assert reacquire.policy_ref == "navigate"
    assert reacquire.action_level == ActionLevel.GROUNDED
    assert reacquire.success_conditions[0].key == "target.tracking_confidence"
    assert reacquire.success_conditions[0].operator == "gte"
    assert reacquire.success_conditions[0].value == 0.65

    respawn = skills.get("respawn_after_death")
    assert respawn.preconditions[0].key == "scene.death"
    assert respawn.success_conditions[0].key == "scene.playable"
    assert respawn.policy_ref == "death_gui"
    assert respawn.action_level == ActionLevel.GUI
    assert respawn.policy_instruction == "respawn"
    assert retreat.preconditions[0].operator == "truthy"
    assert escape.preconditions[0].key == "environment.underwater"
    assert escape.success_conditions[0].operator == "falsy"
    assert escape.policy_instruction == "swim to the surface and leave the water"
