from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

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


def test_supervisor_rejects_host_display_input_even_when_requested() -> None:
    supervisor = Supervisor()
    supervisor.start()

    with pytest.raises(RuntimeError, match="debug-only"):
        supervisor.attach_bedrock_x11(":0", 1, allow_host=True)


def test_progression_skills_are_goal_conditioned_contracts_not_key_scripts() -> None:
    skills = build_bootstrap_skill_library()
    gather = skills.get("gather_nearby_wood")
    crafting = skills.get("craft_crafting_table")
    retreat = skills.get("retreat_from_danger")
    escape = skills.get("escape_submersion")

    assert "visible trunk" in gather.description
    assert gather.success_conditions[0].key == "inventory.logs"
    assert "Bedrock crafting interface" in crafting.description
    assert crafting.success_conditions[0].key == "inventory.crafting_table"
    assert retreat.preconditions[0].key == "danger.immediate"
    assert retreat.preconditions[0].operator == "truthy"
    assert escape.preconditions[0].key == "environment.underwater"
    assert escape.success_conditions[0].operator == "falsy"
    assert escape.policy_instruction == "swim to the surface and leave the water"
