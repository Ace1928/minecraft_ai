from __future__ import annotations

from pathlib import Path
import sqlite3
import time

from minecraft_ai.memory import MemoryKind, MemoryRecord
from minecraft_ai.perception import ScreenRegion, Track
from minecraft_ai.planning import Goal
from minecraft_ai.skills import SkillSpec, SkillStats
from minecraft_ai.social import (
    OperatorMessage,
    OperatorMessageKind,
    OperatorMessageStatus,
    Promise,
    SharedProject,
)
from minecraft_ai.storage import StateDatabase


def test_state_database_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite3"
    with StateDatabase(path) as db:
        db.save_memory(
            MemoryRecord(
                memory_id="m1",
                kind=MemoryKind.SPATIAL,
                text="Base is beside river",
                created_ns=1,
                updated_ns=2,
                location_key="base",
            )
        )
        db.save_skill(SkillSpec(skill_id="walk", name="Walk"))
        db.save_skill_stats(
            "walk",
            "flat",
            SkillStats(successes=7, failures=1, consecutive_failures=1),
        )
        db.save_goal(Goal(goal_id="g1", description="Build roof"))
        db.save_promise(
            Promise(promise_id="p1", player="Alex", summary="Finish roof", created_ns=3)
        )
        db.save_project(SharedProject(project_id="build", name="House", created_ns=4, owner="Alex"))
        db.save_operator_message(
            OperatorMessage(
                message_id="op1",
                created_ns=time.time_ns(),
                kind=OperatorMessageKind.CORRECTION,
                text="Return to the workshop and finish the west wall.",
            )
        )
        db.save_operator_target(
            Track(
                track_id="operator:log-1",
                label="oak_log",
                confidence=1.0,
                region=ScreenRegion(x=0.4, y=0.3, width=0.2, height=0.4),
                first_seen_ns=10,
                last_seen_ns=10,
                attributes={"source": "operator"},
            )
        )

    with StateDatabase(path) as db:
        memories = db.load_memories()
        assert memories.records["m1"].location_key == "base"
        skills = db.load_skills()
        assert skills.get("walk").name == "Walk"
        assert skills.stats[("walk", "flat")].successes == 7
        assert skills.stats[("walk", "flat")].consecutive_failures == 1
        assert db.load_goals()[0].goal_id == "g1"
        social = db.load_social()
        assert social.promises["p1"].player == "Alex"
        assert social.projects["build"].owner == "Alex"
        messages = db.load_operator_messages(limit=10)
        assert messages[0].message_id == "op1"
        assert messages[0].kind == OperatorMessageKind.CORRECTION
        updated = db.update_operator_message_status(
            "op1",
            OperatorMessageStatus.ACKNOWLEDGED,
            timestamp_ns=time.time_ns(),
            response_text="I will finish the west wall.",
        )
        assert updated.response_text == "I will finish the west wall."
        target = db.load_operator_target()
        assert target is not None
        assert target.label == "oak_log"
        assert target.region.width == 0.2
        db.clear_operator_target()
        assert db.load_operator_target() is None


def test_memory_kind_filter(tmp_path: Path) -> None:
    with StateDatabase(tmp_path / "state.sqlite3") as db:
        for index, kind in enumerate((MemoryKind.SPATIAL, MemoryKind.SOCIAL)):
            db.save_memory(
                MemoryRecord(
                    memory_id=str(index),
                    kind=kind,
                    text=str(kind),
                    created_ns=index,
                    updated_ns=index,
                )
            )
        loaded = db.load_memories({MemoryKind.SPATIAL})
        assert set(loaded.records) == {"0"}


def test_current_schema_open_does_not_reexecute_migration_ddl(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "state.sqlite3"
    with StateDatabase(path):
        pass

    calls = 0

    class ConnectionProxy:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self.connection = connection

        def __getattr__(self, name: str):
            return getattr(self.connection, name)

        def executescript(self, script: str):
            nonlocal calls
            calls += 1
            return self.connection.executescript(script)

    real_connect = sqlite3.connect
    monkeypatch.setattr(
        "minecraft_ai.storage.sqlite3.connect",
        lambda *args, **kwargs: ConnectionProxy(real_connect(*args, **kwargs)),
    )
    with StateDatabase(path):
        pass

    assert calls == 0
