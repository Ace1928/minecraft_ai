from __future__ import annotations

from pathlib import Path

from minecraft_ai.memory import MemoryKind, MemoryRecord
from minecraft_ai.planning import Goal
from minecraft_ai.skills import SkillSpec, SkillStats
from minecraft_ai.social import Promise, SharedProject
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
