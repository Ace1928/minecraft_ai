from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .memory import MemoryKind, MemoryRecord, MemoryStore
from .planning import Goal
from .skills import SkillLibrary, SkillSpec, SkillStats
from .social import Promise, SharedProject, SocialState
from .spatial import PlaceRecord, SpatialPlaceMemory


SCHEMA_VERSION = 2


class StateDatabase:
    """Small durable state store for restart-safe agent identity and learning."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> StateDatabase:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _migrate(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                updated_ns INTEGER NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memories_kind_updated
                ON memories(kind, updated_ns DESC);
            CREATE TABLE IF NOT EXISTS skills (
                skill_id TEXT PRIMARY KEY,
                version INTEGER NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS skill_stats (
                skill_id TEXT NOT NULL,
                context_key TEXT NOT NULL,
                successes INTEGER NOT NULL,
                failures INTEGER NOT NULL,
                timeouts INTEGER NOT NULL,
                cancellations INTEGER NOT NULL,
                consecutive_failures INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(skill_id, context_key)
            );
            CREATE TABLE IF NOT EXISTS goals (
                goal_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS promises (
                promise_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS projects (
                project_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS spatial_places (
                place_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                dimension TEXT NOT NULL,
                x REAL,
                y REAL,
                z REAL,
                payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_spatial_places_kind_dim
                ON spatial_places(kind, dimension);
            """
        )
        current = self.connection.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()
        if current is None:
            self.connection.execute(
                "INSERT INTO meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
        elif int(current[0]) == 1:
            self._migrate_v1_to_v2()
            self.connection.execute(
                "UPDATE meta SET value=? WHERE key='schema_version'",
                (str(SCHEMA_VERSION),),
            )
        elif int(current[0]) != SCHEMA_VERSION:
            raise RuntimeError(
                f"unsupported state database schema {current[0]}; expected {SCHEMA_VERSION}"
            )
        self.connection.commit()

    def _migrate_v1_to_v2(self) -> None:
        stats_columns = {
            str(row[1]) for row in self.connection.execute("PRAGMA table_info(skill_stats)")
        }
        if "consecutive_failures" not in stats_columns:
            self.connection.execute(
                "ALTER TABLE skill_stats ADD COLUMN consecutive_failures INTEGER NOT NULL DEFAULT 0"
            )
        self.connection.executescript(
            """
            ALTER TABLE spatial_places RENAME TO spatial_places_v1;
            CREATE TABLE spatial_places (
                place_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                dimension TEXT NOT NULL,
                x REAL,
                y REAL,
                z REAL,
                payload TEXT NOT NULL
            );
            INSERT INTO spatial_places(place_id, kind, dimension, x, y, z, payload)
                SELECT place_id, kind, dimension, x, y, z, payload FROM spatial_places_v1;
            DROP TABLE spatial_places_v1;
            CREATE INDEX idx_spatial_places_kind_dim
                ON spatial_places(kind, dimension);
            """
        )

    def save_memory(self, record: MemoryRecord) -> None:
        self.connection.execute(
            """
            INSERT INTO memories(memory_id, kind, updated_ns, payload)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                kind=excluded.kind,
                updated_ns=excluded.updated_ns,
                payload=excluded.payload
            """,
            (
                record.memory_id,
                record.kind.value,
                record.updated_ns,
                record.model_dump_json(),
            ),
        )
        self.connection.commit()

    def load_memories(self, kinds: set[MemoryKind] | None = None) -> MemoryStore:
        if kinds:
            placeholders = ",".join("?" for _ in kinds)
            rows = self.connection.execute(
                f"SELECT payload FROM memories WHERE kind IN ({placeholders})",
                tuple(kind.value for kind in kinds),
            ).fetchall()
        else:
            rows = self.connection.execute("SELECT payload FROM memories").fetchall()
        store = MemoryStore()
        for (payload,) in rows:
            store.upsert(MemoryRecord.model_validate_json(payload))
        return store

    def save_place(self, record: PlaceRecord) -> None:
        self.connection.execute(
            """
            INSERT INTO spatial_places(place_id, kind, dimension, x, y, z, payload)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(place_id) DO UPDATE SET
                kind=excluded.kind,
                dimension=excluded.dimension,
                x=excluded.x,
                y=excluded.y,
                z=excluded.z,
                payload=excluded.payload
            """,
            (
                record.place_id,
                record.kind.value,
                record.dimension,
                record.x,
                record.y,
                record.z,
                record.model_dump_json(),
            ),
        )
        self.connection.commit()

    def load_places(self) -> SpatialPlaceMemory:
        rows = self.connection.execute("SELECT payload FROM spatial_places").fetchall()
        memory = SpatialPlaceMemory()
        for (payload,) in rows:
            memory.upsert(PlaceRecord.model_validate_json(payload))
        return memory

    def save_skill(self, spec: SkillSpec) -> None:
        self.connection.execute(
            """
            INSERT INTO skills(skill_id, version, payload) VALUES(?, ?, ?)
            ON CONFLICT(skill_id) DO UPDATE SET
                version=excluded.version,
                payload=excluded.payload
            """,
            (spec.skill_id, spec.version, spec.model_dump_json()),
        )
        self.connection.commit()

    def save_skill_stats(self, skill_id: str, context_key: str, stats: SkillStats) -> None:
        self.connection.execute(
            """
            INSERT INTO skill_stats(
                skill_id, context_key, successes, failures, timeouts, cancellations,
                consecutive_failures
            ) VALUES(?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(skill_id, context_key) DO UPDATE SET
                successes=excluded.successes,
                failures=excluded.failures,
                timeouts=excluded.timeouts,
                cancellations=excluded.cancellations,
                consecutive_failures=excluded.consecutive_failures
            """,
            (
                skill_id,
                context_key,
                stats.successes,
                stats.failures,
                stats.timeouts,
                stats.cancellations,
                stats.consecutive_failures,
            ),
        )
        self.connection.commit()

    def load_skills(self) -> SkillLibrary:
        library = SkillLibrary()
        for (payload,) in self.connection.execute("SELECT payload FROM skills"):
            spec = SkillSpec.model_validate_json(payload)
            library.specs[spec.skill_id] = spec
        for row in self.connection.execute(
            """
            SELECT skill_id, context_key, successes, failures, timeouts, cancellations,
                   consecutive_failures
            FROM skill_stats
            """
        ):
            (
                skill_id,
                context_key,
                successes,
                failures,
                timeouts,
                cancellations,
                consecutive_failures,
            ) = row
            library.stats[(str(skill_id), str(context_key))] = SkillStats(
                successes=int(successes),
                failures=int(failures),
                timeouts=int(timeouts),
                cancellations=int(cancellations),
                consecutive_failures=int(consecutive_failures),
            )
        return library

    def save_goal(self, goal: Goal) -> None:
        self._save_json("goals", "goal_id", goal.goal_id, goal.model_dump(mode="json"))

    def load_goals(self) -> tuple[Goal, ...]:
        return tuple(
            Goal.model_validate(json.loads(payload))
            for (payload,) in self.connection.execute("SELECT payload FROM goals")
        )

    def save_promise(self, promise: Promise) -> None:
        self._save_json(
            "promises",
            "promise_id",
            promise.promise_id,
            promise.model_dump(mode="json"),
        )

    def save_project(self, project: SharedProject) -> None:
        self._save_json(
            "projects",
            "project_id",
            project.project_id,
            project.model_dump(mode="json"),
        )

    def load_social(self) -> SocialState:
        state = SocialState()
        for (payload,) in self.connection.execute("SELECT payload FROM promises"):
            promise = Promise.model_validate(json.loads(payload))
            state.promises[promise.promise_id] = promise
        for (payload,) in self.connection.execute("SELECT payload FROM projects"):
            project = SharedProject.model_validate(json.loads(payload))
            state.projects[project.project_id] = project
        return state

    def _save_json(
        self,
        table: str,
        key_column: str,
        key: str,
        payload: object,
    ) -> None:
        if table not in {"goals", "promises", "projects"}:
            raise ValueError("unsupported state table")
        if key_column not in {"goal_id", "promise_id", "project_id"}:
            raise ValueError("unsupported key column")
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        self.connection.execute(
            f"""
            INSERT INTO {table}({key_column}, payload) VALUES(?, ?)
            ON CONFLICT({key_column}) DO UPDATE SET payload=excluded.payload
            """,
            (key, serialized),
        )
        self.connection.commit()
