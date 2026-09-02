from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .memory import MemoryKind, MemoryRecord, MemoryStore
from .planning import Goal
from .skills import SkillLibrary, SkillSpec, SkillStats
from .social import Promise, SharedProject, SocialState


SCHEMA_VERSION = 1


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
        elif int(current[0]) != SCHEMA_VERSION:
            raise RuntimeError(
                f"unsupported state database schema {current[0]}; expected {SCHEMA_VERSION}"
            )
        self.connection.commit()

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
                skill_id, context_key, successes, failures, timeouts, cancellations
            ) VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(skill_id, context_key) DO UPDATE SET
                successes=excluded.successes,
                failures=excluded.failures,
                timeouts=excluded.timeouts,
                cancellations=excluded.cancellations
            """,
            (
                skill_id,
                context_key,
                stats.successes,
                stats.failures,
                stats.timeouts,
                stats.cancellations,
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
            SELECT skill_id, context_key, successes, failures, timeouts, cancellations
            FROM skill_stats
            """
        ):
            skill_id, context_key, successes, failures, timeouts, cancellations = row
            library.stats[(str(skill_id), str(context_key))] = SkillStats(
                successes=int(successes),
                failures=int(failures),
                timeouts=int(timeouts),
                cancellations=int(cancellations),
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
