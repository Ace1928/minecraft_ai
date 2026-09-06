from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from minecraft_ai.datasets.schema import TrajectoryManifest
from minecraft_ai.episodes import RuntimeEvent, RuntimeEventKind
from minecraft_ai.memory import MemoryKind, MemoryRecord, MemoryStore
from minecraft_ai.planning import Goal
from minecraft_ai.perception import Track
from minecraft_ai.skills import SkillLibrary, SkillSpec, SkillStats
from minecraft_ai.social import (
    OperatorMessage,
    OperatorMessageKind,
    OperatorMessageStatus,
    Promise,
    SharedProject,
    SocialState,
)
from minecraft_ai.spatial import PlaceRecord, SpatialPlaceMemory

if TYPE_CHECKING:
    from minecraft_ai.eval.evaluator import BenchmarkReport


SCHEMA_VERSION = 7
MAX_OPERATOR_REVISION = (1 << 63) - 1
_OPERATOR_REVISION_KEY = "operator_authority_revision"


@dataclass(frozen=True)
class OperatorContextSnapshot:
    revision: int
    messages: tuple[OperatorMessage, ...]
    target: Track | None


def _message_authority(message: OperatorMessage) -> dict[str, object]:
    """Delivery receipts do not change intent; consumed corrections do."""
    content = message.model_dump(mode="json", exclude={
        "status", "delivered_ns", "acknowledged_ns", "response_text",
    })
    content["archived"] = message.status == OperatorMessageStatus.ARCHIVED
    content["correction_consumed"] = (
        message.kind == OperatorMessageKind.CORRECTION
        and message.status == OperatorMessageStatus.ACKNOWLEDGED
    )
    return content


class StateDatabase:
    """Small durable state store for restart-safe agent identity and learning."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=15.0)
        self.connection.execute("PRAGMA busy_timeout=15000")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._transaction_depth = 0
        self._migrate()

    def close(self) -> None:
        self.connection.close()

    def set_busy_timeout_ms(self, milliseconds: int) -> None:
        """Bound how long this connection may block its caller on writer contention."""
        if milliseconds < 0 or milliseconds > 60_000:
            raise ValueError("busy timeout must be between 0 and 60000 milliseconds")
        self.connection.execute(f"PRAGMA busy_timeout={milliseconds}")

    def __enter__(self) -> StateDatabase:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _commit(self) -> None:
        if not self._transaction_depth:
            self.connection.commit()

    def _execute_write(
        self, statement: str, parameters: tuple[object, ...], *, commit: bool = True,
    ) -> None:
        """Own new writes while preserving existing caller-owned batch semantics.

        A failed implicit INSERT can leave SQLite in a deferred transaction.
        Owning the transaction before executing guarantees rollback on failure,
        so later operator work can establish its own authority boundary. Inside
        an admission transaction the existing savepoint defers the outer commit.
        ``commit=False`` leaves its batch open. A later ``commit=True`` write
        joins and commits an already open unmanaged batch, as before; failures
        in that path leave rollback of prior caller-owned work to the caller.
        """
        if not commit:
            self.connection.execute(statement, parameters)
            return
        if not self._transaction_depth and self.connection.in_transaction:
            self.connection.execute(statement, parameters)
            self.connection.commit()
            return
        with self._transaction(write=True):
            self.connection.execute(statement, parameters)

    @contextmanager
    def _transaction(self, *, write: bool) -> Iterator[None]:
        """Own a short transaction, or isolate a nested storage operation."""
        outer = self._transaction_depth == 0
        savepoint = f"state_boundary_{self._transaction_depth}"
        if outer:
            if self.connection.in_transaction:
                raise RuntimeError("finish the pending storage transaction before this boundary")
            self.connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        else:
            self.connection.execute(f"SAVEPOINT {savepoint}")
        self._transaction_depth += 1
        try:
            yield
            if outer:
                self.connection.commit()
            else:
                self.connection.execute(f"RELEASE {savepoint}")
        except BaseException:
            if outer:
                self.connection.rollback()
            else:
                self.connection.execute(f"ROLLBACK TO {savepoint}")
                self.connection.execute(f"RELEASE {savepoint}")
            raise
        finally:
            self._transaction_depth -= 1

    def operator_revision(self) -> int:
        """Durable monotonic intent revision; legacy stores start at zero."""
        row = self.connection.execute(
            "SELECT value FROM meta WHERE key=?", (_OPERATOR_REVISION_KEY,),
        ).fetchone()
        if row is None:
            return 0
        raw = row[0]
        if (not isinstance(raw, str) or not 1 <= len(raw) <= 19
                or not raw.isascii() or not raw.isdecimal()):
            raise ValueError("invalid persisted operator authority revision")
        revision = int(raw)
        if not 0 <= revision <= MAX_OPERATOR_REVISION:
            raise ValueError("persisted operator authority revision is out of bounds")
        if str(revision) != raw:
            raise ValueError("invalid persisted operator authority revision")
        return revision

    def _advance_operator_revision(self) -> None:
        revision = self.operator_revision()
        if revision == MAX_OPERATOR_REVISION:
            raise OverflowError("operator authority revision exhausted; mutation rejected")
        self.connection.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (_OPERATOR_REVISION_KEY, str(revision + 1)),
        )

    def load_operator_context(
        self, *, statuses: set[OperatorMessageStatus] | None = None, limit: int = 100,
    ) -> OperatorContextSnapshot:
        """Read revision, messages and target from one short SQLite snapshot.

        No database lock is retained after return: model work uses this value,
        then checks its revision at the separate admission boundary.
        """
        with self._transaction(write=False):
            return OperatorContextSnapshot(
                revision=self.operator_revision(),
                messages=self.load_operator_messages(statuses=statuses, limit=limit),
                target=self.load_operator_target(),
            )

    @contextmanager
    def admit_operator_revision(self, expected: int) -> Iterator[bool]:
        """Serialize a short revision check and decision application with writers.

        Run model work BEFORE entering. Apply the result only when the yielded
        value is true. Storage saves inside the boundary commit together at its
        exit; exceptions roll them back. External pause/lease authority must be
        checked separately by its owner, as it is not stored in this database.
        Do not call connection.commit() directly from within this boundary.
        """
        if type(expected) is not int or not 0 <= expected <= MAX_OPERATOR_REVISION:
            raise ValueError("expected operator authority revision is out of bounds")
        with self._transaction(write=True):
            yield self.operator_revision() == expected

    def _migrate(self) -> None:
        try:
            current = self.connection.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()
        except sqlite3.OperationalError as exc:
            if "no such table" not in str(exc).casefold():
                raise
            current = None
        if current is not None and int(current[0]) == SCHEMA_VERSION:
            return
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
            CREATE TABLE IF NOT EXISTS operator_messages (
                message_id TEXT PRIMARY KEY,
                created_ns INTEGER NOT NULL,
                status TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_operator_messages_status_created
                ON operator_messages(status, created_ns DESC);
            CREATE TABLE IF NOT EXISTS operator_targets (
                target_id TEXT PRIMARY KEY,
                created_ns INTEGER NOT NULL,
                active INTEGER NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_operator_targets_active_created
                ON operator_targets(active, created_ns DESC);
            CREATE TABLE IF NOT EXISTS trajectories (
                trajectory_id TEXT PRIMARY KEY,
                started_ns INTEGER NOT NULL,
                ended_ns INTEGER,
                source_type TEXT NOT NULL,
                game_version TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS trajectory_shards (
                shard_id TEXT PRIMARY KEY,
                trajectory_id TEXT NOT NULL REFERENCES trajectories(trajectory_id),
                path TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                first_step_index INTEGER NOT NULL,
                last_step_index INTEGER NOT NULL,
                step_count INTEGER NOT NULL,
                bytes INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS trajectory_steps_index (
                trajectory_id TEXT NOT NULL REFERENCES trajectories(trajectory_id),
                step_index INTEGER NOT NULL,
                captured_ns INTEGER NOT NULL,
                accepted_ns INTEGER,
                shard_id TEXT NOT NULL REFERENCES trajectory_shards(shard_id),
                sample_key TEXT NOT NULL,
                frame_hash TEXT NOT NULL,
                action_json TEXT NOT NULL,
                action_level TEXT NOT NULL,
                action_origin TEXT NOT NULL DEFAULT 'legacy',
                policy_id TEXT,
                model_version TEXT,
                route_id TEXT,
                policy_action_kind TEXT,
                policy_request_id TEXT,
                prediction_id TEXT,
                condition_id TEXT,
                condition_json TEXT,
                behavior_token INTEGER,
                latent_id TEXT,
                target_track_id TEXT,
                skill_run_id TEXT,
                skill_id TEXT,
                goal_id TEXT,
                plan_node_id TEXT,
                correction_of_step INTEGER,
                PRIMARY KEY(trajectory_id, step_index)
            );
            CREATE INDEX IF NOT EXISTS idx_trajectory_steps_capture
                ON trajectory_steps_index(trajectory_id, captured_ns);
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY,
                trajectory_id TEXT NOT NULL REFERENCES trajectories(trajectory_id),
                started_ns INTEGER NOT NULL,
                ended_ns INTEGER,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                trajectory_id TEXT REFERENCES trajectories(trajectory_id),
                step_index INTEGER,
                observed_ns INTEGER NOT NULL,
                kind TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS benchmark_runs (
                benchmark_run_id TEXT PRIMARY KEY,
                created_ns INTEGER NOT NULL,
                suite_id TEXT NOT NULL,
                git_commit TEXT,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS benchmark_task_results (
                benchmark_run_id TEXT NOT NULL REFERENCES benchmark_runs(benchmark_run_id),
                task_id TEXT NOT NULL,
                repetition INTEGER NOT NULL,
                status TEXT NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY(benchmark_run_id, task_id, repetition)
            );
            CREATE INDEX IF NOT EXISTS idx_benchmark_task_results_status
                ON benchmark_task_results(benchmark_run_id, status);
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
        else:
            version = int(current[0])
            if version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"unsupported state database schema {version}; expected <= {SCHEMA_VERSION}"
                )
            if version == 1:
                self._migrate_v1_to_v2()
                version = 2
            if version == 2:
                self._migrate_v2_to_v3()
                version = 3
            if version == 3:
                self._migrate_v3_to_v4()
                version = 4
            if version == 4:
                self._migrate_v4_to_v5()
                version = 5
            if version == 5:
                self._migrate_v5_to_v6()
                version = 6
            if version == 6:
                self._migrate_v6_to_v7()
                version = 7
            self.connection.execute(
                "UPDATE meta SET value=? WHERE key='schema_version'",
                (str(SCHEMA_VERSION),),
            )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_trajectory_steps_policy_route "
            "ON trajectory_steps_index(policy_id, model_version, route_id)"
        )
        self._commit()

    def _migrate_v2_to_v3(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS operator_messages (
                message_id TEXT PRIMARY KEY,
                created_ns INTEGER NOT NULL,
                status TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_operator_messages_status_created
                ON operator_messages(status, created_ns DESC);
            """
        )

    def _migrate_v3_to_v4(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS trajectories (
                trajectory_id TEXT PRIMARY KEY,
                started_ns INTEGER NOT NULL,
                ended_ns INTEGER,
                source_type TEXT NOT NULL,
                game_version TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS trajectory_shards (
                shard_id TEXT PRIMARY KEY,
                trajectory_id TEXT NOT NULL REFERENCES trajectories(trajectory_id),
                path TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                first_step_index INTEGER NOT NULL,
                last_step_index INTEGER NOT NULL,
                step_count INTEGER NOT NULL,
                bytes INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS trajectory_steps_index (
                trajectory_id TEXT NOT NULL REFERENCES trajectories(trajectory_id),
                step_index INTEGER NOT NULL,
                captured_ns INTEGER NOT NULL,
                shard_id TEXT NOT NULL REFERENCES trajectory_shards(shard_id),
                sample_key TEXT NOT NULL,
                frame_hash TEXT NOT NULL,
                action_json TEXT NOT NULL,
                action_level TEXT NOT NULL,
                skill_run_id TEXT,
                skill_id TEXT,
                goal_id TEXT,
                plan_node_id TEXT,
                correction_of_step INTEGER,
                PRIMARY KEY(trajectory_id, step_index)
            );
            CREATE INDEX IF NOT EXISTS idx_trajectory_steps_capture
                ON trajectory_steps_index(trajectory_id, captured_ns);
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY,
                trajectory_id TEXT NOT NULL REFERENCES trajectories(trajectory_id),
                started_ns INTEGER NOT NULL,
                ended_ns INTEGER,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                trajectory_id TEXT REFERENCES trajectories(trajectory_id),
                step_index INTEGER,
                observed_ns INTEGER NOT NULL,
                kind TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            """
        )

    def _migrate_v4_to_v5(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS operator_targets (
                target_id TEXT PRIMARY KEY,
                created_ns INTEGER NOT NULL,
                active INTEGER NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_operator_targets_active_created
                ON operator_targets(active, created_ns DESC);
            """
        )

    def _migrate_v5_to_v6(self) -> None:
        columns = {
            str(row[1])
            for row in self.connection.execute("PRAGMA table_info(trajectory_steps_index)")
        }
        if "accepted_ns" not in columns:
            self.connection.execute(
                "ALTER TABLE trajectory_steps_index ADD COLUMN accepted_ns INTEGER"
            )
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS benchmark_runs (
                benchmark_run_id TEXT PRIMARY KEY,
                created_ns INTEGER NOT NULL,
                suite_id TEXT NOT NULL,
                git_commit TEXT,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS benchmark_task_results (
                benchmark_run_id TEXT NOT NULL REFERENCES benchmark_runs(benchmark_run_id),
                task_id TEXT NOT NULL,
                repetition INTEGER NOT NULL,
                status TEXT NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY(benchmark_run_id, task_id, repetition)
            );
            CREATE INDEX IF NOT EXISTS idx_benchmark_task_results_status
                ON benchmark_task_results(benchmark_run_id, status);
            """
        )

    def _migrate_v6_to_v7(self) -> None:
        columns = {
            str(row[1])
            for row in self.connection.execute("PRAGMA table_info(trajectory_steps_index)")
        }
        additions = {
            "action_origin": "TEXT NOT NULL DEFAULT 'legacy'",
            "policy_id": "TEXT",
            "model_version": "TEXT",
            "route_id": "TEXT",
            "policy_action_kind": "TEXT",
            "policy_request_id": "TEXT",
            "prediction_id": "TEXT",
            "condition_id": "TEXT",
            "condition_json": "TEXT",
            "behavior_token": "INTEGER",
            "latent_id": "TEXT",
            "target_track_id": "TEXT",
        }
        for name, declaration in additions.items():
            if name not in columns:
                self.connection.execute(
                    f"ALTER TABLE trajectory_steps_index ADD COLUMN {name} {declaration}"
                )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_trajectory_steps_policy_route "
            "ON trajectory_steps_index(policy_id, model_version, route_id)"
        )

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
        self._execute_write(
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

    def save_runtime_event(self, event: RuntimeEvent) -> None:
        """Persist one idempotent append-only runtime observation."""

        self._execute_write(
            """
            INSERT INTO events(event_id, trajectory_id, step_index, observed_ns, kind, payload)
            VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(event_id) DO NOTHING
            """,
            (
                event.event_id,
                event.trajectory_id,
                event.step_index,
                event.observed_ns,
                event.kind.value,
                event.model_dump_json(),
            ),
        )

    def load_runtime_events(
        self,
        *,
        kinds: set[RuntimeEventKind] | None = None,
        limit: int = 100,
    ) -> tuple[RuntimeEvent, ...]:
        if limit < 1 or limit > 1000:
            raise ValueError("runtime event limit must be between 1 and 1000")
        parameters: list[object] = []
        where = ""
        if kinds:
            selected_kinds = sorted(kinds, key=lambda kind: kind.value)
            placeholders = ",".join("?" for _ in selected_kinds)
            where = f" WHERE kind IN ({placeholders})"
            parameters.extend(kind.value for kind in selected_kinds)
        parameters.append(limit)
        rows = self.connection.execute(
            "SELECT payload FROM events"
            + where
            + " ORDER BY observed_ns DESC, event_id DESC LIMIT ?",
            tuple(parameters),
        ).fetchall()
        return tuple(RuntimeEvent.model_validate_json(payload) for (payload,) in rows)

    def save_place(self, record: PlaceRecord) -> None:
        self._execute_write(
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

    def load_places(self) -> SpatialPlaceMemory:
        rows = self.connection.execute("SELECT payload FROM spatial_places").fetchall()
        memory = SpatialPlaceMemory()
        for (payload,) in rows:
            memory.upsert(PlaceRecord.model_validate_json(payload))
        return memory

    def save_skill(self, spec: SkillSpec) -> None:
        self._execute_write(
            """
            INSERT INTO skills(skill_id, version, payload) VALUES(?, ?, ?)
            ON CONFLICT(skill_id) DO UPDATE SET
                version=excluded.version,
                payload=excluded.payload
            """,
            (spec.skill_id, spec.version, spec.model_dump_json()),
        )

    def save_skill_stats(self, skill_id: str, context_key: str, stats: SkillStats) -> None:
        self._execute_write(
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

    def save_operator_message(self, message: OperatorMessage) -> None:
        with self._transaction(write=True):
            self._store_operator_message(message)

    def _store_operator_message(self, message: OperatorMessage) -> None:
        row = self.connection.execute(
            "SELECT payload FROM operator_messages WHERE message_id=?", (message.message_id,),
        ).fetchone()
        current = None if row is None else OperatorMessage.model_validate_json(row[0])
        if current == message:
            return
        if current is None or _message_authority(current) != _message_authority(message):
            self._advance_operator_revision()
        self.connection.execute(
            """
            INSERT INTO operator_messages(message_id, created_ns, status, payload)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(message_id) DO UPDATE SET
                created_ns=excluded.created_ns,
                status=excluded.status,
                payload=excluded.payload
            """,
            (
                message.message_id,
                message.created_ns,
                message.status.value,
                message.model_dump_json(),
            ),
        )

    def save_operator_target(self, target: Track) -> None:
        """Persist one explicit operator grounding target and supersede older ones."""
        with self._transaction(write=True):
            rows = self.connection.execute(
                "SELECT payload FROM operator_targets WHERE active=1",
            ).fetchall()
            if len(rows) == 1 and Track.model_validate_json(rows[0][0]) == target:
                return
            self._advance_operator_revision()
            self.connection.execute("UPDATE operator_targets SET active=0 WHERE active=1")
            self.connection.execute(
                """
                INSERT INTO operator_targets(target_id, created_ns, active, payload)
                VALUES(?, ?, 1, ?)
                ON CONFLICT(target_id) DO UPDATE SET
                    created_ns=excluded.created_ns,
                    active=1,
                    payload=excluded.payload
                """,
                (target.track_id, target.last_seen_ns, target.model_dump_json()),
            )

    def load_operator_target(self) -> Track | None:
        row = self.connection.execute(
            """
            SELECT payload FROM operator_targets
            WHERE active=1
            ORDER BY created_ns DESC
            LIMIT 1
            """
        ).fetchone()
        return None if row is None else Track.model_validate_json(row[0])

    def clear_operator_target(self) -> None:
        with self._transaction(write=True):
            active = self.connection.execute(
                "SELECT 1 FROM operator_targets WHERE active=1 LIMIT 1",
            ).fetchone()
            if active is not None:
                self._advance_operator_revision()
                self.connection.execute("UPDATE operator_targets SET active=0 WHERE active=1")

    def load_operator_messages(
        self,
        *,
        statuses: set[OperatorMessageStatus] | None = None,
        limit: int = 100,
    ) -> tuple[OperatorMessage, ...]:
        if limit < 1 or limit > 1000:
            raise ValueError("operator message limit must be between 1 and 1000")
        parameters: list[object] = []
        where = ""
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            where = f" WHERE status IN ({placeholders})"
            parameters.extend(status.value for status in statuses)
        parameters.append(limit)
        rows = self.connection.execute(
            "SELECT payload FROM operator_messages" + where + " ORDER BY created_ns DESC LIMIT ?",
            tuple(parameters),
        ).fetchall()
        return tuple(OperatorMessage.model_validate_json(payload) for (payload,) in rows)

    def update_operator_message_status(
        self,
        message_id: str,
        status: OperatorMessageStatus,
        *,
        timestamp_ns: int,
        response_text: str | None = None,
    ) -> OperatorMessage:
        with self._transaction(write=True):
            row = self.connection.execute(
                "SELECT payload FROM operator_messages WHERE message_id=?",
                (message_id,),
            ).fetchone()
            if row is None:
                raise KeyError(message_id)
            current = OperatorMessage.model_validate_json(row[0])
            changes: dict[str, object] = {"status": status}
            if status == OperatorMessageStatus.DELIVERED:
                changes["delivered_ns"] = timestamp_ns
            elif status == OperatorMessageStatus.ACKNOWLEDGED:
                changes["acknowledged_ns"] = timestamp_ns
                changes["response_text"] = response_text
            updated = current.model_copy(update=changes)
            self._store_operator_message(updated)
            return updated

    def save_trajectory_manifest(self, manifest: TrajectoryManifest) -> None:
        self._execute_write(
            """
            INSERT INTO trajectories(
                trajectory_id, started_ns, ended_ns, source_type, game_version, payload
            ) VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(trajectory_id) DO UPDATE SET
                ended_ns=excluded.ended_ns,
                payload=excluded.payload
            """,
            (
                manifest.trajectory_id,
                manifest.started_ns,
                manifest.ended_ns,
                manifest.source.source_type.value,
                manifest.game_version,
                manifest.model_dump_json(),
            ),
        )

    def save_trajectory_shard(
        self,
        *,
        shard_id: str,
        trajectory_id: str,
        path: str,
        sha256: str,
        first_step_index: int,
        last_step_index: int,
        step_count: int,
        bytes_count: int,
        commit: bool = True,
    ) -> None:
        self._execute_write(
            """
            INSERT INTO trajectory_shards(
                shard_id, trajectory_id, path, sha256, first_step_index,
                last_step_index, step_count, bytes
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(shard_id) DO UPDATE SET
                path=excluded.path,
                sha256=excluded.sha256,
                first_step_index=excluded.first_step_index,
                last_step_index=excluded.last_step_index,
                step_count=excluded.step_count,
                bytes=excluded.bytes
            """,
            (
                shard_id,
                trajectory_id,
                path,
                sha256,
                first_step_index,
                last_step_index,
                step_count,
                bytes_count,
            ),
            commit=commit,
        )

    def save_trajectory_step_index(
        self,
        *,
        trajectory_id: str,
        step_index: int,
        captured_ns: int,
        accepted_ns: int | None,
        shard_id: str,
        sample_key: str,
        frame_hash: str,
        action_json: str,
        action_level: str,
        action_origin: str,
        policy_id: str | None,
        model_version: str | None,
        route_id: str | None,
        policy_action_kind: str | None,
        policy_request_id: str | None,
        prediction_id: str | None,
        condition_id: str | None,
        condition_json: str | None,
        behavior_token: int | None,
        latent_id: str | None,
        target_track_id: str | None,
        skill_run_id: str | None,
        skill_id: str | None,
        goal_id: str | None,
        plan_node_id: str | None,
        correction_of_step: int | None,
        commit: bool = True,
    ) -> None:
        self._execute_write(
            """
            INSERT INTO trajectory_steps_index(
                trajectory_id, step_index, captured_ns, accepted_ns, shard_id, sample_key,
                frame_hash, action_json, action_level, action_origin, policy_id, model_version,
                route_id, policy_action_kind, policy_request_id, prediction_id, condition_id,
                condition_json, behavior_token, latent_id, target_track_id, skill_run_id,
                skill_id, goal_id, plan_node_id, correction_of_step
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(trajectory_id, step_index) DO UPDATE SET
                captured_ns=excluded.captured_ns,
                accepted_ns=excluded.accepted_ns,
                shard_id=excluded.shard_id,
                sample_key=excluded.sample_key,
                frame_hash=excluded.frame_hash,
                action_json=excluded.action_json,
                action_level=excluded.action_level,
                action_origin=excluded.action_origin,
                policy_id=excluded.policy_id,
                model_version=excluded.model_version,
                route_id=excluded.route_id,
                policy_action_kind=excluded.policy_action_kind,
                policy_request_id=excluded.policy_request_id,
                prediction_id=excluded.prediction_id,
                condition_id=excluded.condition_id,
                condition_json=excluded.condition_json,
                behavior_token=excluded.behavior_token,
                latent_id=excluded.latent_id,
                target_track_id=excluded.target_track_id,
                skill_run_id=excluded.skill_run_id,
                skill_id=excluded.skill_id,
                goal_id=excluded.goal_id,
                plan_node_id=excluded.plan_node_id,
                correction_of_step=excluded.correction_of_step
            """,
            (
                trajectory_id,
                step_index,
                captured_ns,
                accepted_ns,
                shard_id,
                sample_key,
                frame_hash,
                action_json,
                action_level,
                action_origin,
                policy_id,
                model_version,
                route_id,
                policy_action_kind,
                policy_request_id,
                prediction_id,
                condition_id,
                condition_json,
                behavior_token,
                latent_id,
                target_track_id,
                skill_run_id,
                skill_id,
                goal_id,
                plan_node_id,
                correction_of_step,
            ),
            commit=commit,
        )

    def save_benchmark_report(self, report: BenchmarkReport) -> None:
        """Persist an immutable benchmark report and its task-level evidence."""
        payload = report.model_dump_json()
        with (self._transaction(write=True) if self._transaction_depth else self.connection):
            self.connection.execute(
                """
                INSERT INTO benchmark_runs(
                    benchmark_run_id, created_ns, suite_id, git_commit, payload
                ) VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(benchmark_run_id) DO UPDATE SET
                    created_ns=excluded.created_ns,
                    suite_id=excluded.suite_id,
                    git_commit=excluded.git_commit,
                    payload=excluded.payload
                """,
                (
                    report.benchmark_run_id,
                    report.created_ns,
                    report.suite_id,
                    report.git_commit,
                    payload,
                ),
            )
            for result in report.results:
                self.connection.execute(
                    """
                    INSERT INTO benchmark_task_results(
                        benchmark_run_id, task_id, repetition, status, payload
                    ) VALUES(?, ?, ?, ?, ?)
                    ON CONFLICT(benchmark_run_id, task_id, repetition) DO UPDATE SET
                        status=excluded.status,
                        payload=excluded.payload
                    """,
                    (
                        report.benchmark_run_id,
                        result.task_id,
                        result.repetition,
                        result.status.value,
                        result.model_dump_json(),
                    ),
                )

    def load_benchmark_report_payload(self, benchmark_run_id: str) -> dict[str, object]:
        row = self.connection.execute(
            "SELECT payload FROM benchmark_runs WHERE benchmark_run_id=?",
            (benchmark_run_id,),
        ).fetchone()
        if row is None:
            raise KeyError(benchmark_run_id)
        payload = json.loads(str(row[0]))
        if not isinstance(payload, dict):
            raise ValueError(f"invalid benchmark report payload: {benchmark_run_id}")
        return payload

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
        self._execute_write(
            f"""
            INSERT INTO {table}({key_column}, payload) VALUES(?, ?)
            ON CONFLICT({key_column}) DO UPDATE SET payload=excluded.payload
            """,
            (key, serialized),
        )
