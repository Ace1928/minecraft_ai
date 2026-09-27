"""Bounded Directions Gateway.

Provides the secure, deterministic, and idempotent service boundary specified in
PAID_DIRECTIONS_HANDOFF_20260908.md for subscribing paid directions to the live Bedrock agent.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from minecraft_ai.social import OperatorMessage, OperatorMessageKind, OperatorMessageStatus
from minecraft_ai.storage import StateDatabase
from .models import (
    DirectionsDiscovery,
    DirectionsOutcome,
    DirectionsReceipt,
    DirectionsRequest,
    DirectionsState,
    SupportedInstruction,
)


class DirectionsError(Exception):
    """Base error for all directions operations."""


class ConflictError(DirectionsError):
    """Raised when an existing request ID is reused with a conflicting payload."""


class ControlUnavailableError(DirectionsError):
    """Raised when runtime or supervisor control is unavailable, paused, or epoch changed."""


class CapacityExceededError(DirectionsError):
    """Raised when global queue or per-member concurrent limit is exceeded."""


class InvalidInstructionError(DirectionsError):
    """Raised when an unsupported or malformed instruction is submitted."""


class NotFoundError(DirectionsError):
    """Raised when a requested direction receipt cannot be found."""


# Catalog of deterministic, verifiable instructions admitted by the gateway.
SUPPORTED_INSTRUCTIONS: tuple[SupportedInstruction, ...] = (
    SupportedInstruction(
        instruction_id="open_observe_close_inventory",
        name="Open, Observe, and Close Inventory",
        description=(
            "Atomically opens the Bedrock inventory, verifies the inventory overlay detector, "
            "and safely closes the menu to restore playable world view "
            "without any movement or attack."
        ),
        canonical_text="open inventory, observe it, close it; no movement or attack",
        prohibited_actions=(
            "allow_movement",
            "allow_attack",
            "allow_jump",
            "allow_use",
            "allow_drop",
            "allow_hotbar",
        ),
        timeout_s=25.0,
        required_skills=("open_inventory", "close_open_inventory"),
        required_observations=("scene.inventory_overlay", "scene.playable"),
    ),
    SupportedInstruction(
        instruction_id="observe_inventory",
        name="Observe Inventory Contents",
        description=(
            "Opens the inventory overlay to inspect current slots and items "
            "without attack or movement."
        ),
        canonical_text="open inventory once, observe it; no movement or attack",
        prohibited_actions=("allow_attack", "allow_jump", "allow_use"),
        timeout_s=20.0,
        required_skills=("open_inventory",),
        required_observations=("scene.inventory_overlay",),
    ),
    SupportedInstruction(
        instruction_id="explore_forward",
        name="Explore Level Ground Forward",
        description="Explores visible level ground ahead for a bounded duration without attacking.",
        canonical_text="explore forward across visible open ground; no attack or interact",
        prohibited_actions=("allow_attack", "allow_use"),
        timeout_s=30.0,
        required_skills=("explore_forward",),
        required_observations=(),
    ),
)

_SUPPORTED_MAP = {spec.instruction_id: spec for spec in SUPPORTED_INSTRUCTIONS}

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS paid_directions (
    request_id TEXT PRIMARY KEY,
    member_id TEXT NOT NULL,
    server_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    epoch INTEGER NOT NULL,
    instruction_id TEXT NOT NULL,
    instruction_text TEXT NOT NULL,
    arguments_json TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN (
        'queued', 'running', 'succeeded', 'failed', 'cancelled', 'expired', 'unknown'
    )),
    message_id TEXT,
    attempt_id TEXT,
    created_ns INTEGER NOT NULL,
    deadline_ns INTEGER NOT NULL,
    started_ns INTEGER,
    ended_ns INTEGER,
    reason_code TEXT,
    outcome_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_paid_directions_pending ON paid_directions(state, created_ns);
CREATE INDEX IF NOT EXISTS idx_paid_directions_member ON paid_directions(member_id, state);
CREATE TABLE IF NOT EXISTS paid_direction_steps (
    request_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    skill_id TEXT NOT NULL,
    run_id TEXT NOT NULL UNIQUE,
    action_sequence INTEGER,
    action_monotonic_ns INTEGER,
    action_frame_id INTEGER,
    action_json TEXT,
    verified_monotonic_ns INTEGER,
    evidence_json TEXT,
    PRIMARY KEY(request_id, attempt_id, skill_id)
);
"""


def _compute_fingerprint(req: DirectionsRequest) -> str:
    """Deterministic hash of the invariant submitted payload."""
    payload = {
        "member_id": req.member_id,
        "server_id": req.server_id,
        "expected_session_id": req.expected_session_id,
        "expected_epoch": req.expected_epoch,
        "instruction_id": req.instruction_id,
        "instruction_text": req.instruction_text,
        "arguments": req.arguments,
        "deadline_s": req.deadline_s,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _control_epoch(sup):
    # release_count counts ordinary input releases, not authority generations.
    # The exact lease is also retained and compared, so this JS-safe integer
    # is only a public generation reference, never the sole authority check.
    lease = sup.get("motor_lease_id")
    if isinstance(lease, str) and lease:
        identity = f"{sup.get('session_id')}:{lease}".encode()
        return int.from_bytes(hashlib.sha256(identity).digest()[:6], "big") + 1
    return int(sup.get("release_count") or 1)


class DirectionsGateway:
    """Manages discovery, submission, idempotency, and outcome verification."""

    def __init__(
        self,
        state_db_path: str | Path,
        *,
        supervisor_status_fn: Callable[[], dict[str, Any]] | None = None,
        queue_capacity: int = 16,
        per_member_limit: int = 2,
    ) -> None:
        self.db_path = Path(state_db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.supervisor_status_fn = supervisor_status_fn
        self.queue_capacity = queue_capacity
        self.per_member_limit = per_member_limit
        self._init_db()

    def _init_db(self) -> None:
        with StateDatabase(self.db_path) as sdb:
            sdb.connection.executescript(_SCHEMA_SQL)
            columns = {
                row[1] for row in sdb.connection.execute("PRAGMA table_info(paid_directions)")
            }
            for name, declaration in (
                ("authority_revision", "INTEGER"),
                ("execution_mode", "TEXT NOT NULL DEFAULT 'legacy'"),
                ("control_lease_id", "TEXT"),
                ("accepted_action_count", "INTEGER NOT NULL DEFAULT 0"),
            ):
                if name not in columns:
                    sdb.connection.execute(
                        f"ALTER TABLE paid_directions ADD COLUMN {name} {declaration}"
                    )
            sdb.connection.commit()

    def _query_supervisor(self) -> dict[str, Any]:
        if self.supervisor_status_fn is not None:
            return self.supervisor_status_fn()
        try:
            from minecraft_ai.supervisor import send_command

            return send_command("status", timeout_s=1.0)
        except Exception as exc:
            return {"available": False, "error": str(exc), "state": "UNKNOWN"}

    def discover(self) -> DirectionsDiscovery:
        """Query service availability, supervisor epoch, and catalog limits."""
        sup = self._query_supervisor()
        state = sup.get("state", "UNKNOWN")
        paused = state in {"PAUSED", "SAFE_IDLE"} or sup.get("operator_pause_latched", False)
        motor_active = bool(sup.get("motor_lease_active", False))
        emergency = bool(sup.get("emergency_stop_latched", False))
        live_capable = bool(sup.get("live_capable", False))

        available = (
            (state == "RUNNING")
            and motor_active
            and (not paused)
            and (not emergency)
            and live_capable
        )
        reason = None
        if not available:
            if emergency:
                reason = "emergency_stop_latched"
            elif paused:
                reason = "operator_paused"
            elif not motor_active:
                reason = "motor_lease_inactive"
            elif not live_capable:
                reason = "live_incapable"
            else:
                reason = f"supervisor_state_{state.lower()}"

        with StateDatabase(self.db_path) as sdb:
            conn = sdb.connection
            conn.row_factory = sqlite3.Row
            depth_row = conn.execute(
                "SELECT COUNT(*) AS count FROM paid_directions WHERE state IN ('queued', 'running')"
            ).fetchone()
            queue_depth = depth_row["count"] if depth_row else 0

        agent_info = sup.get("agent") or {}
        return DirectionsDiscovery(
            # Public/commercial admission awaits live qualification and private billing binding.
            available=False,
            readiness_reason=reason or "execution_contract_unqualified",
            server_id=sup.get("motor_target_instance")
            or agent_info.get("instance_id")
            or "bedrock-local",
            player_name="Eidos",
            instance_id=agent_info.get("instance_id"),
            session_id=sup.get("session_id"),
            control_epoch=_control_epoch(sup),
            motor_lease_active=motor_active,
            queue_capacity=self.queue_capacity,
            queue_depth=queue_depth,
            per_member_limit=self.per_member_limit,
            supported_instructions=SUPPORTED_INSTRUCTIONS,
        )

    def submit(self, request: DirectionsRequest) -> DirectionsReceipt:
        """Submit a bounded direction with idempotent deduplication and quota checks."""
        return self._submit(request, qualification=False)

    def submit_qualification(self, request: DirectionsRequest) -> DirectionsReceipt:
        """Private operator trial of the single inventory transaction; paid access stays held."""
        if request.instruction_id != "open_observe_close_inventory":
            raise InvalidInstructionError(
                "Only the inventory qualification transaction is admitted."
            )
        return self._submit(request, qualification=True)

    def _submit(self, request: DirectionsRequest, *, qualification: bool) -> DirectionsReceipt:
        spec = _SUPPORTED_MAP.get(request.instruction_id)
        if spec is None:
            raise InvalidInstructionError(f"Unsupported instruction ID: {request.instruction_id!r}")

        fingerprint = _compute_fingerprint(request)
        now_ns = time.time_ns()
        deadline_ns = now_ns + int(min(request.deadline_s, spec.timeout_s) * 1e9)

        # A lost response reconciles even after pause or a terminal outcome.
        with StateDatabase(self.db_path) as sdb:
            sdb.connection.row_factory = sqlite3.Row
            existing = sdb.connection.execute(
                "SELECT * FROM paid_directions WHERE request_id=?", (request.request_id,)
            ).fetchone()
            if existing is not None:
                if existing["fingerprint"] != fingerprint:
                    raise ConflictError("Request identity already binds a different payload.")
                return self._row_to_receipt(existing).model_copy(update={"replayed": True})

        # 1. Pre-check readiness outside transaction
        discovery = self.discover()
        if not discovery.available and not (
            qualification and discovery.readiness_reason == "execution_contract_unqualified"
        ):
            raise ControlUnavailableError(
                f"Agent control unavailable: {discovery.readiness_reason or 'unready'}"
            )
        if request.instruction_text not in (None, spec.canonical_text) or request.arguments:
            raise InvalidInstructionError(
                "Catalogue instructions are immutable and accept no arguments."
            )
        if request.server_id != discovery.server_id:
            raise ControlUnavailableError("Server mismatch: exact live target required.")
        if discovery.session_id != request.expected_session_id:
            raise ControlUnavailableError(
                f"Session mismatch: requested {request.expected_session_id!r}, "
                f"live is {discovery.session_id!r}"
            )
        if discovery.control_epoch != request.expected_epoch:
            raise ControlUnavailableError(
                f"Control epoch mismatch: requested {request.expected_epoch}, "
                f"live is {discovery.control_epoch}"
            )

        with StateDatabase(self.db_path) as sdb:
            conn = sdb.connection
            conn.row_factory = sqlite3.Row
            # Receipt, authority revision and operator queue are one transaction.
            # Nested StateDatabase saves join this boundary and cannot publish early.
            with sdb._transaction(write=True):
                return self._admit_conn(
                    sdb,
                    request,
                    spec,
                    fingerprint,
                    now_ns,
                    deadline_ns,
                    qualification=qualification,
                )

    def _admit_conn(self, sdb, request, spec, fingerprint, now_ns, deadline_ns, *, qualification):
        conn = sdb.connection
        # Check idempotency
        existing = conn.execute(
            "SELECT * FROM paid_directions WHERE request_id = ?",
            (request.request_id,),
        ).fetchone()

        if existing is not None:
            if existing["fingerprint"] != fingerprint:
                raise ConflictError(
                    f"Request ID {request.request_id!r} already exists with a different payload."
                )
            receipt = self._row_to_receipt(existing)
            return receipt.model_copy(update={"replayed": True})

        # Check capacities
        counts = conn.execute(
            """
                SELECT
                    COUNT(*) AS total_pending,
                    SUM(CASE WHEN member_id = ? THEN 1 ELSE 0 END) AS member_pending
                FROM paid_directions
                WHERE state IN ('queued', 'running')
                """,
            (request.member_id,),
        ).fetchone()
        total_pending = counts["total_pending"] or 0
        member_pending = counts["member_pending"] or 0

        if total_pending >= self.queue_capacity:
            raise CapacityExceededError("Directions global queue capacity reached.")
        if qualification and total_pending:
            raise CapacityExceededError("One inventory qualification may own the player at a time.")
        if member_pending >= self.per_member_limit:
            raise CapacityExceededError(
                f"Member active direction limit ({self.per_member_limit}) reached."
            )

        # Dispatch operator instruction
        attempt_id = uuid.uuid4().hex
        message_id = uuid.uuid4().hex
        instruction_text = spec.canonical_text

        message = OperatorMessage(
            message_id=message_id,
            created_ns=now_ns,
            author=f"directions:{request.member_id}",
            text=instruction_text,
            kind=OperatorMessageKind.INSTRUCTION,
            priority=1.0,
            status=OperatorMessageStatus.QUEUED,
            direction_request_id=request.request_id,
            direction_attempt_id=attempt_id,
        )
        sdb.save_operator_message(message)

        # Persist durable receipt
        conn.execute(
            """
                INSERT INTO paid_directions (
                    request_id, member_id, server_id, session_id, epoch,
                    instruction_id, instruction_text, arguments_json, fingerprint,
                    state, message_id, attempt_id, created_ns, deadline_ns,
                    authority_revision, execution_mode, control_lease_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?)
                """,
            (
                request.request_id,
                request.member_id,
                request.server_id,
                request.expected_session_id,
                request.expected_epoch,
                request.instruction_id,
                instruction_text,
                json.dumps(request.arguments),
                fingerprint,
                message_id,
                attempt_id,
                now_ns,
                deadline_ns,
                sdb.operator_revision(),
                "qualification" if qualification else "unqualified",
                self._query_supervisor().get("motor_lease_id"),
            ),
        )
        row = conn.execute(
            "SELECT * FROM paid_directions WHERE request_id=?", (request.request_id,)
        ).fetchone()
        return self._row_to_receipt(row)

    @staticmethod
    def _row(sdb, request_id, member_id=None):
        sdb.connection.row_factory = sqlite3.Row
        row = sdb.connection.execute(
            "SELECT * FROM paid_directions WHERE request_id=?", (request_id,)
        ).fetchone()
        if row is None or (member_id is not None and row["member_id"] != member_id):
            raise NotFoundError("Direction request not found for member.")
        return row

    @staticmethod
    def _terminal(row):
        return row["state"] not in {"queued", "running"}

    def _revoke_conn(self, sdb, row, state, reason):
        if self._terminal(row):
            return
        self._update_state_conn(
            sdb.connection, row["request_id"], state, reason_code=reason, ended_ns=time.time_ns()
        )
        sdb.update_operator_message_status(
            row["message_id"], OperatorMessageStatus.ARCHIVED, timestamp_ns=time.time_ns()
        )

    def _continuity_reason(self, row, sup):
        if time.time_ns() > row["deadline_ns"]:
            return "deadline_exceeded"
        server = (
            sup.get("motor_target_instance")
            or (sup.get("agent") or {}).get("instance_id")
            or "bedrock-local"
        )
        if (
            sup.get("session_id") != row["session_id"]
            or _control_epoch(sup) != row["epoch"]
            or sup.get("motor_lease_id") != row["control_lease_id"]
            or server != row["server_id"]
            or sup.get("state") != "RUNNING"
            or not sup.get("motor_lease_active")
            or not sup.get("live_capable")
            or sup.get("emergency_stop_latched")
            or sup.get("operator_pause_latched")
        ):
            return "control_interrupted"
        return None

    def status(self, request_id: str, member_id: str | None = None) -> DirectionsReceipt:
        """Read the exact durable receipt; generic skill labels never establish completion."""
        with StateDatabase(self.db_path) as sdb, sdb._transaction(write=True):
            row = self._row(sdb, request_id, member_id)
            if not self._terminal(row):
                reason = self._continuity_reason(row, self._query_supervisor())
                if reason:
                    state = (
                        DirectionsState.EXPIRED
                        if reason == "deadline_exceeded"
                        else DirectionsState.CANCELLED
                    )
                    self._revoke_conn(sdb, row, state, reason)
                    row = self._row(sdb, request_id)
            return self._row_to_receipt(row)

    def cancel(
        self, request_id: str, member_id: str | None = None, reason: str = "cancelled_by_user"
    ) -> DirectionsReceipt:
        """Revoke receipt and queued authority under the motor dispatch writer lock.

        An already accepted bounded E pulse may finish. The returned receipt
        reports its count; no later owned input can cross the same boundary.
        """
        with StateDatabase(self.db_path) as sdb, sdb._transaction(write=True):
            row = self._row(sdb, request_id, member_id)
            self._revoke_conn(sdb, row, DirectionsState.CANCELLED, reason)
            return self._row_to_receipt(self._row(sdb, request_id))

    def _evaluate_outcome(self, row, spec, events):
        # Retained compatibility hook: unbound or scene-recovery events cannot
        # turn a legacy receipt into a paid success. finish_step owns evidence.
        return None

    def next_pending(self):
        with StateDatabase(self.db_path) as sdb:
            sdb.connection.row_factory = sqlite3.Row
            row = sdb.connection.execute(
                "SELECT * FROM paid_directions WHERE state IN ('queued','running') "
                "ORDER BY CASE state WHEN 'running' THEN 0 ELSE 1 END, created_ns LIMIT 1"
            ).fetchone()
            return None if row is None else dict(row)

    def claim(self, request_id):
        with StateDatabase(self.db_path) as sdb, sdb._transaction(write=True):
            row = self._row(sdb, request_id)
            reason = self._continuity_reason(row, self._query_supervisor())
            if self._terminal(row):
                return None
            if row["state"] == "running":
                self._revoke_conn(sdb, row, DirectionsState.UNKNOWN, "restart_execution_uncertain")
                return None
            if row["execution_mode"] != "qualification":
                reason = "execution_contract_unqualified"
            elif sdb.operator_revision() != row["authority_revision"]:
                reason = "operator_authority_changed"
            if reason:
                self._revoke_conn(sdb, row, DirectionsState.CANCELLED, reason)
                return None
            sdb.update_operator_message_status(
                row["message_id"], OperatorMessageStatus.DELIVERED, timestamp_ns=time.time_ns()
            )
            self._update_state_conn(
                sdb.connection, request_id, DirectionsState.RUNNING, started_ns=time.time_ns()
            )
            return dict(self._row(sdb, request_id))

    def start_step(self, request_id, attempt_id, skill_id, run_id):
        with StateDatabase(self.db_path) as sdb, sdb._transaction(write=True):
            row = self._authorize_conn(sdb, request_id, attempt_id)
            steps = sdb.connection.execute(
                "SELECT skill_id, verified_monotonic_ns FROM paid_direction_steps "
                "WHERE request_id=? AND attempt_id=?",
                (request_id, attempt_id),
            ).fetchall()
            expected = "open_inventory" if not steps else "close_open_inventory"
            if skill_id != expected or (steps and (len(steps) != 1 or steps[0][1] is None)):
                raise ControlUnavailableError("Inventory step order is not verified.")
            sdb.connection.execute(
                "INSERT INTO paid_direction_steps(request_id,attempt_id,skill_id,run_id) "
                "VALUES(?,?,?,?)",
                (request_id, attempt_id, skill_id, run_id),
            )
            return {
                "direction_request_id": request_id,
                "direction_attempt_id": attempt_id,
                **{key: False for key in _SUPPORTED_MAP[row["instruction_id"]].prohibited_actions},
            }

    def _authorize_conn(self, sdb, request_id, attempt_id):
        row = self._row(sdb, request_id)
        if (
            row["attempt_id"] != attempt_id
            or row["state"] != "running"
            or row["instruction_id"] != "open_observe_close_inventory"
            or row["execution_mode"] != "qualification"
            or sdb.operator_revision() != row["authority_revision"]
            or self._continuity_reason(row, self._query_supervisor())
        ):
            raise ControlUnavailableError("Direction authority was revoked or expired.")
        return row

    @contextmanager
    def motor_authority(self, run, action, capture):
        """Serialize revocation with actual dispatch, enforcing the catalogue at the wire."""
        request_id = run.parameters["direction_request_id"]
        attempt_id = run.parameters["direction_attempt_id"]
        with StateDatabase(self.db_path) as sdb, sdb._transaction(write=True):
            self._authorize_conn(sdb, request_id, attempt_id)
            step = sdb.connection.execute(
                "SELECT * FROM paid_direction_steps "
                "WHERE request_id=? AND attempt_id=? AND run_id=?",
                (request_id, attempt_id, run.run_id),
            ).fetchone()
            if step is None or step["skill_id"] != run.skill_id:
                raise ControlUnavailableError("Action has no matching direction step.")
            if (
                action.keys_down not in ((), ("e",))
                or action.buttons_down
                or action.mouse_dx
                or action.mouse_dy
                or action.cursor_x is not None
                or action.cursor_y is not None
                or (
                    action.keys_down
                    and (not 0 < action.duration_ms <= 150 or "e" not in action.keys_up)
                )
            ):
                raise InvalidInstructionError("Direction action exceeds the inventory-only mask.")
            if action.keys_down and step["action_sequence"] is not None:
                raise ControlUnavailableError("Inventory toggle already dispatched for this step.")
            if action.keys_down and (
                capture is None or not 0 <= time.monotonic_ns() - capture.captured_ns <= 500_000_000
            ):
                raise ControlUnavailableError("Direction action lacks a captured source frame.")
            stamp = time.monotonic_ns()

            def accepted(response):
                accepted_ns = (
                    response.get("accepted_monotonic_ns") if isinstance(response, dict) else None
                )
                if (
                    not isinstance(response, dict)
                    or type(response.get("accepted_sequence")) is not int
                    or response["accepted_sequence"] != action.sequence
                    or type(accepted_ns) is not int
                    or not stamp <= accepted_ns <= time.monotonic_ns()
                    or response.get("lease_active") is not True
                ):
                    raise ControlUnavailableError("Supervisor action acceptance is unverified.")
                if not action.keys_down:
                    return
                if capture is None:
                    raise ControlUnavailableError("Direction action lacks a captured source frame.")
                evidence = {
                    "request_id": request_id,
                    "attempt_id": attempt_id,
                    "run_id": run.run_id,
                    "action": action.model_dump(mode="json"),
                    "supervisor_acceptance": {
                        "accepted_sequence": response["accepted_sequence"],
                        "accepted_monotonic_ns": accepted_ns,
                        "lease_active": True,
                    },
                }
                sdb.connection.execute(
                    "UPDATE paid_direction_steps SET action_sequence=?,action_monotonic_ns=?,"
                    "action_frame_id=?,action_json=? WHERE run_id=?",
                    (
                        action.sequence, accepted_ns, capture.frame_id,
                        json.dumps(evidence), run.run_id,
                    ),
                )
                sdb.connection.execute(
                    "UPDATE paid_directions SET accepted_action_count=accepted_action_count+1 "
                    "WHERE request_id=?",
                    (request_id,),
                )

            dispatch_error = None
            try:
                yield accepted
            except Exception as error:
                self._revoke_conn(
                    sdb,
                    self._row(sdb, request_id),
                    DirectionsState.UNKNOWN,
                    "motor_dispatch_uncertain",
                )
                dispatch_error = error
        if dispatch_error is not None:
            raise dispatch_error

    def finish_step(self, request_id, attempt_id, run, blackboard, capture):
        from minecraft_ai.perception_service import BEDROCK_HUD_SAFETY_SOURCE

        with StateDatabase(self.db_path) as sdb, sdb._transaction(write=True):
            row = self._authorize_conn(sdb, request_id, attempt_id)
            step = sdb.connection.execute(
                "SELECT * FROM paid_direction_steps WHERE run_id=?", (run.run_id,)
            ).fetchone()
            key = (
                "scene.inventory_overlay" if run.skill_id == "open_inventory" else "scene.playable"
            )
            fact = blackboard.fact(key, min_confidence=0.99)
            if (
                step is None
                or step["request_id"] != request_id
                or step["attempt_id"] != attempt_id
                or step["skill_id"] != run.skill_id
                or run.outcome.value != "succeeded"
                or run.parameters.get("direction_request_id") != request_id
                or run.parameters.get("direction_attempt_id") != attempt_id
                or run.context_key != f"operator:{row['message_id']}"
                or step["action_sequence"] is None
                or fact is None
                or fact.value is not True
                or fact.source != BEDROCK_HUD_SAFETY_SOURCE
                or capture is None
                or fact.observed_ns <= step["action_monotonic_ns"]
                or capture.captured_ns <= step["action_monotonic_ns"]
                or capture.frame_id <= step["action_frame_id"]
            ):
                self._revoke_conn(
                    sdb, row, DirectionsState.FAILED, "attributable_observation_missing"
                )
                return False
            evidence = {
                "request_id": request_id,
                "attempt_id": attempt_id,
                "run_id": run.run_id,
                "skill_id": run.skill_id,
                "action_sequence": step["action_sequence"],
                "action_monotonic_ns": step["action_monotonic_ns"],
                "source_frame_id": capture.frame_id,
                "source_captured_ns": capture.captured_ns,
                "observation": fact.model_dump(mode="json"),
            }
            sdb.connection.execute(
                "UPDATE paid_direction_steps SET verified_monotonic_ns=?,evidence_json=? "
                "WHERE run_id=?",
                (fact.observed_ns, json.dumps(evidence), run.run_id),
            )
            if run.skill_id == "close_open_inventory":
                steps = sdb.connection.execute(
                    "SELECT * FROM paid_direction_steps WHERE request_id=? "
                    "AND attempt_id=? ORDER BY action_monotonic_ns",
                    (request_id, attempt_id),
                ).fetchall()
                if (
                    len(steps) != 2
                    or steps[0]["skill_id"] != "open_inventory"
                    or steps[0]["verified_monotonic_ns"] is None
                    or steps[0]["verified_monotonic_ns"] >= steps[1]["action_monotonic_ns"]
                ):
                    self._revoke_conn(
                        sdb, row, DirectionsState.FAILED, "inventory_order_unverified"
                    )
                    return False
                outcome = DirectionsOutcome(
                    success=True,
                    instruction_id=row["instruction_id"],
                    attempt_id=attempt_id,
                    message_id=row["message_id"],
                    started_ns=row["started_ns"],
                    ended_ns=time.time_ns(),
                    duration_ms=(fact.observed_ns - steps[0]["action_monotonic_ns"]) / 1e6,
                    observed_steps=tuple(json.loads(s["evidence_json"]) for s in steps),
                    evidence_receipt={
                        "request_id": request_id,
                        "attempt_id": attempt_id,
                        "verification_contract": "inventory_two_step_v1",
                        "billing_enabled": False,
                    },
                )
                self._update_state_conn(
                    sdb.connection,
                    request_id,
                    DirectionsState.SUCCEEDED,
                    ended_ns=outcome.ended_ns,
                    outcome=outcome,
                )
                sdb.update_operator_message_status(
                    row["message_id"], OperatorMessageStatus.ARCHIVED, timestamp_ns=time.time_ns()
                )
            return True

    def _update_state_conn(
        self,
        conn: sqlite3.Connection,
        request_id: str,
        state: DirectionsState,
        *,
        reason_code: str | None = None,
        started_ns: int | None = None,
        ended_ns: int | None = None,
        outcome: DirectionsOutcome | None = None,
    ) -> None:
        updates = ["state = ?"]
        params: list[Any] = [state.value]
        if reason_code is not None:
            updates.append("reason_code = ?")
            params.append(reason_code)
        if started_ns is not None:
            updates.append("started_ns = ?")
            params.append(started_ns)
        if ended_ns is not None:
            updates.append("ended_ns = ?")
            params.append(ended_ns)
        if outcome is not None:
            updates.append("outcome_json = ?")
            params.append(outcome.model_dump_json())
        params.append(request_id)
        conn.execute(
            f"UPDATE paid_directions SET {', '.join(updates)} WHERE request_id = ?",
            params,
        )

    def _row_to_receipt(self, row: sqlite3.Row) -> DirectionsReceipt:
        outcome = None
        if row["outcome_json"]:
            outcome = DirectionsOutcome.model_validate_json(row["outcome_json"])
        return DirectionsReceipt(
            request_id=row["request_id"],
            member_id=row["member_id"],
            server_id=row["server_id"],
            session_id=row["session_id"],
            epoch=row["epoch"],
            instruction_id=row["instruction_id"],
            state=DirectionsState(row["state"]),
            created_ns=row["created_ns"],
            deadline_ns=row["deadline_ns"],
            started_ns=row["started_ns"],
            ended_ns=row["ended_ns"],
            reason_code=row["reason_code"],
            attempt_id=row["attempt_id"],
            message_id=row["message_id"],
            outcome=outcome,
            accepted_action_count=row["accepted_action_count"],
        )
