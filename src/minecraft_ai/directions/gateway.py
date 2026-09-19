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
        prohibited_actions=("allow_attack", "allow_jump", "allow_use"),
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
            # Receipt atomicity, cancellation and outcome attribution are not qualified.
            available=False,
            readiness_reason=reason or "execution_contract_unqualified",
            server_id=sup.get("motor_target_instance")
            or agent_info.get("instance_id")
            or "bedrock-local",
            player_name="Eidos",
            instance_id=agent_info.get("instance_id"),
            session_id=sup.get("session_id"),
            control_epoch=int(sup.get("release_count") or 1),
            motor_lease_active=motor_active,
            queue_capacity=self.queue_capacity,
            queue_depth=queue_depth,
            per_member_limit=self.per_member_limit,
            supported_instructions=SUPPORTED_INSTRUCTIONS,
        )

    def submit(self, request: DirectionsRequest) -> DirectionsReceipt:
        """Submit a bounded direction with idempotent deduplication and quota checks."""
        spec = _SUPPORTED_MAP.get(request.instruction_id)
        if spec is None:
            raise InvalidInstructionError(f"Unsupported instruction ID: {request.instruction_id!r}")

        fingerprint = _compute_fingerprint(request)
        now_ns = time.time_ns()
        deadline_ns = now_ns + int(request.deadline_s * 1e9)

        # 1. Pre-check readiness outside transaction
        discovery = self.discover()
        if not discovery.available:
            raise ControlUnavailableError(
                f"Agent control unavailable: {discovery.readiness_reason or 'unready'}"
            )
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

            # Check idempotency
            existing = conn.execute(
                "SELECT * FROM paid_directions WHERE request_id = ?",
                (request.request_id,),
            ).fetchone()

            if existing is not None:
                if existing["fingerprint"] != fingerprint:
                    raise ConflictError(
                        f"Request ID {request.request_id!r} already exists "
                        "with a different payload."
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
            if member_pending >= self.per_member_limit:
                raise CapacityExceededError(
                    f"Member active direction limit ({self.per_member_limit}) reached."
                )

            # Dispatch operator instruction
            attempt_id = uuid.uuid4().hex
            message_id = uuid.uuid4().hex
            instruction_text = request.instruction_text or spec.canonical_text

            message = OperatorMessage(
                message_id=message_id,
                created_ns=now_ns,
                author=f"directions:{request.member_id}",
                text=instruction_text,
                kind=OperatorMessageKind.INSTRUCTION,
                priority=1.0,
                status=OperatorMessageStatus.QUEUED,
            )
            sdb.save_operator_message(message)

            # Persist durable receipt
            conn.execute(
                """
                INSERT INTO paid_directions (
                    request_id, member_id, server_id, session_id, epoch,
                    instruction_id, instruction_text, arguments_json, fingerprint,
                    state, message_id, attempt_id, created_ns, deadline_ns
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?)
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
                ),
            )
            conn.commit()

        return self.status(request.request_id, member_id=request.member_id)

    def status(self, request_id: str, member_id: str | None = None) -> DirectionsReceipt:
        """Lookup receipt status, evaluate ongoing progress, and verify terminal outcome."""
        with StateDatabase(self.db_path) as sdb:
            conn = sdb.connection
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM paid_directions WHERE request_id = ?",
                (request_id,),
            ).fetchone()

            if row is None:
                raise NotFoundError(f"Direction request {request_id!r} not found.")
            if member_id is not None and row["member_id"] != member_id:
                raise NotFoundError(f"Direction request {request_id!r} not found for member.")

            state = DirectionsState(row["state"])
            if state in {
                DirectionsState.SUCCEEDED,
                DirectionsState.FAILED,
                DirectionsState.CANCELLED,
                DirectionsState.EXPIRED,
            }:
                return self._row_to_receipt(row)

            now_ns = time.time_ns()
            # Check deadline expiration
            if now_ns > row["deadline_ns"]:
                self._update_state_conn(
                    conn,
                    request_id,
                    DirectionsState.EXPIRED,
                    reason_code="deadline_exceeded",
                    ended_ns=now_ns,
                )
                conn.commit()
                row = conn.execute(
                    "SELECT * FROM paid_directions WHERE request_id = ?", (request_id,)
                ).fetchone()
                return self._row_to_receipt(row)

            # Check supervisor authority continuity
            sup = self._query_supervisor()
            if (
                sup.get("session_id") != row["session_id"]
                or sup.get("emergency_stop_latched")
                or sup.get("operator_pause_latched")
            ):
                self._update_state_conn(
                    conn,
                    request_id,
                    DirectionsState.CANCELLED,
                    reason_code="control_interrupted",
                    ended_ns=now_ns,
                )
                conn.commit()
                row = conn.execute(
                    "SELECT * FROM paid_directions WHERE request_id = ?", (request_id,)
                ).fetchone()
                return self._row_to_receipt(row)

            msgs = sdb.load_operator_messages(limit=50)
            msg = next((m for m in msgs if m.message_id == row["message_id"]), None)
            events = sdb.load_runtime_events(limit=100)

            spec = _SUPPORTED_MAP.get(row["instruction_id"])
            if spec is None:
                self._update_state_conn(
                    conn,
                    request_id,
                    DirectionsState.FAILED,
                    reason_code="unknown_spec",
                    ended_ns=now_ns,
                )
                conn.commit()
                row = conn.execute(
                    "SELECT * FROM paid_directions WHERE request_id = ?", (request_id,)
                ).fetchone()
                return self._row_to_receipt(row)

            # Check if message has been consumed/delivered
            if state == DirectionsState.QUEUED:
                if msg is not None and msg.status in {
                    OperatorMessageStatus.DELIVERED,
                    OperatorMessageStatus.ACKNOWLEDGED,
                }:
                    self._update_state_conn(
                        conn,
                        request_id,
                        DirectionsState.RUNNING,
                        started_ns=msg.delivered_ns or now_ns,
                    )
                    conn.commit()
                    row = conn.execute(
                        "SELECT * FROM paid_directions WHERE request_id = ?", (request_id,)
                    ).fetchone()
                    state = DirectionsState.RUNNING

            # Evaluate outcome proof from verified runtime events
            outcome = self._evaluate_outcome(row, spec, events)
            if outcome is not None:
                if outcome.success:
                    self._update_state_conn(
                        conn,
                        request_id,
                        DirectionsState.SUCCEEDED,
                        ended_ns=outcome.ended_ns,
                        outcome=outcome,
                    )
                else:
                    self._update_state_conn(
                        conn,
                        request_id,
                        DirectionsState.FAILED,
                        reason_code="verification_failed",
                        ended_ns=outcome.ended_ns,
                        outcome=outcome,
                    )
                conn.commit()
                row = conn.execute(
                    "SELECT * FROM paid_directions WHERE request_id = ?", (request_id,)
                ).fetchone()
                return self._row_to_receipt(row)

            return self._row_to_receipt(row)

    def cancel(
        self, request_id: str, member_id: str | None = None, reason: str = "cancelled_by_user"
    ) -> DirectionsReceipt:
        """Cancel an in-flight or queued direction."""
        receipt = self.status(request_id, member_id=member_id)
        if receipt.state in {
            DirectionsState.SUCCEEDED,
            DirectionsState.FAILED,
            DirectionsState.CANCELLED,
            DirectionsState.EXPIRED,
        }:
            return receipt

        now_ns = time.time_ns()
        with StateDatabase(self.db_path) as sdb:
            self._update_state_conn(
                sdb.connection,
                request_id,
                DirectionsState.CANCELLED,
                reason_code=reason,
                ended_ns=now_ns,
            )
            sdb.connection.commit()
        return self.status(request_id, member_id=member_id)

    def _evaluate_outcome(
        self,
        row: sqlite3.Row,
        spec: SupportedInstruction,
        events: tuple[Any, ...],
    ) -> DirectionsOutcome | None:
        """Evaluate legacy skill labels; this is not qualified paid execution proof."""
        message_id = row["message_id"]
        attempt_id = row["attempt_id"]
        expected_context = f"operator:{message_id}"
        created_ns = row["created_ns"]

        # Filter relevant events observed after creation
        relevant = [
            e
            for e in events
            if (e.observed_ns or 0) >= created_ns
            and (
                e.payload.get("context_key") == expected_context
                or e.payload.get("context_key") == "scene-recovery"
            )
        ]

        # Check for inventory open -> close sequence
        if spec.instruction_id == "open_observe_close_inventory":
            open_evt = next(
                (
                    e
                    for e in relevant
                    if e.kind.value == "skill_succeeded"
                    and e.payload.get("skill_id") == "open_inventory"
                ),
                None,
            )

            close_evt = next(
                (
                    e
                    for e in relevant
                    if e.kind.value == "skill_succeeded"
                    and e.payload.get("skill_id") == "close_open_inventory"
                    and (open_evt is not None and e.observed_ns >= open_evt.observed_ns)
                ),
                None,
            )

            if open_evt is not None and close_evt is not None:
                start_ns = open_evt.observed_ns
                end_ns = close_evt.observed_ns
                duration_ms = float(close_evt.payload.get("duration_ms", 0.0)) + float(
                    open_evt.payload.get("duration_ms", 0.0)
                )

                return DirectionsOutcome(
                    success=True,
                    instruction_id=spec.instruction_id,
                    attempt_id=attempt_id,
                    message_id=message_id,
                    observed_steps=(
                        {
                            "step": "open_inventory",
                            "run_id": open_evt.payload.get("run_id"),
                            "outcome": "succeeded",
                        },
                        {
                            "step": "close_open_inventory",
                            "run_id": close_evt.payload.get("run_id"),
                            "outcome": "succeeded",
                        },
                    ),
                    forbidden_actions_detected=(),
                    started_ns=start_ns,
                    ended_ns=end_ns,
                    duration_ms=duration_ms,
                    evidence_receipt={
                        "open_event_id": open_evt.event_id,
                        "close_event_id": close_evt.event_id,
                        "trajectory_id": close_evt.trajectory_id,
                    },
                )

        elif spec.instruction_id in {"explore_forward", "observe_inventory"}:
            req_skill = spec.required_skills[0]
            success_evt = next(
                (
                    e
                    for e in relevant
                    if e.kind.value == "skill_succeeded" and e.payload.get("skill_id") == req_skill
                ),
                None,
            )

            if success_evt is not None:
                return DirectionsOutcome(
                    success=True,
                    instruction_id=spec.instruction_id,
                    attempt_id=attempt_id,
                    message_id=message_id,
                    observed_steps=(
                        {
                            "step": req_skill,
                            "run_id": success_evt.payload.get("run_id"),
                            "outcome": "succeeded",
                        },
                    ),
                    forbidden_actions_detected=(),
                    started_ns=success_evt.observed_ns,
                    ended_ns=success_evt.observed_ns,
                    duration_ms=float(success_evt.payload.get("duration_ms", 0.0)),
                    evidence_receipt={
                        "event_id": success_evt.event_id,
                        "trajectory_id": success_evt.trajectory_id,
                    },
                )

        return None

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
        )
