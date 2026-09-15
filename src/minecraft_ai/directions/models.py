"""Data models and schemas for bounded paid player directions.

Implements the contract defined in PAID_DIRECTIONS_HANDOFF_20260908.md for
versioned, deterministic instruction dispatch and cryptographic execution receipts.
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class DirectionsState(str, Enum):
    """Lifecycle states of a submitted paid direction."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    UNKNOWN = "unknown"


class SupportedInstruction(BaseModel):
    """Specification of a pre-verified bounded direction the agent can execute."""

    model_config = ConfigDict(frozen=True)

    instruction_id: str = Field(description="Unique stable identifier for the instruction.")
    name: str = Field(description="Human-readable title of the instruction.")
    description: str = Field(description="Detailed explanation of execution bounds and checks.")
    canonical_text: str = Field(description="The exact text dispatched into operator cognition.")
    prohibited_actions: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Explicit actuator masks enforced during execution (e.g. allow_attack=False).",
    )
    timeout_s: float = Field(
        default=30.0, description="Maximum wallclock execution timeout in seconds."
    )
    required_skills: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Ordered or set of skill IDs expected to be observed for terminal success.",
    )
    required_observations: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Observation keys required to be truthy during execution "
            "(e.g. scene.inventory_overlay)."
        ),
    )


class DirectionsDiscovery(BaseModel):
    """Readiness, capacity, and supported catalog discovery payload."""

    model_config = ConfigDict(frozen=True)

    protocol_version: str = Field(
        default="neuroforge.directions.v1", description="Stable protocol identifier."
    )
    available: bool = Field(description="Whether the service can currently accept paid work.")
    readiness_reason: str | None = Field(
        default=None,
        description="Safe explanation if unavailable (e.g. 'operator_paused', 'no_motor_lease').",
    )
    server_id: str = Field(description="Target Bedrock server or instance identifier.")
    player_name: str = Field(description="Inhabited player identity in the Bedrock world.")
    instance_id: str | None = Field(
        default=None, description="Current runtime instance identifier."
    )
    session_id: str | None = Field(
        default=None, description="Current supervisor session identifier."
    )
    control_epoch: int = Field(
        default=1, description="Monotonically increasing control authority epoch."
    )
    motor_lease_active: bool = Field(
        description="Whether the agent currently holds an active motor lease."
    )
    queue_capacity: int = Field(
        default=16, description="Total maximum pending directions across all users."
    )
    queue_depth: int = Field(default=0, description="Current number of queued/running directions.")
    per_member_limit: int = Field(
        default=2, description="Maximum active directions allowed per member."
    )
    supported_instructions: tuple[SupportedInstruction, ...] = Field(
        default_factory=tuple,
        description="Catalog of bounded directions supported in the current epoch.",
    )
    validation_limits: dict[str, Any] = Field(
        default_factory=lambda: {
            "max_text_chars": 256,
            "min_deadline_s": 5.0,
            "max_deadline_s": 300.0,
        },
        description="Input boundary constraints.",
    )


class DirectionsRequest(BaseModel):
    """Authenticated caller request to enqueue a bounded player direction."""

    model_config = ConfigDict(frozen=True)

    request_id: str = Field(description="Caller-generated unique idempotent identifier.")
    member_id: str = Field(
        description="Authenticated member/subscriber reference from private business store."
    )
    server_id: str = Field(description="Exact target server identifier.")
    expected_session_id: str = Field(description="Expected supervisor session ID for epoch gating.")
    expected_epoch: int = Field(description="Expected control epoch number.")
    instruction_id: str = Field(description="ID of the supported instruction to execute.")
    instruction_text: str | None = Field(
        default=None,
        description="Optional custom text or override; defaults to canonical_text if omitted.",
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Typed arguments for parameterized instructions."
    )
    deadline_s: float = Field(
        default=60.0,
        ge=5.0,
        le=300.0,
        description="Maximum wallclock duration before automatic expiration.",
    )


class DirectionsOutcome(BaseModel):
    """Terminal proof evidence verifying observed game events and constraints."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(
        description="Whether the requested direction achieved all target criteria."
    )
    instruction_id: str = Field(description="Executed instruction ID.")
    attempt_id: str = Field(description="Runtime assigned attempt ID.")
    message_id: str = Field(description="Dispatched operator message ID.")
    observed_steps: tuple[dict[str, Any], ...] = Field(
        default_factory=tuple,
        description="Chronological sequence of verified skill runs and observations.",
    )
    forbidden_actions_detected: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Any prohibited actuator events observed (must be empty for success).",
    )
    started_ns: int = Field(description="Monotonic or epoch start timestamp in nanoseconds.")
    ended_ns: int = Field(description="Monotonic or epoch end timestamp in nanoseconds.")
    duration_ms: float = Field(description="Total observed elapsed execution time in milliseconds.")
    evidence_receipt: dict[str, Any] = Field(
        default_factory=dict,
        description="Cryptographic and trajectory metadata linking this outcome to local storage.",
    )


class DirectionsReceipt(BaseModel):
    """Durable receipt tracking submission, state transitions, and terminal outcome."""

    model_config = ConfigDict(frozen=True)

    request_id: str = Field(description="Idempotent caller request ID.")
    member_id: str = Field(description="Authenticated member ID.")
    server_id: str = Field(description="Server identifier.")
    session_id: str = Field(description="Supervisor session ID under which this was admitted.")
    epoch: int = Field(description="Control epoch under which this was admitted.")
    instruction_id: str = Field(description="Instruction ID.")
    state: DirectionsState = Field(description="Current lifecycle state.")
    replayed: bool = Field(
        default=False, description="True if returned from an idempotent deduplication cache."
    )
    created_ns: int = Field(description="Creation timestamp in epoch nanoseconds.")
    deadline_ns: int = Field(description="Hard expiration deadline in epoch nanoseconds.")
    started_ns: int | None = Field(default=None, description="Timestamp when execution began.")
    ended_ns: int | None = Field(
        default=None, description="Timestamp when execution reached terminal state."
    )
    reason_code: str | None = Field(
        default=None, description="Failure, cancellation, or expiration reason code."
    )
    attempt_id: str | None = Field(
        default=None, description="Runtime execution attempt identifier."
    )
    message_id: str | None = Field(
        default=None, description="Underlying operator message identifier."
    )
    outcome: DirectionsOutcome | None = Field(
        default=None, description="Terminal outcome evidence once resolved."
    )
