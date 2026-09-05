from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SupervisorState(StrEnum):
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    SAFE_IDLE = "SAFE_IDLE"
    ARMED = "ARMED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    FAILSAFE = "FAILSAFE"


_ALLOWED_TRANSITIONS: dict[SupervisorState, frozenset[SupervisorState]] = {
    SupervisorState.STOPPED: frozenset({SupervisorState.STARTING}),
    SupervisorState.STARTING: frozenset(
        {
            SupervisorState.SAFE_IDLE,
            SupervisorState.FAILSAFE,
            SupervisorState.STOPPING,
        }
    ),
    SupervisorState.SAFE_IDLE: frozenset(
        {
            SupervisorState.ARMED,
            SupervisorState.PAUSED,
            SupervisorState.STOPPING,
            SupervisorState.FAILSAFE,
        }
    ),
    SupervisorState.ARMED: frozenset(
        {
            SupervisorState.RUNNING,
            SupervisorState.PAUSED,
            SupervisorState.STOPPING,
            SupervisorState.FAILSAFE,
        }
    ),
    SupervisorState.RUNNING: frozenset(
        {
            SupervisorState.PAUSED,
            SupervisorState.STOPPING,
            SupervisorState.FAILSAFE,
        }
    ),
    SupervisorState.PAUSED: frozenset(
        {
            SupervisorState.SAFE_IDLE,
            SupervisorState.STOPPING,
            SupervisorState.FAILSAFE,
        }
    ),
    SupervisorState.STOPPING: frozenset({SupervisorState.STOPPED, SupervisorState.FAILSAFE}),
    SupervisorState.FAILSAFE: frozenset({SupervisorState.STOPPED}),
}


class TransitionError(RuntimeError):
    pass


class MotorRejected(RuntimeError):
    pass


MIN_MOTOR_LEASE_TTL_MS = 50
MAX_MOTOR_LEASE_TTL_MS = 5_000


def _validate_motor_lease_ttl(ttl_ms: int) -> None:
    if ttl_ms < MIN_MOTOR_LEASE_TTL_MS or ttl_ms > MAX_MOTOR_LEASE_TTL_MS:
        raise MotorRejected("lease ttl outside safety bounds")


class MotorAction(BaseModel):
    """Bounded human-style input semantics accepted by a motor backend."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int = Field(ge=0)
    keys_down: tuple[str, ...] = ()
    keys_up: tuple[str, ...] = ()
    buttons_down: tuple[str, ...] = ()
    buttons_up: tuple[str, ...] = ()
    mouse_dx: int = Field(default=0, ge=-4096, le=4096)
    mouse_dy: int = Field(default=0, ge=-4096, le=4096)
    cursor_x: float | None = Field(default=None, ge=0.0, le=1.0)
    cursor_y: float | None = Field(default=None, ge=0.0, le=1.0)
    camera_semantics: Literal["world", "cursor"] = "world"
    duration_ms: int = Field(default=0, ge=0, le=1000)

    @field_validator("keys_down", "keys_up", "buttons_down", "buttons_up")
    @classmethod
    def _bounded_tokens(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) > 16:
            raise ValueError("too many simultaneous input tokens")
        normalized = tuple(token.strip().lower() for token in value)
        if any(not token or len(token) > 32 for token in normalized):
            raise ValueError("invalid input token")
        return normalized

    @model_validator(mode="after")
    def _absolute_cursor_is_gui_only(self) -> MotorAction:
        if (self.cursor_x is None) != (self.cursor_y is None):
            raise ValueError("cursor_x and cursor_y must be supplied together")
        if self.cursor_x is None:
            return self
        if self.camera_semantics != "cursor":
            raise ValueError("absolute cursor targets require cursor semantics")
        if self.mouse_dx or self.mouse_dy:
            raise ValueError("absolute and relative pointer motion cannot be combined")
        return self

    def action_kinds(self) -> frozenset[str]:
        kinds: set[str] = set()
        if self.keys_down or self.keys_up:
            kinds.add("keyboard")
        if self.buttons_down or self.buttons_up:
            kinds.add("button")
        if self.mouse_dx or self.mouse_dy or self.cursor_x is not None:
            kinds.add("mouse")
        return frozenset(kinds)


@dataclass(frozen=True)
class MotorLease:
    lease_id: str
    session_id: str
    target_instance: str
    backend_id: str
    expires_monotonic_ns: int
    allowed_actions: frozenset[str]
    max_action_duration_ms: int
    first_sequence: int

    def expired(self, now_ns: int | None = None) -> bool:
        now = time.monotonic_ns() if now_ns is None else now_ns
        return now >= self.expires_monotonic_ns


class InputBackend(Protocol):
    backend_id: str
    live_capable: bool

    def bind_lease(self, lease: MotorLease) -> None: ...

    def apply(self, action: MotorAction) -> None: ...

    def release_all(self) -> None: ...

    def clear_lease(self) -> None: ...


@dataclass
class FakeInputBackend:
    """Deterministic non-live backend used to prove safety behavior first."""

    backend_id: str = "fake"
    live_capable: bool = False
    held_keys: set[str] = field(default_factory=set)
    held_buttons: set[str] = field(default_factory=set)
    actions: list[MotorAction] = field(default_factory=list)
    release_count: int = 0
    lease_id: str | None = None

    def bind_lease(self, lease: MotorLease) -> None:
        self.lease_id = lease.lease_id

    def apply(self, action: MotorAction) -> None:
        if self.lease_id is None:
            raise MotorRejected("backend has no active motor lease")
        self.held_keys.update(action.keys_down)
        self.held_keys.difference_update(action.keys_up)
        self.held_buttons.update(action.buttons_down)
        self.held_buttons.difference_update(action.buttons_up)
        self.actions.append(action)

    def release_all(self) -> None:
        self.held_keys.clear()
        self.held_buttons.clear()
        self.release_count += 1

    def clear_lease(self) -> None:
        self.lease_id = None


class MotorGate:
    """Capability gate between cognition/skills and any concrete input backend.

    A lease is deliberately short lived. Stale, replayed, oversized or disallowed
    actions are rejected before they reach the backend. The same lease is bound
    into the backend so a scoped remote bridge can independently enforce expiry
    and sequence checks. Every revocation releases all input unconditionally.
    """

    def __init__(self, backend: InputBackend) -> None:
        self.backend = backend
        self._lock = threading.RLock()
        self._lease: MotorLease | None = None
        self._last_sequence = -1
        self._revocation_reason = "not-issued"

    @property
    def lease(self) -> MotorLease | None:
        with self._lock:
            return self._lease

    @property
    def revocation_reason(self) -> str:
        with self._lock:
            return self._revocation_reason

    def issue(
        self,
        *,
        session_id: str,
        target_instance: str,
        ttl_ms: int = 750,
        allowed_actions: frozenset[str] = frozenset({"keyboard", "button", "mouse"}),
        max_action_duration_ms: int = 250,
        first_sequence: int = 0,
    ) -> MotorLease:
        if not target_instance:
            raise MotorRejected("target instance identity is required")
        _validate_motor_lease_ttl(ttl_ms)
        if max_action_duration_ms < 1 or max_action_duration_ms > 1000:
            raise MotorRejected("action duration outside safety bounds")
        with self._lock:
            self._revoke_locked("lease-issue")
            lease = MotorLease(
                lease_id=uuid.uuid4().hex,
                session_id=session_id,
                target_instance=target_instance,
                backend_id=self.backend.backend_id,
                expires_monotonic_ns=time.monotonic_ns() + ttl_ms * 1_000_000,
                allowed_actions=allowed_actions,
                max_action_duration_ms=max_action_duration_ms,
                first_sequence=first_sequence,
            )
            try:
                self.backend.bind_lease(lease)
            except Exception as exc:
                self._revoke_after_failure_locked("lease-issue-failed", exc)
                raise
            self._lease = lease
            self._last_sequence = first_sequence - 1
            self._revocation_reason = "active"
            return lease

    def renew(self, lease_id: str, *, ttl_ms: int = 750) -> MotorLease:
        _validate_motor_lease_ttl(ttl_ms)
        with self._lock:
            lease = self._require_lease(lease_id)
            if lease.expired():
                self._revoke_locked("heartbeat-expired")
                raise MotorRejected("motor lease expired")
            renewed = MotorLease(
                lease_id=lease.lease_id,
                session_id=lease.session_id,
                target_instance=lease.target_instance,
                backend_id=lease.backend_id,
                expires_monotonic_ns=time.monotonic_ns() + ttl_ms * 1_000_000,
                allowed_actions=lease.allowed_actions,
                max_action_duration_ms=lease.max_action_duration_ms,
                first_sequence=lease.first_sequence,
            )
            # Do not publish the old capability while the backend is replacing
            # its binding. Successful renewal must not release held inputs.
            self._lease = None
            self._revocation_reason = "lease-renewal"
            try:
                self.backend.bind_lease(renewed)
            except Exception as exc:
                self._revoke_after_failure_locked("lease-renewal-failed", exc)
                raise
            self._lease = renewed
            self._revocation_reason = "active"
            return renewed

    def apply(
        self,
        lease_id: str,
        action: MotorAction,
        *,
        accepted_action_ttl_ms: int | None = None,
    ) -> None:
        """Apply one action and optionally refresh its authenticated lease.

        Refresh happens only after the backend accepts a valid, in-order action
        and remains bounded by the same fixed TTL ceiling as an explicit
        heartbeat. The gate lock makes acceptance and refresh atomic with
        respect to the watchdog, while inactivity still expires normally.
        """
        if accepted_action_ttl_ms is not None:
            _validate_motor_lease_ttl(accepted_action_ttl_ms)
        with self._lock:
            lease = self._require_lease(lease_id)
            if lease.expired():
                self._revoke_locked("lease-expired")
                raise MotorRejected("motor lease expired")
            if action.sequence <= self._last_sequence:
                self._revoke_locked("replay-or-out-of-order-action")
                raise MotorRejected("action sequence must increase monotonically")
            if action.duration_ms > lease.max_action_duration_ms:
                self._revoke_locked("action-duration-limit")
                raise MotorRejected("action exceeds lease duration limit")
            if not action.action_kinds().issubset(lease.allowed_actions):
                self._revoke_locked("action-not-allowed")
                raise MotorRejected("action kind not allowed by lease")
            try:
                self.backend.apply(action)
            except Exception as exc:
                # A physical backend can fail after emitting only part of an
                # action. Revoke synchronously so a pressed key/button is not
                # left behind while the runtime attempts a separate fault IPC.
                self._revoke_after_failure_locked("backend-apply-failed", exc)
                raise
            self._last_sequence = action.sequence
            if accepted_action_ttl_ms is not None:
                refreshed = MotorLease(
                    lease_id=lease.lease_id,
                    session_id=lease.session_id,
                    target_instance=lease.target_instance,
                    backend_id=lease.backend_id,
                    expires_monotonic_ns=(
                        time.monotonic_ns() + accepted_action_ttl_ms * 1_000_000
                    ),
                    allowed_actions=lease.allowed_actions,
                    max_action_duration_ms=lease.max_action_duration_ms,
                    first_sequence=lease.first_sequence,
                )
                try:
                    self.backend.bind_lease(refreshed)
                except Exception as exc:
                    self._revoke_after_failure_locked("accepted-action-refresh-failed", exc)
                    raise
                self._lease = refreshed

    def release_inputs(self, lease_id: str) -> None:
        """Release held input without revoking an otherwise healthy lease."""
        with self._lock:
            lease = self._require_lease(lease_id)
            if lease.expired():
                self._revoke_locked("lease-expired")
                raise MotorRejected("motor lease expired")
            try:
                self.backend.release_all()
            except Exception as exc:
                self._revoke_after_failure_locked("input-release-failed", exc)
                raise

    def check_expiry(self, now_ns: int | None = None) -> bool:
        with self._lock:
            if self._lease is None:
                return False
            if self._lease.expired(now_ns):
                self._revoke_locked("watchdog-expired")
                return True
            return False

    def revoke(self, reason: str) -> None:
        with self._lock:
            self._revoke_locked(reason)

    def _require_lease(self, lease_id: str) -> MotorLease:
        if self._lease is None or self._lease.lease_id != lease_id:
            error = MotorRejected("invalid or missing motor lease")
            self._revoke_after_failure_locked("invalid-or-missing-lease", error)
            raise error
        return self._lease

    def _revoke_after_failure_locked(self, reason: str, error: Exception) -> None:
        """Retain the original failure if emergency cleanup also fails."""
        try:
            self._revoke_locked(reason)
        except Exception as cleanup_error:
            error.add_note(f"motor cleanup also failed: {type(cleanup_error).__name__}")

    def _revoke_locked(self, reason: str) -> None:
        self._lease = None
        self._revocation_reason = reason
        release_error: BaseException | None = None
        try:
            self.backend.release_all()
        except BaseException as exc:
            release_error = exc
            raise
        finally:
            # Capability clearing is independent of whether physical release
            # succeeded. Neither a release nor a clear failure proves neutrality.
            try:
                self.backend.clear_lease()
            except BaseException as clear_error:
                if release_error is None:
                    raise
                release_error.add_note(
                    f"motor lease clearing also failed: {type(clear_error).__name__}"
                )


def allowed_targets(current: SupervisorState) -> frozenset[SupervisorState]:
    return _ALLOWED_TRANSITIONS[current]


def validate_transition(current: SupervisorState, target: SupervisorState) -> None:
    if target not in allowed_targets(current):
        raise TransitionError(f"invalid supervisor transition: {current} -> {target}")
