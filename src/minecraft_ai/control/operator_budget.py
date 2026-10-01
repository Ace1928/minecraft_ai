"""Request-owned bounds; no policy, model, actuator or clock calls in this module."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from minecraft_ai.social import OperatorMessage, OperatorMessageStatus


def authority_digest(message: OperatorMessage) -> str:
    content = message.model_dump(
        mode="json",
        exclude={
            "status",
            "delivered_ns",
            "acknowledged_ns",
            "response_text",
        },
    )
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass
class OperatorAttempt:
    """One owner clock and count; children cannot refresh either.

    A consumed request cannot create a replacement owner after a restart.
    A new request ID is required instead of pretending count continuation.
    """

    authority_sha256: str
    deadline_ns: int
    max_skills: int
    skills_started: int = 0
    terminal_reason: str | None = None

    @classmethod
    def admit(cls, message: OperatorMessage, *, wall_ns: int, monotonic_ns: int) -> OperatorAttempt:
        budget = message.execution_budget
        if budget is None:
            raise ValueError("explicit execution budget required")
        remaining = message.created_ns + budget.timeout_ms * 1_000_000 - wall_ns
        owner = cls(authority_digest(message), monotonic_ns + max(0, remaining), budget.max_skills)
        if message.status not in {OperatorMessageStatus.QUEUED, OperatorMessageStatus.DELIVERED}:
            owner.terminal_reason = "operator.attempt_owner_not_retained"
        elif message.created_ns > wall_ns:
            owner.terminal_reason = "operator.attempt_clock_invalid"
        elif remaining <= 0:
            owner.terminal_reason = "operator.attempt_deadline"
        return owner

    def bind(self, message: OperatorMessage) -> None:
        if authority_digest(message) != self.authority_sha256:
            self.terminal_reason = "operator.attempt_authority_changed"

    def check(self, *, now_ns: int) -> str | None:
        if self.terminal_reason is None and now_ns >= self.deadline_ns:
            self.terminal_reason = "operator.attempt_deadline"
        return self.terminal_reason

    def start(self, *, now_ns: int) -> str | None:
        reason = self.check(now_ns=now_ns)
        if reason is None and self.skills_started >= self.max_skills:
            reason = self.terminal_reason = "operator.attempt_skill_limit"
        if reason is None:
            self.skills_started += 1
        return reason
