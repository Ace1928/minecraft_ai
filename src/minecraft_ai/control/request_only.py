"""One prospective literal survey; waiting and terminal states grant no work.

This is an optional runtime admission fence, not a policy or actuator. The
first supported scope is deliberately a camera-only survey. Other skills,
natural-language planning and typed directions need separate qualification.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from minecraft_ai.control.operator_budget import OperatorAttempt, authority_digest
from minecraft_ai.skills import SkillSpec
from minecraft_ai.social import OperatorMessage, OperatorMessageKind, OperatorMessageStatus


class RequestOnlyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    skill_id: Literal["survey_surroundings"] = "survey_surroundings"
    wait_timeout_ms: StrictInt = Field(default=60_000, ge=50, le=180_000)


def skill_digest(spec: SkillSpec) -> str:
    return hashlib.sha256(json.dumps(
        spec.model_dump(mode="json"), sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def camera_only_spec(spec: SkillSpec) -> bool:
    permissions = spec.action_permissions.model_dump()
    return len(permissions) == 7 and all(value is False for value in permissions.values())


@dataclass
class LiteralRequestFence:
    config: RequestOnlyConfig
    started_wall_ns: int
    waiting_deadline_ns: int
    state: str = "waiting"
    reason: str | None = None
    message_id: str | None = None
    context_key: str | None = None
    run_id: str | None = None
    authority_sha256: str | None = None
    spec_sha256: str | None = None
    operator_revision: int | None = None
    attempt: OperatorAttempt | None = None

    def admission_reason(self, message: OperatorMessage, spec: SkillSpec) -> str | None:
        if self.state != "waiting":
            return "owner_already_consumed"
        if (message.status != OperatorMessageStatus.QUEUED
                or message.created_ns < self.started_wall_ns):
            return "request_not_fresh"
        if (message.kind not in {OperatorMessageKind.INSTRUCTION, OperatorMessageKind.CORRECTION}
                or message.direction_request_id is not None
                or message.direction_attempt_id is not None
                or message.text != self.config.skill_id or spec.skill_id != self.config.skill_id):
            return "literal_request_required"
        if (message.execution_budget is None or message.execution_budget.max_skills != 1
                or message.execution_budget.timeout_ms > 12_000):
            return "one_bounded_survey_required"
        if not camera_only_spec(spec):
            return "camera_permissions_changed"
        return None

    def bind(self, message: OperatorMessage, spec: SkillSpec, attempt: OperatorAttempt,
             *, revision: int, run_id: str) -> None:
        if self.admission_reason(message, spec) is not None or attempt.terminal_reason is not None:
            raise ValueError("request-only admission refused")
        self.message_id = message.message_id
        self.context_key = f"operator:{message.message_id}"
        self.run_id = run_id
        self.authority_sha256 = authority_digest(message)
        self.spec_sha256 = skill_digest(spec)
        self.operator_revision = revision
        self.attempt = attempt
        self.state = "running"

    def check(self, snapshot, run, spec: SkillSpec, *, now_ns: int) -> str | None:
        if self.state != "running" or self.attempt is None:
            return "request_not_running"
        if (snapshot.revision != self.operator_revision
                or run is None or run.run_id != self.run_id or run.context_key != self.context_key
                or run.skill_id != self.config.skill_id):
            return "request_authority_changed"
        matching = tuple(message for message in snapshot.messages if message.message_id == self.message_id)
        if (len(matching) != 1 or matching[0].status != OperatorMessageStatus.ACKNOWLEDGED
                or authority_digest(matching[0]) != self.authority_sha256):
            return "request_authority_changed"
        if not camera_only_spec(spec) or skill_digest(spec) != self.spec_sha256:
            return "camera_permissions_changed"
        return self.attempt.check(now_ns=now_ns)

    def finish(self, reason: str) -> None:
        # An expiry, refusal or completed survey never turns the fence off.
        if self.state != "terminal":
            self.state, self.reason = "terminal", reason

    def status(self) -> dict[str, str | None]:
        return {"state": self.state, "reason": self.reason, "skill_id": self.config.skill_id}
