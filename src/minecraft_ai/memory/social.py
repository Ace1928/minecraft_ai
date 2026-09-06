from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class PromiseStatus(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ChatEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str
    observed_ns: int
    speaker: str | None = None
    text: str
    channel: str = "chat"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class PlayerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str
    player: str
    summary: str
    created_ns: int
    priority: float = Field(default=0.7, ge=0.0, le=1.0)
    target_node: str | None = None
    project_id: str | None = None
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class OperatorMessageKind(StrEnum):
    INSTRUCTION = "instruction"
    QUESTION = "question"
    FEEDBACK = "feedback"
    CORRECTION = "correction"


class OperatorMessageStatus(StrEnum):
    QUEUED = "queued"
    DELIVERED = "delivered"
    ACKNOWLEDGED = "acknowledged"
    ARCHIVED = "archived"


class OperatorMessage(BaseModel):
    """Durable high-level input from the local operator surface.

    These messages enter cognition as social/task context. They never bypass the
    supervisor or translate directly into keyboard/mouse events.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    message_id: str
    created_ns: int
    author: str = "operator"
    text: str = Field(min_length=1, max_length=2000)
    kind: OperatorMessageKind = OperatorMessageKind.INSTRUCTION
    priority: float = Field(default=0.8, ge=0.0, le=1.0)
    status: OperatorMessageStatus = OperatorMessageStatus.QUEUED
    delivered_ns: int | None = None
    acknowledged_ns: int | None = None
    response_text: str | None = Field(default=None, max_length=2000)


class Promise(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    promise_id: str
    player: str
    summary: str
    created_ns: int
    status: PromiseStatus = PromiseStatus.PENDING
    goal_id: str | None = None
    project_id: str | None = None
    updated_ns: int | None = None


class SharedProject(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str
    name: str
    created_ns: int
    owner: str | None = None
    participants: tuple[str, ...] = ()
    description: str = ""
    status: str = "active"
    goal_ids: tuple[str, ...] = ()
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


@dataclass
class SocialState:
    chat_history: list[ChatEvent] = field(default_factory=list)
    requests: dict[str, PlayerRequest] = field(default_factory=dict)
    promises: dict[str, Promise] = field(default_factory=dict)
    projects: dict[str, SharedProject] = field(default_factory=dict)

    def add_chat(self, event: ChatEvent, *, max_history: int = 1000) -> None:
        self.chat_history.append(event)
        if len(self.chat_history) > max_history:
            del self.chat_history[: len(self.chat_history) - max_history]

    def add_request(self, request: PlayerRequest) -> None:
        self.requests[request.request_id] = request

    def add_project(self, project: SharedProject) -> None:
        self.projects[project.project_id] = project

    def promise(self, promise: Promise) -> None:
        self.promises[promise.promise_id] = promise

    def set_promise_status(
        self,
        promise_id: str,
        status: PromiseStatus,
        *,
        updated_ns: int,
    ) -> Promise:
        current = self.promises[promise_id]
        updated = current.model_copy(update={"status": status, "updated_ns": updated_ns})
        self.promises[promise_id] = updated
        return updated

    def active_promises(self) -> tuple[Promise, ...]:
        return tuple(
            promise
            for promise in self.promises.values()
            if promise.status in {PromiseStatus.PENDING, PromiseStatus.ACTIVE}
        )
