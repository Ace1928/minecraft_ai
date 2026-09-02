from __future__ import annotations

import secrets
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

PROTOCOL_VERSION = 1
_ALLOWED_ACTIONS = frozenset({"keyboard", "button", "mouse"})


class BridgeCapability(StrEnum):
    INPUT = "input"
    FRAME = "frame"
    CHAT = "chat"
    WINDOW_IDENTITY = "window_identity"


class InstanceIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    edition: Literal["java", "bedrock"]
    version: str
    instance_id: str = Field(min_length=8, max_length=256)
    process_id: int | None = Field(default=None, ge=1)
    profile: str | None = Field(default=None, max_length=256)


class BridgeHello(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["hello"] = "hello"
    protocol_version: int = PROTOCOL_VERSION
    nonce: str = Field(default_factory=lambda: secrets.token_urlsafe(24), min_length=16)
    identity: InstanceIdentity
    capabilities: frozenset[BridgeCapability]


class Authenticate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["authenticate"] = "authenticate"
    protocol_version: int = PROTOCOL_VERSION
    token: str = Field(min_length=24, max_length=512)
    expected_instance_id: str = Field(min_length=8, max_length=256)


class LeaseBind(BaseModel):
    """Replicate a lease using a relative TTL, never a cross-runtime timestamp."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["lease_bind"] = "lease_bind"
    protocol_version: int = PROTOCOL_VERSION
    lease_id: str = Field(min_length=16, max_length=128)
    supervisor_session_id: str = Field(min_length=16, max_length=128)
    target_instance_id: str = Field(min_length=8, max_length=256)
    ttl_ms: int = Field(ge=1, le=5000)
    allowed_actions: frozenset[str]
    max_action_duration_ms: int = Field(ge=1, le=1000)
    first_sequence: int = Field(ge=0)

    @field_validator("allowed_actions")
    @classmethod
    def _validate_allowed_actions(cls, value: frozenset[str]) -> frozenset[str]:
        if not value.issubset(_ALLOWED_ACTIONS):
            raise ValueError("unsupported action kind")
        return value


class LeaseClear(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["lease_clear"] = "lease_clear"
    protocol_version: int = PROTOCOL_VERSION
    lease_id: str | None = Field(default=None, max_length=128)
    reason: str = Field(min_length=1, max_length=256)


class InputCommand(BaseModel):
    """Bounded per-instance input semantics with a relative execution TTL."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["input"] = "input"
    protocol_version: int = PROTOCOL_VERSION
    lease_id: str = Field(min_length=16, max_length=128)
    sequence: int = Field(ge=0)
    ttl_ms: int = Field(ge=1, le=1000)
    keys_down: tuple[str, ...] = ()
    keys_up: tuple[str, ...] = ()
    buttons_down: tuple[str, ...] = ()
    buttons_up: tuple[str, ...] = ()
    mouse_dx: int = Field(default=0, ge=-4096, le=4096)
    mouse_dy: int = Field(default=0, ge=-4096, le=4096)
    duration_ms: int = Field(default=0, ge=0, le=1000)

    @field_validator("keys_down", "keys_up", "buttons_down", "buttons_up")
    @classmethod
    def _validate_tokens(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) > 16:
            raise ValueError("too many simultaneous inputs")
        normalized = tuple(token.strip().lower() for token in value)
        if any(not token or len(token) > 32 for token in normalized):
            raise ValueError("invalid input token")
        return normalized


class ReleaseAll(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["release_all"] = "release_all"
    protocol_version: int = PROTOCOL_VERSION
    reason: str = Field(min_length=1, max_length=256)


class Heartbeat(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["heartbeat"] = "heartbeat"
    protocol_version: int = PROTOCOL_VERSION
    sequence: int = Field(ge=0)


class BridgeAck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["ack"] = "ack"
    protocol_version: int = PROTOCOL_VERSION
    sequence: int = Field(ge=0)
    instance_id: str = Field(min_length=8, max_length=256)


class BridgeError(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["error"] = "error"
    protocol_version: int = PROTOCOL_VERSION
    code: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=1024)
    release_all: bool = True


class ChatEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["chat"] = "chat"
    protocol_version: int = PROTOCOL_VERSION
    sequence: int = Field(ge=0)
    speaker: str = Field(max_length=128)
    message: str = Field(max_length=2048)


class FrameDescriptor(BaseModel):
    """Metadata for a frame transported out-of-band/shared-memory later."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["frame"] = "frame"
    protocol_version: int = PROTOCOL_VERSION
    sequence: int = Field(ge=0)
    width: int = Field(gt=0, le=16384)
    height: int = Field(gt=0, le=16384)
    pixel_format: Literal["rgb24", "rgba32", "bgra32"]
    timestamp_monotonic_ns: int = Field(gt=0)
    transport_ref: str = Field(min_length=1, max_length=1024)
