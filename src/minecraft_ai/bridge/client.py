from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..safety import MotorAction, MotorLease, MotorRejected
from .protocol import (
    Authenticate,
    BridgeAck,
    BridgeCapability,
    BridgeError,
    BridgeHello,
    InputCommand,
    InstanceIdentity,
    LeaseBind,
    LeaseClear,
    ReleaseAll,
)


class BridgeEndpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    host: str = "127.0.0.1"
    port: int = Field(ge=1, le=65535)
    token: str = Field(min_length=24, max_length=512)
    instance_id: str = Field(min_length=8, max_length=256)

    @classmethod
    def load(cls, path: Path) -> BridgeEndpoint:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("bridge endpoint descriptor must be a JSON object")
        return cls.model_validate(raw)


@dataclass
class ScopedBridgeBackend:
    """InputBackend implementation that controls one authenticated game instance."""

    endpoint: BridgeEndpoint
    connect_timeout_s: float = 1.0
    request_timeout_s: float = 1.0
    backend_id: str = "scoped-bridge"
    live_capable: bool = True

    def __post_init__(self) -> None:
        self._lock = threading.RLock()
        self._sock: socket.socket | None = None
        self._hello: BridgeHello | None = None
        self._lease: MotorLease | None = None
        self._authenticated = False

    @property
    def identity(self) -> InstanceIdentity | None:
        with self._lock:
            return None if self._hello is None else self._hello.identity

    def connect(self) -> BridgeHello:
        with self._lock:
            self.close()
            sock = socket.create_connection(
                (self.endpoint.host, self.endpoint.port),
                timeout=self.connect_timeout_s,
            )
            sock.settimeout(self.request_timeout_s)
            try:
                hello_raw = _recv_message(sock)
                hello = BridgeHello.model_validate(hello_raw)
                if hello.protocol_version != 1:
                    raise MotorRejected("unsupported bridge protocol version")
                if hello.identity.instance_id != self.endpoint.instance_id:
                    raise MotorRejected("bridge instance identity mismatch")
                if BridgeCapability.INPUT not in hello.capabilities:
                    raise MotorRejected("bridge does not advertise input capability")
                _send_model(
                    sock,
                    Authenticate(
                        token=self.endpoint.token,
                        expected_instance_id=self.endpoint.instance_id,
                    ),
                )
                self._expect_ack(sock, expected_sequence=0)
            except Exception:
                sock.close()
                raise
            self._sock = sock
            self._hello = hello
            self._authenticated = True
            return hello

    def bind_lease(self, lease: MotorLease) -> None:
        with self._lock:
            sock = self._require_socket()
            identity = self._require_identity()
            if lease.target_instance != identity.instance_id:
                self._fail_closed("lease-target-mismatch")
                raise MotorRejected("motor lease targets another Minecraft instance")
            remaining_ms = _remaining_lease_ms(lease)
            message = LeaseBind(
                lease_id=lease.lease_id,
                supervisor_session_id=lease.session_id,
                target_instance_id=lease.target_instance,
                ttl_ms=remaining_ms,
                allowed_actions=frozenset(lease.allowed_actions),
                max_action_duration_ms=lease.max_action_duration_ms,
                first_sequence=lease.first_sequence,
            )
            try:
                _send_model(sock, message)
                self._expect_ack(sock, expected_sequence=lease.first_sequence)
            except Exception:
                self._fail_closed("lease-bind-failed")
                raise
            self._lease = lease

    def apply(self, action: MotorAction) -> None:
        with self._lock:
            sock = self._require_socket()
            lease = self._lease
            if lease is None:
                self._fail_closed("missing-lease")
                raise MotorRejected("scoped bridge has no bound lease")
            remaining_ms = _remaining_lease_ms(lease)
            action_ttl_ms = max(1, min(remaining_ms, max(action.duration_ms, 1)))
            command = InputCommand(
                lease_id=lease.lease_id,
                sequence=action.sequence,
                ttl_ms=action_ttl_ms,
                keys_down=action.keys_down,
                keys_up=action.keys_up,
                buttons_down=action.buttons_down,
                buttons_up=action.buttons_up,
                mouse_dx=action.mouse_dx,
                mouse_dy=action.mouse_dy,
                duration_ms=action.duration_ms,
            )
            try:
                _send_model(sock, command)
                self._expect_ack(sock, expected_sequence=action.sequence)
            except Exception:
                self._fail_closed("input-command-failed")
                raise

    def release_all(self) -> None:
        with self._lock:
            sock = self._sock
            if sock is None or not self._authenticated:
                return
            try:
                _send_model(sock, ReleaseAll(reason="supervisor-release"))
                self._expect_ack(sock, expected_sequence=0)
            except (OSError, TimeoutError, ValueError, TypeError, MotorRejected):
                self.close()

    def clear_lease(self) -> None:
        with self._lock:
            lease = self._lease
            sock = self._sock
            self._lease = None
            if sock is None or not self._authenticated:
                return
            try:
                _send_model(
                    sock,
                    LeaseClear(
                        lease_id=None if lease is None else lease.lease_id,
                        reason="supervisor-revoke",
                    ),
                )
                self._expect_ack(sock, expected_sequence=0)
            except (OSError, TimeoutError, ValueError, TypeError, MotorRejected):
                self.close()

    def close(self) -> None:
        sock = getattr(self, "_sock", None)
        self._sock = None
        self._hello = None
        self._lease = None
        self._authenticated = False
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def _require_socket(self) -> socket.socket:
        if self._sock is None or not self._authenticated:
            raise MotorRejected("scoped bridge is not authenticated")
        return self._sock

    def _require_identity(self) -> InstanceIdentity:
        if self._hello is None:
            raise MotorRejected("bridge identity is unavailable")
        return self._hello.identity

    def _expect_ack(self, sock: socket.socket, *, expected_sequence: int) -> BridgeAck:
        raw = _recv_message(sock)
        if raw.get("kind") == "error":
            error = BridgeError.model_validate(raw)
            if error.release_all:
                self.close()
            raise MotorRejected(f"bridge rejected request: {error.code}: {error.message}")
        ack = BridgeAck.model_validate(raw)
        if ack.instance_id != self.endpoint.instance_id:
            self._fail_closed("ack-instance-mismatch")
            raise MotorRejected("bridge acknowledgement instance mismatch")
        if ack.sequence != expected_sequence:
            self._fail_closed("ack-sequence-mismatch")
            raise MotorRejected("bridge acknowledgement sequence mismatch")
        return ack

    def _fail_closed(self, reason: str) -> None:
        sock = self._sock
        if sock is not None and self._authenticated:
            try:
                _send_model(sock, ReleaseAll(reason=reason))
            except OSError:
                pass
        self.close()


def _remaining_lease_ms(lease: MotorLease) -> int:
    remaining_ns = lease.expires_monotonic_ns - time.monotonic_ns()
    if remaining_ns <= 0:
        raise MotorRejected("scoped bridge lease expired")
    return max(1, min(5000, remaining_ns // 1_000_000))


def _send_model(sock: socket.socket, message: BaseModel) -> None:
    payload = message.model_dump(mode="json")
    sock.sendall(json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n")


def _recv_message(sock: socket.socket, *, limit: int = 64 * 1024) -> dict[str, Any]:
    data = bytearray()
    while len(data) < limit:
        chunk = sock.recv(min(4096, limit - len(data)))
        if not chunk:
            raise ConnectionError("bridge connection closed")
        data.extend(chunk)
        if b"\n" in chunk:
            break
    if len(data) >= limit:
        raise ValueError("bridge message exceeds size limit")
    raw = bytes(data).split(b"\n", 1)[0]
    decoded = json.loads(raw.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise TypeError("bridge message must be a JSON object")
    return decoded
