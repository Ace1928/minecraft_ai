from __future__ import annotations

import json
import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from minecraft_ai.bridge import BridgeEndpoint, ScopedBridgeBackend
from minecraft_ai.safety import MotorAction, MotorGate, MotorRejected


class FakeBridgeServer:
    def __init__(self, *, instance_id: str = "java-instance-test", token: str = "t" * 32) -> None:
        self.instance_id = instance_id
        self.token = token
        self.messages: list[dict[str, Any]] = []
        self.held_keys: set[str] = set()
        self.held_buttons: set[str] = set()
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.bind(("127.0.0.1", 0))
        self._server.listen(1)
        self.host, self.port = self._server.getsockname()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        try:
            self._server.close()
        except OSError:
            pass
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        try:
            conn, _ = self._server.accept()
        except OSError:
            return
        with conn:
            conn.settimeout(1.0)
            self._send(
                conn,
                {
                    "kind": "hello",
                    "protocol_version": 1,
                    "nonce": "n" * 24,
                    "identity": {
                        "edition": "java",
                        "version": "test",
                        "instance_id": self.instance_id,
                        "process_id": 1234,
                        "profile": "test",
                    },
                    "capabilities": ["input", "window_identity"],
                },
            )
            while not self._stop.is_set():
                try:
                    message = self._recv(conn)
                except (ConnectionError, OSError, TimeoutError):
                    return
                self.messages.append(message)
                kind = message.get("kind")
                if kind == "authenticate":
                    if (
                        message.get("token") != self.token
                        or message.get("expected_instance_id") != self.instance_id
                    ):
                        self._error(conn, "unauthorized", "authentication failed")
                        return
                    self._ack(conn, 0)
                elif kind == "lease_bind":
                    if message.get("target_instance_id") != self.instance_id:
                        self._release()
                        self._error(conn, "target_mismatch", "wrong target")
                        return
                    self._ack(conn, int(message.get("first_sequence", 0)))
                elif kind == "input":
                    self.held_keys.update(message.get("keys_down", []))
                    self.held_keys.difference_update(message.get("keys_up", []))
                    self.held_buttons.update(message.get("buttons_down", []))
                    self.held_buttons.difference_update(message.get("buttons_up", []))
                    self._ack(conn, int(message["sequence"]))
                elif kind in {"release_all", "lease_clear"}:
                    self._release()
                    self._ack(conn, 0)
                else:
                    self._release()
                    self._error(conn, "unknown", "unknown command")
                    return

    def _release(self) -> None:
        self.held_keys.clear()
        self.held_buttons.clear()

    def _ack(self, conn: socket.socket, sequence: int) -> None:
        self._send(
            conn,
            {
                "kind": "ack",
                "protocol_version": 1,
                "sequence": sequence,
                "instance_id": self.instance_id,
            },
        )

    def _error(self, conn: socket.socket, code: str, message: str) -> None:
        self._send(
            conn,
            {
                "kind": "error",
                "protocol_version": 1,
                "code": code,
                "message": message,
                "release_all": True,
            },
        )

    @staticmethod
    def _send(conn: socket.socket, payload: dict[str, Any]) -> None:
        conn.sendall(json.dumps(payload).encode("utf-8") + b"\n")

    @staticmethod
    def _recv(conn: socket.socket) -> dict[str, Any]:
        data = bytearray()
        while b"\n" not in data:
            chunk = conn.recv(4096)
            if not chunk:
                raise ConnectionError("closed")
            data.extend(chunk)
        payload = json.loads(bytes(data).split(b"\n", 1)[0].decode("utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("expected object")
        return payload


@contextmanager
def running_bridge(**kwargs: str) -> Iterator[FakeBridgeServer]:
    bridge = FakeBridgeServer(**kwargs)
    bridge.start()
    try:
        yield bridge
    finally:
        bridge.close()


def test_scoped_bridge_controls_only_authenticated_instance() -> None:
    with running_bridge() as bridge:
        backend = ScopedBridgeBackend(
            BridgeEndpoint(
                host=bridge.host,
                port=bridge.port,
                token=bridge.token,
                instance_id=bridge.instance_id,
            )
        )
        hello = backend.connect()
        assert hello.identity.instance_id == bridge.instance_id

        gate = MotorGate(backend)
        lease = gate.issue(session_id="s" * 16, target_instance=bridge.instance_id)
        gate.apply(
            lease.lease_id,
            MotorAction(sequence=0, keys_down=("w",), buttons_down=("left",)),
        )
        assert bridge.held_keys == {"w"}
        assert bridge.held_buttons == {"left"}

        gate.revoke("test-stop")
        assert bridge.held_keys == set()
        assert bridge.held_buttons == set()
        backend.close()


def test_instance_identity_mismatch_rejected_before_lease() -> None:
    with running_bridge() as bridge:
        backend = ScopedBridgeBackend(
            BridgeEndpoint(
                host=bridge.host,
                port=bridge.port,
                token=bridge.token,
                instance_id="different-instance",
            )
        )
        with pytest.raises(MotorRejected, match="identity mismatch"):
            backend.connect()


def test_wrong_lease_target_fails_closed() -> None:
    with running_bridge() as bridge:
        backend = ScopedBridgeBackend(
            BridgeEndpoint(
                host=bridge.host,
                port=bridge.port,
                token=bridge.token,
                instance_id=bridge.instance_id,
            )
        )
        backend.connect()
        gate = MotorGate(backend)
        with pytest.raises(MotorRejected, match="targets another"):
            gate.issue(session_id="s" * 16, target_instance="wrong-target-instance")
        assert bridge.held_keys == set()
        assert bridge.held_buttons == set()


def test_bridge_uses_relative_ttls_bounded_by_local_lease() -> None:
    with running_bridge() as bridge:
        backend = ScopedBridgeBackend(
            BridgeEndpoint(
                host=bridge.host,
                port=bridge.port,
                token=bridge.token,
                instance_id=bridge.instance_id,
            )
        )
        backend.connect()
        gate = MotorGate(backend)
        lease = gate.issue(
            session_id="s" * 16,
            target_instance=bridge.instance_id,
            ttl_ms=100,
            max_action_duration_ms=90,
        )
        gate.apply(lease.lease_id, MotorAction(sequence=0, keys_down=("w",), duration_ms=90))

        lease_messages = [msg for msg in bridge.messages if msg.get("kind") == "lease_bind"]
        input_messages = [msg for msg in bridge.messages if msg.get("kind") == "input"]
        assert len(lease_messages) == 1
        assert len(input_messages) == 1
        assert 1 <= int(lease_messages[0]["ttl_ms"]) <= 100
        assert 1 <= int(input_messages[0]["ttl_ms"]) <= 90
        assert "expires_monotonic_ns" not in lease_messages[0]
        assert "deadline_monotonic_ns" not in input_messages[0]


def test_expired_gate_revokes_remote_inputs() -> None:
    with running_bridge() as bridge:
        backend = ScopedBridgeBackend(
            BridgeEndpoint(
                host=bridge.host,
                port=bridge.port,
                token=bridge.token,
                instance_id=bridge.instance_id,
            )
        )
        backend.connect()
        gate = MotorGate(backend)
        lease = gate.issue(
            session_id="s" * 16,
            target_instance=bridge.instance_id,
            ttl_ms=50,
        )
        gate.apply(lease.lease_id, MotorAction(sequence=0, keys_down=("w",)))
        assert bridge.held_keys == {"w"}
        # Drive the watchdog with its explicit clock input. Sleeping only ten
        # milliseconds past the lease made this safety test scheduler-dependent
        # on busy Windows CI hosts without exercising any additional behavior.
        assert gate.check_expiry(now_ns=lease.expires_monotonic_ns)
        assert bridge.held_keys == set()
