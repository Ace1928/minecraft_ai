from __future__ import annotations

import json
import os
import secrets
import signal
import socket
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from platformdirs import user_runtime_dir

from .emergency import emergency_reason, emergency_stop_latched
from .safety import (
    FakeInputBackend,
    InputBackend,
    MotorAction,
    MotorGate,
    SupervisorState,
    allowed_targets,
    validate_transition,
)

APP_NAME = "minecraft-ai"
RUNTIME_DIR = Path(user_runtime_dir(APP_NAME))
CONTROL_FILE = RUNTIME_DIR / "control.json"
STATUS_FILE = RUNTIME_DIR / "supervisor-state.json"


@dataclass(frozen=True)
class ControlEndpoint:
    host: str
    port: int
    token: str
    pid: int
    session_id: str

    @classmethod
    def load(cls, path: Path = CONTROL_FILE) -> ControlEndpoint:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            host=str(raw["host"]),
            port=int(raw["port"]),
            token=str(raw["token"]),
            pid=int(raw["pid"]),
            session_id=str(raw["session_id"]),
        )


class Supervisor:
    """Independent lifecycle owner for Minecraft AI."""

    def __init__(self, *, role: str = "generalist", watchdog_interval_s: float = 0.05) -> None:
        self.role = role
        self.session_id = secrets.token_hex(16)
        self.state = SupervisorState.STOPPED
        self.backend: InputBackend = FakeInputBackend()
        self.motor = MotorGate(self.backend)
        self.watchdog_interval_s = watchdog_interval_s
        self.started_monotonic_ns = time.monotonic_ns()
        self.last_command_monotonic_ns = self.started_monotonic_ns
        self.last_fault: str | None = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._server: socket.socket | None = None
        self._endpoint: ControlEndpoint | None = None

    def transition(self, target: SupervisorState) -> None:
        with self._lock:
            validate_transition(self.state, target)
            self.state = target
            self._persist_status()

    def start(self) -> None:
        if emergency_stop_latched():
            self.last_fault = emergency_reason() or "emergency-stop-latched"
            self.motor.revoke("emergency-stop-latched")
            self._persist_status()
            raise RuntimeError("emergency stop is latched; explicitly reset it before starting")
        self.transition(SupervisorState.STARTING)
        self.transition(SupervisorState.SAFE_IDLE)

    def pause(self) -> None:
        with self._lock:
            if self.state == SupervisorState.PAUSED:
                return
            if self.state in {SupervisorState.STOPPED, SupervisorState.STOPPING}:
                return
            self.motor.revoke("operator-pause")
            if self.state == SupervisorState.FAILSAFE:
                return
            validate_transition(self.state, SupervisorState.PAUSED)
            self.state = SupervisorState.PAUSED
            self._persist_status()

    def resume(self) -> None:
        with self._lock:
            if emergency_stop_latched():
                raise RuntimeError("emergency stop is latched")
            if self.state == SupervisorState.SAFE_IDLE:
                return
            if self.state != SupervisorState.PAUSED:
                raise RuntimeError(f"cannot resume from {self.state}")
            validate_transition(self.state, SupervisorState.SAFE_IDLE)
            self.state = SupervisorState.SAFE_IDLE
            self._persist_status()

    def replace_backend(self, backend: InputBackend) -> None:
        """Replace the motor backend only while completely unarmed."""
        with self._lock:
            if self.state != SupervisorState.SAFE_IDLE:
                raise RuntimeError(f"cannot replace backend from {self.state}")
            self.motor.revoke("backend-replacement")
            old_backend = self.backend
            try:
                old_backend.release_all()
            finally:
                close = getattr(old_backend, "close", None)
                if callable(close):
                    close()
            self.backend = backend
            self.motor = MotorGate(backend)
            self._persist_status()

    def attach_bedrock_x11(
        self, display: str, window_id: int, *, allow_host: bool = False
    ) -> dict[str, Any]:
        if emergency_stop_latched():
            raise RuntimeError("emergency stop is latched")
        if allow_host:
            raise RuntimeError("host-display input is debug-only and cannot be armed")
        from .platforms.bedrock_x11 import IsolatedX11InputBackend

        backend = IsolatedX11InputBackend(display, target_window_id=window_id, allow_host=False)
        try:
            self.replace_backend(backend)
        except Exception:
            backend.close()
            raise
        return self.status()

    def arm(self, target_instance: str) -> dict[str, Any]:
        with self._lock:
            if emergency_stop_latched():
                raise RuntimeError("emergency stop is latched")
            if self.state != SupervisorState.SAFE_IDLE:
                raise RuntimeError(f"cannot arm from {self.state}")
            validate_transition(self.state, SupervisorState.ARMED)
            lease = self.motor.issue(
                session_id=self.session_id,
                target_instance=target_instance,
                ttl_ms=3000,
            )
            self.state = SupervisorState.ARMED
            self._persist_status()
            return {
                "lease_id": lease.lease_id,
                "backend": lease.backend_id,
                "target_instance": lease.target_instance,
                "live_capable": self.backend.live_capable,
                "expires_monotonic_ns": lease.expires_monotonic_ns,
            }

    def renew(self, lease_id: str, *, ttl_ms: int = 750) -> dict[str, Any]:
        with self._lock:
            if self.state not in {SupervisorState.ARMED, SupervisorState.RUNNING}:
                raise RuntimeError(f"cannot renew motor lease from {self.state}")
            lease = self.motor.renew(lease_id, ttl_ms=ttl_ms)
            return {
                "lease_id": lease.lease_id,
                "expires_monotonic_ns": lease.expires_monotonic_ns,
            }

    def apply_motor_action(self, lease_id: str, raw_action: object) -> dict[str, Any]:
        with self._lock:
            self._require_running_lease(lease_id)
            action = MotorAction.model_validate(raw_action)
            self.motor.apply(lease_id, action)
            return {
                "accepted_sequence": action.sequence,
                "lease_active": self.motor.lease is not None,
            }

    def send_chat(self, lease_id: str, text: str) -> dict[str, Any]:
        """Send player chat through the already scoped Minecraft input backend."""
        with self._lock:
            self._require_running_lease(lease_id)
            if not text or len(text) > 256:
                raise ValueError("chat text must contain 1..256 characters")
            if any(ord(char) < 32 or ord(char) > 126 for char in text):
                raise ValueError("chat currently supports printable ASCII only")
            actuator = getattr(self.backend, "type_chat", None)
            if not callable(actuator):
                raise RuntimeError("active motor backend does not support scoped chat typing")
            try:
                # Chat typing is synchronous and can outlive the normal heartbeat.
                # Extend only this already-authenticated lease around the transaction.
                self.motor.renew(lease_id, ttl_ms=5000)
                actuator(text)
                self.motor.renew(lease_id, ttl_ms=3000)
            except Exception as exc:
                self.fail(f"chat-backend-fault:{type(exc).__name__}")
                raise
            return {"sent": True, "characters": len(text)}

    def _require_running_lease(self, lease_id: str) -> None:
        if emergency_stop_latched():
            self.fail("emergency-stop-latched")
            raise RuntimeError("emergency stop is latched")
        if self.state != SupervisorState.RUNNING:
            raise RuntimeError(f"motor interaction requires RUNNING, got {self.state}")
        lease = self.motor.lease
        if lease is None or lease.lease_id != lease_id or lease.expired():
            self.motor.revoke("invalid-runtime-lease")
            self.fail("invalid-runtime-lease")
            raise RuntimeError("invalid or expired runtime motor lease")

    def activate(self) -> None:
        with self._lock:
            if emergency_stop_latched():
                self.fail("emergency-stop-latched")
                return
            if self.state != SupervisorState.ARMED:
                raise RuntimeError(f"cannot activate from {self.state}")
            lease = self.motor.lease
            if lease is None or lease.expired():
                self.motor.revoke("activate-without-live-lease")
                self.fail("activate-without-live-lease")
                return
            validate_transition(self.state, SupervisorState.RUNNING)
            self.state = SupervisorState.RUNNING
            self._persist_status()

    def disarm(self, reason: str = "operator-disarm") -> None:
        with self._lock:
            self.motor.revoke(reason)
            if self.state in {SupervisorState.ARMED, SupervisorState.RUNNING}:
                validate_transition(self.state, SupervisorState.PAUSED)
                self.state = SupervisorState.PAUSED
            self._persist_status()

    def fail(self, reason: str) -> None:
        with self._lock:
            if self.state == SupervisorState.STOPPED:
                return
            self.last_fault = reason
            self.motor.revoke(reason)
            if (
                self.state != SupervisorState.FAILSAFE
                and SupervisorState.FAILSAFE in allowed_targets(self.state)
            ):
                self.state = SupervisorState.FAILSAFE
            self._persist_status()

    def stop(self) -> None:
        with self._lock:
            if self.state == SupervisorState.STOPPED:
                self.motor.revoke("operator-stop")
                self._stop.set()
                self._persist_status()
                return
            self.motor.revoke("operator-stop")
            if self.state == SupervisorState.FAILSAFE:
                self.state = SupervisorState.STOPPED
            else:
                if self.state != SupervisorState.STOPPING:
                    validate_transition(self.state, SupervisorState.STOPPING)
                    self.state = SupervisorState.STOPPING
                validate_transition(self.state, SupervisorState.STOPPED)
                self.state = SupervisorState.STOPPED
            self._stop.set()
            self._persist_status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            lease = self.motor.lease
            held_keys = sorted(getattr(self.backend, "held_keys", ()))
            held_buttons = sorted(getattr(self.backend, "held_buttons", ()))
            release_count = getattr(self.backend, "release_count", None)
            return {
                "state": self.state.value,
                "role": self.role,
                "session_id": self.session_id,
                "pid": os.getpid(),
                "backend": self.backend.backend_id,
                "live_capable": self.backend.live_capable,
                "motor_lease_active": lease is not None and not lease.expired(),
                "motor_target_instance": lease.target_instance if lease is not None else None,
                "motor_revocation_reason": self.motor.revocation_reason,
                "held_keys": held_keys,
                "held_buttons": held_buttons,
                "release_count": release_count,
                "last_fault": self.last_fault,
                "emergency_stop_latched": emergency_stop_latched(),
                "uptime_s": round(
                    (time.monotonic_ns() - self.started_monotonic_ns) / 1e9,
                    3,
                ),
            }

    def serve_forever(self) -> None:
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        self.start()
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(8)
        server.settimeout(self.watchdog_interval_s)
        self._server = server
        host, port = server.getsockname()
        endpoint = ControlEndpoint(
            host=str(host),
            port=int(port),
            token=secrets.token_urlsafe(32),
            pid=os.getpid(),
            session_id=self.session_id,
        )
        self._endpoint = endpoint
        _atomic_json_write(CONTROL_FILE, asdict(endpoint), mode=0o600)
        self._persist_status()

        try:
            while not self._stop.is_set():
                if emergency_stop_latched():
                    self.fail(emergency_reason() or "emergency-stop-latched")
                    self.stop()
                    break
                if self.motor.check_expiry():
                    self.fail("motor-lease-watchdog-expired")
                try:
                    conn, _ = server.accept()
                except TimeoutError:
                    continue
                with conn:
                    conn.settimeout(1.0)
                    self._handle_connection(conn)
        finally:
            self.motor.revoke("supervisor-exit")
            try:
                server.close()
            finally:
                close = getattr(self.backend, "close", None)
                if callable(close):
                    close()
                self._remove_control_file_if_owned()
                if self.state != SupervisorState.STOPPED:
                    self.state = SupervisorState.STOPPED
                self._persist_status()

    def _handle_connection(self, conn: socket.socket) -> None:
        try:
            payload = _recv_json_line(conn)
            if payload.get("token") != (self._endpoint.token if self._endpoint else None):
                _send_json_line(conn, {"ok": False, "error": "unauthorized"})
                return
            self.last_command_monotonic_ns = time.monotonic_ns()
            command = str(payload.get("command", ""))
            if command == "status":
                result = self.status()
            elif command == "pause":
                self.pause()
                result = self.status()
            elif command == "resume":
                self.resume()
                result = self.status()
            elif command == "stop":
                self.stop()
                result = self.status()
            elif command == "attach-bedrock-x11":
                display = str(payload.get("display", ""))
                window_id = int(payload.get("window_id", 0))
                allow_host = bool(payload.get("allow_host", False))
                result = self.attach_bedrock_x11(display, window_id, allow_host=allow_host)
            elif command in {"arm-fake", "arm"}:
                target_instance = str(payload.get("target_instance", "fake-instance"))
                result = {"lease": self.arm(target_instance), "status": self.status()}
            elif command in {"activate-fake", "activate"}:
                self.activate()
                result = self.status()
            elif command == "renew":
                lease_id = str(payload.get("lease_id", ""))
                ttl_ms = int(payload.get("ttl_ms", 750))
                result = self.renew(lease_id, ttl_ms=ttl_ms)
            elif command == "motor-action":
                lease_id = str(payload.get("lease_id", ""))
                result = self.apply_motor_action(lease_id, payload.get("action"))
            elif command == "chat":
                lease_id = str(payload.get("lease_id", ""))
                text = str(payload.get("text", ""))
                result = self.send_chat(lease_id, text)
            elif command == "disarm":
                self.disarm()
                result = self.status()
            elif command == "fault":
                reason = str(payload.get("reason", "injected-test-fault"))
                self.fail(reason)
                result = self.status()
            else:
                raise RuntimeError(f"unknown command: {command}")
            _send_json_line(conn, {"ok": True, "result": result})
        except Exception as exc:
            _send_json_line(conn, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    def _persist_status(self) -> None:
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        _atomic_json_write(STATUS_FILE, self.status(), mode=0o600)

    def _remove_control_file_if_owned(self) -> None:
        try:
            current = ControlEndpoint.load(CONTROL_FILE)
        except Exception:
            return
        if current.pid == os.getpid() and current.session_id == self.session_id:
            try:
                CONTROL_FILE.unlink()
            except FileNotFoundError:
                pass


def send_command(command: str, **payload: Any) -> dict[str, Any]:
    endpoint = ControlEndpoint.load()
    request = {"token": endpoint.token, "command": command, **payload}
    with socket.create_connection((endpoint.host, endpoint.port), timeout=1.5) as sock:
        _send_json_line(sock, request)
        response = _recv_json_line(sock)
    if not bool(response.get("ok")):
        raise RuntimeError(str(response.get("error", "supervisor command failed")))
    result = response.get("result")
    if not isinstance(result, dict):
        raise TypeError("invalid supervisor response")
    return result


def supervisor_alive() -> bool:
    try:
        send_command("status")
    except Exception:
        return False
    return True


def install_signal_handlers(supervisor: Supervisor) -> None:
    def _handler(signum: int, _frame: Any) -> None:
        supervisor.fail(f"signal-{signum}")
        supervisor.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handler)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    role = "generalist"
    if args[:1] == ["--role"] and len(args) >= 2:
        role = args[1]
    supervisor = Supervisor(role=role)
    install_signal_handlers(supervisor)
    supervisor.serve_forever()
    return 0


def _atomic_json_write(path: Path, payload: dict[str, Any], *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    try:
        os.chmod(temp, mode)
    except OSError:
        pass
    temp.replace(path)


def _recv_json_line(sock: socket.socket, *, limit: int = 64 * 1024) -> dict[str, Any]:
    data = bytearray()
    while len(data) < limit:
        chunk = sock.recv(min(4096, limit - len(data)))
        if not chunk:
            break
        data.extend(chunk)
        if b"\n" in chunk:
            break
    if len(data) >= limit:
        raise RuntimeError("control message too large")
    line = bytes(data).split(b"\n", 1)[0]
    parsed = json.loads(line.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise TypeError("control message must be a JSON object")
    return parsed


def _send_json_line(sock: socket.socket, payload: dict[str, Any]) -> None:
    sock.sendall(json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n")


if __name__ == "__main__":
    raise SystemExit(main())
