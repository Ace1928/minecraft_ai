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

from .safety import (
    FakeInputBackend,
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
    def load(cls, path: Path = CONTROL_FILE) -> "ControlEndpoint":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            host=str(raw["host"]),
            port=int(raw["port"]),
            token=str(raw["token"]),
            pid=int(raw["pid"]),
            session_id=str(raw["session_id"]),
        )


class Supervisor:
    """Independent lifecycle owner for Minecraft AI.

    Phase 0 uses FakeInputBackend only. Concrete Minecraft input backends must
    satisfy docs/SAFETY.md and can be wired in later without changing control
    ownership.
    """

    def __init__(self, *, role: str = "generalist", watchdog_interval_s: float = 0.05) -> None:
        self.role = role
        self.session_id = secrets.token_hex(16)
        self.state = SupervisorState.STOPPED
        self.backend = FakeInputBackend()
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
            if self.state == SupervisorState.SAFE_IDLE:
                return
            if self.state != SupervisorState.PAUSED:
                raise RuntimeError(f"cannot resume from {self.state}")
            validate_transition(self.state, SupervisorState.SAFE_IDLE)
            self.state = SupervisorState.SAFE_IDLE
            self._persist_status()

    def arm(self, target_instance: str) -> dict[str, Any]:
        """Create a fake motor lease for Phase-0 testing only."""
        with self._lock:
            if self.state != SupervisorState.SAFE_IDLE:
                raise RuntimeError(f"cannot arm from {self.state}")
            validate_transition(self.state, SupervisorState.ARMED)
            self.state = SupervisorState.ARMED
            lease = self.motor.issue(session_id=self.session_id, target_instance=target_instance)
            self._persist_status()
            return {
                "lease_id": lease.lease_id,
                "backend": lease.backend_id,
                "target_instance": lease.target_instance,
                "live_capable": self.backend.live_capable,
            }

    def disarm(self, reason: str = "operator-disarm") -> None:
        with self._lock:
            self.motor.revoke(reason)
            if self.state in {SupervisorState.ARMED, SupervisorState.RUNNING}:
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
            return {
                "state": self.state.value,
                "role": self.role,
                "session_id": self.session_id,
                "pid": os.getpid(),
                "backend": self.backend.backend_id,
                "live_capable": self.backend.live_capable,
                "motor_lease_active": lease is not None and not lease.expired(),
                "motor_revocation_reason": self.motor.revocation_reason,
                "held_keys": sorted(self.backend.held_keys),
                "held_buttons": sorted(self.backend.held_buttons),
                "release_count": self.backend.release_count,
                "last_fault": self.last_fault,
                "uptime_s": round((time.monotonic_ns() - self.started_monotonic_ns) / 1e9, 3),
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
            elif command == "arm-fake":
                target_instance = str(payload.get("target_instance", "fake-instance"))
                result = {"lease": self.arm(target_instance), "status": self.status()}
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
        endpoint = ControlEndpoint.load()
    except Exception:
        return False
    try:
        os.kill(endpoint.pid, 0)
    except (OSError, ProcessLookupError):
        return False
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
