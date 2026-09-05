from __future__ import annotations

import json
import os
import secrets
import signal
import socket
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

from platformdirs import user_data_dir, user_runtime_dir

from .agent_lifecycle import (
    AGENT_FILE,
    GRACEFUL_AGENT_STOP_TIMEOUT_S,
    _command_sha256,
    _linux_process_identity,
    _pid_alive,
    stop_agent_process,
)
from .emergency import emergency_reason, emergency_stop_latched
from .safety import (
    MAX_MOTOR_LEASE_TTL_MS,
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
LOCK_FILE = RUNTIME_DIR / "supervisor.lock"
OPERATOR_PAUSE_FILE = Path(user_data_dir(APP_NAME)) / "OPERATOR_PAUSE"
_IS_LINUX = sys.platform.startswith("linux")


def _set_private_descriptor_mode(fd: int) -> None:
    fchmod = getattr(os, "fchmod", None)
    if callable(fchmod):
        fchmod(fd, 0o600)


def operator_pause_latched() -> bool:
    """Return whether an operator explicitly requested persistent suspension.

    Supervisor state alone is intentionally insufficient: the realtime agent
    disarms its lease during normal fault cleanup, which also leaves the
    supervisor in ``PAUSED``.  A malformed marker fails closed so a damaged
    control file cannot silently re-arm gameplay.
    """
    # Existence is the authority bit. The JSON body is audit metadata only, so
    # corruption or an interrupted older writer can never turn a pause into a
    # permission to re-arm.
    return OPERATOR_PAUSE_FILE.exists()


def latch_operator_pause() -> None:
    _atomic_json_write(
        OPERATOR_PAUSE_FILE,
        {"paused": True, "requested_at_ns": time.time_ns()},
        mode=0o600,
    )


def clear_operator_pause() -> None:
    try:
        OPERATOR_PAUSE_FILE.unlink()
    except FileNotFoundError:
        pass


if sys.platform == "win32":

    def _acquire_file_lock(handle: BinaryIO) -> None:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise BlockingIOError from exc

    def _release_file_lock(handle: BinaryIO) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

else:

    def _acquire_file_lock(handle: BinaryIO) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _release_file_lock(handle: BinaryIO) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def operator_intent_lock(*, timeout_s: float = 5.0) -> Iterator[None]:
    """Serialize durable pause/stop/resume transactions across processes."""

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNTIME_DIR / "operator-intent.lock"
    handle: BinaryIO = path.open("a+b")
    deadline = time.monotonic() + timeout_s
    acquired = False
    try:
        while True:
            try:
                _acquire_file_lock(handle)
                acquired = True
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise RuntimeError("operator intent transaction is busy") from exc
                time.sleep(0.01)
        try:
            _set_private_descriptor_mode(handle.fileno())
        except OSError:
            pass
        yield
    finally:
        if acquired:
            try:
                _release_file_lock(handle)
            except OSError:
                pass
        handle.close()


def _bounded_camera_calibration_deltas(
    pitch_counts_per_degree: float,
    *,
    max_step: int = 96,
) -> tuple[int, ...]:
    """Home at the upper pitch pole, then return exactly 90 degrees to horizon."""
    if not 0.0 < pitch_counts_per_degree <= 100.0:
        raise ValueError("camera pitch_counts_per_degree must be in (0, 100]")
    if max_step < 1 or max_step > 4096:
        raise ValueError("camera calibration max_step must be in [1, 4096]")

    def split(total: int) -> list[int]:
        sign = 1 if total >= 0 else -1
        remaining = abs(total)
        parts: list[int] = []
        while remaining:
            step = min(max_step, remaining)
            parts.append(sign * step)
            remaining -= step
        return parts

    # From any legal pitch, 200 degrees upward is guaranteed to hit the -90
    # pole. Moving 90 measured degrees down from that pole establishes a
    # reproducible physical horizon independently of prior supervisor state.
    home = -round(pitch_counts_per_degree * 200.0)
    horizon = round(pitch_counts_per_degree * 90.0)
    return tuple((*split(home), *split(horizon)))


@dataclass(frozen=True)
class ControlEndpoint:
    host: str
    port: int
    token: str
    pid: int
    session_id: str
    proc_start_ticks: int | None = None
    command_sha256: str | None = None

    @classmethod
    def load(cls, path: Path | None = None) -> ControlEndpoint:
        path = CONTROL_FILE if path is None else path
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            host=str(raw["host"]),
            port=int(raw["port"]),
            token=str(raw["token"]),
            pid=int(raw["pid"]),
            session_id=str(raw["session_id"]),
            proc_start_ticks=(
                None
                if raw.get("proc_start_ticks") is None
                else int(raw["proc_start_ticks"])
            ),
            command_sha256=(
                None if raw.get("command_sha256") is None else str(raw["command_sha256"])
            ),
        )


def _supervisor_command(command: tuple[str, ...]) -> bool:
    return len(command) >= 3 and command[1:3] == ("-m", "minecraft_ai.supervisor")


def control_endpoint_process_state(endpoint: ControlEndpoint) -> str:
    """Classify the descriptor without ever treating a PID alone as identity."""
    if not _IS_LINUX:
        return "unverifiable"
    if not _pid_alive(endpoint.pid):
        return "dead"
    if endpoint.proc_start_ticks is None or not endpoint.command_sha256:
        return "unverifiable"
    identity = _linux_process_identity(endpoint.pid)
    if identity is None:
        return "unverifiable"
    start_ticks, command = identity
    if (
        start_ticks == endpoint.proc_start_ticks
        and _command_sha256(command) == endpoint.command_sha256
        and _supervisor_command(command)
    ):
        return "verified-live"
    if _supervisor_command(command):
        # A canonical supervisor with metadata drift is ambiguous ownership,
        # not proof that the PID belongs to an unrelated process.
        return "unverifiable"
    return "mismatch"


def current_control_owner_state() -> str:
    if not CONTROL_FILE.exists():
        return "absent"
    try:
        endpoint = ControlEndpoint.load(CONTROL_FILE)
    except (OSError, ValueError, TypeError, KeyError):
        return "unreadable"
    return control_endpoint_process_state(endpoint)


def remove_control_endpoint_if_owned(endpoint: ControlEndpoint) -> bool:
    """Compare the complete descriptor before removing a stale endpoint."""
    try:
        current = ControlEndpoint.load(CONTROL_FILE)
    except (OSError, ValueError, TypeError, KeyError):
        return False
    if current != endpoint:
        return False
    try:
        CONTROL_FILE.unlink()
    except FileNotFoundError:
        return False
    return True


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
        self.world_camera_pitch_units = 0
        self.world_camera_updates = 0
        self.world_camera_origin_calibrated = False
        self.world_camera_pitch_counts_per_degree: float | None = None
        self.world_camera_calibration_id: str | None = None
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
        if operator_pause_latched():
            self.last_fault = "operator-pause-latched"
            self.motor.revoke("operator-pause")
            self._persist_status()
            raise RuntimeError("operator pause is latched; explicitly resume before starting")
        self.transition(SupervisorState.STARTING)
        self.transition(SupervisorState.SAFE_IDLE)

    @staticmethod
    def _actuation_permitted() -> bool:
        return not emergency_stop_latched() and not operator_pause_latched()

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
            if self.state == SupervisorState.FAILSAFE:
                # A faulted generation cannot safely return to SAFE_IDLE. Retire
                # it so the persistent owner can launch a clean generation once
                # the resume command clears the durable operator-pause marker.
                self.stop()
                return
            if self.state != SupervisorState.PAUSED:
                raise RuntimeError(f"cannot resume from {self.state}")
            validate_transition(self.state, SupervisorState.SAFE_IDLE)
            self.state = SupervisorState.SAFE_IDLE
            self._persist_status()

    def replace_backend(
        self,
        backend: InputBackend,
        *,
        preserve_world_camera: bool = False,
    ) -> None:
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
            if not preserve_world_camera:
                self.world_camera_pitch_units = 0
                self.world_camera_updates = 0
                self.world_camera_origin_calibrated = False
                self.world_camera_pitch_counts_per_degree = None
                self.world_camera_calibration_id = None
            self._persist_status()

    def attach_bedrock_x11(
        self,
        display: str,
        window_id: int,
        *,
        allow_host: bool = False,
        host_monitor_binding: object = None,
    ) -> dict[str, Any]:
        if emergency_stop_latched():
            raise RuntimeError("emergency stop is latched")
        from .platforms.bedrock_x11 import (
            HostMonitorBinding,
            IsolatedX11InputBackend,
        )

        binding = (
            None
            if host_monitor_binding is None
            else HostMonitorBinding.from_payload(host_monitor_binding)
        )
        if allow_host and binding is None:
            # Unbound host-display play: input is targeted at the exact window
            # (XSendEvent, no focus steal), so no monitor binding is required.
            pass
        if binding is not None and not allow_host:
            raise RuntimeError("dedicated-monitor binding requires explicit host access")

        same_physical_target = (
            getattr(self.backend, "display_name", None) == display
            and getattr(self.backend, "target_window_id", None) == window_id
        )
        backend = IsolatedX11InputBackend(
            display,
            target_window_id=window_id,
            allow_host=allow_host,
            host_monitor_binding=binding,
            input_permitted=self._actuation_permitted,
        )
        try:
            self.replace_backend(
                backend,
                preserve_world_camera=same_physical_target,
            )
        except Exception:
            backend.close()
            raise
        return self.status()

    def calibrate_world_camera(
        self,
        *,
        pitch_counts_per_degree: float,
        calibration_id: str,
    ) -> dict[str, Any]:
        """Establish a physical pitch origin under a one-use mouse-only lease."""
        with self._lock:
            if self.state != SupervisorState.SAFE_IDLE:
                raise RuntimeError(f"cannot calibrate camera from {self.state}")
            if not self.backend.live_capable:
                raise RuntimeError("camera calibration requires a live isolated backend")
            target_window_id = getattr(self.backend, "target_window_id", None)
            display_name = getattr(self.backend, "display_name", None)
            if target_window_id is None or not display_name:
                raise RuntimeError("camera calibration requires a bound Bedrock target")
            if not calibration_id or len(calibration_id) > 128:
                raise ValueError("camera calibration identity is required")
            deltas = _bounded_camera_calibration_deltas(pitch_counts_per_degree)
            lease = self.motor.issue(
                session_id=self.session_id,
                target_instance=f"camera-calibration:{display_name}:{target_window_id}",
                ttl_ms=5000,
                allowed_actions=frozenset({"mouse"}),
                max_action_duration_ms=50,
            )
            try:
                if not self._actuation_permitted():
                    raise RuntimeError("actuation interlock was latched during calibration")
                # The measured sensitivity remains useful, but the old origin
                # ceases to be trustworthy as soon as calibration can move.
                # Persist this before motion so an interrupted run cannot
                # leave its previous valid origin in the status artifact.
                self.world_camera_origin_calibrated = False
                self._persist_status()
                return_phase = False
                for sequence, mouse_dy in enumerate(deltas):
                    if not self._actuation_permitted():
                        raise RuntimeError("actuation interlock was latched during calibration")
                    if mouse_dy > 0 and not return_phase:
                        # X11 and a grabbed Bedrock pointer may coalesce a burst
                        # of relative events. Let the game consume the complete
                        # home phase at its pitch pole before issuing the
                        # measured 90-degree return; otherwise only the net
                        # delta is observed and no physical origin is created.
                        time.sleep(0.1)
                        if not self._actuation_permitted():
                            raise RuntimeError(
                                "actuation interlock was latched during calibration"
                            )
                        return_phase = True
                    self.motor.apply(
                        lease.lease_id,
                        MotorAction(sequence=sequence, mouse_dy=mouse_dy),
                    )
                    # Relative events sent in one CPU burst are not a measured
                    # actuator trajectory: Xwayland/Wine/Bedrock can collapse
                    # or sample that queue at render cadence. Pace every chunk
                    # across input frames and renew the private lease during a
                    # long machine calibration.
                    if sequence % 25 == 24:
                        self.motor.renew(lease.lease_id, ttl_ms=5000)
                    time.sleep(0.02)
                    if not self._actuation_permitted():
                        raise RuntimeError("actuation interlock was latched during calibration")
                # Do not hand the lease to a policy before the return phase has
                # crossed at least one Bedrock render/input boundary.
                time.sleep(0.1)
            except Exception as exc:
                if emergency_stop_latched():
                    self.fail("emergency-stop-latched")
                elif operator_pause_latched():
                    self.motor.revoke("operator-pause")
                else:
                    self.fail(f"camera-calibration:{type(exc).__name__}")
                raise
            finally:
                self.motor.revoke(
                    "operator-pause"
                    if operator_pause_latched()
                    else "camera-calibration-complete"
                )
            # The final settle and lease cleanup are still interruptible;
            # never publish a valid origin after either latches operator intent.
            if not self._actuation_permitted():
                if emergency_stop_latched():
                    self.fail("emergency-stop-latched")
                elif operator_pause_latched():
                    self.motor.revoke("operator-pause")
                raise RuntimeError("actuation interlock was latched during calibration")
            self.world_camera_pitch_units = 0
            self.world_camera_updates = 0
            self.world_camera_origin_calibrated = True
            self.world_camera_pitch_counts_per_degree = pitch_counts_per_degree
            self.world_camera_calibration_id = calibration_id
            self._persist_status()
            return self.status()

    def arm(self, target_instance: str) -> dict[str, Any]:
        with self._lock:
            if emergency_stop_latched():
                raise RuntimeError("emergency stop is latched")
            if operator_pause_latched():
                self.motor.revoke("operator-pause")
                raise RuntimeError("operator pause is latched")
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
            if emergency_stop_latched():
                self.fail("emergency-stop-latched")
                raise RuntimeError("emergency stop is latched")
            if operator_pause_latched():
                self.disarm("operator-pause")
                raise RuntimeError("operator pause is latched")
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
            # A valid, accepted action proves that the authenticated runtime is
            # still alive. Refresh the same capability atomically with action
            # acceptance so the 20 Hz motor stream cannot starve a separately
            # queued heartbeat. Silence still expires at the fixed safety cap.
            self.motor.apply(
                lease_id,
                action,
                accepted_action_ttl_ms=MAX_MOTOR_LEASE_TTL_MS,
            )
            if action.camera_semantics == "world" and (action.mouse_dx or action.mouse_dy):
                self.world_camera_pitch_units += action.mouse_dy
                self.world_camera_updates += 1
            return {
                "accepted_sequence": action.sequence,
                # This is captured only after the scoped backend accepts the
                # action, so trajectory latency and action labels refer to the
                # physical supervisor boundary rather than policy intent.
                "accepted_monotonic_ns": time.monotonic_ns(),
                "lease_active": self.motor.lease is not None,
                # Return the camera state from this same locked acceptance
                # transaction. The agent can rebind a policy action that was
                # filtered or replaced without racing a second status call.
                "world_camera": {
                    "estimated_pitch_units": self.world_camera_pitch_units,
                    "accepted_updates": self.world_camera_updates,
                    "origin_calibrated": self.world_camera_origin_calibrated,
                    "pitch_counts_per_degree": self.world_camera_pitch_counts_per_degree,
                    "calibration_id": self.world_camera_calibration_id,
                },
            }

    def release_inputs(self, lease_id: str) -> dict[str, Any]:
        """Neutralize current input while preserving the scoped runtime lease."""
        with self._lock:
            self._require_running_lease(lease_id)
            self.motor.release_inputs(lease_id)
            self._persist_status()
            return {
                "released": True,
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
                actuator(text, input_permitted=self._actuation_permitted)
                if not self._actuation_permitted():
                    raise RuntimeError("actuation interlock was latched during chat")
                self.motor.renew(lease_id, ttl_ms=3000)
            except Exception as exc:
                if emergency_stop_latched():
                    self.fail("emergency-stop-latched")
                elif operator_pause_latched():
                    self.disarm("operator-pause")
                else:
                    self.fail(f"chat-backend-fault:{type(exc).__name__}")
                raise
            return {"sent": True, "characters": len(text)}

    def _require_running_lease(self, lease_id: str) -> None:
        if emergency_stop_latched():
            self.fail("emergency-stop-latched")
            raise RuntimeError("emergency stop is latched")
        if operator_pause_latched():
            self.disarm("operator-pause")
            raise RuntimeError("operator pause is latched")
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
                raise RuntimeError("emergency stop is latched")
            if operator_pause_latched():
                self.disarm("operator-pause")
                raise RuntimeError("operator pause is latched")
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
            effective_reason = (
                "operator-pause"
                if self.state == SupervisorState.PAUSED and operator_pause_latched()
                else reason
            )
            self.motor.revoke(effective_reason)
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
            input_window_id = getattr(self.backend, "input_window_id", None)
            return {
                "state": self.state.value,
                "role": self.role,
                "session_id": self.session_id,
                "pid": os.getpid(),
                "agent_reload_resume_supported": True,
                "backend": self.backend.backend_id,
                "live_capable": self.backend.live_capable,
                "motor_lease_active": lease is not None and not lease.expired(),
                "motor_lease_id": lease.lease_id if lease is not None else None,
                "motor_target_instance": lease.target_instance if lease is not None else None,
                "motor_revocation_reason": self.motor.revocation_reason,
                "held_keys": held_keys,
                "held_buttons": held_buttons,
                "release_count": release_count,
                "input_window_id": input_window_id,
                "world_camera": {
                    "estimated_pitch_units": self.world_camera_pitch_units,
                    "accepted_updates": self.world_camera_updates,
                    "origin_calibrated": self.world_camera_origin_calibrated,
                    "pitch_counts_per_degree": (self.world_camera_pitch_counts_per_degree),
                    "calibration_id": self.world_camera_calibration_id,
                },
                "last_fault": self.last_fault,
                "emergency_stop_latched": emergency_stop_latched(),
                "operator_pause_latched": operator_pause_latched(),
                "uptime_s": round(
                    (time.monotonic_ns() - self.started_monotonic_ns) / 1e9,
                    3,
                ),
            }

    def serve_forever(self) -> None:
        with _exclusive_runtime_lock():
            self._serve_forever_owned()

    def _serve_forever_owned(self) -> None:
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        self.start()
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(8)
        server.settimeout(self.watchdog_interval_s)
        self._server = server
        host, port = server.getsockname()
        identity = _linux_process_identity(os.getpid()) if _IS_LINUX else None
        if _IS_LINUX and (identity is None or not _supervisor_command(identity[1])):
            raise RuntimeError("could not establish supervisor process identity")
        endpoint = ControlEndpoint(
            host=str(host),
            port=int(port),
            token=secrets.token_urlsafe(32),
            pid=os.getpid(),
            session_id=self.session_id,
            proc_start_ticks=None if identity is None else identity[0],
            command_sha256=None if identity is None else _command_sha256(identity[1]),
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
                if operator_pause_latched() and (
                    self.state != SupervisorState.PAUSED or self.motor.lease is not None
                ):
                    self.disarm("operator-pause")
                if self.motor.check_expiry():
                    self.fail("motor-lease-watchdog-expired")
                try:
                    conn, _ = server.accept()
                except TimeoutError:
                    continue
                # Handle each command on its own daemon thread. The motor loop
                # floods motor-action commands at 20 Hz; servicing them serially
                # in this accept loop starved the heartbeat's renew, so the lease
                # lapsed and the watchdog expired it. Motor/lease state is
                # protected by the supervisor lock, so concurrent handling is
                # safe: the renew, motor-action, and status planes interleave.
                conn.settimeout(1.0)
                threading.Thread(
                    target=self._handle_connection,
                    args=(conn,),
                    name="minecraft-ai-cmd",
                    daemon=True,
                ).start()
        finally:
            self.motor.revoke("supervisor-exit")
            try:
                server.close()
            finally:
                close = getattr(self.backend, "close", None)
                if callable(close):
                    close()
                if self.state != SupervisorState.STOPPED:
                    self.state = SupervisorState.STOPPED
                self._persist_status()
                self._remove_control_file_if_owned()

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
                with operator_intent_lock():
                    pause_persisted = True
                    try:
                        latch_operator_pause()
                    except OSError:
                        pause_persisted = False
                    # Revoke actuation first, then leave the supervisor lock
                    # available while SIGTERM cleanup disarms its lease and
                    # seals the trajectory/learning buffers.
                    self.pause()
                    stop_agent_process(timeout_s=GRACEFUL_AGENT_STOP_TIMEOUT_S)
                    result = self.status()
                    result["operator_pause_persisted"] = pause_persisted
                    result["agent_containment_confirmed"] = not AGENT_FILE.exists()
            elif command == "resume":
                with operator_intent_lock(), self._lock:
                    self.resume()
                    clear_operator_pause()
                    result = self.status()
            elif command == "resume-for-agent-reload":
                # Cleanup pause is not renewed operator permission. Admit only
                # this unarmed generation, under the same lock as durable intent.
                with operator_intent_lock(), self._lock:
                    requested_session = payload.get("session_id")
                    if (
                        not isinstance(requested_session, str)
                        or requested_session != self.session_id
                    ):
                        raise RuntimeError("agent reload supervisor session mismatch")
                    if self.state != SupervisorState.PAUSED:
                        raise RuntimeError(f"cannot resume agent reload from {self.state}")
                    if self.motor.lease is not None:
                        raise RuntimeError("agent reload requires a revoked motor lease")
                    if AGENT_FILE.exists() or AGENT_FILE.is_symlink():
                        raise RuntimeError("agent reload requires confirmed agent retirement")
                    if emergency_stop_latched():
                        raise RuntimeError("emergency stop is latched")
                    if operator_pause_latched():
                        raise RuntimeError("operator pause is latched")
                    # Do not call resume(): its FAILSAFE recovery may stop this
                    # supervisor, and its IPC wrapper clears persistent intent.
                    validate_transition(self.state, SupervisorState.SAFE_IDLE)
                    self.state = SupervisorState.SAFE_IDLE
                    self._persist_status()
                    result = self.status()
            elif command == "stop":
                with operator_intent_lock():
                    pause_persisted = True
                    if bool(payload.get("persistent_intent", True)):
                        try:
                            latch_operator_pause()
                        except OSError:
                            pause_persisted = False
                    # Keep the control endpoint alive until the agent has had
                    # a chance to disarm and flush its durable state. The
                    # supervisor itself retires only after that bounded wait.
                    self.pause()
                    stop_agent_process(timeout_s=GRACEFUL_AGENT_STOP_TIMEOUT_S)
                    self.stop()
                    result = self.status()
                    result["operator_pause_persisted"] = pause_persisted
                    result["agent_containment_confirmed"] = not AGENT_FILE.exists()
            elif command == "attach-bedrock-x11":
                display = str(payload.get("display", ""))
                window_id = int(payload.get("window_id", 0))
                allow_host = bool(payload.get("allow_host", False))
                result = self.attach_bedrock_x11(
                    display,
                    window_id,
                    allow_host=allow_host,
                    host_monitor_binding=payload.get("host_monitor_binding"),
                )
            elif command == "calibrate-world-camera":
                pitch_counts_per_degree = float(payload.get("pitch_counts_per_degree", 0.0))
                calibration_id = str(payload.get("calibration_id", ""))
                result = self.calibrate_world_camera(
                    pitch_counts_per_degree=pitch_counts_per_degree,
                    calibration_id=calibration_id,
                )
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
            elif command == "release-inputs":
                lease_id = str(payload.get("lease_id", ""))
                result = self.release_inputs(lease_id)
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
            try:
                _send_json_line(conn, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            except Exception:
                pass
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _persist_status(self) -> None:
        # Pure in-process Supervisor instances (including unit tests) do not own
        # public runtime state. A serving process may publish only while its
        # exact PID/session endpoint is still the registered owner, preventing
        # an older supervisor from clobbering a newer live session on exit.
        if not self._owns_control_file():
            return
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        _atomic_json_write(STATUS_FILE, self.status(), mode=0o600)

    def _owns_control_file(self) -> bool:
        endpoint = self._endpoint
        if endpoint is None:
            return False
        try:
            current = ControlEndpoint.load(CONTROL_FILE)
        except Exception:
            return False
        return current == endpoint

    def _remove_control_file_if_owned(self) -> None:
        endpoint = self._endpoint
        if endpoint is not None:
            remove_control_endpoint_if_owned(endpoint)


@contextmanager
def _exclusive_runtime_lock() -> Iterator[None]:
    """Permit exactly one supervisor process to own the public control plane."""
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    handle: BinaryIO = LOCK_FILE.open("a+b")
    acquired = False
    try:
        try:
            _acquire_file_lock(handle)
            acquired = True
        except BlockingIOError as exc:
            raise RuntimeError("another minecraft-ai supervisor already owns the runtime") from exc
        yield
    finally:
        try:
            if acquired:
                _release_file_lock(handle)
        finally:
            handle.close()


def send_command(
    command: str,
    *,
    timeout_s: float = 1.5,
    **payload: Any,
) -> dict[str, Any]:
    if not 0.05 <= timeout_s <= 30.0:
        raise ValueError("supervisor command timeout must be in [0.05, 30] seconds")
    endpoint = ControlEndpoint.load()
    if _IS_LINUX and control_endpoint_process_state(endpoint) != "verified-live":
        raise RuntimeError("supervisor control endpoint does not match a live owned process")
    request = {"token": endpoint.token, "command": command, **payload}
    with socket.create_connection((endpoint.host, endpoint.port), timeout=timeout_s) as sock:
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
