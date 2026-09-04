from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

from platformdirs import user_data_dir, user_runtime_dir

from .agent_lifecycle import _command_sha256, _linux_process_identity

APP_NAME = "minecraft-ai"
DATA_DIR = Path(user_data_dir(APP_NAME))
RUNTIME_DIR = Path(user_runtime_dir(APP_NAME))
EMERGENCY_STOP_FILE = DATA_DIR / "EMERGENCY_STOP"
CONTROL_FILE = RUNTIME_DIR / "control.json"
_IS_LINUX = sys.platform.startswith("linux")


def emergency_stop_latched() -> bool:
    return EMERGENCY_STOP_FILE.exists()


def engage_emergency_stop(reason: str = "operator-emergency-stop") -> None:
    """Latch the stop independently of the supervisor control socket."""
    EMERGENCY_STOP_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp = EMERGENCY_STOP_FILE.with_suffix(".tmp")
    temp.write_text(reason.strip() or "operator-emergency-stop", encoding="utf-8")
    temp.replace(EMERGENCY_STOP_FILE)


def clear_emergency_stop() -> None:
    """Explicitly permit future starts again. This never starts the agent."""
    try:
        EMERGENCY_STOP_FILE.unlink()
    except FileNotFoundError:
        pass


def emergency_reason() -> str | None:
    try:
        return EMERGENCY_STOP_FILE.read_text(encoding="utf-8").strip() or "emergency-stop"
    except FileNotFoundError:
        return None


def registered_supervisor_pid() -> int | None:
    """Return a PID only when the runtime descriptor proves exact ownership."""
    identity = _registered_supervisor_identity()
    return None if identity is None else identity[0]


def _registered_supervisor_identity() -> tuple[int, int, str] | None:
    try:
        raw = json.loads(CONTROL_FILE.read_text(encoding="utf-8"))
        pid = int(raw["pid"])
        start_ticks = int(raw["proc_start_ticks"])
        expected_digest = str(raw["command_sha256"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if not _IS_LINUX or pid <= 0 or start_ticks <= 0 or not expected_digest:
        return None
    identity = _linux_process_identity(pid)
    if identity is None:
        return None
    observed_ticks, command = identity
    if len(command) < 3 or command[1:3] != ("-m", "minecraft_ai.supervisor"):
        return None
    if observed_ticks != start_ticks or _command_sha256(command) != expected_digest:
        return None
    return pid, start_ticks, expected_digest


def _supervisor_identity_matches(pid: int, start_ticks: int, expected_digest: str) -> bool:
    identity = _linux_process_identity(pid)
    if identity is None:
        return False
    observed_ticks, command = identity
    return (
        len(command) >= 3
        and command[1:3] == ("-m", "minecraft_ai.supervisor")
        and observed_ticks == start_ticks
        and _command_sha256(command) == expected_digest
    )


def terminate_registered_supervisor() -> bool:
    """Best-effort OS-level stop that deliberately bypasses supervisor IPC."""
    registered = _registered_supervisor_identity()
    if registered is None:
        return False
    pid, start_ticks, expected_digest = registered
    if not _IS_LINUX:
        return False
    try:
        pidfd = os.pidfd_open(pid, 0)
    except (AttributeError, OSError):
        return False
    try:
        # Opening a pidfd reserves this exact process object, but its identity
        # still must match the persisted descriptor before any signal is sent.
        if not _supervisor_identity_matches(pid, start_ticks, expected_digest):
            return False
        try:
            signal.pidfd_send_signal(pidfd, signal.SIGTERM)
        except ProcessLookupError:
            return not _supervisor_identity_matches(pid, start_ticks, expected_digest)
        except (AttributeError, OSError):
            return False
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if not _supervisor_identity_matches(pid, start_ticks, expected_digest):
                return True
            time.sleep(0.05)
        if not _supervisor_identity_matches(pid, start_ticks, expected_digest):
            return True
        try:
            signal.pidfd_send_signal(pidfd, getattr(signal, "SIGKILL", signal.SIGTERM))
        except ProcessLookupError:
            return not _supervisor_identity_matches(pid, start_ticks, expected_digest)
        except (AttributeError, OSError):
            return False
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if not _supervisor_identity_matches(pid, start_ticks, expected_digest):
                return True
            time.sleep(0.05)
        return not _supervisor_identity_matches(pid, start_ticks, expected_digest)
    finally:
        os.close(pidfd)
