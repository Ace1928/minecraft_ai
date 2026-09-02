from __future__ import annotations

import json
import os
import signal
import subprocess
from pathlib import Path

from platformdirs import user_data_dir, user_runtime_dir

APP_NAME = "minecraft-ai"
DATA_DIR = Path(user_data_dir(APP_NAME))
RUNTIME_DIR = Path(user_runtime_dir(APP_NAME))
EMERGENCY_STOP_FILE = DATA_DIR / "EMERGENCY_STOP"
CONTROL_FILE = RUNTIME_DIR / "control.json"


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
    """Read only the PID from the runtime descriptor; no socket is required."""
    try:
        raw = json.loads(CONTROL_FILE.read_text(encoding="utf-8"))
        pid = int(raw["pid"])
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return pid if pid > 0 else None


def terminate_registered_supervisor() -> bool:
    """Best-effort OS-level stop that deliberately bypasses supervisor IPC."""
    pid = registered_supervisor_pid()
    if pid is None:
        return False
    if os.name == "nt":
        completed = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.returncode == 0
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return False
    except OSError:
        return False
    return True
