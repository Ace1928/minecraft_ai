from __future__ import annotations

import shutil
import subprocess
import sys


PERSISTENT_AGENT_SERVICE = "minecraft-ai-agent-live.service"


def persistent_agent_service_state() -> str:
    """Return active, inactive, or unknown without collapsing query failures."""

    if not sys.platform.startswith("linux") or shutil.which("systemctl") is None:
        return "unknown"
    try:
        completed = subprocess.run(
            ["systemctl", "--user", "is-active", PERSISTENT_AGENT_SERVICE],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    state = completed.stdout.strip().casefold()
    if completed.returncode == 0 and state == "active":
        return "active"
    if state in {"inactive", "failed"}:
        return "inactive"
    # A transition is neither a stable owner nor a confirmed stopped owner.
    # Callers use ``unknown`` fail-closed rather than racing systemd cleanup.
    if state in {"activating", "deactivating", "reloading"}:
        return "unknown"
    return "unknown"


def persistent_agent_service_active() -> bool:
    return persistent_agent_service_state() == "active"


def persistent_agent_service_load_state() -> str:
    """Return loaded, not-found, or unknown for standalone-safe resume behavior."""

    if not sys.platform.startswith("linux") or shutil.which("systemctl") is None:
        return "not-found"
    try:
        completed = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                "--property=LoadState",
                "--value",
                PERSISTENT_AGENT_SERVICE,
            ],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    state = completed.stdout.strip().casefold()
    if completed.returncode == 0 and state == "loaded":
        return "loaded"
    if state == "not-found":
        return "not-found"
    return "unknown"


def stop_persistent_agent_service() -> bool:
    """Synchronously establish that the persistent recovery owner is inactive."""

    if not sys.platform.startswith("linux") or shutil.which("systemctl") is None:
        return False
    try:
        subprocess.run(
            ["systemctl", "--user", "stop", PERSISTENT_AGENT_SERVICE],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return persistent_agent_service_state() == "inactive"


def start_persistent_agent_service() -> bool:
    """Start the installed recovery owner and confirm it reached an active state."""

    if not sys.platform.startswith("linux") or shutil.which("systemctl") is None:
        return False
    try:
        subprocess.run(
            ["systemctl", "--user", "start", PERSISTENT_AGENT_SERVICE],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return persistent_agent_service_state() == "active"
