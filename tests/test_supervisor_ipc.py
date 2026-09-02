from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

import minecraft_ai.supervisor as supervisor_module
from minecraft_ai.supervisor import send_command, supervisor_alive


def test_supervisor_process_ipc_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_dir = tmp_path / "minecraft-ai"
    control_file = runtime_dir / "control.json"
    monkeypatch.setattr(supervisor_module, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(supervisor_module, "CONTROL_FILE", control_file)
    monkeypatch.setattr(supervisor_module, "STATUS_FILE", runtime_dir / "supervisor-state.json")
    monkeypatch.setattr(supervisor_module, "LOCK_FILE", runtime_dir / "supervisor.lock")
    env = os.environ.copy()
    env["XDG_RUNTIME_DIR"] = str(tmp_path)

    process = subprocess.Popen(
        [sys.executable, "-m", "minecraft_ai.supervisor", "--role", "builder"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not supervisor_alive():
            time.sleep(0.05)
        assert supervisor_alive()

        status = send_command("status")
        assert status["state"] == "SAFE_IDLE"
        assert status["role"] == "builder"
        assert status["live_capable"] is False

        paused = send_command("pause")
        assert paused["state"] == "PAUSED"
        assert paused["motor_lease_active"] is False

        resumed = send_command("resume")
        assert resumed["state"] == "SAFE_IDLE"

        stopped = send_command("stop")
        assert stopped["state"] == "STOPPED"

        process.wait(timeout=5.0)
        assert process.returncode == 0
        assert not supervisor_alive()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5.0)
        control_file.unlink(missing_ok=True)
