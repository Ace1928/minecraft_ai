from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from platformdirs import user_data_dir, user_runtime_dir

import minecraft_ai.supervisor as supervisor_module
from minecraft_ai.supervisor import send_command, supervisor_alive


def test_supervisor_process_ipc_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name == "nt":
        monkeypatch.setenv("WIN_PD_OVERRIDE_LOCAL_APPDATA", str(tmp_path))
    else:
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    runtime_dir = Path(user_runtime_dir("minecraft-ai"))
    data_dir = Path(user_data_dir("minecraft-ai"))
    control_file = runtime_dir / "control.json"
    monkeypatch.setattr(supervisor_module, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(supervisor_module, "CONTROL_FILE", control_file)
    monkeypatch.setattr(supervisor_module, "STATUS_FILE", runtime_dir / "supervisor-state.json")
    monkeypatch.setattr(supervisor_module, "LOCK_FILE", runtime_dir / "supervisor.lock")
    monkeypatch.setattr(
        supervisor_module,
        "OPERATOR_PAUSE_FILE",
        data_dir / "OPERATOR_PAUSE",
    )
    env = os.environ.copy()

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
        assert (data_dir / "OPERATOR_PAUSE").exists()

        process.wait(timeout=5.0)
        assert process.returncode == 0
        assert not supervisor_alive()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5.0)
        control_file.unlink(missing_ok=True)
