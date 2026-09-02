from __future__ import annotations

import subprocess
import sys
import time

from minecraft_ai.supervisor import CONTROL_FILE, send_command, supervisor_alive


def test_supervisor_process_ipc_lifecycle() -> None:
    try:
        CONTROL_FILE.unlink()
    except FileNotFoundError:
        pass

    process = subprocess.Popen(
        [sys.executable, "-m", "minecraft_ai.supervisor", "--role", "builder"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
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
        try:
            CONTROL_FILE.unlink()
        except FileNotFoundError:
            pass
