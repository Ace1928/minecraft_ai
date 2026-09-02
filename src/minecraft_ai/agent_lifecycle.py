from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from platformdirs import user_data_dir, user_runtime_dir


RUNTIME_DIR = Path(user_runtime_dir("minecraft-ai"))
DATA_DIR = Path(user_data_dir("minecraft-ai"))
AGENT_FILE = RUNTIME_DIR / "agent-process.json"
AGENT_LOG = DATA_DIR / "logs" / "agent.log"


@dataclass(frozen=True)
class AgentProcess:
    pid: int
    started_ns: int
    display: str
    window_id: int
    instance_id: str
    role: str

    @classmethod
    def load(cls, path: Path = AGENT_FILE) -> AgentProcess:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("invalid agent process descriptor")
        return cls(
            pid=int(raw["pid"]),
            started_ns=int(raw["started_ns"]),
            display=str(raw["display"]),
            window_id=int(raw["window_id"]),
            instance_id=str(raw["instance_id"]),
            role=str(raw["role"]),
        )

    def persist(self, path: Path = AGENT_FILE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        staged = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        staged.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")
        try:
            staged.chmod(0o600)
        except OSError:
            pass
        staged.replace(path)


def agent_alive(process: AgentProcess | None = None) -> bool:
    current = process
    if current is None:
        try:
            current = AgentProcess.load()
        except (OSError, ValueError, TypeError, KeyError):
            return False
    return _pid_alive(current.pid)


def launch_agent_process(
    *,
    lease_id: str,
    display: str,
    window_id: int,
    instance_id: str,
    role: str,
    config_file: Path | None = None,
) -> AgentProcess:
    if agent_alive():
        raise RuntimeError("agent process is already running")
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    AGENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "minecraft_ai.agent_process",
        "--lease-id",
        lease_id,
        "--display",
        display,
        "--window-id",
        str(window_id),
        "--instance-id",
        instance_id,
        "--role",
        role,
    ]
    if config_file is not None:
        command.extend(("--config", str(config_file)))
    with AGENT_LOG.open("ab", buffering=0) as log:
        child = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=(os.name != "nt"),
        )
    process = AgentProcess(
        pid=child.pid,
        started_ns=time.monotonic_ns(),
        display=display,
        window_id=window_id,
        instance_id=instance_id,
        role=role,
    )
    process.persist()
    time.sleep(0.05)
    if child.poll() is not None:
        _remove_descriptor_if_owned(process)
        raise RuntimeError(f"agent process exited immediately; inspect {AGENT_LOG}")
    return process


def stop_agent_process(process: AgentProcess | None = None, *, timeout_s: float = 3.0) -> bool:
    current = process
    if current is None:
        try:
            current = AgentProcess.load()
        except (OSError, ValueError, TypeError, KeyError):
            return False
    if not _pid_alive(current.pid):
        _remove_descriptor_if_owned(current)
        return False
    _terminate_group(current.pid, timeout_s=timeout_s)
    _remove_descriptor_if_owned(current)
    return True


def _remove_descriptor_if_owned(process: AgentProcess) -> None:
    try:
        current = AgentProcess.load()
    except (OSError, ValueError, TypeError, KeyError):
        return
    if current.pid == process.pid and current.started_ns == process.started_ns:
        try:
            AGENT_FILE.unlink()
        except FileNotFoundError:
            pass


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        # The realtime Bedrock reference runtime is Linux; avoid POSIX-style
        # os.kill(pid, 0) semantics on Windows.
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _terminate_group(pid: int, *, timeout_s: float) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        return
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return
        time.sleep(0.05)
    try:
        os.killpg(pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass
