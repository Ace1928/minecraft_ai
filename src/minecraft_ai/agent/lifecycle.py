from __future__ import annotations

import hashlib
import json
import math
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
_IS_LINUX = sys.platform.startswith("linux")
GRACEFUL_AGENT_STOP_TIMEOUT_S = 25.0


def _signal_process_group(process_group_id: int, sent_signal: int) -> None:
    """Call the optional POSIX process-group API without platform stub errors."""

    kill_group = getattr(os, "killpg", None)
    if not callable(kill_group):
        raise OSError("process-group signaling is unavailable")
    kill_group(process_group_id, sent_signal)


@dataclass(frozen=True)
class AgentProcess:
    pid: int
    started_ns: int
    display: str
    window_id: int
    instance_id: str
    role: str
    allow_host_capture: bool = False
    capture_source: str = "pipewire"
    proc_start_ticks: int | None = None
    command_sha256: str | None = None

    @classmethod
    def load(cls, path: Path | None = None) -> AgentProcess:
        selected = AGENT_FILE if path is None else path
        raw = json.loads(selected.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("invalid agent process descriptor")
        return cls(
            pid=int(raw["pid"]),
            started_ns=int(raw["started_ns"]),
            display=str(raw["display"]),
            window_id=int(raw["window_id"]),
            instance_id=str(raw["instance_id"]),
            role=str(raw["role"]),
            allow_host_capture=bool(raw.get("allow_host_capture", False)),
            capture_source=str(raw.get("capture_source", "pipewire")),
            proc_start_ticks=(
                None
                if raw.get("proc_start_ticks") is None
                else int(raw["proc_start_ticks"])
            ),
            command_sha256=(
                None if raw.get("command_sha256") is None else str(raw["command_sha256"])
            ),
        )

    def persist(self, path: Path | None = None) -> None:
        selected = AGENT_FILE if path is None else path
        selected.parent.mkdir(parents=True, exist_ok=True)
        staged = selected.with_name(f".{selected.name}.{os.getpid()}.tmp")
        staged.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")
        try:
            staged.chmod(0o600)
        except OSError:
            pass
        staged.replace(selected)


def agent_alive(process: AgentProcess | None = None) -> bool:
    current = process
    if current is None:
        try:
            current = AgentProcess.load()
        except (OSError, ValueError, TypeError, KeyError):
            return False
    state = _agent_process_state(current)
    if state == "verified-live":
        return True
    if state in {"dead", "mismatch"} and not _process_group_alive(current.pid):
        _remove_descriptor_if_owned(current)
    return False


def launch_agent_process(
    *,
    lease_id: str,
    display: str,
    window_id: int,
    instance_id: str,
    role: str,
    config_file: Path | None = None,
    allow_host_capture: bool = False,
    capture_source: str = "pipewire",
) -> AgentProcess:
    try:
        existing = AgentProcess.load()
    except FileNotFoundError:
        existing = None
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise RuntimeError(
            f"existing agent process descriptor is unreadable; refusing replacement: {exc}"
        ) from exc
    if existing is not None:
        state = _agent_process_state(existing)
        if state == "verified-live":
            raise RuntimeError("agent process is already running")
        if state == "unverifiable" or _process_group_alive(existing.pid):
            raise RuntimeError(
                "existing agent ownership is unverifiable; refusing duplicate launch"
            )
        _remove_descriptor_if_owned(existing)
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
    if allow_host_capture:
        command.append("--allow-host-capture")
    command.extend(("--capture-source", capture_source))
    with AGENT_LOG.open("ab", buffering=0) as log:
        child = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=(os.name != "nt"),
        )
    process: AgentProcess | None = None
    try:
        command_sha256 = _command_sha256(tuple(command))
        identity = None
        if _IS_LINUX:
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                candidate = _linux_process_identity(child.pid)
                if candidate is not None and _command_sha256(candidate[1]) == command_sha256:
                    identity = candidate
                    break
                if child.poll() is not None:
                    break
                time.sleep(0.02)
        proc_start_ticks = None if identity is None else identity[0]
        process = AgentProcess(
            pid=child.pid,
            started_ns=time.monotonic_ns(),
            display=display,
            window_id=window_id,
            instance_id=instance_id,
            role=role,
            allow_host_capture=allow_host_capture,
            capture_source=capture_source,
            proc_start_ticks=proc_start_ticks,
            command_sha256=command_sha256,
        )
        if _IS_LINUX and (
            identity is None
            or _command_sha256(identity[1]) != command_sha256
            or not _command_matches_descriptor(identity[1], process)
        ):
            raise RuntimeError("could not establish agent process identity after launch")
        process.persist()
        time.sleep(0.05)
        if child.poll() is not None:
            raise RuntimeError(f"agent process exited immediately; inspect {AGENT_LOG}")
        return process
    except Exception as exc:
        cleanup_ok = _terminate_spawned_agent_group(child)
        if process is not None and cleanup_ok:
            _remove_descriptor_if_owned(process)
        if not cleanup_ok:
            raise RuntimeError("failed agent launch left process cleanup unconfirmed") from exc
        raise


def stop_agent_process(
    process: AgentProcess | None = None,
    *,
    timeout_s: float = GRACEFUL_AGENT_STOP_TIMEOUT_S,
    planned: bool = False,
) -> bool:
    """Stop within one wait budget, optionally letting the parent drain workers.

    Emergency and orphan cleanup remain group-first. A planned stop sends TERM
    only to the verified parent; its workers are contained only if draining
    exceeds the grace portion of the same total timeout.
    """

    if not math.isfinite(timeout_s) or timeout_s < 0:
        raise ValueError("agent stop timeout must be finite and nonnegative")
    current = process
    if current is None:
        try:
            current = AgentProcess.load()
        except (OSError, ValueError, TypeError, KeyError):
            return False
    state = _agent_process_state(current)
    original_zombie = state == "unverifiable" and _original_agent_zombie(current)
    if state == "unverifiable" and not original_zombie:
        return False
    if state == "mismatch":
        _remove_descriptor_if_owned(current)
        return False
    if state == "dead" or original_zombie:
        if not _process_group_alive(current.pid):
            _remove_descriptor_if_owned(current)
            return False
        if not _descriptor_has_process_identity(current):
            return False
        stopped = _terminate_orphaned_group(current, timeout_s=timeout_s)
    else:
        stopped = _terminate_group(current, timeout_s=timeout_s, planned=planned)
    if stopped and not _process_group_alive(current.pid):
        _remove_descriptor_if_owned(current)
        return True
    return False


def _remove_descriptor_if_owned(process: AgentProcess) -> None:
    try:
        current = AgentProcess.load()
    except (OSError, ValueError, TypeError, KeyError):
        return
    if current == process:
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
    except PermissionError:
        # A permissions failure is not evidence that the recorded leader died.
        return True
    except OSError:
        return False
    return True


def _agent_identity_matches(process: AgentProcess) -> bool:
    return _agent_process_state(process) == "verified-live"


def _descriptor_has_process_identity(process: AgentProcess) -> bool:
    digest = process.command_sha256
    return bool(
        _IS_LINUX
        and process.proc_start_ticks is not None
        and process.proc_start_ticks > 0
        and digest is not None
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    )


def _agent_process_state(process: AgentProcess) -> str:
    if not _pid_alive(process.pid):
        return "dead"
    if not _IS_LINUX:
        return "unverifiable"
    if not _descriptor_has_process_identity(process):
        return "unverifiable"
    identity = _linux_process_identity(process.pid)
    if identity is None:
        return "unverifiable"
    start_ticks, command = identity
    if (
        start_ticks == process.proc_start_ticks
        and _command_sha256(command) == process.command_sha256
        and _command_matches_descriptor(command, process)
    ):
        return "verified-live"
    if len(command) >= 3 and command[1:3] == ("-m", "minecraft_ai.agent_process"):
        # This is still an agent process, but altered/stale metadata cannot
        # prove whether it is the recorded owner or a replacement generation.
        # Keep the descriptor and fail closed instead of reporting PID reuse.
        return "unverifiable"
    return "mismatch"


def _linux_process_identity(pid: int) -> tuple[int, tuple[str, ...]] | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        command_raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (OSError, UnicodeError):
        return None
    close_paren = stat.rfind(")")
    if close_paren < 0:
        return None
    # Fields after the process name begin at proc(5) field 3 (state), so
    # zero-based index 19 is field 22 (starttime in clock ticks since boot).
    fields = stat[close_paren + 1 :].split()
    if len(fields) <= 19 or not command_raw:
        return None
    try:
        start_ticks = int(fields[19])
    except ValueError:
        return None
    command = tuple(
        os.fsdecode(part) for part in command_raw.rstrip(b"\0").split(b"\0") if part
    )
    if start_ticks <= 0 or not command:
        return None
    return start_ticks, command


def _command_sha256(command: tuple[str, ...]) -> str:
    encoded = b"\0".join(os.fsencode(argument) for argument in command)
    return hashlib.sha256(encoded).hexdigest()


def _command_matches_descriptor(command: tuple[str, ...], process: AgentProcess) -> bool:
    if len(command) < 3 or command[1:3] != ("-m", "minecraft_ai.agent_process"):
        return False
    expected_options = {
        "--display": process.display,
        "--window-id": str(process.window_id),
        "--instance-id": process.instance_id,
        "--role": process.role,
        "--capture-source": process.capture_source,
    }
    for option, expected in expected_options.items():
        positions = [index for index, item in enumerate(command) if item == option]
        if len(positions) != 1 or positions[0] + 1 >= len(command):
            return False
        if command[positions[0] + 1] != expected:
            return False
    lease_positions = [index for index, item in enumerate(command) if item == "--lease-id"]
    if len(lease_positions) != 1 or lease_positions[0] + 1 >= len(command):
        return False
    if not command[lease_positions[0] + 1]:
        return False
    return ("--allow-host-capture" in command) == process.allow_host_capture


def _original_agent_zombie(process: AgentProcess) -> bool:
    """Recognize the recorded dead leader when /proc/cmdline is already empty."""

    if not _descriptor_has_process_identity(process):
        return False
    try:
        stat = Path(f"/proc/{process.pid}/stat").read_text(encoding="utf-8")
        close_paren = stat.rfind(")")
        if close_paren < 0:
            return False
        fields = stat[close_paren + 1 :].split()
        return (
            len(fields) > 19
            and fields[0] in {"Z", "X"}
            and int(fields[2]) == process.pid
            and int(fields[3]) == process.pid
            and int(fields[19]) == process.proc_start_ticks
        )
    except (OSError, UnicodeError, ValueError):
        return False


def _group_signal_authorized(process: AgentProcess) -> bool:
    if not _descriptor_has_process_identity(process):
        return False
    state = _agent_process_state(process)
    return state in {"verified-live", "dead"} or (
        state == "unverifiable" and _original_agent_zombie(process)
    )


def _wait_for_group_exit(process_group_id: int, *, deadline: float) -> bool:
    while _process_group_alive(process_group_id):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.05, remaining))
    return True


def _finish_group_stop(process: AgentProcess, *, deadline: float, grace_deadline: float) -> bool:
    if _wait_for_group_exit(process.pid, deadline=grace_deadline):
        return True
    # The recorded leader may have exited while its workers drained. A reused
    # or uninspectable live PID is never authority to kill the retained group.
    if not _group_signal_authorized(process):
        return False
    try:
        _signal_process_group(process.pid, getattr(signal, "SIGKILL", signal.SIGTERM))
    except ProcessLookupError:
        return not _process_group_alive(process.pid)
    except OSError:
        return False
    return _wait_for_group_exit(process.pid, deadline=deadline)


def _terminate_group(process: AgentProcess, *, timeout_s: float, planned: bool = False) -> bool:
    deadline = time.monotonic() + timeout_s
    grace_deadline = deadline - min(1.0, timeout_s / 4)
    if not _agent_identity_matches(process):
        return False
    kill_group = getattr(os, "killpg", None)
    if not callable(kill_group):
        return False
    try:
        if planned:
            os.kill(process.pid, signal.SIGTERM)
        else:
            kill_group(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        # The identity-checked parent may finish before TERM is delivered.
        # Retained workers still get their bounded drain interval.
        pass
    except OSError:
        return False
    return _finish_group_stop(process, deadline=deadline, grace_deadline=grace_deadline)


def _terminate_orphaned_group(process: AgentProcess, *, timeout_s: float) -> bool:
    """Boundedly stop members after their recorded Linux group leader exited."""

    deadline = time.monotonic() + timeout_s
    grace_deadline = deadline - min(1.0, timeout_s / 4)
    if not _IS_LINUX or process.pid <= 0 or not _process_group_alive(process.pid):
        return False
    if not _group_signal_authorized(process):
        return False
    try:
        _signal_process_group(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return not _process_group_alive(process.pid)
    except OSError:
        return False
    return _finish_group_stop(process, deadline=deadline, grace_deadline=grace_deadline)


def _process_group_alive(process_group_id: int) -> bool:
    if process_group_id <= 0 or os.name != "posix":
        return False
    try:
        _signal_process_group(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    if not _IS_LINUX:
        return True
    found_member = False
    try:
        entries = tuple(Path("/proc").iterdir())
    except OSError:
        return True
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            stat = entry.joinpath("stat").read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        close_paren = stat.rfind(")")
        if close_paren < 0:
            continue
        fields = stat[close_paren + 1 :].split()
        if len(fields) <= 2:
            continue
        try:
            group_id = int(fields[2])
        except ValueError:
            continue
        if group_id != process_group_id:
            continue
        found_member = True
        if fields[0] not in {"Z", "X"}:
            return True
    return not found_member


def _terminate_spawned_agent_group(child: subprocess.Popen[bytes]) -> bool:
    """Best-effort cleanup backed by the Popen handle owned by this launch."""

    if os.name == "posix":
        try:
            _signal_process_group(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            return not _process_group_alive(child.pid)
        except OSError:
            return False
    else:
        try:
            child.terminate()
        except OSError:
            return False

    def contained() -> bool:
        if os.name == "posix":
            return not _process_group_alive(child.pid)
        return child.poll() is not None

    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        child.poll()
        if contained():
            try:
                child.wait(timeout=0.1)
            except (subprocess.TimeoutExpired, ChildProcessError, OSError):
                pass
            return True
        time.sleep(0.05)

    if os.name == "posix":
        try:
            _signal_process_group(
                child.pid,
                getattr(signal, "SIGKILL", signal.SIGTERM),
            )
        except ProcessLookupError:
            pass
        except OSError:
            return False
    else:
        try:
            child.kill()
        except OSError:
            return False
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        child.poll()
        if contained():
            try:
                child.wait(timeout=0.1)
            except (subprocess.TimeoutExpired, ChildProcessError, OSError):
                pass
            return True
        time.sleep(0.05)
    return contained()
