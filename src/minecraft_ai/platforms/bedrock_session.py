from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterator

from platformdirs import user_runtime_dir

from ..agent_lifecycle import _command_sha256, _linux_process_identity
from .bedrock_linux import discover_bedrock_linux_install
from .bedrock_x11 import (
    HostMonitorBinding,
    IsolationError,
    ScreenRect,
    _prepare_new_isolated_window_geometry,
    bind_host_monitor,
    find_minecraft_window,
    request_window_close,
    require_isolated_display,
)
from .weston_seat import (
    HeadlessSeatArtifact,
    build_headless_seat_module,
    require_loaded_headless_seat,
)
from .xwayland_identity import (
    XwaylandIdentity,
    capture_xwayland_identity,
    require_owned_xwayland,
)


RUNTIME_DIR = Path(user_runtime_dir("minecraft-ai"))
BEDROCK_SESSION_FILE = RUNTIME_DIR / "bedrock-session.json"
BEDROCK_LIFECYCLE_LOCK = RUNTIME_DIR / "bedrock-session.lock"
DEFAULT_BEDROCK_WIDTH = 1920
DEFAULT_BEDROCK_HEIGHT = 1080
HEADLESS_INPUT_ISOLATION = "headless-virtual-seat-v1"
_HEADLESS_FORBIDDEN_ENVIRONMENT = frozenset({
    "DISPLAY", "WAYLAND_DISPLAY", "WAYLAND_SOCKET", "WAYLAND_SERVER_SOCKET",
    "XAUTHORITY", "BOL_DISPLAY",
    "WESTON_MODULE_MAP", "WESTON_MODULE_DIR", "WESTON_XWAYLAND_PATH",
    "LD_PRELOAD", "LD_AUDIT", "LD_LIBRARY_PATH",
})
_IS_LINUX = sys.platform.startswith("linux")


def _required_int_attribute(module: object, name: str) -> int:
    value = getattr(module, name, None)
    if not isinstance(value, int):
        raise OSError(f"platform lock constant {name} is unavailable")
    return value


def _flock_descriptor(lock_module: object, fd: int, *, unlock: bool) -> None:
    flock = getattr(lock_module, "flock", None)
    if not callable(flock):
        raise OSError("POSIX file locking is unavailable")
    if unlock:
        operation = _required_int_attribute(lock_module, "LOCK_UN")
    else:
        operation = _required_int_attribute(
            lock_module, "LOCK_EX"
        ) | _required_int_attribute(lock_module, "LOCK_NB")
    flock(fd, operation)


def _set_private_descriptor_mode(fd: int) -> None:
    fchmod = getattr(os, "fchmod", None)
    if callable(fchmod):
        fchmod(fd, 0o600)


def _signal_process_group(process_group_id: int, sent_signal: int) -> None:
    kill_group = getattr(os, "killpg", None)
    if not callable(kill_group):
        raise OSError("process-group signaling is unavailable")
    kill_group(process_group_id, sent_signal)


def _open_pidfd(pid: int) -> int:
    opener = getattr(os, "pidfd_open", None)
    if not callable(opener):
        raise AttributeError("pidfd_open is unavailable")
    return int(opener(pid, 0))


def _send_pidfd_signal(pidfd: int, sent_signal: int) -> None:
    sender = getattr(signal, "pidfd_send_signal", None)
    if not callable(sender):
        raise AttributeError("pidfd_send_signal is unavailable")
    sender(pidfd, sent_signal)


@contextmanager
def bedrock_lifecycle_lock(*, wait_timeout_s: float = 0.0) -> Iterator[None]:
    """Serialize check/stop/launch/persist across CLI and recovery processes."""

    if os.name != "posix":
        raise IsolationError("managed Bedrock lifecycle locking is Linux-only")
    if wait_timeout_s < 0:
        raise ValueError("Bedrock lifecycle lock timeout cannot be negative")
    import fcntl

    BEDROCK_LIFECYCLE_LOCK.parent.mkdir(parents=True, exist_ok=True)
    handle = BEDROCK_LIFECYCLE_LOCK.open("a+b")
    try:
        deadline = time.monotonic() + wait_timeout_s
        while True:
            try:
                _flock_descriptor(fcntl, handle.fileno(), unlock=False)
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise IsolationError(
                        "another Bedrock lifecycle operation is already active"
                    ) from exc
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        try:
            _set_private_descriptor_mode(handle.fileno())
        except OSError:
            pass
        yield
    finally:
        try:
            _flock_descriptor(fcntl, handle.fileno(), unlock=True)
        except OSError:
            pass
        handle.close()


@dataclass(frozen=True)
class BedrockSession:
    display: str
    host_display: str
    xserver_pid: int
    launcher_pid: int
    width: int
    height: int
    created_ns: int
    launcher_command: tuple[str, ...]
    xserver_proc_start_ticks: int | None = None
    xserver_command_sha256: str | None = None
    launcher_proc_start_ticks: int | None = None
    launcher_command_sha256: str | None = None
    mode: str = "xephyr"
    compositor_fullscreen: bool = False
    wayland_socket: str | None = None
    compositor_log: str | None = None
    launcher_log: str | None = None
    host_monitor_name: str | None = None
    host_monitor_x: int | None = None
    host_monitor_y: int | None = None
    host_monitor_width: int | None = None
    host_monitor_height: int | None = None
    host_monitor_window_id: int | None = None
    host_monitor_bound_ns: int | None = None
    input_isolation: str = "unverified"
    input_isolation_module: str | None = None
    input_isolation_module_sha256: str | None = None
    input_isolation_source_sha256: str | None = None
    xwayland_pid: int | None = None
    xwayland_proc_start_ticks: int | None = None
    xwayland_command_sha256: str | None = None

    @classmethod
    def load(cls, path: Path | None = None) -> BedrockSession:
        selected = BEDROCK_SESSION_FILE if path is None else path
        raw = json.loads(selected.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("invalid Bedrock session descriptor")
        command = raw.get("launcher_command")
        if not isinstance(command, list) or not all(isinstance(value, str) for value in command):
            raise TypeError("invalid launcher command in Bedrock session descriptor")
        return cls(
            display=str(raw["display"]),
            host_display=str(raw["host_display"]),
            xserver_pid=int(raw["xserver_pid"]),
            launcher_pid=int(raw["launcher_pid"]),
            width=int(raw["width"]),
            height=int(raw["height"]),
            created_ns=int(raw["created_ns"]),
            launcher_command=tuple(command),
            xserver_proc_start_ticks=(
                None
                if raw.get("xserver_proc_start_ticks") is None
                else int(raw["xserver_proc_start_ticks"])
            ),
            xserver_command_sha256=(
                None
                if raw.get("xserver_command_sha256") is None
                else str(raw["xserver_command_sha256"])
            ),
            launcher_proc_start_ticks=(
                None
                if raw.get("launcher_proc_start_ticks") is None
                else int(raw["launcher_proc_start_ticks"])
            ),
            launcher_command_sha256=(
                None
                if raw.get("launcher_command_sha256") is None
                else str(raw["launcher_command_sha256"])
            ),
            mode=str(raw.get("mode", "xephyr")),
            input_isolation=str(raw.get("input_isolation", "unverified")),
            input_isolation_module=(
                None if raw.get("input_isolation_module") is None
                else str(raw["input_isolation_module"])
            ),
            input_isolation_module_sha256=(
                None if raw.get("input_isolation_module_sha256") is None
                else str(raw["input_isolation_module_sha256"])
            ),
            input_isolation_source_sha256=(
                None if raw.get("input_isolation_source_sha256") is None
                else str(raw["input_isolation_source_sha256"])
            ),
            xwayland_pid=None if raw.get("xwayland_pid") is None else int(raw["xwayland_pid"]),
            xwayland_proc_start_ticks=(
                None if raw.get("xwayland_proc_start_ticks") is None
                else int(raw["xwayland_proc_start_ticks"])
            ),
            xwayland_command_sha256=(
                None if raw.get("xwayland_command_sha256") is None
                else str(raw["xwayland_command_sha256"])
            ),
            compositor_fullscreen=bool(raw.get("compositor_fullscreen", False)),
            wayland_socket=None
            if raw.get("wayland_socket") is None
            else str(raw["wayland_socket"]),
            compositor_log=None
            if raw.get("compositor_log") is None
            else str(raw["compositor_log"]),
            launcher_log=None if raw.get("launcher_log") is None else str(raw["launcher_log"]),
            host_monitor_name=None
            if raw.get("host_monitor_name") is None
            else str(raw["host_monitor_name"]),
            host_monitor_x=None
            if raw.get("host_monitor_x") is None
            else int(raw["host_monitor_x"]),
            host_monitor_y=None
            if raw.get("host_monitor_y") is None
            else int(raw["host_monitor_y"]),
            host_monitor_width=None
            if raw.get("host_monitor_width") is None
            else int(raw["host_monitor_width"]),
            host_monitor_height=None
            if raw.get("host_monitor_height") is None
            else int(raw["host_monitor_height"]),
            host_monitor_window_id=None
            if raw.get("host_monitor_window_id") is None
            else int(raw["host_monitor_window_id"]),
            host_monitor_bound_ns=None
            if raw.get("host_monitor_bound_ns") is None
            else int(raw["host_monitor_bound_ns"]),
        )

    def persist(self, path: Path | None = None) -> None:
        selected = BEDROCK_SESSION_FILE if path is None else path
        selected.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = asdict(self)
        payload["launcher_command"] = list(self.launcher_command)
        staged = selected.with_name(f".{selected.name}.{os.getpid()}.tmp")
        staged.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        try:
            os.chmod(staged, 0o600)
        except OSError:
            pass
        staged.replace(selected)

    def find_window(self) -> int | None:
        return find_minecraft_window(
            self.display,
            host_display=self.host_display,
            allow_host=(self.mode in {"direct", "host-monitor"}),
        )

    def host_monitor_binding(self) -> HostMonitorBinding | None:
        name = self.host_monitor_name
        x = self.host_monitor_x
        y = self.host_monitor_y
        width = self.host_monitor_width
        height = self.host_monitor_height
        window_id = self.host_monitor_window_id
        bound_ns = self.host_monitor_bound_ns
        values = (name, x, y, width, height, window_id, bound_ns)
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise IsolationError("incomplete host-monitor binding in session descriptor")
        assert name is not None
        assert x is not None
        assert y is not None
        assert width is not None
        assert height is not None
        assert window_id is not None
        assert bound_ns is not None
        return HostMonitorBinding(
            display=self.display,
            output_name=name,
            monitor=ScreenRect(
                x=x,
                y=y,
                width=width,
                height=height,
            ),
            window_id=window_id,
            bound_ns=bound_ns,
        )


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _session_process_state(session: BedrockSession, *, launcher: bool) -> str:
    pid = session.launcher_pid if launcher else session.xserver_pid
    if pid <= 0:
        return "absent"
    if not _pid_alive(pid):
        return "dead"
    if not _IS_LINUX:
        return "unverifiable"
    start_ticks = (
        session.launcher_proc_start_ticks if launcher else session.xserver_proc_start_ticks
    )
    expected_digest = (
        session.launcher_command_sha256 if launcher else session.xserver_command_sha256
    )
    if start_ticks is None or not expected_digest:
        return "unverifiable"
    identity = _linux_process_identity(pid)
    if identity is None:
        return "unverifiable"
    observed_ticks, command = identity
    if launcher:
        configured_name = Path(session.launcher_command[0]).name.casefold()
        command_ok = "play" in command and any(
            Path(argument).name.casefold() == configured_name for argument in command
        )
    else:
        expected_name = "weston" if session.mode == "weston" else "xephyr"
        command_ok = bool(command) and Path(command[0]).name.casefold() == expected_name
    if (
        observed_ticks == start_ticks
        and _command_sha256(command) == expected_digest
        and command_ok
    ):
        return "verified-live"
    return "mismatch"


def _session_identity_recorded(session: BedrockSession, *, launcher: bool) -> bool:
    start_ticks = (
        session.launcher_proc_start_ticks if launcher else session.xserver_proc_start_ticks
    )
    digest = session.launcher_command_sha256 if launcher else session.xserver_command_sha256
    return (
        start_ticks is not None
        and start_ticks > 0
        and digest is not None
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest.casefold())
    )


def _required_process_identity(
    pid: int,
    *,
    label: str,
    expected_program: str | None = None,
    required_argument: str | None = None,
) -> tuple[int, str]:
    """Capture identity after a script launcher has completed its initial exec."""

    deadline = time.monotonic() + 1.0
    stable_identity: tuple[int, tuple[str, ...]] | None = None
    while True:
        identity = _linux_process_identity(pid) if _IS_LINUX else None
        if identity is not None:
            start_ticks, command = identity
            executable_name = Path(command[0]).name.casefold() if command else ""
            expected_name = expected_program.casefold() if expected_program is not None else None
            interpreter_script = (
                expected_name is not None
                and executable_name.startswith(("python", "pypy"))
                and len(command) > 1
                and Path(command[1]).name.casefold() == expected_name
            )
            program_ready = (
                expected_name is None
                or executable_name == expected_name
                or interpreter_script
            )
            argument_ready = required_argument is None or required_argument in command
            if program_ready and argument_ready:
                candidate = (start_ticks, command)
                if candidate == stable_identity:
                    return start_ticks, _command_sha256(command)
                stable_identity = candidate
            else:
                stable_identity = None
        if not _pid_alive(pid) or time.monotonic() >= deadline:
            break
        time.sleep(0.01)
    raise IsolationError(f"could not establish stable {label} process identity")


def bedrock_session_alive(session: BedrockSession | None = None) -> bool:
    current = session
    if current is None:
        try:
            current = BedrockSession.load()
        except (OSError, ValueError, TypeError, KeyError):
            return False
    if current.mode in {"direct", "host-monitor"}:
        return _session_process_state(current, launcher=True) == "verified-live"
    return (
        _session_process_state(current, launcher=False) == "verified-live"
        and _session_process_state(current, launcher=True) == "verified-live"
        and _x_socket(current.display).exists()
    )


def bedrock_session_resources_absent(session: BedrockSession) -> bool:
    """Prove there are no recorded resources before permitting a fresh launch.

    Failed combined liveness does not imply an empty session: the launcher may
    have died while its compositor or descendants still own the game. Only
    kernel-confirmed missing leaders/groups and a missing private X socket
    establish absence. Permission failures and reused PIDs remain a hold.
    """
    if not _IS_LINUX or session.mode not in {"weston", "xephyr", "direct", "host-monitor"}:
        return False
    private_display = session.mode in {"weston", "xephyr"}
    pids = (
        (session.launcher_pid, session.xserver_pid)
        if private_display else (session.launcher_pid,)
    )
    for pid in pids:
        if pid < 0:
            return False
        if pid == 0:
            continue
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        except OSError:
            return False
        else:
            return False
        try:
            _signal_process_group(pid, 0)
        except ProcessLookupError:
            pass
        except OSError:
            return False
        else:
            return False
    if private_display:
        try:
            require_isolated_display(session.display, session.host_display)
            _x_socket(session.display).lstat()
        except FileNotFoundError:
            return True
        except (OSError, IsolationError):
            return False
        return False
    return True


def require_autonomous_input_isolation(session: BedrockSession) -> None:
    """Verify the live compositor cannot import a host keyboard/pointer seat.

    A separate X display prevents outgoing XTEST from reaching the host, but
    nested Wayland/Xephyr still forward host input into that display.  Legacy
    sessions remain inspectable/stoppable; they cannot acquire motor authority
    by adopting a new descriptor label.  This is a read-only readiness check.
    """
    _require_headless_compositor_identity(session)
    if (
        session.xwayland_pid is None
        or session.xwayland_proc_start_ticks is None
        or not session.xwayland_command_sha256
        or session.xserver_proc_start_ticks is None
    ):
        raise IsolationError("managed Xwayland display ownership identity is missing")
    require_owned_xwayland(
        session.xserver_pid,
        session.display,
        XwaylandIdentity(
            session.xwayland_pid,
            session.xwayland_proc_start_ticks,
            session.xwayland_command_sha256,
        ),
        compositor_start_ticks=session.xserver_proc_start_ticks,
    )


def _require_headless_compositor_identity(session: BedrockSession) -> None:
    """Check compositor provenance before fresh geometry starts lazy Xwayland."""
    if session.mode != "weston" or session.input_isolation != HEADLESS_INPUT_ISOLATION:
        raise IsolationError(
            "autonomous input requires a verified headless Weston session without a host "
            "input seat; the existing session is preserved for observation"
        )
    require_isolated_display(session.display, session.host_display)
    if (
        not session.wayland_socket
        or not session.compositor_log
        or not session.input_isolation_module
        or not session.input_isolation_module_sha256
        or not session.input_isolation_source_sha256
        or session.compositor_fullscreen
        or _session_process_state(session, launcher=False) != "verified-live"
    ):
        raise IsolationError("headless compositor input isolation identity is unverifiable")
    identity = _linux_process_identity(session.xserver_pid)
    if identity is None:
        raise IsolationError("headless compositor input isolation identity is unavailable")
    ticks, command = identity
    if not command:
        raise IsolationError("headless compositor command is unavailable")
    expected = tuple(_weston_command(
        weston=command[0],
        wayland_socket=session.wayland_socket,
        width=session.width,
        height=session.height,
        fullscreen=False,
        compositor_log=Path(session.compositor_log),
        seat_module=Path(session.input_isolation_module),
    ))
    if (
        ticks != session.xserver_proc_start_ticks
        or _command_sha256(command) != session.xserver_command_sha256
        or command != expected
    ):
        raise IsolationError("live compositor command does not prove headless input isolation")
    _require_headless_compositor_environment(session.xserver_pid)
    require_loaded_headless_seat(session.xserver_pid, HeadlessSeatArtifact(
        session.input_isolation_module,
        session.input_isolation_module_sha256,
        session.input_isolation_source_sha256,
    ))


def _display_number(display: str) -> int:
    identity = display.split(".", 1)[0]
    if not identity.startswith(":") or not identity[1:].isdigit():
        raise IsolationError(f"unsupported local X display name: {display!r}")
    return int(identity[1:])


def _x_socket(display: str) -> Path:
    return Path("/tmp/.X11-unix") / f"X{_display_number(display)}"


def choose_free_display(*, start: int = 70, stop: int = 199) -> str:
    for number in range(start, stop + 1):
        display = f":{number}"
        if not _x_socket(display).exists():
            return display
    raise IsolationError("no free isolated X display number is available")


def _prepare_new_isolated_bedrock_geometry(session: BedrockSession) -> None:
    """Fresh-launch-only preflight, before a descriptor enables capture/attach."""
    from ..emergency import emergency_stop_latched
    from ..supervisor import operator_pause_latched

    if session.mode not in {"weston", "xephyr"}:
        raise IsolationError("geometry preparation requires a new isolated session")
    _prepare_new_isolated_window_geometry(
        session.display,
        session.host_display,
        preparation_permitted=lambda: (
            bedrock_session_alive(session)
            and not emergency_stop_latched()
            and not operator_pause_latched()
        ),
    )


def launch_xephyr_bedrock_session(
    *,
    width: int = DEFAULT_BEDROCK_WIDTH,
    height: int = DEFAULT_BEDROCK_HEIGHT,
    display: str | None = None,
    launcher_command: tuple[str, ...] | None = None,
) -> BedrockSession:
    """Launch BedrockOnLinux inside a dedicated nested X server.

    Xephyr is the conservative reference implementation because it creates a
    genuinely distinct X input namespace that is easy to validate. A faster
    compositor backend may replace it after equivalent hardware isolation tests.
    """

    if os.name != "posix" or not Path("/proc").is_dir():
        raise IsolationError("managed Bedrock isolation is currently Linux-only")
    host_display = os.environ.get("DISPLAY", "").strip()
    if not host_display:
        raise IsolationError("host DISPLAY is required to show the nested Bedrock session")
    chosen = choose_free_display() if display is None else display
    require_isolated_display(chosen, host_display)
    xephyr = shutil.which("Xephyr")
    if xephyr is None:
        raise IsolationError("Xephyr is not installed; install the Xephyr/Xserver package")

    if launcher_command is None:
        install = discover_bedrock_linux_install()
        command = install.launcher_command if install is not None else None
        command = command or shutil.which("bedrock-on-linux")
        if command is None:
            raise IsolationError("bedrock-on-linux launcher was not found")
        launcher_command = (command, "play")
    if width < 320 or height < 240:
        raise IsolationError("isolated Bedrock resolution is too small")

    xproc = subprocess.Popen(
        [
            xephyr,
            chosen,
            "-screen",
            f"{width}x{height}",
            "-resizeable",
            "-nolisten",
            "tcp",
            "-noreset",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    launcher: subprocess.Popen[bytes] | None = None
    session: BedrockSession | None = None
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if xproc.poll() is not None:
                raise IsolationError("Xephyr exited before its display became available")
            if _x_socket(chosen).exists():
                break
            time.sleep(0.05)
        else:
            raise IsolationError("timed out waiting for isolated X server")

        env = dict(os.environ)
        env["DISPLAY"] = chosen
        env.pop("WAYLAND_DISPLAY", None)
        env["XDG_SESSION_TYPE"] = "x11"
        # Prevent a user-level custom environment from intentionally selecting
        # the host display after this point. BedrockOnLinux receives the nested
        # display as its process environment and all descendants inherit it.
        env.pop("BOL_DISPLAY", None)
        launcher = subprocess.Popen(
            list(launcher_command),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        xserver_start, xserver_digest = _required_process_identity(
            xproc.pid,
            label="Xephyr",
            expected_program=Path(xephyr).name,
        )
        launcher_start, launcher_digest = _required_process_identity(
            launcher.pid,
            label="Bedrock launcher",
            expected_program=Path(launcher_command[0]).name,
            required_argument="play",
        )
        session = BedrockSession(
            display=chosen,
            host_display=host_display,
            xserver_pid=xproc.pid,
            launcher_pid=launcher.pid,
            width=width,
            height=height,
            created_ns=time.monotonic_ns(),
            launcher_command=launcher_command,
            xserver_proc_start_ticks=xserver_start,
            xserver_command_sha256=xserver_digest,
            launcher_proc_start_ticks=launcher_start,
            launcher_command_sha256=launcher_digest,
        )
        _prepare_new_isolated_bedrock_geometry(session)
        session.persist()
        return session
    except Exception as exc:
        cleanup_ok = True
        if launcher is not None:
            cleanup_ok = _terminate_spawned_process_group(launcher) and cleanup_ok
        cleanup_ok = _terminate_spawned_process_group(xproc) and cleanup_ok
        if session is not None and cleanup_ok:
            _remove_session_descriptor_if_owned(session)
        if not cleanup_ok:
            raise IsolationError("failed Xephyr launch left process cleanup unconfirmed") from exc
        raise


def launch_weston_bedrock_session(
    *,
    width: int = DEFAULT_BEDROCK_WIDTH,
    height: int = DEFAULT_BEDROCK_HEIGHT,
    fullscreen: bool = True,
    launcher_command: tuple[str, ...] | None = None,
) -> BedrockSession:
    """Launch Bedrock in headless GPU Weston with no host input transport."""
    if os.name != "posix" or not Path("/proc").is_dir():
        raise IsolationError("managed Bedrock isolation is currently Linux-only")
    host_display = os.environ.get("DISPLAY", "").strip()
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR", "").strip()
    if not runtime_dir:
        raise IsolationError("XDG_RUNTIME_DIR is required for the private Wayland socket")
    weston = shutil.which("weston")
    if weston is None:
        raise IsolationError("Weston is not installed")
    if width < 320 or height < 240:
        raise IsolationError("isolated Bedrock resolution is too small")
    if launcher_command is None:
        install = discover_bedrock_linux_install()
        command = install.launcher_command if install is not None else None
        command = command or shutil.which("bedrock-on-linux")
        if command is None:
            raise IsolationError("bedrock-on-linux launcher was not found")
        launcher_command = (command, "play")

    seat_artifact = build_headless_seat_module()
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    nonce = f"{os.getpid()}-{time.monotonic_ns()}"
    wayland_socket = f"minecraft-ai-{nonce}"
    compositor_log = RUNTIME_DIR / f"weston-{nonce}.log"
    launcher_log = RUNTIME_DIR / f"bedrock-launcher-{nonce}.log"
    compositor_env = _headless_compositor_environment()
    compositor = subprocess.Popen(
        _weston_command(
            weston=weston,
            wayland_socket=wayland_socket,
            width=width,
            height=height,
            fullscreen=fullscreen,
            compositor_log=compositor_log,
            seat_module=Path(seat_artifact.module_path),
        ),
        env=compositor_env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    launcher: subprocess.Popen[bytes] | None = None
    session: BedrockSession | None = None
    try:
        display = _wait_for_weston_xwayland(compositor, compositor_log)
        require_isolated_display(display, host_display)
        env = dict(compositor_env)
        env["DISPLAY"] = display
        env["WAYLAND_DISPLAY"] = wayland_socket
        env["XDG_SESSION_TYPE"] = "wayland"
        env["BOL_INPUT"] = "x11"
        env.pop("BOL_DISPLAY", None)
        with launcher_log.open("ab", buffering=0) as log:
            launcher = subprocess.Popen(
                list(launcher_command),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        xserver_start, xserver_digest = _required_process_identity(
            compositor.pid,
            label="Weston",
            expected_program=Path(weston).name,
        )
        launcher_start, launcher_digest = _required_process_identity(
            launcher.pid,
            label="Bedrock launcher",
            expected_program=Path(launcher_command[0]).name,
            required_argument="play",
        )
        session = BedrockSession(
            display=display,
            host_display=host_display,
            xserver_pid=compositor.pid,
            launcher_pid=launcher.pid,
            width=width,
            height=height,
            created_ns=time.monotonic_ns(),
            launcher_command=launcher_command,
            xserver_proc_start_ticks=xserver_start,
            xserver_command_sha256=xserver_digest,
            launcher_proc_start_ticks=launcher_start,
            launcher_command_sha256=launcher_digest,
            mode="weston",
            compositor_fullscreen=False,
            input_isolation=HEADLESS_INPUT_ISOLATION,
            input_isolation_module=seat_artifact.module_path,
            input_isolation_module_sha256=seat_artifact.module_sha256,
            input_isolation_source_sha256=seat_artifact.source_sha256,
            wayland_socket=wayland_socket,
            compositor_log=str(compositor_log),
            launcher_log=str(launcher_log),
        )
        _require_headless_compositor_identity(session)
        _prepare_new_isolated_bedrock_geometry(session)
        xwayland = capture_xwayland_identity(
            compositor.pid, display, compositor_start_ticks=xserver_start,
        )
        session = replace(
            session,
            xwayland_pid=xwayland.pid,
            xwayland_proc_start_ticks=xwayland.proc_start_ticks,
            xwayland_command_sha256=xwayland.command_sha256,
        )
        require_autonomous_input_isolation(session)
        session.persist()
        return session
    except Exception as exc:
        cleanup_ok = True
        if launcher is not None:
            cleanup_ok = _terminate_spawned_process_group(launcher) and cleanup_ok
        cleanup_ok = _terminate_spawned_process_group(compositor) and cleanup_ok
        if session is not None and cleanup_ok:
            _remove_session_descriptor_if_owned(session)
        if not cleanup_ok:
            raise IsolationError("failed Weston launch left process cleanup unconfirmed") from exc
        raise


def _weston_command(
    *,
    weston: str,
    wayland_socket: str,
    width: int,
    height: int,
    fullscreen: bool,
    compositor_log: Path,
    seat_module: Path,
) -> list[str]:
    """Keep the complete fixed-size HUD surface without a host input seat.

    ``fullscreen`` remains accepted for launcher API compatibility. A headless
    output always has the requested dimensions and never creates a host window.
    """
    return [
        weston,
        "--backend=headless",
        f"--socket={wayland_socket}",
        f"--width={width}",
        f"--height={height}",
        "--renderer=gl",
        "--xwayland",
        "--shell=kiosk",
        "--no-config",
        "--idle-time=0",
        f"--modules={seat_module}",
        f"--log={compositor_log}",
    ]


def _headless_compositor_environment() -> dict[str, str]:
    """Remove host transports and ambient code-loading overrides for the private server."""
    env = dict(os.environ)
    for name in _HEADLESS_FORBIDDEN_ENVIRONMENT:
        env.pop(name, None)
    return env


def _require_headless_compositor_environment(pid: int) -> None:
    """Check the actual compositor startup environment without exposing its values.

    An exact command and the expected virtual-seat mapping are insufficient if
    an inherited Weston module map or dynamic-loader override can substitute
    another backend. Do not infer a running process's environment from ours.
    """
    try:
        environment = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError as exc:
        raise IsolationError("headless compositor startup environment is unverifiable") from exc
    forbidden = {name.encode("ascii") for name in _HEADLESS_FORBIDDEN_ENVIRONMENT}
    for entry in environment.split(b"\0"):
        if not entry:
            continue
        name, separator, _value = entry.partition(b"=")
        if not name or not separator:
            raise IsolationError("headless compositor startup environment is malformed")
        if name in forbidden:
            raise IsolationError(
                "headless compositor inherited a forbidden display or code-loading override"
            )


def _wait_for_weston_xwayland(
    compositor: subprocess.Popen[bytes],
    log_path: Path,
    *,
    timeout_s: float = 10.0,
) -> str:
    pattern = re.compile(r"xserver listening on display (:\d+)")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if compositor.poll() is not None:
            detail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            raise IsolationError(f"Weston exited before Xwayland was ready: {detail}")
        try:
            detail = log_path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            detail = ""
        match = pattern.search(detail)
        if match is not None and _x_socket(match.group(1)).exists():
            return match.group(1)
        time.sleep(0.05)
    raise IsolationError(f"timed out waiting for Weston Xwayland; inspect {log_path}")


def launch_isolated_bedrock_session(
    *,
    width: int = DEFAULT_BEDROCK_WIDTH,
    height: int = DEFAULT_BEDROCK_HEIGHT,
    fullscreen: bool = True,
    launcher_command: tuple[str, ...] | None = None,
) -> BedrockSession:
    """Require headless Weston; never fall back to a host-fed input namespace."""
    return launch_weston_bedrock_session(
        width=width,
        height=height,
        fullscreen=fullscreen,
        launcher_command=launcher_command,
    )


def launch_direct_bedrock_session(
    *,
    width: int = DEFAULT_BEDROCK_WIDTH,
    height: int = DEFAULT_BEDROCK_HEIGHT,
    launcher_command: tuple[str, ...] | None = None,
) -> BedrockSession:
    """Launch a manual debug session on the host display.

    The returned mode is intentionally rejected by the autonomous run path.
    """
    if os.name != "posix" or not Path("/proc").is_dir():
        raise IsolationError("managed Bedrock execution is currently Linux-only")
    host_display = os.environ.get("DISPLAY", ":0").strip() or ":0"
    if launcher_command is None:
        install = discover_bedrock_linux_install()
        command = install.launcher_command if install is not None else None
        command = command or shutil.which("bedrock-on-linux")
        if command is None:
            raise IsolationError("bedrock-on-linux launcher was not found")
        launcher_command = (command, "play")

    env = dict(os.environ)
    env["DISPLAY"] = host_display
    launcher = subprocess.Popen(
        list(launcher_command),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    session: BedrockSession | None = None
    try:
        launcher_start, launcher_digest = _required_process_identity(
            launcher.pid,
            label="Bedrock launcher",
            expected_program=Path(launcher_command[0]).name,
            required_argument="play",
        )
        session = BedrockSession(
            display=host_display,
            host_display=host_display,
            xserver_pid=0,
            launcher_pid=launcher.pid,
            width=width,
            height=height,
            created_ns=time.monotonic_ns(),
            launcher_command=launcher_command,
            launcher_proc_start_ticks=launcher_start,
            launcher_command_sha256=launcher_digest,
            mode="direct",
        )
        session.persist()
        return session
    except Exception as exc:
        cleanup_ok = _terminate_spawned_process_group(launcher)
        if session is not None and cleanup_ok:
            _remove_session_descriptor_if_owned(session)
        if not cleanup_ok:
            raise IsolationError("failed direct launch left process cleanup unconfirmed") from exc
        raise


def bind_direct_session_to_monitor(
    session: BedrockSession,
    *,
    output_name: str,
    path: Path = BEDROCK_SESSION_FILE,
) -> BedrockSession:
    """Promote a debug session only after proving exclusive monitor occupancy."""

    if session.mode not in {"direct", "host-monitor"}:
        raise IsolationError("only a direct host-display session can bind to a monitor")
    if not bedrock_session_alive(session):
        raise IsolationError("cannot bind a stopped Bedrock session")
    window_id = wait_for_minecraft_window(session, timeout_s=30.0)
    binding = bind_host_monitor(session.display, window_id, output_name)
    bound = replace(
        session,
        mode="host-monitor",
        host_monitor_name=binding.output_name,
        host_monitor_x=binding.monitor.x,
        host_monitor_y=binding.monitor.y,
        host_monitor_width=binding.monitor.width,
        host_monitor_height=binding.monitor.height,
        host_monitor_window_id=binding.window_id,
        host_monitor_bound_ns=binding.bound_ns,
    )
    bound.persist(path)
    return bound


def wait_for_minecraft_window(
    session: BedrockSession,
    *,
    timeout_s: float = 120.0,
) -> int:
    require_isolated_display(
        session.display,
        session.host_display,
        allow_host=(session.mode in {"direct", "host-monitor"}),
    )
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not bedrock_session_alive(session):
            raise IsolationError(
                "Bedrock server/launcher process stopped while waiting for Minecraft"
            )
        window_id = session.find_window()
        if window_id is not None:
            return window_id
        time.sleep(0.25)
    raise IsolationError("timed out waiting for Minecraft window on display")


def _request_bedrock_prefix_stop(session: BedrockSession) -> bool:
    """Ask BedrockOnLinux to stop its exact Wine prefix before group fallback."""

    command = list(session.launcher_command)
    if not command:
        return False
    launcher_index = 0
    launcher_name = Path(command[launcher_index]).name.casefold()
    if launcher_name.startswith(("python", "pypy")) and len(command) > 1:
        launcher_index = 1
        launcher_name = Path(command[launcher_index]).name.casefold()
    if launcher_name != "bedrock-on-linux":
        return False
    try:
        play_index = command.index("play", launcher_index + 1)
    except ValueError:
        return False
    stop_command = [*command[:play_index], "stop"]
    try:
        completed = subprocess.run(
            stop_command,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def stop_bedrock_session(session: BedrockSession | None = None) -> None:
    current = session
    if current is None:
        try:
            current = BedrockSession.load()
        except FileNotFoundError:
            return
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise IsolationError(
                "managed Bedrock descriptor is unreadable; refusing to claim containment: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    if current.mode not in {"xephyr", "weston", "direct", "host-monitor"}:
        raise IsolationError(f"unsupported Bedrock session mode: {current.mode!r}")
    if current.mode in {"xephyr", "weston"}:
        require_isolated_display(current.display, current.host_display)
    launcher_state = _session_process_state(current, launcher=True)
    xserver_state = (
        _session_process_state(current, launcher=False)
        if current.mode in {"xephyr", "weston"}
        else "absent"
    )
    blocked = [
        label
        for label, state in (("launcher", launcher_state), ("compositor", xserver_state))
        if state in {"unverifiable", "mismatch"}
    ]
    if blocked:
        labels = ", ".join(blocked)
        raise IsolationError(
            f"refusing to stop Bedrock: {labels} process identity is unverifiable or changed"
        )

    errors: list[str] = []
    launcher_stopped = launcher_state == "dead" and not _process_group_alive(
        current.launcher_pid
    )
    if launcher_state == "verified-live":
        try:
            window_id = current.find_window()
        except (OSError, IsolationError):
            window_id = None
        if window_id is not None:
            try:
                request_window_close(
                    current.display,
                    window_id,
                    host_display=current.host_display,
                    allow_host=(current.mode in {"direct", "host-monitor"}),
                )
            except (OSError, IsolationError):
                pass
            else:
                deadline = time.monotonic() + 8.0
                while time.monotonic() < deadline:
                    leader_state = _session_process_state(current, launcher=True)
                    if leader_state == "dead" and not _process_group_alive(
                        current.launcher_pid
                    ):
                        launcher_stopped = True
                        break
                    if leader_state in {"mismatch", "unverifiable"}:
                        errors.append("launcher identity changed during graceful shutdown")
                        break
                    time.sleep(0.1)
        if not launcher_stopped and not errors and _request_bedrock_prefix_stop(current):
            deadline = time.monotonic() + 12.0
            while time.monotonic() < deadline:
                leader_state = _session_process_state(current, launcher=True)
                if leader_state == "dead" and not _process_group_alive(
                    current.launcher_pid
                ):
                    launcher_stopped = True
                    break
                if leader_state in {"mismatch", "unverifiable"}:
                    errors.append("launcher identity changed during prefix shutdown")
                    break
                time.sleep(0.1)
        if not launcher_stopped and not errors:
            leader_state = _session_process_state(current, launcher=True)
            try:
                if leader_state == "verified-live":
                    _terminate_verified_session_group(current, launcher=True)
                elif leader_state == "dead" and _process_group_alive(
                    current.launcher_pid
                ):
                    if not _session_identity_recorded(current, launcher=True):
                        raise IsolationError(
                            "refusing orphaned launcher cleanup without modern identity metadata"
                        )
                    _terminate_orphaned_process_group(
                        current.launcher_pid,
                        label="launcher",
                    )
                else:
                    raise IsolationError("launcher identity changed during shutdown")
                launcher_stopped = True
            except IsolationError as exc:
                errors.append(str(exc))
    elif launcher_state == "dead" and _process_group_alive(current.launcher_pid):
        try:
            if not _session_identity_recorded(current, launcher=True):
                raise IsolationError(
                    "refusing orphaned launcher cleanup without modern identity metadata"
                )
            _terminate_orphaned_process_group(current.launcher_pid, label="launcher")
            launcher_stopped = True
        except IsolationError as exc:
            errors.append(str(exc))

    verified_private_processes = (
        _private_display_processes(current) if xserver_state == "verified-live" else {}
    )
    dead_weston_processes = (
        _private_display_processes(current, require_wayland_nonce=True)
        if xserver_state == "dead"
        and current.mode == "weston"
        and current.wayland_socket is not None
        else {}
    )
    compositor_stopped = xserver_state == "dead" and not _process_group_alive(
        current.xserver_pid
    )
    if xserver_state == "verified-live":
        try:
            _terminate_verified_session_group(current, launcher=False)
            compositor_stopped = True
        except IsolationError as exc:
            errors.append(str(exc))
    elif xserver_state == "dead" and _process_group_alive(current.xserver_pid):
        try:
            if not _session_identity_recorded(current, launcher=False):
                raise IsolationError(
                    "refusing orphaned compositor cleanup without modern identity metadata"
                )
            _terminate_orphaned_process_group(current.xserver_pid, label="compositor")
            compositor_stopped = True
        except IsolationError as exc:
            errors.append(str(exc))

    if (
        current.mode in {"xephyr", "weston"}
        and xserver_state == "verified-live"
        and compositor_stopped
    ):
        try:
            _terminate_private_display_processes(
                current,
                expected=verified_private_processes,
            )
            launcher_stopped = True
        except IsolationError as exc:
            errors.append(str(exc))
    elif current.mode == "weston" and xserver_state == "dead" and dead_weston_processes:
        try:
            _terminate_private_display_processes(
                current,
                expected=dead_weston_processes,
                require_wayland_nonce=True,
            )
            launcher_stopped = not _process_group_alive(current.launcher_pid)
        except IsolationError as exc:
            errors.append(str(exc))
    elif (
        current.mode in {"xephyr", "weston"}
        and xserver_state == "dead"
        and _private_display_processes(current)
    ):
        errors.append("private Bedrock clients remain after their compositor leader died")

    if not launcher_stopped:
        errors.append("Bedrock launcher containment could not be confirmed stopped")

    if errors:
        raise IsolationError("; ".join(errors))
    _remove_session_descriptor_if_owned(current)


def _remove_session_descriptor_if_owned(session: BedrockSession) -> None:
    try:
        persisted = BedrockSession.load()
    except (OSError, ValueError, TypeError, KeyError):
        return
    if persisted != session:
        return
    try:
        BEDROCK_SESSION_FILE.unlink()
    except FileNotFoundError:
        pass


def _process_group_alive(group_id: int) -> bool:
    if group_id <= 0 or os.name != "posix":
        return False
    try:
        _signal_process_group(group_id, 0)
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
            process_group = int(fields[2])
        except ValueError:
            continue
        if process_group != group_id:
            continue
        found_member = True
        if fields[0] not in {"Z", "X"}:
            return True
    return not found_member


def _wait_for_process_group_exit(group_id: int, *, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not _process_group_alive(group_id):
            return True
        time.sleep(0.05)
    return not _process_group_alive(group_id)


def _terminate_spawned_process_group(process: subprocess.Popen[bytes]) -> bool:
    """Best-effort cleanup for a process created by this invocation."""

    group_id = process.pid
    if group_id <= 0 or os.name != "posix":
        return False
    try:
        _signal_process_group(group_id, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    if not _wait_for_process_group_exit(group_id, timeout_s=2.0):
        try:
            _signal_process_group(
                group_id,
                getattr(signal, "SIGKILL", signal.SIGTERM),
            )
        except (OSError, ProcessLookupError):
            pass
        _wait_for_process_group_exit(group_id, timeout_s=1.0)
    try:
        process.wait(timeout=0.2)
    except (subprocess.TimeoutExpired, ChildProcessError, OSError):
        pass
    return not _process_group_alive(group_id)


def _terminate_verified_session_group(
    session: BedrockSession,
    *,
    launcher: bool,
) -> None:
    label = "launcher" if launcher else "compositor"
    pid = session.launcher_pid if launcher else session.xserver_pid
    if _session_process_state(session, launcher=launcher) != "verified-live":
        if not _pid_alive(pid) and not _process_group_alive(pid):
            return
        raise IsolationError(f"refusing to signal {label}: process identity changed")
    try:
        _signal_process_group(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError as exc:
        raise IsolationError(f"could not stop Bedrock {label}: {exc}") from exc
    if _wait_for_process_group_exit(pid, timeout_s=3.0):
        return
    # A non-empty process group retains its identity after the leader exits, so
    # this escalation cannot cross into a newly reused numeric PID group.
    try:
        _signal_process_group(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
    except ProcessLookupError:
        return
    except OSError as exc:
        raise IsolationError(f"could not kill Bedrock {label} group: {exc}") from exc
    if not _wait_for_process_group_exit(pid, timeout_s=2.0):
        raise IsolationError(f"Bedrock {label} process group did not stop")


def _terminate_orphaned_process_group(group_id: int, *, label: str) -> None:
    """Stop descendants while their non-empty Linux PGID remains reserved."""

    if _pid_alive(group_id) or not _process_group_alive(group_id):
        raise IsolationError(f"refusing to signal {label}: orphaned group state changed")
    try:
        _signal_process_group(group_id, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError as exc:
        raise IsolationError(f"could not stop orphaned Bedrock {label}: {exc}") from exc
    if _wait_for_process_group_exit(group_id, timeout_s=3.0):
        return
    try:
        _signal_process_group(
            group_id,
            getattr(signal, "SIGKILL", signal.SIGTERM),
        )
    except ProcessLookupError:
        return
    except OSError as exc:
        raise IsolationError(f"could not kill orphaned Bedrock {label}: {exc}") from exc
    if not _wait_for_process_group_exit(group_id, timeout_s=2.0):
        raise IsolationError(f"orphaned Bedrock {label} process group did not stop")


def _private_display_processes(
    session: BedrockSession,
    *,
    require_wayland_nonce: bool = False,
) -> dict[int, int]:
    """Return exact identities for clients of this unique nested display."""

    if not _IS_LINUX or session.mode not in {"xephyr", "weston"}:
        return {}
    expected_display = os.fsencode(f"DISPLAY={session.display}")
    expected_wayland = (
        None
        if not session.wayland_socket
        else os.fsencode(f"WAYLAND_DISPLAY={session.wayland_socket}")
    )
    matches: dict[int, int] = {}
    for proc_dir in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(proc_dir.name)
        except ValueError:
            continue
        if pid in {os.getpid(), session.launcher_pid, session.xserver_pid}:
            continue
        try:
            values = proc_dir.joinpath("environ").read_bytes().split(b"\0")
        except OSError:
            continue
        if require_wayland_nonce:
            matched = expected_wayland is not None and expected_wayland in values
        else:
            matched = expected_display in values or (
                expected_wayland is not None and expected_wayland in values
            )
        if not matched:
            continue
        identity = _linux_process_identity(pid)
        if identity is not None:
            matches[pid] = identity[0]
    return matches


def _private_process_identity_matches(
    session: BedrockSession,
    pid: int,
    start_ticks: int,
    *,
    require_wayland_nonce: bool = False,
) -> bool:
    return (
        _private_display_processes(
            session,
            require_wayland_nonce=require_wayland_nonce,
        ).get(pid)
        == start_ticks
    )


def _terminate_private_display_processes(
    session: BedrockSession,
    *,
    expected: dict[int, int] | None = None,
    require_wayland_nonce: bool = False,
) -> None:
    """Stop residual clients only after proving the nested display owner."""

    def reject_unexpected(current: dict[int, int]) -> None:
        if expected is None:
            return
        unexpected = sorted(
            pid for pid, start_ticks in current.items() if expected.get(pid) != start_ticks
        )
        if unexpected:
            detail = ", ".join(str(pid) for pid in unexpected[:8])
            raise IsolationError(
                f"unexpected private Bedrock display processes appeared: {detail}"
            )

    for sig, timeout_s in (
        (signal.SIGTERM, 2.0),
        (getattr(signal, "SIGKILL", signal.SIGTERM), 1.0),
    ):
        current = _private_display_processes(
            session,
            require_wayland_nonce=require_wayland_nonce,
        )
        reject_unexpected(current)
        snapshot = (
            current
            if expected is None
            else {pid: ticks for pid, ticks in expected.items() if current.get(pid) == ticks}
        )
        if not snapshot:
            return
        for pid, start_ticks in snapshot.items():
            try:
                pidfd = _open_pidfd(pid)
            except ProcessLookupError:
                continue
            except AttributeError as exc:
                raise IsolationError(
                    "pidfd process signaling is unavailable; refusing residual-client cleanup"
                ) from exc
            except OSError as exc:
                raise IsolationError(
                    f"could not acquire private Bedrock process handle {pid}: {exc}"
                ) from exc
            try:
                # The pidfd pins this exact process object across PID reuse;
                # still revalidate the display identity after acquiring it.
                if not _private_process_identity_matches(
                    session,
                    pid,
                    start_ticks,
                    require_wayland_nonce=require_wayland_nonce,
                ):
                    continue
                try:
                    _send_pidfd_signal(pidfd, sig)
                except ProcessLookupError:
                    continue
                except (AttributeError, OSError) as exc:
                    raise IsolationError(
                        f"could not stop private Bedrock display process {pid}: {exc}"
                    ) from exc
            finally:
                os.close(pidfd)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            current = _private_display_processes(
                session,
                require_wayland_nonce=require_wayland_nonce,
            )
            reject_unexpected(current)
            if not any(current.get(pid) == ticks for pid, ticks in snapshot.items()):
                return
            time.sleep(0.05)
    current = _private_display_processes(
        session,
        require_wayland_nonce=require_wayland_nonce,
    )
    reject_unexpected(current)
    candidates = current if expected is None else expected
    survivors = sorted(pid for pid, ticks in candidates.items() if current.get(pid) == ticks)
    if survivors:
        detail = ", ".join(str(pid) for pid in survivors[:8])
        raise IsolationError(f"private Bedrock display still has processes: {detail}")
