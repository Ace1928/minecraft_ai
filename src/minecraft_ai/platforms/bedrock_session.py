from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from platformdirs import user_runtime_dir

from .bedrock_linux import discover_bedrock_linux_install
from .bedrock_x11 import (
    IsolationError,
    find_minecraft_window,
    request_window_close,
    require_isolated_display,
)


RUNTIME_DIR = Path(user_runtime_dir("minecraft-ai"))
BEDROCK_SESSION_FILE = RUNTIME_DIR / "bedrock-session.json"
DEFAULT_BEDROCK_WIDTH = 1920
DEFAULT_BEDROCK_HEIGHT = 1080


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
    mode: str = "xephyr"
    wayland_socket: str | None = None
    compositor_log: str | None = None
    launcher_log: str | None = None

    @classmethod
    def load(cls, path: Path = BEDROCK_SESSION_FILE) -> BedrockSession:
        raw = json.loads(path.read_text(encoding="utf-8"))
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
            mode=str(raw.get("mode", "xephyr")),
            wayland_socket=None
            if raw.get("wayland_socket") is None
            else str(raw["wayland_socket"]),
            compositor_log=None
            if raw.get("compositor_log") is None
            else str(raw["compositor_log"]),
            launcher_log=None if raw.get("launcher_log") is None else str(raw["launcher_log"]),
        )

    def persist(self, path: Path = BEDROCK_SESSION_FILE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = asdict(self)
        payload["launcher_command"] = list(self.launcher_command)
        staged = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        staged.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        try:
            os.chmod(staged, 0o600)
        except OSError:
            pass
        staged.replace(path)

    def find_window(self) -> int | None:
        return find_minecraft_window(
            self.display,
            host_display=self.host_display,
            allow_host=(self.mode == "direct"),
        )


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def bedrock_session_alive(session: BedrockSession | None = None) -> bool:
    current = session
    if current is None:
        try:
            current = BedrockSession.load()
        except (OSError, ValueError, TypeError, KeyError):
            return False
    if current.mode == "direct":
        return _pid_alive(current.launcher_pid)
    return (
        _pid_alive(current.xserver_pid)
        and _pid_alive(current.launcher_pid)
        and _x_socket(current.display).exists()
    )


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
        session = BedrockSession(
            display=chosen,
            host_display=host_display,
            xserver_pid=xproc.pid,
            launcher_pid=launcher.pid,
            width=width,
            height=height,
            created_ns=time.monotonic_ns(),
            launcher_command=launcher_command,
        )
        session.persist()
        return session
    except Exception:
        _terminate_process_group(xproc.pid)
        raise


def launch_weston_bedrock_session(
    *,
    width: int = DEFAULT_BEDROCK_WIDTH,
    height: int = DEFAULT_BEDROCK_HEIGHT,
    launcher_command: tuple[str, ...] | None = None,
) -> BedrockSession:
    """Launch Bedrock in a GPU-accelerated nested Weston/Xwayland compositor."""
    if os.name != "posix" or not Path("/proc").is_dir():
        raise IsolationError("managed Bedrock isolation is currently Linux-only")
    host_display = os.environ.get("DISPLAY", "").strip()
    host_wayland = os.environ.get("WAYLAND_DISPLAY", "").strip()
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR", "").strip()
    if not host_display or not host_wayland or not runtime_dir:
        raise IsolationError("a host Wayland session with DISPLAY is required")
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

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    nonce = f"{os.getpid()}-{time.monotonic_ns()}"
    wayland_socket = f"minecraft-ai-{nonce}"
    compositor_log = RUNTIME_DIR / f"weston-{nonce}.log"
    launcher_log = RUNTIME_DIR / f"bedrock-launcher-{nonce}.log"
    compositor = subprocess.Popen(
        [
            weston,
            "--backend=wayland",
            f"--socket={wayland_socket}",
            f"--width={width}",
            f"--height={height}",
            "--renderer=gl",
            "--xwayland",
            "--shell=kiosk",
            "--no-config",
            f"--log={compositor_log}",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        display = _wait_for_weston_xwayland(compositor, compositor_log)
        require_isolated_display(display, host_display)
        env = dict(os.environ)
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
        session = BedrockSession(
            display=display,
            host_display=host_display,
            xserver_pid=compositor.pid,
            launcher_pid=launcher.pid,
            width=width,
            height=height,
            created_ns=time.monotonic_ns(),
            launcher_command=launcher_command,
            mode="weston",
            wayland_socket=wayland_socket,
            compositor_log=str(compositor_log),
            launcher_log=str(launcher_log),
        )
        session.persist()
        return session
    except Exception:
        _terminate_process_group(compositor.pid)
        raise


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
    launcher_command: tuple[str, ...] | None = None,
) -> BedrockSession:
    """Prefer the accelerated compositor, retaining Xephyr as a compatibility fallback."""
    if shutil.which("weston") and os.environ.get("WAYLAND_DISPLAY"):
        return launch_weston_bedrock_session(
            width=width,
            height=height,
            launcher_command=launcher_command,
        )
    return launch_xephyr_bedrock_session(
        width=width,
        height=height,
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
    session = BedrockSession(
        display=host_display,
        host_display=host_display,
        xserver_pid=0,
        launcher_pid=launcher.pid,
        width=width,
        height=height,
        created_ns=time.monotonic_ns(),
        launcher_command=launcher_command,
        mode="direct",
    )
    session.persist()
    return session


def wait_for_minecraft_window(
    session: BedrockSession,
    *,
    timeout_s: float = 120.0,
) -> int:
    require_isolated_display(
        session.display,
        session.host_display,
        allow_host=(session.mode == "direct"),
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


def stop_bedrock_session(session: BedrockSession | None = None) -> None:
    current = session
    if current is None:
        try:
            current = BedrockSession.load()
        except (OSError, ValueError, TypeError, KeyError):
            return
    if current.mode not in {"xephyr", "weston", "direct"}:
        raise IsolationError(f"unsupported Bedrock session mode: {current.mode!r}")
    if current.mode in {"xephyr", "weston"}:
        require_isolated_display(current.display, current.host_display)
    window_id = current.find_window() if bedrock_session_alive(current) else None
    if window_id is not None:
        request_window_close(
            current.display,
            window_id,
            host_display=current.host_display,
            allow_host=(current.mode == "direct"),
        )
        deadline = time.monotonic() + 12.0
        while time.monotonic() < deadline and _pid_alive(current.launcher_pid):
            time.sleep(0.1)
    if _pid_alive(current.launcher_pid):
        _terminate_process_group(current.launcher_pid)
    if current.mode in {"xephyr", "weston"}:
        _terminate_process_group(current.xserver_pid)
    try:
        persisted = BedrockSession.load()
    except (OSError, ValueError, TypeError, KeyError):
        persisted = None
    if persisted is not None and persisted.created_ns == current.created_ns:
        try:
            BEDROCK_SESSION_FILE.unlink()
        except FileNotFoundError:
            pass


def _terminate_process_group(pid: int) -> None:
    if pid <= 0:
        return
    kill_group = getattr(os, "killpg", None)
    if not callable(kill_group):
        return
    try:
        kill_group(pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        return
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return
        time.sleep(0.05)
    try:
        kill_group(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
    except (OSError, ProcessLookupError):
        pass
