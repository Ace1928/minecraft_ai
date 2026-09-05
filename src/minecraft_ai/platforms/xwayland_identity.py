"""Read-only binding of a private X display to its managed Xwayland process."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .bedrock_x11 import IsolationError


PROC_ROOT = Path("/proc")
MAX_CHILDREN = 64
MAX_FDS = 512
MAX_UNIX_TABLE_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class XwaylandIdentity:
    pid: int
    proc_start_ticks: int
    command_sha256: str


@dataclass(frozen=True)
class _Process:
    parent_pid: int
    start_ticks: int
    command: tuple[str, ...]

    @property
    def command_sha256(self) -> str:
        return hashlib.sha256(b"\0".join(os.fsencode(arg) for arg in self.command)).hexdigest()


def _read(path: Path, limit: int) -> bytes:
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise IsolationError("Xwayland ownership observation exceeded its bounded size")
    return data


def _stat(pid: int) -> tuple[int, int]:
    data = _read(PROC_ROOT / str(pid) / "stat", 8192).decode("ascii")
    fields = data.rsplit(")", 1)[1].split()
    if fields[0] in {"Z", "X"}:
        raise IsolationError("managed Xwayland/compositor process has exited")
    return int(fields[1]), int(fields[19])


def _process(pid: int) -> _Process:
    if pid <= 0:
        raise IsolationError("managed Xwayland process identity is missing")
    parent, ticks = _stat(pid)
    raw = _read(PROC_ROOT / str(pid) / "cmdline", 65536)
    command = tuple(os.fsdecode(arg) for arg in raw.rstrip(b"\0").split(b"\0"))
    if not raw or not command[0] or ticks <= 0 or _stat(pid) != (parent, ticks):
        raise IsolationError("managed Xwayland process identity changed during observation")
    return _Process(parent, ticks, command)


def _display_number(display: str) -> str:
    match = re.fullmatch(r":(0|[1-9][0-9]*)(?:\.[0-9]+)?", display)
    if match is None:
        raise IsolationError("Xwayland ownership requires an exact local X display")
    return match.group(1)


def _require_safe_command(command: tuple[str, ...]) -> None:
    """Do not enroll a weakened initial command merely because it stays unchanged.

    Weston 13 uses Xwayland's deprecated ``-listen <fd>`` spelling. Numeric
    inherited descriptors are not the generic X server ``-listen tcp`` option.
    Listener ownership is checked separately; this is not a live ACL query.
    """
    for index, argument in enumerate(command[2:], start=2):
        if argument in {"-ac", "-query", "-broadcast", "-indirect"}:
            raise IsolationError("Xwayland command weakens private display access")
        if argument in {"-listen", "-listenfd"} and (
            index + 1 >= len(command)
            or re.fullmatch(r"[0-9]+", command[index + 1]) is None
        ):
            raise IsolationError("Xwayland listener must be an inherited descriptor")


def _socket_inodes(pid: int) -> set[int]:
    result: set[int] = set()
    with os.scandir(PROC_ROOT / str(pid) / "fd") as entries:
        for index, entry in enumerate(entries):
            if index >= MAX_FDS:
                raise IsolationError("Xwayland descriptor inventory exceeds the admission bound")
            try:
                target = os.readlink(entry.path)
            except FileNotFoundError:
                continue  # An unrelated connection may close during the observation.
            match = re.fullmatch(r"socket:\[([0-9]+)\]", target)
            if match is not None:
                result.add(int(match.group(1)))
    return result


def _listener_inodes(number: str) -> set[int]:
    paths = {f"/tmp/.X11-unix/X{number}", f"@/tmp/.X11-unix/X{number}"}
    listeners: set[int] = set()
    table = os.fsdecode(_read(PROC_ROOT / "net" / "unix", MAX_UNIX_TABLE_BYTES))
    for line in table.splitlines()[1:]:
        fields = line.split(maxsplit=7)
        if len(fields) != 8 or fields[7] not in paths:
            continue
        # Linux SOCK_STREAM + SO_ACCEPTCON; accepted clients may have the same path.
        if int(fields[4], 16) == 1 and int(fields[3], 16) & 0x10000:
            listeners.add(int(fields[6]))
    if not listeners:
        raise IsolationError("claimed X display has no observable Unix listener")
    return listeners


def _validate(
    compositor_pid: int,
    compositor_start_ticks: int,
    display: str,
    identity: XwaylandIdentity,
) -> None:
    number = _display_number(display)
    if compositor_pid <= 0 or compositor_start_ticks <= 0:
        raise IsolationError("managed compositor identity is missing")
    if _stat(compositor_pid)[1] != compositor_start_ticks:
        raise IsolationError("managed compositor identity changed")
    before = _process(identity.pid)
    if (
        before.parent_pid != compositor_pid
        or before.start_ticks != identity.proc_start_ticks
        or before.command_sha256 != identity.command_sha256
        or Path(before.command[0]).name.casefold() != "xwayland"
        or len(before.command) < 2
        or before.command[1] != f":{number}"
    ):
        raise IsolationError("claimed X display does not match its managed Xwayland child")
    _require_safe_command(before.command)
    listeners = _listener_inodes(number)
    if not listeners.issubset(_socket_inodes(identity.pid)):
        raise IsolationError("claimed X display has a listener outside the managed Xwayland child")
    if _process(identity.pid) != before or _stat(compositor_pid)[1] != compositor_start_ticks:
        raise IsolationError("Xwayland/compositor identity changed during ownership observation")


def require_owned_xwayland(
    compositor_pid: int,
    display: str,
    identity: XwaylandIdentity,
    *,
    compositor_start_ticks: int,
) -> None:
    """Require exact child identity and ownership of every local listener alias.

    No X connection, event selection or process-wide inventory is performed.
    Weston may also retain the inherited listening descriptors. A competing
    abstract/pathname listener, stale PID or substituted display is rejected.
    """
    try:
        _validate(compositor_pid, compositor_start_ticks, display, identity)
    except (OSError, ValueError, IndexError, UnicodeError) as exc:
        raise IsolationError("managed Xwayland display ownership is unverifiable") from exc


def capture_xwayland_identity(
    compositor_pid: int,
    display: str,
    *,
    compositor_start_ticks: int,
) -> XwaylandIdentity:
    """Bind once after a fresh launch has started lazy Xwayland through capture."""
    try:
        number = _display_number(display)
        path = PROC_ROOT / str(compositor_pid) / "task" / str(compositor_pid) / "children"
        children = _read(path, 8192).split()
        if len(children) > MAX_CHILDREN:
            raise IsolationError("managed compositor child inventory exceeds the admission bound")
        matches: list[XwaylandIdentity] = []
        for child in children:
            pid = int(child)
            try:
                process = _process(pid)
            except FileNotFoundError:
                continue
            if (
                process.parent_pid == compositor_pid
                and len(process.command) >= 2
                and Path(process.command[0]).name.casefold() == "xwayland"
                and process.command[1] == f":{number}"
            ):
                matches.append(XwaylandIdentity(pid, process.start_ticks, process.command_sha256))
        if len(matches) != 1:
            raise IsolationError("fresh display has no unique managed Xwayland child")
        require_owned_xwayland(
            compositor_pid, display, matches[0], compositor_start_ticks=compositor_start_ticks,
        )
        return matches[0]
    except (OSError, ValueError, IndexError, UnicodeError) as exc:
        raise IsolationError("fresh Xwayland display ownership is unverifiable") from exc
