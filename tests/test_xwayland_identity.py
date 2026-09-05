from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from minecraft_ai.platforms import xwayland_identity as ownership
from minecraft_ai.platforms.bedrock_x11 import IsolationError


def _process(root: Path, pid: int, parent: int, ticks: int, command: tuple[str, ...]) -> None:
    directory = root / str(pid)
    directory.mkdir(exist_ok=True)
    fields = ["S", str(parent), *(["0"] * 17), str(ticks)]
    (directory / "stat").write_text(f"{pid} (test process) {' '.join(fields)}\n")
    (directory / "cmdline").write_bytes(b"\0".join(arg.encode() for arg in command) + b"\0")


def _table(root: Path, listeners: list[tuple[int, str, bool]]) -> None:
    (root / "net").mkdir(exist_ok=True)
    lines = ["Num RefCount Protocol Flags Type St Inode Path"]
    for inode, path, listening in listeners:
        flag = "00010000" if listening else "00000000"
        lines.append(f"000: 00000002 00000000 {flag} 0001 01 {inode} {path}")
    (root / "net" / "unix").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def namespace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(ownership, "PROC_ROOT", tmp_path)
    _process(tmp_path, 300, 1, 100, ("/usr/bin/weston", "--backend=headless"))
    _process(tmp_path, 301, 300, 200, ("/usr/bin/Xwayland", ":71", "-rootless"))
    children = tmp_path / "300" / "task" / "300"
    children.mkdir(parents=True)
    (children / "children").write_text("301 ")
    descriptors = tmp_path / "301" / "fd"
    descriptors.mkdir()
    for inode in (101, 102):
        # Model procfs readlink results without creating Windows-invalid targets
        # or depending on permission to create symlinks on the CI host.
        (descriptors / str(inode)).write_text(f"socket:[{inode}]", encoding="ascii")
    original_readlink = ownership.os.readlink

    def readlink(path: str, *args: object, **kwargs: object) -> str:
        selected = Path(path)
        if selected.parent == descriptors:
            return selected.read_text(encoding="ascii")
        return original_readlink(path, *args, **kwargs)

    monkeypatch.setattr(ownership.os, "readlink", readlink)
    _table(tmp_path, [
        (101, "/tmp/.X11-unix/X71", True),
        (102, "@/tmp/.X11-unix/X71", True),
    ])
    return tmp_path


def _capture() -> ownership.XwaylandIdentity:
    return ownership.capture_xwayland_identity(300, ":71", compositor_start_ticks=100)


def _require(identity: ownership.XwaylandIdentity, display: str = ":71.0") -> None:
    ownership.require_owned_xwayland(300, display, identity, compositor_start_ticks=100)


def test_binds_exact_child_and_both_listener_aliases(namespace: Path) -> None:
    identity = _capture()
    assert identity.pid == 301
    assert identity.proc_start_ticks == 200
    _require(identity)


@pytest.mark.parametrize("arguments", [
    ("-ac",), ("-query", "remote"), ("-broadcast",), ("-multicast",), ("-indirect", "remote"),
    ("-listen", "tcp"), ("-listen",), ("-listenfd", "invalid"), ("-enable-ei-portal",),
])
def test_unsafe_initial_command_cannot_be_enrolled(
    namespace: Path, arguments: tuple[str, ...],
) -> None:
    _process(namespace, 301, 300, 200, ("/usr/bin/Xwayland", ":71", "-rootless", *arguments))
    with pytest.raises(IsolationError, match="private display access|inherited descriptor"):
        _capture()


@pytest.mark.parametrize("listen_option", ["-listen", "-listenfd"])
def test_weston_inherited_listener_spelling_remains_supported(
    namespace: Path, listen_option: str,
) -> None:
    _process(namespace, 301, 300, 200, (
        "/usr/bin/Xwayland", ":71", "-rootless", listen_option, "23",
        listen_option, "24", "-displayfd", "31", "-wm", "29", "-terminate",
    ))
    _require(_capture())


@pytest.mark.parametrize("display", [":72", ":0", "localhost:71", "host:71", ":071"])
def test_descriptor_display_substitution_cannot_retarget_valid_child(
    namespace: Path, display: str,
) -> None:
    identity = _capture()
    with pytest.raises(IsolationError):
        _require(identity, display)


@pytest.mark.parametrize("path", ["/tmp/.X11-unix/X71", "@/tmp/.X11-unix/X71"])
def test_unowned_alternate_listener_rejects_display(namespace: Path, path: str) -> None:
    identity = _capture()
    _table(namespace, [(101, "/tmp/.X11-unix/X71", True), (202, path, True)])
    with pytest.raises(IsolationError, match="listener outside"):
        _require(identity)


def test_accepted_clients_at_same_path_do_not_require_their_descriptors(namespace: Path) -> None:
    identity = _capture()
    _table(namespace, [
        (101, "/tmp/.X11-unix/X71", True),
        (102, "@/tmp/.X11-unix/X71", True),
        (999, "/tmp/.X11-unix/X71", False),
        (888, "/tmp/unrelated-\N{SNOWMAN}", True),
    ])
    _require(identity)


@pytest.mark.parametrize("mutation", ["reused-pid", "reparented", "command", "compositor"])
def test_saved_process_identity_cannot_be_reused(namespace: Path, mutation: str) -> None:
    identity = _capture()
    if mutation == "compositor":
        _process(namespace, 300, 1, 101, ("/usr/bin/weston", "--backend=headless"))
    else:
        _process(
            namespace, 301, 1 if mutation == "reparented" else 300,
            201 if mutation == "reused-pid" else 200,
            ("/usr/bin/Xwayland", ":71", "-rootless", "-ac")
            if mutation == "command" else ("/usr/bin/Xwayland", ":71", "-rootless"),
        )
    with pytest.raises(IsolationError):
        _require(identity)


def test_matching_display_on_another_compositors_child_is_rejected(namespace: Path) -> None:
    identity = _capture()
    _process(namespace, 401, 400, 201, ("/usr/bin/Xwayland", ":71", "-rootless"))
    substituted = replace(identity, pid=401, proc_start_ticks=201)
    with pytest.raises(IsolationError, match="managed Xwayland child"):
        _require(substituted)


def test_closed_listener_descriptor_cannot_reuse_stale_binding(namespace: Path) -> None:
    identity = _capture()
    (namespace / "301" / "fd" / "102").unlink()
    with pytest.raises(IsolationError, match="listener outside"):
        _require(identity)


def test_no_listener_cannot_be_promoted_from_a_display_name(namespace: Path) -> None:
    identity = _capture()
    _table(namespace, [])
    with pytest.raises(IsolationError, match="no observable Unix listener"):
        _require(identity)


def test_lazy_xwayland_requires_a_started_child_before_binding(namespace: Path) -> None:
    children = namespace / "300" / "task" / "300" / "children"
    children.write_text("")
    with pytest.raises(IsolationError, match="no unique managed Xwayland child"):
        _capture()
    children.write_text("301 ")
    assert _capture().pid == 301


@pytest.mark.parametrize("inventory", ["children", "fds", "unix-table"])
def test_ownership_observation_has_explicit_work_bounds(
    namespace: Path, monkeypatch: pytest.MonkeyPatch, inventory: str,
) -> None:
    if inventory == "children":
        monkeypatch.setattr(ownership, "MAX_CHILDREN", 0)
    elif inventory == "fds":
        monkeypatch.setattr(ownership, "MAX_FDS", 1)
    else:
        monkeypatch.setattr(ownership, "MAX_UNIX_TABLE_BYTES", 16)
    with pytest.raises(IsolationError, match="bound"):
        _capture()


def test_child_identity_is_rechecked_after_listener_observation(
    namespace: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _capture()
    original = ownership._socket_inodes

    def changed_after_descriptors(pid: int) -> set[int]:
        result = original(pid)
        _process(namespace, 301, 300, 201, ("/usr/bin/Xwayland", ":71", "-rootless"))
        return result

    monkeypatch.setattr(ownership, "_socket_inodes", changed_after_descriptors)
    with pytest.raises(IsolationError, match="changed during ownership"):
        _require(identity)
