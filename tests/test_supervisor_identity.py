from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import typer

import minecraft_ai.cli as cli
import minecraft_ai.supervisor as supervisor
from minecraft_ai.supervisor import ControlEndpoint


def _command() -> tuple[str, ...]:
    return (sys.executable, "-m", "minecraft_ai.supervisor", "--role", "builder")


def _endpoint(*, session_id: str = "session-a") -> ControlEndpoint:
    command = _command()
    return ControlEndpoint(
        host="127.0.0.1",
        port=12345,
        token="secret",
        pid=4242,
        session_id=session_id,
        proc_start_ticks=9876,
        command_sha256=supervisor._command_sha256(command),
    )


@pytest.mark.parametrize(
    ("identity", "expected"),
    [
        (None, "unverifiable"),
        ((1111, _command()), "unverifiable"),
        ((9876, (sys.executable, "-c", "pass")), "mismatch"),
        ((9876, _command()), "verified-live"),
    ],
)
def test_control_endpoint_binds_pid_start_and_command(
    monkeypatch: pytest.MonkeyPatch,
    identity: tuple[int, tuple[str, ...]] | None,
    expected: str,
) -> None:
    monkeypatch.setattr(supervisor, "_IS_LINUX", True)
    monkeypatch.setattr(supervisor, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(supervisor, "_linux_process_identity", lambda _pid: identity)

    assert supervisor.control_endpoint_process_state(_endpoint()) == expected


def test_non_linux_endpoint_is_never_classified_dead_from_linux_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(supervisor, "_IS_LINUX", False)
    monkeypatch.setattr(supervisor, "_pid_alive", lambda _pid: False)

    assert supervisor.control_endpoint_process_state(_endpoint()) == "unverifiable"


def test_control_descriptor_compare_and_unlink_preserves_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "control.json"
    original = _endpoint()
    replacement = _endpoint(session_id="session-b")
    path.write_text(json.dumps(asdict(replacement)), encoding="utf-8")
    monkeypatch.setattr(supervisor, "CONTROL_FILE", path)

    assert supervisor.remove_control_endpoint_if_owned(original) is False
    assert ControlEndpoint.load(path) == replacement


def test_start_refuses_to_unlink_verified_unresponsive_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = _endpoint()
    monkeypatch.setattr(
        cli.ControlEndpoint,
        "load",
        classmethod(lambda _cls, _path=None: existing),
    )
    monkeypatch.setattr(cli, "control_endpoint_process_state", lambda _item: "verified-live")
    monkeypatch.setattr(
        cli.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("must not spawn replacement"),
    )

    with pytest.raises(typer.BadParameter, match="refusing"):
        cli._start_supervisor("builder")
