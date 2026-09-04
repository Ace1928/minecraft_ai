from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

import minecraft_ai.service_control as service_control


@pytest.fixture(autouse=True)
def _simulate_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service_control.sys, "platform", "linux")


def _completed(returncode: int, stdout: str = "") -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stdout=stdout)


def test_inactive_service_still_receives_idempotent_stop_to_close_start_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(service_control.shutil, "which", lambda _name: "/usr/bin/systemctl")
    results = iter((_completed(0), _completed(3, "inactive\n")))
    monkeypatch.setattr(
        service_control.subprocess,
        "run",
        lambda command, **_kwargs: calls.append(command) or next(results),
    )

    assert service_control.stop_persistent_agent_service() is True
    assert [call[2] for call in calls] == ["stop", "is-active"]


def test_unknown_status_still_attempts_stop_but_never_claims_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = iter((_completed(0), _completed(4, "unknown\n")))
    calls: list[list[str]] = []
    monkeypatch.setattr(service_control.shutil, "which", lambda _name: "/usr/bin/systemctl")
    monkeypatch.setattr(
        service_control.subprocess,
        "run",
        lambda command, **_kwargs: calls.append(command) or next(results),
    )

    assert service_control.stop_persistent_agent_service() is False
    assert [call[2] for call in calls] == ["stop", "is-active"]


def test_status_timeout_cannot_be_misreported_as_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fail_run(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise subprocess.TimeoutExpired("systemctl", 3.0)

    monkeypatch.setattr(service_control.shutil, "which", lambda _name: "/usr/bin/systemctl")
    monkeypatch.setattr(service_control.subprocess, "run", fail_run)

    assert service_control.stop_persistent_agent_service() is False
    assert calls == 1


def test_active_service_must_reach_explicit_inactive_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = iter((_completed(0), _completed(3, "inactive\n")))
    monkeypatch.setattr(service_control.shutil, "which", lambda _name: "/usr/bin/systemctl")
    monkeypatch.setattr(
        service_control.subprocess,
        "run",
        lambda *_args, **_kwargs: next(results),
    )

    assert service_control.stop_persistent_agent_service() is True


def test_start_requires_explicit_active_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = iter((_completed(0), _completed(0, "active\n")))
    calls: list[list[str]] = []
    monkeypatch.setattr(service_control.shutil, "which", lambda _name: "/usr/bin/systemctl")
    monkeypatch.setattr(
        service_control.subprocess,
        "run",
        lambda command, **_kwargs: calls.append(command) or next(results),
    )

    assert service_control.start_persistent_agent_service() is True
    assert [call[2] for call in calls] == ["start", "is-active"]
