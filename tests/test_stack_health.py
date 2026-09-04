from __future__ import annotations

from pathlib import Path

import pytest

import minecraft_ai.agent_lifecycle as agent_lifecycle
import minecraft_ai.emergency as emergency
import minecraft_ai.stack_health as stack_health
import minecraft_ai.supervisor as supervisor


def test_start_permission_requires_both_durable_interlocks_clear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(emergency, "emergency_stop_latched", lambda: False)
    monkeypatch.setattr(emergency, "emergency_reason", lambda: None)
    monkeypatch.setattr(supervisor, "operator_pause_latched", lambda: True)

    permitted, detail = stack_health._start_permitted()

    assert permitted is False
    assert detail["operator_pause_latched"] is True


def test_stack_cleanup_never_resumes_paused_supervisor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(agent_lifecycle, "AGENT_FILE", tmp_path / "missing-agent.json")
    monkeypatch.setattr(agent_lifecycle, "stop_agent_process", lambda: False)
    monkeypatch.setattr(supervisor, "supervisor_alive", lambda: True)
    monkeypatch.setattr(
        supervisor,
        "send_command",
        lambda command: calls.append(command) or {"state": "PAUSED"},
    )

    healthy, detail = stack_health._stop_agent()

    assert healthy is True
    assert calls == ["status"]
    assert detail["supervisor_state"] == "PAUSED"


def test_stack_cleanup_fails_when_agent_descriptor_remains(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = tmp_path / "agent-process.json"
    descriptor.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(agent_lifecycle, "AGENT_FILE", descriptor)
    monkeypatch.setattr(agent_lifecycle, "stop_agent_process", lambda: False)
    monkeypatch.setattr(supervisor, "supervisor_alive", lambda: False)

    healthy, detail = stack_health._stop_agent()

    assert healthy is False
    assert detail["agent_containment_confirmed"] is False


def test_stack_cleanup_disarms_before_graceful_agent_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(agent_lifecycle, "AGENT_FILE", tmp_path / "missing-agent.json")
    monkeypatch.setattr(
        agent_lifecycle,
        "stop_agent_process",
        lambda: calls.append("agent-stop") or True,
    )
    monkeypatch.setattr(supervisor, "supervisor_alive", lambda: True)
    monkeypatch.setattr(
        supervisor,
        "send_command",
        lambda command: calls.append(command)
        or {"state": "RUNNING" if command == "status" else "PAUSED"},
    )

    healthy, _detail = stack_health._stop_agent()

    assert healthy is True
    assert calls == ["status", "disarm", "agent-stop"]
