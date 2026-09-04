from __future__ import annotations

from pathlib import Path

import pytest

import minecraft_ai.agent_lifecycle as agent_lifecycle_module
import minecraft_ai.cli as cli_module
import minecraft_ai.emergency as emergency_module
import minecraft_ai.operator_server as operator_server_module
import minecraft_ai.platforms.bedrock_session as bedrock_session_module
import minecraft_ai.supervisor as supervisor_module


@pytest.fixture(autouse=True)
def isolate_live_safety_latches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep every test process descriptor away from the live desktop runtime."""

    runtime_dir = tmp_path / "live-runtime"
    data_dir = tmp_path / "live-data"
    control_file = runtime_dir / "control.json"
    status_file = runtime_dir / "supervisor-state.json"
    agent_file = runtime_dir / "agent-process.json"
    bedrock_file = runtime_dir / "bedrock-session.json"

    monkeypatch.setattr(supervisor_module, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(supervisor_module, "CONTROL_FILE", control_file)
    monkeypatch.setattr(supervisor_module, "STATUS_FILE", status_file)
    monkeypatch.setattr(supervisor_module, "LOCK_FILE", runtime_dir / "supervisor.lock")
    monkeypatch.setattr(supervisor_module, "OPERATOR_PAUSE_FILE", data_dir / "OPERATOR_PAUSE")
    monkeypatch.setattr(supervisor_module, "AGENT_FILE", agent_file)
    monkeypatch.setattr(operator_server_module, "AGENT_FILE", agent_file)

    monkeypatch.setattr(emergency_module, "DATA_DIR", data_dir)
    monkeypatch.setattr(emergency_module, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(emergency_module, "EMERGENCY_STOP_FILE", data_dir / "EMERGENCY_STOP")
    monkeypatch.setattr(emergency_module, "CONTROL_FILE", control_file)

    monkeypatch.setattr(agent_lifecycle_module, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(agent_lifecycle_module, "DATA_DIR", data_dir)
    monkeypatch.setattr(agent_lifecycle_module, "AGENT_FILE", agent_file)
    monkeypatch.setattr(agent_lifecycle_module, "AGENT_LOG", data_dir / "logs" / "agent.log")

    monkeypatch.setattr(bedrock_session_module, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(bedrock_session_module, "BEDROCK_SESSION_FILE", bedrock_file)
    monkeypatch.setattr(
        bedrock_session_module,
        "BEDROCK_LIFECYCLE_LOCK",
        runtime_dir / "bedrock-session.lock",
    )

    # The CLI imports descriptor paths by value; isolate those aliases too.
    monkeypatch.setattr(cli_module, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(cli_module, "DATA_DIR", data_dir)
    monkeypatch.setattr(cli_module, "CONTROL_FILE", control_file)
    monkeypatch.setattr(cli_module, "STATUS_FILE", status_file)
    monkeypatch.setattr(cli_module, "AGENT_FILE", agent_file)
    monkeypatch.setattr(cli_module, "BEDROCK_SESSION_FILE", bedrock_file)
