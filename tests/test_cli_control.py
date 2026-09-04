from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import pytest
import typer

import minecraft_ai.cli as cli


def test_live_agent_launch_is_serialized_with_operator_intent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    lock_held = False

    @contextmanager
    def intent_lock() -> Iterator[None]:
        nonlocal lock_held
        assert not lock_held
        lock_held = True
        calls.append("lock-enter")
        try:
            yield
        finally:
            calls.append("lock-exit")
            lock_held = False

    def launch_agent_process(**kwargs: object) -> SimpleNamespace:
        assert lock_held
        assert kwargs["lease_id"] == "lease-1"
        calls.append("launch")
        return SimpleNamespace(pid=1234)

    monkeypatch.setattr(cli, "operator_intent_lock", intent_lock)
    monkeypatch.setattr(cli, "operator_pause_latched", lambda: False)
    monkeypatch.setattr(cli, "emergency_stop_latched", lambda: False)
    monkeypatch.setattr(cli, "agent_alive", lambda: False)
    monkeypatch.setattr(
        cli,
        "_command",
        lambda command, **_kwargs: calls.append(command)
        or ({"lease": {"lease_id": "lease-1"}} if command == "arm" else {}),
    )
    monkeypatch.setattr(cli, "launch_agent_process", launch_agent_process)

    result = cli._launch_realtime_agent_transaction(
        target="bedrock:test",
        display=":2",
        window_id=42,
        role="creative_builder",
        allow_host_capture=False,
        capture_source="x11",
    )

    assert result.pid == 1234
    assert calls == ["lock-enter", "arm", "activate", "launch", "lock-exit"]


def test_live_agent_launch_rechecks_pause_inside_intent_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    @contextmanager
    def intent_lock() -> Iterator[None]:
        yield

    monkeypatch.setattr(cli, "operator_intent_lock", intent_lock)
    monkeypatch.setattr(cli, "emergency_stop_latched", lambda: False)
    monkeypatch.setattr(cli, "operator_pause_latched", lambda: True)
    monkeypatch.setattr(cli, "agent_alive", lambda: False)
    monkeypatch.setattr(cli, "_command", lambda command, **_kwargs: calls.append(command) or {})
    monkeypatch.setattr(
        cli,
        "launch_agent_process",
        lambda **_kwargs: calls.append("launch") or SimpleNamespace(pid=1234),
    )

    with pytest.raises(typer.BadParameter, match="pause"):
        cli._launch_realtime_agent_transaction(
            target="bedrock:test",
            display=":2",
            window_id=42,
            role="creative_builder",
            allow_host_capture=False,
            capture_source="x11",
        )

    assert calls == []


def test_human_takeover_serializes_pause_revocation_and_preserves_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    @contextmanager
    def intent_lock() -> Iterator[None]:
        calls.append("lock-enter")
        try:
            yield
        finally:
            calls.append("lock-exit")

    monkeypatch.setattr(cli, "operator_intent_lock", intent_lock)
    monkeypatch.setattr(cli, "latch_operator_pause", lambda: calls.append("latch"))
    monkeypatch.setattr(cli, "supervisor_alive", lambda: True)
    monkeypatch.setattr(
        cli,
        "_command",
        lambda command, **_kwargs: calls.append(command)
        or {"motor_lease_active": False},
    )
    monkeypatch.setattr(
        cli,
        "stop_agent_process",
        lambda **_kwargs: calls.append("agent-stop") or True,
    )
    monkeypatch.setattr(
        cli,
        "_persistent_agent_service_state",
        lambda: calls.append("service-state") or "active",
    )
    monkeypatch.setattr(
        cli,
        "current_control_owner_state",
        lambda: calls.append("owner-state") or "verified-live",
    )
    monkeypatch.setattr(
        cli,
        "operator_pause_latched",
        lambda: calls.append("pause-check") or True,
    )
    monkeypatch.setattr(cli, "AGENT_FILE", tmp_path / "missing-agent.json")

    cli._prepare_human_recording_takeover()

    assert calls == [
        "lock-enter",
        "latch",
        "disarm",
        "agent-stop",
        "service-state",
        "agent-stop",
        "owner-state",
        "pause-check",
        "lock-exit",
    ]


def test_human_takeover_refuses_recording_when_service_state_is_unconfirmed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "_ensure_dirs", lambda: None)
    monkeypatch.setattr(cli, "load_config", lambda: object())
    monkeypatch.setattr(cli, "agent_alive", lambda: False)
    monkeypatch.setattr(cli, "supervisor_alive", lambda: False)
    monkeypatch.setattr(cli, "latch_operator_pause", lambda: None)
    monkeypatch.setattr(cli, "stop_agent_process", lambda **_kwargs: True)
    monkeypatch.setattr(cli, "_persistent_agent_service_state", lambda: "unknown")
    monkeypatch.setattr(cli, "current_control_owner_state", lambda: "absent")
    monkeypatch.setattr(cli, "operator_pause_latched", lambda: True)
    monkeypatch.setattr(cli, "AGENT_FILE", tmp_path / "missing-agent.json")
    monkeypatch.setattr(
        cli,
        "record_human_session",
        lambda _request: pytest.fail("human input must not begin before containment"),
    )

    with pytest.raises(typer.BadParameter, match="service state"):
        cli.record_human(
            duration_s=1.0,
            capture_hz=20.0,
            label="test",
            task_id=None,
            fov=None,
            mouse_sensitivity=None,
            takeover=True,
            resume_live=False,
        )


@pytest.mark.parametrize("ambiguous_owner", ["agent", "supervisor"])
def test_record_human_requires_takeover_for_unreadable_process_ownership(
    ambiguous_owner: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "_ensure_dirs", lambda: None)
    monkeypatch.setattr(cli, "load_config", lambda: object())
    monkeypatch.setattr(cli, "supervisor_alive", lambda: False)
    monkeypatch.setattr(cli, "_persistent_agent_service_state", lambda: "inactive")
    if ambiguous_owner == "agent":
        monkeypatch.setattr(
            cli.AgentProcess,
            "load",
            classmethod(lambda _cls: (_ for _ in ()).throw(ValueError("malformed"))),
        )
        monkeypatch.setattr(cli, "current_control_owner_state", lambda: "absent")
    else:
        monkeypatch.setattr(
            cli.AgentProcess,
            "load",
            classmethod(lambda _cls: (_ for _ in ()).throw(FileNotFoundError())),
        )
        monkeypatch.setattr(cli, "current_control_owner_state", lambda: "unreadable")
    monkeypatch.setattr(
        cli,
        "record_human_session",
        lambda _request: pytest.fail("ambiguous autonomous ownership must block human input"),
    )

    with pytest.raises(typer.BadParameter, match="active or unconfirmed"):
        cli.record_human(
            duration_s=1.0,
            capture_hz=20.0,
            label="test",
            task_id=None,
            fov=None,
            mouse_sensitivity=None,
            takeover=False,
            resume_live=False,
        )


def test_record_human_resume_live_uses_shared_safe_resume_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    config = SimpleNamespace(
        role="generalist",
        trajectory=SimpleNamespace(shard_steps=64, queue_size=8),
    )
    session = SimpleNamespace(display=":8", mode="weston")
    manifest = SimpleNamespace(
        trajectory_id="human-test",
        accepted_steps=1,
        dropped_steps=0,
    )
    paths = SimpleNamespace(data_dir=tmp_path, state_db=tmp_path / "state.db")

    monkeypatch.setattr(cli, "_ensure_dirs", lambda: None)
    monkeypatch.setattr(cli, "load_config", lambda: config)
    monkeypatch.setattr(cli, "agent_alive", lambda: False)
    monkeypatch.setattr(cli, "supervisor_alive", lambda: False)
    monkeypatch.setattr(cli, "_persistent_agent_service_state", lambda: "inactive")
    monkeypatch.setattr(
        cli.BedrockSession,
        "load",
        classmethod(lambda _cls: session),
    )
    monkeypatch.setattr(cli, "bedrock_session_alive", lambda _session: True)
    monkeypatch.setattr(cli, "wait_for_minecraft_window", lambda *_args, **_kwargs: 42)
    monkeypatch.setattr(cli, "discover_bedrock_linux_install", lambda: None)
    monkeypatch.setattr(cli, "app_paths", lambda: paths)
    monkeypatch.setattr(
        cli,
        "record_human_session",
        lambda _request: calls.append("record") or manifest,
    )
    monkeypatch.setattr(
        cli,
        "_resume_operator_intent",
        lambda: calls.append("safe-resume"),
    )
    monkeypatch.setattr(
        cli,
        "clear_operator_pause",
        lambda: pytest.fail("record-human must not directly clear durable intent"),
    )
    monkeypatch.setattr(
        cli,
        "run",
        lambda **_kwargs: calls.append("run"),
    )

    cli.record_human(
        duration_s=1.0,
        capture_hz=20.0,
        label="test",
        task_id=None,
        fov=None,
        mouse_sensitivity=None,
        takeover=False,
        resume_live=True,
    )

    assert calls == ["record", "safe-resume", "run"]


def test_resume_waits_for_faulted_supervisor_generation_to_retire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(cli, "emergency_stop_latched", lambda: False)
    monkeypatch.setattr(cli, "supervisor_alive", lambda: True)
    monkeypatch.setattr(
        cli,
        "_command",
        lambda command, **_kwargs: {
            "state": "STOPPED",
            "session_id": "retiring-generation",
        },
    )
    monkeypatch.setattr(
        cli,
        "_wait_for_supervisor_generation_retirement",
        lambda session_id: calls.append(session_id),
    )

    cli._resume_operator_intent()

    assert calls == ["retiring-generation"]


def test_late_resume_race_also_waits_for_faulted_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alive = iter((False, True))
    waited: list[str] = []

    @contextmanager
    def intent_lock() -> Iterator[None]:
        yield

    monkeypatch.setattr(cli, "emergency_stop_latched", lambda: False)
    monkeypatch.setattr(cli, "supervisor_alive", lambda: next(alive))
    monkeypatch.setattr(cli, "operator_intent_lock", intent_lock)
    monkeypatch.setattr(
        cli,
        "_command",
        lambda _command_name, **_kwargs: {
            "state": "STOPPED",
            "session_id": "late-retiring-generation",
        },
    )
    monkeypatch.setattr(
        cli,
        "_wait_for_supervisor_generation_retirement",
        lambda session_id: waited.append(session_id),
    )

    cli._resume_operator_intent()

    assert waited == ["late-retiring-generation"]


def test_retirement_wait_accepts_only_absence_or_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = iter(
        (
            {"state": "STOPPED", "session_id": "old"},
            {"state": "SAFE_IDLE", "session_id": "new"},
        )
    )
    calls: list[str] = []

    def status(command: str, **_kwargs: object) -> dict[str, object]:
        calls.append(command)
        return next(observed)

    monkeypatch.setattr(cli, "send_command", status)

    cli._wait_for_supervisor_generation_retirement("old", timeout_s=1.0)

    assert calls == ["status", "status"]


def test_retirement_wait_fails_closed_while_exact_generation_lingers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli,
        "send_command",
        lambda *_args, **_kwargs: {"state": "STOPPED", "session_id": "old"},
    )

    with pytest.raises(typer.BadParameter, match="did not release"):
        cli._wait_for_supervisor_generation_retirement("old", timeout_s=0.01)


def test_pause_control_timeout_covers_graceful_agent_flush(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_timeout: list[float] = []
    monkeypatch.setattr(cli, "supervisor_alive", lambda: True)
    monkeypatch.setattr(cli, "AGENT_FILE", tmp_path / "missing-agent.json")
    monkeypatch.setattr(
        cli,
        "_command",
        lambda _command_name, *, timeout_s, **_kwargs: observed_timeout.append(timeout_s)
        or {
            "state": "PAUSED",
            "operator_pause_persisted": True,
            "agent_containment_confirmed": True,
        },
    )

    cli.pause()

    assert observed_timeout == [30.0]


def test_stop_control_timeout_covers_graceful_agent_flush(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_timeout: list[float] = []
    monkeypatch.setattr(cli, "supervisor_alive", lambda: True)
    monkeypatch.setattr(cli, "AGENT_FILE", tmp_path / "missing-agent.json")
    monkeypatch.setattr(cli, "current_control_owner_state", lambda: "absent")
    monkeypatch.setattr(
        cli,
        "send_command",
        lambda _command_name, *, timeout_s, **_kwargs: observed_timeout.append(timeout_s)
        or {
            "state": "STOPPED",
            "operator_pause_persisted": True,
            "agent_containment_confirmed": True,
        },
    )

    cli.stop(transient=False)

    assert observed_timeout == [30.0]


def test_manual_stop_latches_before_supervisor_and_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    control = tmp_path / "control.json"
    control.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli, "CONTROL_FILE", control)
    monkeypatch.setattr(cli, "AGENT_FILE", tmp_path / "missing-agent.json")
    monkeypatch.setattr(cli, "latch_operator_pause", lambda: calls.append("latch"))
    monkeypatch.setattr(
        cli, "send_command", lambda command, **_kwargs: calls.append(command) or {}
    )
    monkeypatch.setattr(
        cli, "stop_agent_process", lambda **_kwargs: calls.append("agent") or True
    )
    monkeypatch.setattr(cli, "supervisor_alive", lambda: False)
    monkeypatch.setattr(cli, "current_control_owner_state", lambda: "dead")

    cli.stop(transient=False)

    assert calls == ["latch", "agent"]


def test_transient_stop_does_not_change_operator_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    control = tmp_path / "missing-control.json"
    monkeypatch.setattr(cli, "CONTROL_FILE", control)
    monkeypatch.setattr(cli, "AGENT_FILE", tmp_path / "missing-agent.json")
    monkeypatch.setattr(cli, "latch_operator_pause", lambda: calls.append("latch"))
    monkeypatch.setattr(
        cli, "stop_agent_process", lambda **_kwargs: calls.append("agent") or True
    )
    monkeypatch.setattr(cli, "supervisor_alive", lambda: False)

    cli.stop(transient=True)

    assert calls == ["agent"]


def test_manual_bedrock_stop_revokes_before_waiting_for_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(cli, "latch_operator_pause", lambda: calls.append("latch"))
    monkeypatch.setattr(cli, "AGENT_FILE", tmp_path / "missing-agent.json")
    monkeypatch.setattr(cli, "supervisor_alive", lambda: True)
    monkeypatch.setattr(
        cli,
        "send_command",
        lambda command, **_kwargs: calls.append(command)
        or ({"live_capable": True} if command == "status" else {}),
    )
    monkeypatch.setattr(
        cli, "stop_agent_process", lambda **_kwargs: calls.append("agent") or True
    )
    monkeypatch.setattr(cli, "stop_bedrock_session", lambda: calls.append("bedrock"))

    cli.bedrock_stop(transient=False)

    assert calls == ["latch", "status", "disarm", "agent", "bedrock"]


def test_emergency_stop_latches_and_stops_owner_before_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(cli, "latch_operator_pause", lambda: calls.append("pause-latch"))
    monkeypatch.setattr(
        cli, "engage_emergency_stop", lambda reason: calls.append(f"emergency:{reason}")
    )
    monkeypatch.setattr(
        cli, "_stop_persistent_agent_service", lambda: calls.append("service") or True
    )
    monkeypatch.setattr(
        cli, "terminate_registered_supervisor", lambda: calls.append("supervisor") or True
    )
    monkeypatch.setattr(
        cli, "stop_agent_process", lambda **_kwargs: calls.append("agent") or True
    )

    cli.emergency_stop(reason="test-stop")

    assert calls == [
        "pause-latch",
        "emergency:test-stop",
        "supervisor",
        "agent",
        "service",
        "supervisor",
        "agent",
    ]


def test_reset_emergency_refuses_while_persistent_owner_is_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleared: list[bool] = []
    monkeypatch.setattr(cli, "persistent_agent_service_load_state", lambda: "loaded")
    monkeypatch.setattr(cli, "_persistent_agent_service_state", lambda: "active")
    monkeypatch.setattr(cli, "clear_emergency_stop", lambda: cleared.append(True))

    with pytest.raises(typer.BadParameter, match="persistent"):
        cli.reset_emergency_stop()

    assert cleared == []


def test_pause_still_revokes_when_durable_marker_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fail_latch() -> None:
        raise OSError("read-only data directory")

    monkeypatch.setattr(cli, "latch_operator_pause", fail_latch)
    monkeypatch.setattr(cli, "AGENT_FILE", tmp_path / "missing-agent.json")
    monkeypatch.setattr(cli, "supervisor_alive", lambda: True)
    monkeypatch.setattr(
        cli,
        "_command",
        lambda command, **_kwargs: calls.append(command)
        or {
            "state": "PAUSED",
            "operator_pause_persisted": False,
            "agent_containment_confirmed": True,
        },
    )
    monkeypatch.setattr(
        cli,
        "_stop_persistent_agent_service",
        lambda: calls.append("service") or True,
    )

    with pytest.raises(typer.BadParameter, match="service is confirmed stopped"):
        cli.pause()

    assert calls == ["pause", "service"]


def test_pause_fails_closed_for_unreadable_supervisor_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "AGENT_FILE", tmp_path / "missing-agent.json")
    monkeypatch.setattr(cli, "supervisor_alive", lambda: False)
    monkeypatch.setattr(cli, "latch_operator_pause", lambda: None)
    monkeypatch.setattr(cli, "terminate_registered_supervisor", lambda: False)
    monkeypatch.setattr(cli, "stop_agent_process", lambda **_kwargs: False)
    monkeypatch.setattr(cli, "current_control_owner_state", lambda: "unreadable")

    with pytest.raises(typer.BadParameter, match="revocation is unconfirmed"):
        cli.pause()


def test_emergency_fallbacks_run_when_both_latch_writes_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fail_latch(*_args: object) -> None:
        raise OSError("read-only data directory")

    monkeypatch.setattr(cli, "latch_operator_pause", fail_latch)
    monkeypatch.setattr(cli, "engage_emergency_stop", fail_latch)
    monkeypatch.setattr(
        cli, "_stop_persistent_agent_service", lambda: calls.append("service") or False
    )
    monkeypatch.setattr(
        cli, "terminate_registered_supervisor", lambda: calls.append("supervisor") or True
    )
    monkeypatch.setattr(
        cli, "stop_agent_process", lambda **_kwargs: calls.append("agent") or True
    )

    with pytest.raises(typer.BadParameter, match="neither durable stop marker"):
        cli.emergency_stop(reason="test-stop")

    assert calls == ["supervisor", "agent", "service", "supervisor", "agent"]


def test_reset_emergency_never_clears_operator_pause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(cli, "CONTROL_FILE", tmp_path / "missing-control.json")
    monkeypatch.setattr(cli, "persistent_agent_service_load_state", lambda: "loaded")
    monkeypatch.setattr(cli, "_persistent_agent_service_state", lambda: "inactive")
    monkeypatch.setattr(cli, "supervisor_alive", lambda: False)
    monkeypatch.setattr(cli, "agent_alive", lambda: False)
    monkeypatch.setattr(cli, "latch_operator_pause", lambda: calls.append("pause-latch"))
    monkeypatch.setattr(cli, "operator_pause_latched", lambda: True)
    monkeypatch.setattr(cli, "clear_emergency_stop", lambda: calls.append("emergency"))
    monkeypatch.setattr(cli, "clear_operator_pause", lambda: calls.append("pause"))

    cli.reset_emergency_stop()

    assert calls == ["pause-latch", "emergency"]


def test_reset_emergency_retains_latch_when_operator_pause_cannot_be_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleared: list[bool] = []
    monkeypatch.setattr(cli, "CONTROL_FILE", tmp_path / "missing-control.json")
    monkeypatch.setattr(cli, "persistent_agent_service_load_state", lambda: "loaded")
    monkeypatch.setattr(cli, "_persistent_agent_service_state", lambda: "inactive")
    monkeypatch.setattr(cli, "supervisor_alive", lambda: False)
    monkeypatch.setattr(cli, "agent_alive", lambda: False)
    monkeypatch.setattr(
        cli,
        "latch_operator_pause",
        lambda: (_ for _ in ()).throw(OSError("read-only filesystem")),
    )
    monkeypatch.setattr(cli, "clear_emergency_stop", lambda: cleared.append(True))

    with pytest.raises(typer.BadParameter, match="operator-pause"):
        cli.reset_emergency_stop()

    assert cleared == []


def test_resume_starts_persistent_service_when_supervisor_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(cli, "emergency_stop_latched", lambda: False)
    monkeypatch.setattr(cli, "supervisor_alive", lambda: False)
    monkeypatch.setattr(cli, "current_control_owner_state", lambda: "absent")
    monkeypatch.setattr(cli, "persistent_agent_service_load_state", lambda: "loaded")
    monkeypatch.setattr(cli, "clear_operator_pause", lambda: calls.append("clear"))
    monkeypatch.setattr(
        cli,
        "start_persistent_agent_service",
        lambda: calls.append("start-service") or True,
    )

    cli.resume()

    assert calls == ["start-service", "clear"]


def test_resume_keeps_pause_if_persistent_service_does_not_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(cli, "emergency_stop_latched", lambda: False)
    monkeypatch.setattr(cli, "supervisor_alive", lambda: False)
    monkeypatch.setattr(cli, "current_control_owner_state", lambda: "absent")
    monkeypatch.setattr(cli, "persistent_agent_service_load_state", lambda: "loaded")
    monkeypatch.setattr(cli, "clear_operator_pause", lambda: calls.append("clear"))
    monkeypatch.setattr(cli, "start_persistent_agent_service", lambda: False)
    monkeypatch.setattr(cli, "latch_operator_pause", lambda: calls.append("restore"))

    with pytest.raises(typer.BadParameter, match="did not start"):
        cli.resume()

    assert calls == []


def test_resume_without_installed_service_permits_manual_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(cli, "emergency_stop_latched", lambda: False)
    monkeypatch.setattr(cli, "supervisor_alive", lambda: False)
    monkeypatch.setattr(cli, "current_control_owner_state", lambda: "absent")
    monkeypatch.setattr(cli, "persistent_agent_service_load_state", lambda: "not-found")
    monkeypatch.setattr(cli, "clear_operator_pause", lambda: calls.append("clear"))
    monkeypatch.setattr(
        cli,
        "start_persistent_agent_service",
        lambda: pytest.fail("standalone resume must not start a missing service"),
    )

    cli.resume()

    assert calls == ["clear"]


def test_bedrock_launch_refuses_to_overwrite_malformed_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli.BedrockSession,
        "load",
        classmethod(lambda _cls: (_ for _ in ()).throw(ValueError("malformed"))),
    )
    monkeypatch.setattr(
        cli,
        "launch_isolated_bedrock_session",
        lambda **_kwargs: pytest.fail("must not launch over ambiguous ownership"),
    )

    with pytest.raises(typer.BadParameter, match="descriptor is unreadable"):
        cli._bedrock_launch_locked(
            width=1280,
            height=720,
            fullscreen=True,
            direct=False,
        )
