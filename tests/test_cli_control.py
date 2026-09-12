from __future__ import annotations

from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import pytest
import typer
from typer.testing import CliRunner

import minecraft_ai.cli as cli
from minecraft_ai.agent.process import build_parser
from minecraft_ai.platforms.capture_source import BedrockCaptureSource


@pytest.mark.parametrize("custom_config", [False, True])
def test_live_agent_launch_is_serialized_with_operator_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    custom_config: bool,
) -> None:
    calls: list[str] = []
    lock_held = False
    config_file = tmp_path / "selected.yaml" if custom_config else None

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
        assert kwargs["config_file"] == config_file
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
        config_file=config_file,
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


@pytest.mark.parametrize("source", ["typo", "", "PIPEWIRE", "x11 "])
@pytest.mark.parametrize("live", [False, True])
def test_run_rejects_invalid_capture_source_before_any_mutation(
    monkeypatch: pytest.MonkeyPatch, source: str, live: bool,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("invalid capture source reached a runtime operation")

    for name in (
        "_ensure_dirs", "ensure_default_config", "load_config", "_start_supervisor",
        "send_command", "_command", "create_bedrock_capture", "launch_agent_process",
        "wait_for_minecraft_window", "load_camera_calibration",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    args = ["run", "--capture-source", source] + (["--live"] if live else [])
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 2, result.output
    assert "--capture-source" in result.output
    assert "pipewire" in result.output and "x11" in result.output


@pytest.mark.parametrize("source", list(BedrockCaptureSource))
def test_parent_and_child_accept_supported_capture_sources(
    source: BedrockCaptureSource, live_config_launch, monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = live_config_launch
    capture_sources = []
    original = cli.create_bedrock_capture

    def capture(*args, **kwargs):
        capture_sources.append(kwargs["source"])
        return original(*args, **kwargs)

    monkeypatch.setattr(cli, "create_bedrock_capture", capture)
    result = CliRunner().invoke(cli.app, ["run", "--live", "--capture-source", source.value])
    assert result.exit_code == 0, result.output
    assert capture_sources == [source]
    assert h.launches[0]["capture_source"] == source.value
    args = build_parser().parse_args([
        "--lease-id", "test", "--display", ":12", "--window-id", "42",
        "--instance-id", "bedrock:test", "--capture-source", source.value,
    ])
    assert args.capture_source == source


def test_child_parser_rejects_invalid_capture_source() -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([
            "--lease-id", "test", "--display", ":12", "--window-id", "42",
            "--instance-id", "bedrock:test", "--capture-source", "typo",
        ])
    assert exc.value.code == 2


@pytest.mark.parametrize("contents", [None, "policy: [", "[1, 2]", "policy:\n  camera_scale: -1\n"])
def test_run_rejects_invalid_explicit_config_before_any_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, contents: str | None,
) -> None:
    selected = tmp_path / "selected.yaml"
    if contents is not None:
        selected.write_text(contents, encoding="utf-8")

    def forbidden(*_args, **_kwargs):
        pytest.fail("invalid explicit configuration must fail before any runtime mutation")

    for name in (
        "_ensure_dirs", "ensure_default_config", "_start_supervisor", "send_command",
        "_command", "launch_agent_process",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    result = CliRunner().invoke(cli.app, ["run", "--live", "--config", str(selected)])
    assert result.exit_code == 2, result.output
    # Rich may wrap a long temporary path in the middle of its filename.
    assert "--config" in result.output or "configuration" in result.output


@pytest.fixture
def live_config_launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Exercise CLI config selection with every game/process operation faked."""
    default_path = tmp_path / "default.yaml"
    default_path.write_text(
        "policy:\n  enabled: true\n  camera_scale: 6.0\n  camera_pitch_scale: 7.0\n",
        encoding="utf-8",
    )
    default_bytes = default_path.read_bytes()
    calls = []
    compatibility = []
    launches = []
    real_load = cli.load_config

    def load(path=None):
        calls.append(("load", path))
        return real_load(default_path if path is None else path)

    monkeypatch.setattr(cli, "load_config", load)
    monkeypatch.setattr(cli, "_ensure_dirs", lambda: calls.append(("dirs",)))
    monkeypatch.setattr(cli, "ensure_default_config", lambda: calls.append(("default",)))
    monkeypatch.setattr(cli, "emergency_stop_latched", lambda: False)
    monkeypatch.setattr(cli, "operator_pause_latched", lambda: False)
    monkeypatch.setattr(cli, "operator_intent_lock", lambda: nullcontext())
    monkeypatch.setattr(cli, "agent_alive", lambda: False)
    monkeypatch.setattr(cli, "supervisor_alive", lambda: True)
    monkeypatch.setattr(cli, "send_command", lambda *_args, **_kw: {"state": "SAFE_IDLE"})
    session = SimpleNamespace(display=":isolated-test")
    monkeypatch.setattr(cli, "BedrockSession", SimpleNamespace(load=lambda: session))
    monkeypatch.setattr(cli, "bedrock_session_alive", lambda _session: True)
    monkeypatch.setattr(cli, "_require_autonomous_isolated_session", lambda _session: None)
    monkeypatch.setattr(cli, "wait_for_minecraft_window", lambda *_args, **_kw: 42)
    monkeypatch.setattr(cli, "discover_bedrock_linux_install", lambda: SimpleNamespace(
        selected_build=SimpleNamespace(version="test-version"), wine_prefix=tmp_path / "wine",
    ))
    frame = SimpleNamespace(width=1280, height=720)
    monkeypatch.setattr(cli, "create_bedrock_capture", lambda *_args, **_kw: SimpleNamespace(
        capture=lambda: frame, close=lambda: None,
    ))
    monkeypatch.setattr(cli, "live_control_arm_reason", lambda _frame: "world")
    profile = SimpleNamespace(
        profile_id="test-profile", pitch_counts_per_degree=3.5,
        require_compatible=lambda **kw: compatibility.append(kw),
    )
    monkeypatch.setattr(cli, "load_camera_calibration", lambda *_args, **_kw: profile)
    monkeypatch.setattr(cli, "read_bedrock_mouse_sensitivity", lambda _prefix: 50)
    monkeypatch.setattr(cli, "app_paths", lambda: SimpleNamespace(data_dir=tmp_path))

    def command(name, **_kwargs):
        calls.append((name,))
        if name == "attach-bedrock-x11":
            return {"world_camera": {
                "origin_calibrated": True, "calibration_id": profile.profile_id,
                "pitch_counts_per_degree": profile.pitch_counts_per_degree,
            }}
        return {"lease": {"lease_id": "lease-test"}} if name == "arm" else {}

    monkeypatch.setattr(cli, "_command", command)
    monkeypatch.setattr(cli, "launch_agent_process", lambda **kw: (
        launches.append(kw) or SimpleNamespace(pid=1234)
    ))
    return SimpleNamespace(
        calls=calls, compatibility=compatibility, launches=launches,
        default_path=default_path, default_bytes=default_bytes,
    )


def test_run_uses_one_resolved_config_for_calibration_and_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, live_config_launch,
) -> None:
    h = live_config_launch
    selected = tmp_path / "selected.yaml"
    selected.write_text(
        "policy:\n  enabled: true\n  camera_scale: 2.75\n  camera_pitch_scale: 3.5\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli.app, ["run", "--live", "--config", "selected.yaml"])
    assert result.exit_code == 0, result.output
    assert h.calls[0] == ("load", selected.resolve())
    assert [call for call in h.calls if call[0] == "load"] == [h.calls[0]]
    assert ("default",) not in h.calls
    assert h.compatibility[0]["configured_yaw_counts_per_degree"] == 2.75
    assert h.compatibility[0]["configured_pitch_counts_per_degree"] == 3.5
    assert h.launches[0]["config_file"] == selected.resolve()
    assert h.default_path.read_bytes() == h.default_bytes


def test_run_without_config_keeps_default_selection(live_config_launch) -> None:
    h = live_config_launch
    result = CliRunner().invoke(cli.app, ["run", "--live"])
    assert result.exit_code == 0, result.output
    assert h.calls[:2] == [("dirs",), ("default",)]
    assert [call for call in h.calls if call[0] == "load"] == [("load", None)]
    assert h.compatibility[0]["configured_yaw_counts_per_degree"] == 6.0
    assert h.compatibility[0]["configured_pitch_counts_per_degree"] == 7.0
    assert h.launches[0]["config_file"] is None


def test_explicit_config_does_not_bypass_operator_pause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, live_config_launch,
) -> None:
    h = live_config_launch
    selected = tmp_path / "selected.yaml"
    selected.write_text("role: generalist\n", encoding="utf-8")
    monkeypatch.setattr(cli, "operator_pause_latched", lambda: True)
    result = CliRunner().invoke(cli.app, ["run", "--live", "--config", str(selected)])
    assert result.exit_code == 2, result.output
    assert "pause" in result.output.casefold()
    assert not h.launches and not h.compatibility
    assert ("arm",) not in h.calls and ("attach-bedrock-x11",) not in h.calls


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

    @contextmanager
    def bedrock_lock(*, wait_timeout_s: float = 0.0) -> Iterator[None]:
        assert wait_timeout_s == 30.0
        yield

    monkeypatch.setattr(cli, "latch_operator_pause", lambda: calls.append("latch"))
    monkeypatch.setattr(cli, "bedrock_lifecycle_lock", bedrock_lock)
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
