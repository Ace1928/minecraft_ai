from __future__ import annotations

import os
import signal
import sys
from pathlib import Path

import pytest

import minecraft_ai.agent_lifecycle as lifecycle
from minecraft_ai.agent_lifecycle import AgentProcess


_SIGKILL = getattr(signal, "SIGKILL", signal.SIGTERM)


def _agent_command(*, display: str = ":2") -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "minecraft_ai.agent_process",
        "--lease-id",
        "lease-secret",
        "--display",
        display,
        "--window-id",
        "6291460",
        "--instance-id",
        "bedrock:test",
        "--role",
        "creative_builder",
        "--capture-source",
        "x11",
    )


def _process(command: tuple[str, ...], *, start_ticks: int = 1234) -> AgentProcess:
    return AgentProcess(
        pid=4242,
        started_ns=10,
        display=":2",
        window_id=6291460,
        instance_id="bedrock:test",
        role="creative_builder",
        capture_source="x11",
        proc_start_ticks=start_ticks,
        command_sha256=lifecycle._command_sha256(command),
    )


def test_launch_persists_linux_start_ticks_and_exact_agent_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[tuple[str, ...]] = []

    class _Child:
        pid = 4242

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            pytest.fail("verified child must not be terminated")

        def wait(self, *, timeout: float) -> int:
            pytest.fail(f"verified child must not be waited during launch: {timeout}")

        def kill(self) -> None:
            pytest.fail("verified child must not be killed")

    def fake_popen(command: list[str], **_kwargs: object) -> _Child:
        commands.append(tuple(command))
        return _Child()

    descriptor = tmp_path / "agent-process.json"
    monkeypatch.setattr(lifecycle, "_IS_LINUX", True)
    monkeypatch.setattr(lifecycle, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(lifecycle, "AGENT_FILE", descriptor)
    monkeypatch.setattr(lifecycle, "AGENT_LOG", tmp_path / "agent.log")
    monkeypatch.setattr(lifecycle.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        lifecycle,
        "_linux_process_identity",
        lambda _pid: (9876, commands[0]),
    )
    monkeypatch.setattr(lifecycle.time, "sleep", lambda _seconds: None)

    process = lifecycle.launch_agent_process(
        lease_id="lease-secret",
        display=":2",
        window_id=6291460,
        instance_id="bedrock:test",
        role="creative_builder",
        capture_source="x11",
    )

    assert process.proc_start_ticks == 9876
    assert process.command_sha256 == lifecycle._command_sha256(commands[0])
    assert AgentProcess.load(descriptor) == process


def test_launch_refuses_to_replace_malformed_agent_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = tmp_path / "agent-process.json"
    descriptor.write_text("{malformed", encoding="utf-8")
    monkeypatch.setattr(lifecycle, "AGENT_FILE", descriptor)
    monkeypatch.setattr(
        lifecycle.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("malformed ownership must block process launch"),
    )

    with pytest.raises(RuntimeError, match="descriptor is unreadable; refusing replacement"):
        lifecycle.launch_agent_process(
            lease_id="lease-secret",
            display=":2",
            window_id=6291460,
            instance_id="bedrock:test",
            role="creative_builder",
            capture_source="x11",
        )

    assert descriptor.read_text(encoding="utf-8") == "{malformed"


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups")
def test_launch_persist_failure_terminates_owned_agent_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[tuple[str, ...]] = []
    running = True
    signals: list[tuple[int, signal.Signals]] = []

    class _Child:
        pid = 4242

        def poll(self) -> None:
            return None

        def wait(self, *, timeout: float) -> int:
            assert timeout == 0.1
            assert not running
            return 0

    def fake_popen(command: list[str], **_kwargs: object) -> _Child:
        commands.append(tuple(command))
        return _Child()

    def fake_killpg(pid: int, sent_signal: signal.Signals) -> None:
        nonlocal running
        if sent_signal != 0:
            signals.append((pid, sent_signal))
            running = False

    descriptor = tmp_path / "agent-process.json"
    monkeypatch.setattr(lifecycle, "_IS_LINUX", True)
    monkeypatch.setattr(lifecycle, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(lifecycle, "AGENT_FILE", descriptor)
    monkeypatch.setattr(lifecycle, "AGENT_LOG", tmp_path / "agent.log")
    monkeypatch.setattr(lifecycle.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        lifecycle,
        "_linux_process_identity",
        lambda _pid: (9876, commands[0]),
    )
    monkeypatch.setattr(lifecycle.os, "killpg", fake_killpg, raising=False)
    monkeypatch.setattr(lifecycle, "_process_group_alive", lambda _pid: running)
    monkeypatch.setattr(
        lifecycle.AgentProcess,
        "persist",
        lambda _self, _path=None: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        lifecycle.launch_agent_process(
            lease_id="lease-secret",
            display=":2",
            window_id=6291460,
            instance_id="bedrock:test",
            role="creative_builder",
            capture_source="x11",
        )

    assert signals == [(4242, signal.SIGTERM)]
    assert not running
    assert not descriptor.exists()


def test_immediate_exit_with_surviving_group_retains_recovery_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[tuple[str, ...]] = []
    cleanup_calls: list[int] = []

    class _Child:
        pid = 4242

        def poll(self) -> int:
            return 1

    def fake_popen(command: list[str], **_kwargs: object) -> _Child:
        commands.append(tuple(command))
        return _Child()

    descriptor = tmp_path / "agent-process.json"
    monkeypatch.setattr(lifecycle, "_IS_LINUX", True)
    monkeypatch.setattr(lifecycle, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(lifecycle, "AGENT_FILE", descriptor)
    monkeypatch.setattr(lifecycle, "AGENT_LOG", tmp_path / "agent.log")
    monkeypatch.setattr(lifecycle.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        lifecycle,
        "_linux_process_identity",
        lambda _pid: (9876, commands[0]),
    )
    monkeypatch.setattr(lifecycle.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        lifecycle,
        "_terminate_spawned_agent_group",
        lambda child: cleanup_calls.append(child.pid) or False,
    )

    with pytest.raises(RuntimeError, match="process cleanup unconfirmed"):
        lifecycle.launch_agent_process(
            lease_id="lease-secret",
            display=":2",
            window_id=6291460,
            instance_id="bedrock:test",
            role="creative_builder",
            capture_source="x11",
        )

    assert cleanup_calls == [4242]
    assert AgentProcess.load(descriptor).pid == 4242


@pytest.mark.parametrize(
    "identity",
    [
        (9999, _agent_command()),
        (1234, (sys.executable, "-c", "import time; time.sleep(60)")),
        None,
    ],
    ids=("pid-reused", "wrong-command", "proc-unverifiable"),
)
def test_stale_or_unverifiable_descriptor_is_never_signaled(
    identity: tuple[int, tuple[str, ...]] | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _agent_command()
    process = _process(command)
    descriptor = tmp_path / "agent-process.json"
    monkeypatch.setattr(lifecycle, "AGENT_FILE", descriptor)
    monkeypatch.setattr(lifecycle, "_IS_LINUX", True)
    monkeypatch.setattr(lifecycle, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(lifecycle, "_linux_process_identity", lambda _pid: identity)
    signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(
        lifecycle.os,
        "killpg",
        lambda pid, sent_signal: signals.append((pid, sent_signal)),
        raising=False,
    )
    process.persist()

    assert lifecycle.stop_agent_process() is False
    assert signals == []
    if identity is None:
        assert descriptor.exists()
    else:
        assert not descriptor.exists()


def test_stop_rechecks_identity_immediately_before_signaling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _agent_command()
    process = _process(command)
    identities = iter(((1234, command), None))
    descriptor = tmp_path / "agent-process.json"
    monkeypatch.setattr(lifecycle, "AGENT_FILE", descriptor)
    monkeypatch.setattr(lifecycle, "_IS_LINUX", True)
    monkeypatch.setattr(lifecycle, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(lifecycle, "_linux_process_identity", lambda _pid: next(identities))
    signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(
        lifecycle.os,
        "killpg",
        lambda pid, sent_signal: signals.append((pid, sent_signal)),
        raising=False,
    )
    process.persist()

    assert lifecycle.stop_agent_process() is False
    assert signals == []
    assert descriptor.exists()


def test_verified_agent_is_signaled_and_descriptor_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _agent_command()
    process = _process(command)
    running = True
    descriptor = tmp_path / "agent-process.json"
    monkeypatch.setattr(lifecycle, "AGENT_FILE", descriptor)
    monkeypatch.setattr(lifecycle, "_IS_LINUX", True)
    monkeypatch.setattr(lifecycle, "_pid_alive", lambda _pid: running)
    monkeypatch.setattr(
        lifecycle,
        "_linux_process_identity",
        lambda _pid: (1234, command) if running else None,
    )
    signals: list[tuple[int, signal.Signals]] = []

    def fake_killpg(pid: int, sent_signal: signal.Signals) -> None:
        nonlocal running
        signals.append((pid, sent_signal))
        running = False

    monkeypatch.setattr(lifecycle.os, "killpg", fake_killpg, raising=False)
    monkeypatch.setattr(lifecycle, "_process_group_alive", lambda _pid: running)
    process.persist()

    assert lifecycle.stop_agent_process(timeout_s=0.1) is True
    assert signals == [(4242, signal.SIGTERM)]
    assert not descriptor.exists()


def test_legacy_descriptor_without_os_identity_is_never_treated_as_alive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = tmp_path / "agent-process.json"
    legacy = AgentProcess(
        pid=4242,
        started_ns=10,
        display=":2",
        window_id=6291460,
        instance_id="bedrock:test",
        role="creative_builder",
    )
    monkeypatch.setattr(lifecycle, "AGENT_FILE", descriptor)
    monkeypatch.setattr(lifecycle, "_IS_LINUX", True)
    monkeypatch.setattr(lifecycle, "_pid_alive", lambda _pid: True)
    legacy.persist()

    assert lifecycle.agent_alive() is False
    assert descriptor.exists()


def test_agent_group_is_killed_after_leader_exits_and_descriptor_then_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _agent_command()
    process = _process(command)
    leader_running = True
    group_running = True
    descriptor = tmp_path / "agent-process.json"
    signals: list[signal.Signals] = []
    monkeypatch.setattr(lifecycle, "AGENT_FILE", descriptor)
    monkeypatch.setattr(lifecycle, "_IS_LINUX", True)
    monkeypatch.setattr(lifecycle, "_pid_alive", lambda _pid: leader_running)
    monkeypatch.setattr(
        lifecycle,
        "_linux_process_identity",
        lambda _pid: (1234, command) if leader_running else None,
    )
    monkeypatch.setattr(lifecycle, "_process_group_alive", lambda _pid: group_running)

    def fake_killpg(_pid: int, sent_signal: signal.Signals) -> None:
        nonlocal leader_running, group_running
        signals.append(sent_signal)
        if len(signals) == 1:
            leader_running = False
        else:
            group_running = False

    monkeypatch.setattr(lifecycle.os, "killpg", fake_killpg, raising=False)
    monkeypatch.setattr(lifecycle.time, "sleep", lambda _seconds: None)
    process.persist()

    assert lifecycle.stop_agent_process(timeout_s=0.001) is True
    assert signals == [signal.SIGTERM, _SIGKILL]
    assert not descriptor.exists()


def test_orphaned_agent_group_is_killed_and_descriptor_then_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _process(_agent_command())
    descriptor = tmp_path / "agent-process.json"
    group_running = True
    signals: list[signal.Signals] = []
    monkeypatch.setattr(lifecycle, "AGENT_FILE", descriptor)
    monkeypatch.setattr(lifecycle, "_IS_LINUX", True)
    monkeypatch.setattr(lifecycle, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(lifecycle, "_process_group_alive", lambda _pid: group_running)

    def fake_killpg(_pid: int, sent_signal: signal.Signals) -> None:
        nonlocal group_running
        signals.append(sent_signal)
        if len(signals) == 2:
            group_running = False

    monkeypatch.setattr(lifecycle.os, "killpg", fake_killpg, raising=False)
    monkeypatch.setattr(lifecycle.time, "sleep", lambda _seconds: None)
    process.persist()

    assert lifecycle.stop_agent_process(timeout_s=0.001) is True
    assert signals == [signal.SIGTERM, _SIGKILL]
    assert not descriptor.exists()


def test_orphaned_agent_group_survivor_retains_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _process(_agent_command())
    descriptor = tmp_path / "agent-process.json"
    signals: list[signal.Signals] = []
    monkeypatch.setattr(lifecycle, "AGENT_FILE", descriptor)
    monkeypatch.setattr(lifecycle, "_IS_LINUX", True)
    monkeypatch.setattr(lifecycle, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(lifecycle, "_process_group_alive", lambda _pid: True)
    monkeypatch.setattr(
        lifecycle.os,
        "killpg",
        lambda _pid, sent_signal: signals.append(sent_signal),
        raising=False,
    )
    monkeypatch.setattr(lifecycle.time, "sleep", lambda _seconds: None)
    process.persist()

    assert lifecycle.stop_agent_process(timeout_s=0.001) is False
    assert signals == [signal.SIGTERM, _SIGKILL]
    assert AgentProcess.load(descriptor) == process


def test_legacy_dead_leader_group_is_never_signaled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = AgentProcess(
        pid=4242,
        started_ns=10,
        display=":2",
        window_id=6291460,
        instance_id="bedrock:test",
        role="creative_builder",
    )
    descriptor = tmp_path / "agent-process.json"
    signals: list[signal.Signals] = []
    monkeypatch.setattr(lifecycle, "AGENT_FILE", descriptor)
    monkeypatch.setattr(lifecycle, "_IS_LINUX", True)
    monkeypatch.setattr(lifecycle, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(lifecycle, "_process_group_alive", lambda _pid: True)
    monkeypatch.setattr(
        lifecycle.os,
        "killpg",
        lambda _pid, sent_signal: signals.append(sent_signal),
        raising=False,
    )
    process.persist()

    assert lifecycle.stop_agent_process(timeout_s=0.001) is False
    assert signals == []
    assert AgentProcess.load(descriptor) == process


def test_agent_signal_failure_retains_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _agent_command()
    process = _process(command)
    descriptor = tmp_path / "agent-process.json"
    monkeypatch.setattr(lifecycle, "AGENT_FILE", descriptor)
    monkeypatch.setattr(lifecycle, "_IS_LINUX", True)
    monkeypatch.setattr(lifecycle, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(lifecycle, "_linux_process_identity", lambda _pid: (1234, command))
    monkeypatch.setattr(lifecycle, "_process_group_alive", lambda _pid: True)
    monkeypatch.setattr(
        lifecycle.os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(PermissionError("denied")),
        raising=False,
    )
    process.persist()

    assert lifecycle.stop_agent_process(timeout_s=0.01) is False
    assert AgentProcess.load(descriptor) == process
