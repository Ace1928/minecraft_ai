from __future__ import annotations

import os
import signal
import sys
from dataclasses import dataclass, field
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
    if identity is None or (
        identity is not None
        and len(identity[1]) >= 3
        and identity[1][1:3] == ("-m", "minecraft_ai.agent_process")
    ):
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


@dataclass
class _StopHarness:
    process: AgentProcess
    descriptor: Path
    identity: tuple[int, tuple[str, ...]] | None
    now: float = 0.0
    leader_running: bool = True
    group_running: bool = True
    leader_exits_on_term: bool = False
    group_exits_on_kill: bool = False
    group_exit_at: float | None = None
    signals: list[tuple[str, int, float]] = field(default_factory=list)
    sleeps: list[float] = field(default_factory=list)

    def sleep(self, seconds: float) -> None:
        assert 0 < seconds <= 0.05
        self.sleeps.append(seconds)
        self.now += seconds

    def group_alive(self, _pid: int) -> bool:
        return self.group_running and (
            self.group_exit_at is None or self.now < self.group_exit_at
        )

    def signal_parent(self, pid: int, sent_signal: int) -> None:
        assert pid == self.process.pid
        self.signals.append(("parent", sent_signal, self.now))
        if self.leader_exits_on_term and sent_signal == signal.SIGTERM:
            self.leader_running = False

    def signal_group(self, pid: int, sent_signal: int) -> None:
        assert pid == self.process.pid
        self.signals.append(("group", sent_signal, self.now))
        if self.group_exits_on_kill and sent_signal == _SIGKILL:
            self.group_running = False


@pytest.fixture
def stop_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _StopHarness:
    command = _agent_command()
    process = _process(command)
    descriptor = tmp_path / "agent-process.json"
    harness = _StopHarness(process, descriptor, (1234, command))
    monkeypatch.setattr(lifecycle, "AGENT_FILE", descriptor)
    monkeypatch.setattr(lifecycle, "_IS_LINUX", True)
    monkeypatch.setattr(lifecycle, "_pid_alive", lambda _pid: harness.leader_running)
    monkeypatch.setattr(lifecycle, "_linux_process_identity", lambda _pid: harness.identity)
    monkeypatch.setattr(lifecycle, "_process_group_alive", harness.group_alive)
    monkeypatch.setattr(lifecycle.os, "kill", harness.signal_parent)
    monkeypatch.setattr(lifecycle.os, "killpg", harness.signal_group, raising=False)
    monkeypatch.setattr(lifecycle.time, "monotonic", lambda: harness.now)
    monkeypatch.setattr(lifecycle.time, "sleep", harness.sleep)
    process.persist()
    return harness


def test_planned_stop_only_signals_parent_while_workers_drain(stop_harness: _StopHarness) -> None:
    stop_harness.group_exit_at = 0.2

    assert lifecycle.stop_agent_process(timeout_s=2, planned=True)

    assert stop_harness.signals == [("parent", signal.SIGTERM, 0.0)]
    assert stop_harness.now == pytest.approx(0.2)
    assert not stop_harness.descriptor.exists()


def test_planned_parent_exit_does_not_cut_short_worker_drain(stop_harness: _StopHarness) -> None:
    stop_harness.leader_exits_on_term = True
    stop_harness.group_exit_at = 0.2

    assert lifecycle.stop_agent_process(timeout_s=2, planned=True)

    assert stop_harness.signals == [("parent", signal.SIGTERM, 0.0)]
    assert stop_harness.now == pytest.approx(0.2)
    assert not stop_harness.descriptor.exists()


@pytest.mark.parametrize("leader_exits", [False, True])
def test_planned_worker_containment_waits_until_grace_deadline(
    stop_harness: _StopHarness, leader_exits: bool
) -> None:
    stop_harness.leader_exits_on_term = leader_exits
    stop_harness.group_exits_on_kill = True

    assert lifecycle.stop_agent_process(timeout_s=2, planned=True)

    assert stop_harness.signals == [
        ("parent", signal.SIGTERM, 0.0),
        ("group", _SIGKILL, 1.5),
    ]
    assert stop_harness.now == 1.5
    assert not stop_harness.descriptor.exists()


@pytest.mark.parametrize("planned,orphan", [(True, False), (False, False), (True, True)])
@pytest.mark.parametrize("timeout_s", [0.0, 0.01, 2.0, 25.0])
def test_stop_has_one_total_wait_budget_including_containment(
    stop_harness: _StopHarness, planned: bool, orphan: bool, timeout_s: float
) -> None:
    stop_harness.leader_running = not orphan

    assert not lifecycle.stop_agent_process(timeout_s=timeout_s, planned=planned)

    grace_deadline = timeout_s - min(1.0, timeout_s / 4)
    assert stop_harness.signals == [
        ("group" if orphan or not planned else "parent", signal.SIGTERM, 0.0),
        ("group", _SIGKILL, grace_deadline),
    ]
    assert stop_harness.now == timeout_s
    assert sum(stop_harness.sleeps) == pytest.approx(timeout_s)
    assert AgentProcess.load(stop_harness.descriptor) == stop_harness.process


@pytest.mark.parametrize("planned,orphan", [(True, False), (False, False), (True, True)])
@pytest.mark.parametrize(
    "replacement",
    [None, (9999, _agent_command()), (9999, (sys.executable, "-c", "pass"))],
    ids=["unverifiable", "different-agent-generation", "unrelated-pid-reuse"],
)
def test_delayed_kill_rechecks_identity_and_preserves_descriptor(
    stop_harness: _StopHarness,
    monkeypatch: pytest.MonkeyPatch,
    planned: bool,
    orphan: bool,
    replacement: tuple[int, tuple[str, ...]] | None,
) -> None:
    stop_harness.leader_running = not orphan
    monkeypatch.setattr(lifecycle, "_original_agent_zombie", lambda _process: False)

    def replace_after_term(seconds: float) -> None:
        stop_harness.sleep(seconds)
        stop_harness.leader_running = True
        stop_harness.identity = replacement

    monkeypatch.setattr(lifecycle.time, "sleep", replace_after_term)

    assert not lifecycle.stop_agent_process(timeout_s=2, planned=planned)

    assert stop_harness.signals == [
        ("group" if orphan or not planned else "parent", signal.SIGTERM, 0.0)
    ]
    assert stop_harness.now == 1.5
    assert AgentProcess.load(stop_harness.descriptor) == stop_harness.process


def test_planned_stop_rechecks_identity_before_parent_term(
    stop_harness: _StopHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    identities = iter((stop_harness.identity, None))
    monkeypatch.setattr(lifecycle, "_linux_process_identity", lambda _pid: next(identities))

    assert not lifecycle.stop_agent_process(timeout_s=2, planned=True)

    assert stop_harness.signals == []
    assert stop_harness.descriptor.exists()


def test_planned_parent_signal_permission_failure_keeps_descriptor(
    stop_harness: _StopHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        lifecycle.os, "kill", lambda *_args: (_ for _ in ()).throw(PermissionError("denied"))
    )

    assert not lifecycle.stop_agent_process(timeout_s=2, planned=True)

    assert stop_harness.signals == []
    assert stop_harness.descriptor.exists()


def test_parent_exit_between_identity_check_and_term_preserves_worker_grace(
    stop_harness: _StopHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    def exit_before_term(_pid: int, _sent_signal: int) -> None:
        stop_harness.leader_running = False
        raise ProcessLookupError

    monkeypatch.setattr(lifecycle.os, "kill", exit_before_term)
    stop_harness.group_exit_at = 0.2

    assert lifecycle.stop_agent_process(timeout_s=2, planned=True)

    assert stop_harness.signals == []
    assert stop_harness.now == pytest.approx(0.2)
    assert not stop_harness.descriptor.exists()


@pytest.mark.parametrize(
    "state,group,session,start_ticks,expected",
    [
        ("Z", 4242, 4242, 1234, True),
        ("X", 4242, 4242, 1234, True),
        ("S", 4242, 4242, 1234, False),
        ("Z", 4243, 4242, 1234, False),
        ("Z", 4242, 4243, 1234, False),
        ("Z", 4242, 4242, 5678, False),
    ],
)
def test_zombie_leader_requires_exact_start_group_and_session_before_delayed_kill(
    stop_harness: _StopHarness,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
    group: int,
    session: int,
    start_ticks: int,
    expected: bool,
) -> None:
    fields = [state, "1", str(group), str(session)] + ["0"] * 15 + [str(start_ticks)]
    stat = f"4242 (python (agent)) {' '.join(fields)}"
    original_read = Path.read_text

    def read_stat(path: Path, *args: object, **kwargs: object) -> str:
        if path == Path("/proc/4242/stat"):
            return stat
        return original_read(path, *args, **kwargs)  # type: ignore[arg-type]

    def become_zombie(seconds: float) -> None:
        stop_harness.sleep(seconds)
        stop_harness.identity = None

    monkeypatch.setattr(Path, "read_text", read_stat)
    monkeypatch.setattr(lifecycle.time, "sleep", become_zombie)
    stop_harness.group_exits_on_kill = True

    assert lifecycle.stop_agent_process(timeout_s=2, planned=True) is expected

    assert stop_harness.signals[0] == ("parent", signal.SIGTERM, 0.0)
    assert stop_harness.signals[1:] == ([("group", _SIGKILL, 1.5)] if expected else [])
    assert stop_harness.descriptor.exists() is not expected


def test_recorded_zombie_at_stop_entry_uses_orphan_group_containment(
    stop_harness: _StopHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    fields = ["Z", "1", "4242", "4242"] + ["0"] * 15 + ["1234"]
    stat = f"4242 (python) {' '.join(fields)}"
    original_read = Path.read_text

    def read_stat(path: Path, *args: object, **kwargs: object) -> str:
        if path == Path("/proc/4242/stat"):
            return stat
        return original_read(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", read_stat)
    stop_harness.identity = None
    stop_harness.group_exits_on_kill = True

    assert lifecycle.stop_agent_process(timeout_s=2, planned=True)

    # Windows has no SIGKILL: the fallback is SIGTERM, so this fake group's
    # configured kill response already completes on the first group signal.
    delayed_kill = [] if _SIGKILL == signal.SIGTERM else [("group", _SIGKILL, 1.5)]
    assert stop_harness.signals == [
        ("group", signal.SIGTERM, 0.0),
    ] + delayed_kill
    assert not stop_harness.descriptor.exists()


@pytest.mark.parametrize("timeout_s", [-1, float("inf"), float("nan")])
def test_invalid_stop_budget_never_signals(stop_harness: _StopHarness, timeout_s: float) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        lifecycle.stop_agent_process(timeout_s=timeout_s, planned=True)
    assert stop_harness.signals == []
    assert stop_harness.now == 0
    assert stop_harness.descriptor.exists()


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX signal-zero semantics")
def test_signal_zero_permission_error_is_not_evidence_of_dead_leader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        lifecycle.os, "kill", lambda *_args: (_ for _ in ()).throw(PermissionError("denied"))
    )
    assert lifecycle._pid_alive(4242)
