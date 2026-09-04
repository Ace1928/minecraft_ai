from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import minecraft_ai.stack_lifecycle as stack_module
from minecraft_ai.config import ModelConfig, RuntimeConfig
from minecraft_ai.stack_lifecycle import (
    FileProbe,
    HttpProbe,
    ManagedModelServer,
    PortableStackLauncher,
    ProcessProbe,
    ServiceMode,
    ServiceRecord,
    ServiceSpec,
    StackManifest,
    StackPhase,
    StackPlan,
    StackStartError,
    build_bedrock_stack_plan,
)


def _sleeper_command(ready: Path, *, marker: Path | None = None) -> tuple[str, ...]:
    program = (
        "import os,pathlib,time; "
        f"pathlib.Path({str(ready)!r}).write_text(str(os.getpid())); "
        + ("" if marker is None else f"pathlib.Path({str(marker)!r}).write_text('started'); ")
        + "time.sleep(60)"
    )
    return (sys.executable, "-c", program)


def _wait_dead(pid: int, timeout_s: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not stack_module._pid_alive(pid):
            return
        time.sleep(0.02)
    pytest.fail(f"process {pid} remained alive")


def test_windows_process_probe_is_non_destructive(monkeypatch: pytest.MonkeyPatch) -> None:
    probed: list[int] = []

    def fake_windows_pid_alive(pid: int) -> bool:
        probed.append(pid)
        return False

    monkeypatch.setattr(stack_module, "_IS_WINDOWS", True)
    monkeypatch.setattr(
        stack_module,
        "_windows_pid_alive",
        fake_windows_pid_alive,
    )

    assert stack_module._pid_alive(42) is False
    assert probed == [42]


def test_windows_spawn_avoids_runner_incompatible_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []
    sentinel = object()

    def fake_popen(command: tuple[str, ...], **kwargs: object) -> object:
        calls.append((command, kwargs))
        return sentinel

    service = ServiceSpec(
        service_id="daemon",
        mode=ServiceMode.DAEMON,
        command=("program",),
        probes=(ProcessProbe(),),
        log_path=tmp_path / "daemon.log",
    )
    launcher = PortableStackLauncher(
        StackPlan(profile_id="windows-spawn", services=(service,)),
        runtime_dir=tmp_path / "runtime",
    )
    monkeypatch.setattr(stack_module, "_IS_WINDOWS", True)
    monkeypatch.setattr("minecraft_ai.stack_lifecycle.subprocess.Popen", fake_popen)

    result = launcher._spawn(service, service.command)

    assert result is sentinel
    assert len(calls) == 1
    assert calls[0][0] == ("program",)
    assert calls[0][1]["close_fds"] is True
    assert "creationflags" not in calls[0][1]


def test_windows_termination_stops_tracked_process_tree_and_checks_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    taskkill_calls: list[list[str]] = []

    class _Child:
        def poll(self) -> None:
            return None

        def wait(self, *, timeout: float) -> int:
            calls.append(f"wait:{timeout}")
            return 0

        def kill(self) -> None:
            calls.append("kill")

    service = ServiceSpec(
        service_id="daemon",
        mode=ServiceMode.DAEMON,
        command=("program",),
        probes=(ProcessProbe(),),
    )
    launcher = PortableStackLauncher(
        StackPlan(profile_id="windows-stop", services=(service,)),
        runtime_dir=tmp_path / "runtime",
    )
    launcher._children[42] = _Child()  # type: ignore[assignment]
    monkeypatch.setattr(stack_module, "_IS_WINDOWS", True)
    monkeypatch.setattr(stack_module, "_pid_alive", lambda _pid: True)

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        taskkill_calls.append(command)
        return subprocess.CompletedProcess(command, 255, b"", b"runner-specific status")

    monkeypatch.setattr("minecraft_ai.stack_lifecycle.subprocess.run", fake_run)

    launcher._terminate_owned_pid(42, 2.0, None)

    assert taskkill_calls == [["taskkill", "/PID", "42", "/T", "/F"]]
    assert calls == ["wait:2.0"]
    assert 42 not in launcher._children


def test_windows_reconstructed_process_is_never_signaled_from_pid_alone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ServiceSpec(
        service_id="daemon",
        mode=ServiceMode.DAEMON,
        command=("program",),
        probes=(ProcessProbe(),),
    )
    launcher = PortableStackLauncher(
        StackPlan(profile_id="windows-persisted-stop", services=(service,)),
        runtime_dir=tmp_path / "runtime",
    )
    monkeypatch.setattr(stack_module, "_IS_WINDOWS", True)
    monkeypatch.setattr(stack_module, "_pid_alive", lambda _pid: True)
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 255, b"", b"runner-specific status")

    monkeypatch.setattr("minecraft_ai.stack_lifecycle.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="identity is unverifiable"):
        launcher._terminate_owned_pid(42, 2.0, None)

    assert calls == []


@pytest.mark.parametrize(
    ("command_digest", "proc_start_ticks", "identity"),
    [
        (None, 1234, (1234, ("daemon",))),
        (stack_module._command_digest(("daemon",)), None, (1234, ("daemon",))),
        (stack_module._command_digest(("daemon",)), 1234, None),
        (stack_module._command_digest(("daemon",)), 1234, (9999, ("daemon",))),
        (stack_module._command_digest(("daemon",)), 1234, (1234, ("replacement",))),
    ],
    ids=(
        "missing-command",
        "legacy-missing-start-token",
        "missing-proc-identity",
        "reused-pid",
        "changed-command",
    ),
)
def test_posix_reconstructed_process_requires_complete_matching_identity(
    command_digest: str | None,
    proc_start_ticks: int | None,
    identity: tuple[int, tuple[str, ...]] | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ServiceSpec(
        service_id="daemon",
        mode=ServiceMode.DAEMON,
        command=("daemon",),
        probes=(ProcessProbe(),),
    )
    launcher = PortableStackLauncher(
        StackPlan(profile_id="posix-identity", services=(service,)),
        runtime_dir=tmp_path / "runtime",
    )
    monkeypatch.setattr(stack_module, "_IS_WINDOWS", False)
    monkeypatch.setattr(stack_module, "_IS_LINUX", True)
    monkeypatch.setattr(stack_module, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(stack_module, "_process_group_alive", lambda _pgid: True)
    monkeypatch.setattr(stack_module, "_linux_process_identity", lambda _pid: identity)
    monkeypatch.setattr(
        "minecraft_ai.stack_lifecycle.os.killpg",
        lambda _pid, _signal: pytest.fail("unverified process group must not be signaled"),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="identity changed or is unverifiable"):
        launcher._terminate_owned_pid(
            42,
            0.1,
            command_digest,
            proc_start_ticks,
        )


def test_posix_reconstructed_process_identity_is_rechecked_before_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = ("daemon", "--serve")
    digest = stack_module._command_digest(command)
    identities = iter(((1234, command), (9999, command)))
    service = ServiceSpec(
        service_id="daemon",
        mode=ServiceMode.DAEMON,
        command=command,
        probes=(ProcessProbe(),),
    )
    launcher = PortableStackLauncher(
        StackPlan(profile_id="posix-recheck", services=(service,)),
        runtime_dir=tmp_path / "runtime",
    )
    monkeypatch.setattr(stack_module, "_IS_WINDOWS", False)
    monkeypatch.setattr(stack_module, "_IS_LINUX", True)
    monkeypatch.setattr(stack_module, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(stack_module, "_process_group_alive", lambda _pgid: True)
    monkeypatch.setattr(
        stack_module,
        "_linux_process_identity",
        lambda _pid: next(identities),
    )
    monkeypatch.setattr(
        "minecraft_ai.stack_lifecycle.os.killpg",
        lambda _pid, _signal: pytest.fail("reused process group must not be signaled"),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="identity changed or is unverifiable"):
        launcher._terminate_owned_pid(42, 0.1, digest, 1234)


def test_linux_spawn_retries_transient_proc_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = ("daemon", "--serve")

    class _Child:
        pid = 42
        returncode: int | None = None

        def poll(self) -> None:
            return None

    service = ServiceSpec(
        service_id="daemon",
        mode=ServiceMode.DAEMON,
        command=command,
        probes=(ProcessProbe(),),
    )
    launcher = PortableStackLauncher(
        StackPlan(profile_id="identity-retry", services=(service,)),
        runtime_dir=tmp_path / "runtime",
    )
    identities = iter((None, (1234, command)))
    monkeypatch.setattr(stack_module, "_IS_LINUX", True)
    monkeypatch.setattr(launcher, "_spawn", lambda _service, _command: _Child())
    monkeypatch.setattr(
        launcher,
        "_healthy",
        lambda _service, *, pid, reuse_check=False: not reuse_check,
    )
    monkeypatch.setattr(
        stack_module,
        "_linux_process_identity",
        lambda _pid: next(identities),
    )

    record = launcher._start_service(service)

    assert record.pid == 42
    assert record.proc_start_ticks == 1234
    assert record.command_digest == stack_module._command_digest(command)


def test_posix_tracked_group_is_terminated_after_leader_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ExitedChild:
        def poll(self) -> int:
            return 0

    service = ServiceSpec(
        service_id="daemon",
        mode=ServiceMode.DAEMON,
        command=("daemon",),
        probes=(ProcessProbe(),),
    )
    launcher = PortableStackLauncher(
        StackPlan(profile_id="orphan-group", services=(service,)),
        runtime_dir=tmp_path / "runtime",
        poll_interval_s=0.001,
    )
    launcher._children[42] = _ExitedChild()  # type: ignore[assignment]
    group_alive = True
    signals: list[tuple[int, signal.Signals]] = []

    def fake_killpg(pid: int, sent_signal: signal.Signals) -> None:
        nonlocal group_alive
        signals.append((pid, sent_signal))
        group_alive = False

    monkeypatch.setattr(stack_module, "_IS_WINDOWS", False)
    monkeypatch.setattr(stack_module, "_process_group_alive", lambda _pgid: group_alive)
    monkeypatch.setattr(
        stack_module,
        "_persisted_process_or_orphan_group_matches",
        lambda _pid, **_kwargs: True,
    )
    monkeypatch.setattr(
        "minecraft_ai.stack_lifecycle.os.killpg",
        fake_killpg,
        raising=False,
    )

    launcher._terminate_owned_pid(42, 0.1, "a" * 64, 1234)

    assert signals == [(42, signal.SIGTERM)]
    assert 42 not in launcher._children


def test_posix_group_waits_for_descendants_and_escalates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ExitedChild:
        def poll(self) -> int:
            return 0

    service = ServiceSpec(
        service_id="daemon",
        mode=ServiceMode.DAEMON,
        command=("daemon",),
        probes=(ProcessProbe(),),
    )
    launcher = PortableStackLauncher(
        StackPlan(profile_id="orphan-escalation", services=(service,)),
        runtime_dir=tmp_path / "runtime",
        poll_interval_s=0.001,
    )
    launcher._children[42] = _ExitedChild()  # type: ignore[assignment]
    group_alive = True
    signals: list[tuple[int, signal.Signals]] = []
    kill_signal = getattr(signal, "SIGKILL", signal.SIGTERM)

    def fake_killpg(pid: int, sent_signal: signal.Signals) -> None:
        nonlocal group_alive
        signals.append((pid, sent_signal))
        if len(signals) == 2:
            group_alive = False

    monkeypatch.setattr(stack_module, "_IS_WINDOWS", False)
    monkeypatch.setattr(stack_module, "_process_group_alive", lambda _pgid: group_alive)
    monkeypatch.setattr(
        stack_module,
        "_persisted_process_or_orphan_group_matches",
        lambda _pid, **_kwargs: True,
    )
    monkeypatch.setattr(
        "minecraft_ai.stack_lifecycle.os.killpg",
        fake_killpg,
        raising=False,
    )

    launcher._terminate_owned_pid(42, 0.01, "a" * 64, 1234)

    assert signals == [(42, signal.SIGTERM), (42, kill_signal)]
    assert 42 not in launcher._children


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="requires Linux procfs")
def test_real_reconstructed_orphan_group_is_killed_after_leader_exits(tmp_path: Path) -> None:
    descendant_ready = tmp_path / "descendant.ready"
    leader_exit = tmp_path / "leader.exit"
    descendant_program = (
        "import os,pathlib,signal,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"pathlib.Path({str(descendant_ready)!r}).write_text(str(os.getpid())); "
        "time.sleep(60)"
    )
    leader_program = (
        "import pathlib,subprocess,sys,time\n"
        f"subprocess.Popen([sys.executable, '-c', {descendant_program!r}])\n"
        f"p=pathlib.Path({str(descendant_ready)!r})\n"
        f"exit_gate=pathlib.Path({str(leader_exit)!r})\n"
        "deadline=time.monotonic()+3\n"
        "while not p.exists() and time.monotonic()<deadline:\n"
        "    time.sleep(0.01)\n"
        "while not exit_gate.exists():\n"
        "    time.sleep(0.01)\n"
    )
    service = ServiceSpec(
        service_id="real-orphan",
        mode=ServiceMode.DAEMON,
        command=(sys.executable, "-c", leader_program),
        probes=(ProcessProbe(),),
        log_path=tmp_path / "real-orphan.log",
    )
    launcher = PortableStackLauncher(
        StackPlan(profile_id="real-orphan", services=(service,)),
        runtime_dir=tmp_path / "runtime",
        poll_interval_s=0.005,
    )
    leader = launcher._spawn(service, service.command)
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and not descendant_ready.exists():
        time.sleep(0.01)
    assert descendant_ready.exists()
    identity = stack_module._linux_process_identity(leader.pid)
    assert identity is not None
    proc_start_ticks, actual_command = identity
    command_digest = stack_module._command_digest(actual_command)
    leader_exit.write_text("exit", encoding="utf-8")
    assert leader.wait(timeout=3.0) == 0
    assert not stack_module._pid_alive(leader.pid)
    assert stack_module._process_group_alive(leader.pid)

    try:
        reconstructed = PortableStackLauncher(
            launcher.plan,
            runtime_dir=tmp_path / "reconstructed-runtime",
            poll_interval_s=0.005,
        )
        reconstructed._terminate_owned_pid(
            leader.pid,
            0.05,
            command_digest,
            proc_start_ticks,
        )
        assert not stack_module._process_group_alive(leader.pid)
    finally:
        if stack_module._process_group_alive(leader.pid):
            os.killpg(leader.pid, signal.SIGKILL)


def test_reconstructed_orphan_group_survivor_retains_owned_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ServiceSpec(
        service_id="daemon",
        mode=ServiceMode.DAEMON,
        command=("daemon",),
        probes=(ProcessProbe(),),
        stop_timeout_s=0.1,
    )
    plan = StackPlan(profile_id="orphan-survivor", services=(service,))
    launcher = PortableStackLauncher(
        plan,
        runtime_dir=tmp_path / "runtime",
        poll_interval_s=0.05,
    )
    record = ServiceRecord(
        service_id="daemon",
        mode=ServiceMode.DAEMON,
        owned=True,
        pid=42,
        started_wall_ns=1,
        command_digest=stack_module._command_digest(service.command),
        proc_start_ticks=1234,
    )
    StackManifest(
        schema_version=1,
        transaction_id="transaction",
        profile_id=plan.profile_id,
        plan_digest=plan.digest,
        phase=StackPhase.RUNNING,
        created_wall_ns=1,
        updated_wall_ns=1,
        services=(record,),
    ).persist(launcher.manifest_path)
    clock = 0.0
    signals: list[tuple[int, signal.Signals]] = []

    def fake_monotonic() -> float:
        return clock

    def fake_sleep(seconds: float) -> None:
        nonlocal clock
        clock += seconds

    monkeypatch.setattr(stack_module, "_IS_WINDOWS", False)
    monkeypatch.setattr(stack_module, "_IS_LINUX", True)
    monkeypatch.setattr(stack_module, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(stack_module, "_process_group_alive", lambda _pgid: True)
    monkeypatch.setattr(
        stack_module,
        "_linux_process_group_members",
        lambda _pgid: ((43, "S"),),
    )
    monkeypatch.setattr(
        "minecraft_ai.stack_lifecycle.os.killpg",
        lambda pid, sent_signal: signals.append((pid, sent_signal)),
        raising=False,
    )
    monkeypatch.setattr("minecraft_ai.stack_lifecycle.time.monotonic", fake_monotonic)
    monkeypatch.setattr("minecraft_ai.stack_lifecycle.time.sleep", fake_sleep)

    with pytest.raises(StackStartError, match="process group 42 did not terminate"):
        launcher.stop()

    failed = StackManifest.load(launcher.manifest_path)
    assert failed.phase == StackPhase.FAILED
    assert failed.services == (record,)
    assert signals == [
        (42, signal.SIGTERM),
        (42, getattr(signal, "SIGKILL", signal.SIGTERM)),
    ]


def test_stack_starts_in_dependency_order_is_idempotent_and_stops(tmp_path: Path) -> None:
    first_ready = tmp_path / "first.ready"
    second_ready = tmp_path / "second.ready"
    plan = StackPlan(
        profile_id="test-live",
        services=(
            ServiceSpec(
                service_id="second",
                mode=ServiceMode.DAEMON,
                command=_sleeper_command(second_ready),
                probes=(ProcessProbe(), FileProbe(second_ready)),
                dependencies=("first",),
                log_path=tmp_path / "second.log",
                ready_timeout_s=3.0,
            ),
            ServiceSpec(
                service_id="first",
                mode=ServiceMode.DAEMON,
                command=_sleeper_command(first_ready),
                probes=(ProcessProbe(), FileProbe(first_ready)),
                log_path=tmp_path / "first.log",
                ready_timeout_s=3.0,
            ),
        ),
    )
    assert [service.service_id for service in plan.ordered_services()] == ["first", "second"]
    launcher = PortableStackLauncher(plan, runtime_dir=tmp_path / "runtime")
    manifest = launcher.start()
    pids = tuple(record.pid for record in manifest.services if record.pid is not None)
    try:
        assert manifest.phase == StackPhase.RUNNING
        assert len(pids) == 2
        again = launcher.start()
        assert again.transaction_id == manifest.transaction_id
        assert tuple(record.pid for record in again.services if record.pid is not None) == pids
        persisted = json.loads(launcher.manifest_path.read_text(encoding="utf-8"))
        assert persisted["plan_digest"] == plan.digest
        if sys.platform.startswith("linux"):
            for record in manifest.services:
                assert record.pid is not None
                assert record.proc_start_ticks is not None
                assert record.proc_start_ticks > 0
                assert record.command_digest is not None
                assert stack_module._persisted_pid_matches(
                    record.pid,
                    command_digest=record.command_digest,
                    proc_start_ticks=record.proc_start_ticks,
                )
        if os.name != "nt":
            assert launcher.manifest_path.stat().st_mode & 0o777 == 0o600
    finally:
        stopped = launcher.stop()
    assert stopped is not None and stopped.phase == StackPhase.STOPPED
    for pid in pids:
        _wait_dead(pid)


def test_stack_kernel_lock_cannot_be_replaced_by_concurrent_launcher(
    tmp_path: Path,
) -> None:
    service = ServiceSpec(
        service_id="external",
        mode=ServiceMode.EXTERNAL,
        probes=(FileProbe(tmp_path / "ready"),),
    )
    plan = StackPlan(profile_id="lock-test", services=(service,))
    first = PortableStackLauncher(plan, runtime_dir=tmp_path / "runtime")
    second = PortableStackLauncher(plan, runtime_dir=tmp_path / "runtime")

    with first._lock():
        with pytest.raises(RuntimeError, match="lifecycle operation is active"):
            with second._lock():
                pytest.fail("concurrent lifecycle operation acquired the same kernel lock")


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="requires Linux procfs")
def test_legacy_manifest_loads_but_cannot_signal_unverifiable_process(tmp_path: Path) -> None:
    ready = tmp_path / "legacy.ready"
    service = ServiceSpec(
        service_id="daemon",
        mode=ServiceMode.DAEMON,
        command=_sleeper_command(ready),
        probes=(ProcessProbe(), FileProbe(ready)),
        log_path=tmp_path / "legacy.log",
        ready_timeout_s=3.0,
    )
    plan = StackPlan(profile_id="legacy-manifest", services=(service,))
    launcher = PortableStackLauncher(plan, runtime_dir=tmp_path / "runtime")
    manifest = launcher.start()
    record = manifest.services[0]
    assert record.pid is not None
    assert record.proc_start_ticks is not None
    raw = json.loads(launcher.manifest_path.read_text(encoding="utf-8"))
    del raw["services"][0]["proc_start_ticks"]
    launcher.manifest_path.write_text(json.dumps(raw), encoding="utf-8")
    reconstructed = PortableStackLauncher(plan, runtime_dir=tmp_path / "runtime")

    try:
        loaded, health = reconstructed.status()
        assert loaded is not None
        assert loaded.services[0].proc_start_ticks is None
        assert health == {"daemon": False}
        with pytest.raises(StackStartError, match="persisted process identity"):
            reconstructed.stop()
        assert stack_module._pid_alive(record.pid)
    finally:
        launcher._terminate_owned_pid(
            record.pid,
            service.stop_timeout_s,
            record.command_digest,
            record.proc_start_ticks,
        )
    _wait_dead(record.pid)


def test_stack_failure_rolls_back_only_owned_services(tmp_path: Path) -> None:
    daemon_ready = tmp_path / "daemon.ready"
    external_ready = tmp_path / "external.ready"
    external_ready.write_text("operator-owned", encoding="utf-8")
    never_healthy = tmp_path / "missing.ready"
    plan = StackPlan(
        profile_id="test-rollback",
        services=(
            ServiceSpec(
                service_id="operator-model",
                mode=ServiceMode.EXTERNAL,
                probes=(FileProbe(external_ready),),
            ),
            ServiceSpec(
                service_id="owned-daemon",
                mode=ServiceMode.DAEMON,
                command=_sleeper_command(daemon_ready),
                probes=(ProcessProbe(), FileProbe(daemon_ready)),
                dependencies=("operator-model",),
                log_path=tmp_path / "owned.log",
                ready_timeout_s=3.0,
            ),
            ServiceSpec(
                service_id="required-endpoint",
                mode=ServiceMode.EXTERNAL,
                probes=(FileProbe(never_healthy),),
                dependencies=("owned-daemon",),
                ready_timeout_s=0.1,
            ),
        ),
    )
    launcher = PortableStackLauncher(plan, runtime_dir=tmp_path / "runtime")

    with pytest.raises(StackStartError, match="required external service"):
        launcher.start()

    manifest, health = launcher.status()
    assert manifest is not None and manifest.phase == StackPhase.FAILED
    owned_pid = int(daemon_ready.read_text(encoding="utf-8"))
    rolled_back = next(
        record for record in manifest.services if record.service_id == "owned-daemon"
    )
    assert rolled_back.owned is False and rolled_back.pid is None
    _wait_dead(owned_pid)
    reused = next(record for record in manifest.services if record.service_id == "operator-model")
    assert reused.owned is False
    assert external_ready.read_text(encoding="utf-8") == "operator-owned"
    assert health["operator-model"] is True


def test_failed_oneshot_is_provisionally_owned_before_wait_and_runs_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start_command = ("resource-helper", "start")
    stop_command = ("resource-helper", "stop")
    service = ServiceSpec(
        service_id="provisional-oneshot",
        mode=ServiceMode.ONESHOT,
        command=start_command,
        stop_command=stop_command,
        probes=(FileProbe(tmp_path / "never-ready"),),
        log_path=tmp_path / "oneshot.log",
        ready_timeout_s=0.1,
    )
    launcher = PortableStackLauncher(
        StackPlan(profile_id="provisional-oneshot", services=(service,)),
        runtime_dir=tmp_path / "runtime",
    )
    spawned_commands: list[tuple[str, ...]] = []
    stop_ran = False

    class _Process:
        returncode: int | None = None

        def __init__(self, pid: int, command: tuple[str, ...]) -> None:
            self.pid = pid
            self.command = command

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, *, timeout: float) -> int:
            nonlocal stop_ran
            assert timeout == service.ready_timeout_s
            if self.command == start_command:
                provisional = StackManifest.load(launcher.manifest_path)
                assert len(provisional.services) == 1
                assert provisional.services[0].service_id == service.service_id
                assert provisional.services[0].owned is True
                self.returncode = 17
            else:
                assert self.command == stop_command
                stop_ran = True
                self.returncode = 0
            return self.returncode

    def fake_spawn(
        _service: ServiceSpec,
        command: tuple[str, ...],
    ) -> _Process:
        spawned_commands.append(command)
        return _Process(40 + len(spawned_commands), command)

    monkeypatch.setattr(stack_module, "_IS_LINUX", False)
    monkeypatch.setattr(launcher, "_spawn", fake_spawn)

    with pytest.raises(StackStartError, match="exited with code 17"):
        launcher.start()

    assert spawned_commands == [start_command, stop_command]
    assert stop_ran is True
    failed = StackManifest.load(launcher.manifest_path)
    assert failed.phase == StackPhase.FAILED
    assert len(failed.services) == 1
    assert failed.services[0].owned is False


def test_oneshot_provisional_persist_failure_contains_helper_and_runs_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start_command = ("resource-helper", "start")
    stop_command = ("resource-helper", "stop")
    service = ServiceSpec(
        service_id="provisional-persist-failure",
        mode=ServiceMode.ONESHOT,
        command=start_command,
        stop_command=stop_command,
        probes=(FileProbe(tmp_path / "never-ready"),),
        log_path=tmp_path / "oneshot.log",
        ready_timeout_s=0.1,
    )
    launcher = PortableStackLauncher(
        StackPlan(profile_id="oneshot-persist-failure", services=(service,)),
        runtime_dir=tmp_path / "runtime",
    )
    original_persist = stack_module.StackManifest.persist
    persist_calls = 0
    spawned_commands: list[tuple[str, ...]] = []
    contained: list[tuple[int, str | None, int | None]] = []

    class _Process:
        returncode: int | None = None

        def __init__(self, pid: int, command: tuple[str, ...]) -> None:
            self.pid = pid
            self.command = command

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, *, timeout: float) -> int:
            assert timeout == service.ready_timeout_s
            assert self.command == stop_command
            self.returncode = 0
            return self.returncode

    def fake_spawn(
        _service: ServiceSpec,
        command: tuple[str, ...],
    ) -> _Process:
        spawned_commands.append(command)
        return _Process(50 + len(spawned_commands), command)

    def fail_provisional_persist(self: StackManifest, path: Path) -> None:
        nonlocal persist_calls
        persist_calls += 1
        if persist_calls == 2:
            raise OSError("injected provisional persistence failure")
        original_persist(self, path)

    def fake_terminate(
        pid: int,
        _timeout_s: float,
        command_digest: str | None,
        proc_start_ticks: int | None = None,
    ) -> None:
        contained.append((pid, command_digest, proc_start_ticks))

    monkeypatch.setattr(stack_module, "_IS_LINUX", False)
    monkeypatch.setattr(launcher, "_spawn", fake_spawn)
    monkeypatch.setattr(launcher, "_terminate_owned_pid", fake_terminate)
    monkeypatch.setattr(stack_module.StackManifest, "persist", fail_provisional_persist)

    with pytest.raises(StackStartError, match="injected provisional persistence failure"):
        launcher.start()

    assert spawned_commands == [start_command, stop_command]
    assert contained == [(51, stack_module._command_digest(start_command), None)]
    failed = StackManifest.load(launcher.manifest_path)
    assert failed.phase == StackPhase.FAILED
    assert len(failed.services) == 1
    assert failed.services[0].owned is False


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="requires Linux procfs")
def test_post_spawn_manifest_failure_always_rolls_back_owned_daemon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ready = tmp_path / "daemon.ready"
    service = ServiceSpec(
        service_id="owned-daemon",
        mode=ServiceMode.DAEMON,
        command=_sleeper_command(ready),
        probes=(ProcessProbe(), FileProbe(ready)),
        log_path=tmp_path / "owned.log",
        ready_timeout_s=3.0,
    )
    launcher = PortableStackLauncher(
        StackPlan(profile_id="persist-failure", services=(service,)),
        runtime_dir=tmp_path / "runtime",
        poll_interval_s=0.005,
    )
    original_persist = stack_module.StackManifest.persist
    original_spawn = launcher._spawn
    persist_calls = 0
    spawned_pids: list[int] = []

    def capture_spawn(
        selected_service: ServiceSpec,
        command: tuple[str, ...],
    ) -> subprocess.Popen[bytes]:
        process = original_spawn(selected_service, command)
        spawned_pids.append(process.pid)
        return process

    def fail_after_spawn(self: object, path: Path) -> None:
        nonlocal persist_calls
        persist_calls += 1
        if persist_calls >= 2:
            raise OSError("injected manifest persistence failure")
        original_persist(self, path)  # type: ignore[arg-type]

    monkeypatch.setattr(launcher, "_spawn", capture_spawn)
    monkeypatch.setattr(stack_module.StackManifest, "persist", fail_after_spawn)

    with pytest.raises(StackStartError, match="manifest persistence failure") as raised:
        launcher.start()

    assert len(spawned_pids) == 1
    pid = spawned_pids[0]
    _wait_dead(pid)
    assert not stack_module._process_group_alive(pid)
    assert any("could not persist" in error for error in raised.value.rollback_errors)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="requires Linux procfs")
def test_pre_ready_daemon_exit_reaps_same_group_descendant(tmp_path: Path) -> None:
    leader_pid_path = tmp_path / "leader.pid"
    descendant_pid_path = tmp_path / "descendant.pid"
    never_ready = tmp_path / "never.ready"
    descendant_program = (
        "import os,pathlib,signal,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"pathlib.Path({str(descendant_pid_path)!r}).write_text(str(os.getpid())); "
        "time.sleep(60)"
    )
    leader_program = (
        "import os,pathlib,subprocess,sys,time\n"
        f"pathlib.Path({str(leader_pid_path)!r}).write_text(str(os.getpid()))\n"
        f"subprocess.Popen([sys.executable, '-c', {descendant_program!r}])\n"
        f"descendant=pathlib.Path({str(descendant_pid_path)!r})\n"
        "deadline=time.monotonic()+3\n"
        "while not descendant.exists() and time.monotonic()<deadline:\n"
        "    time.sleep(0.01)\n"
        "time.sleep(0.2)\n"
    )
    service = ServiceSpec(
        service_id="pre-ready-orphan",
        mode=ServiceMode.DAEMON,
        command=(sys.executable, "-c", leader_program),
        probes=(ProcessProbe(), FileProbe(never_ready)),
        log_path=tmp_path / "pre-ready-orphan.log",
        ready_timeout_s=2.0,
        stop_timeout_s=0.1,
    )
    launcher = PortableStackLauncher(
        StackPlan(profile_id="pre-ready-orphan", services=(service,)),
        runtime_dir=tmp_path / "runtime",
        poll_interval_s=0.005,
    )

    try:
        with pytest.raises(StackStartError, match="exited with code 0"):
            launcher.start()
        leader_pid = int(leader_pid_path.read_text(encoding="utf-8"))
        assert descendant_pid_path.exists()
        assert not stack_module._process_group_alive(leader_pid)
        assert not stack_module._pid_alive(leader_pid)
        failed = StackManifest.load(launcher.manifest_path)
        assert failed.phase == StackPhase.FAILED
        assert len(failed.services) == 1
        assert failed.services[0].owned is False
        assert failed.services[0].pid is None
    finally:
        if leader_pid_path.exists():
            leader_pid = int(leader_pid_path.read_text(encoding="utf-8"))
            if stack_module._process_group_alive(leader_pid):
                os.killpg(leader_pid, signal.SIGKILL)


@pytest.mark.parametrize("identity_mismatch", [False, True])
def test_pre_ready_cleanup_survivor_retains_provisional_identity(
    identity_mismatch: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = ("daemon", "--serve")
    observed_command = ("replacement", "--serve") if identity_mismatch else command

    class _ExitedChild:
        pid = 42
        returncode = 17

        def poll(self) -> int:
            return self.returncode

    service = ServiceSpec(
        service_id="provisional-daemon",
        mode=ServiceMode.DAEMON,
        command=command,
        probes=(ProcessProbe(),),
        ready_timeout_s=0.1,
        stop_timeout_s=0.1,
    )
    plan = StackPlan(profile_id="provisional-survivor", services=(service,))
    launcher = PortableStackLauncher(
        plan,
        runtime_dir=tmp_path / "runtime",
        poll_interval_s=0.05,
    )
    clock = 0.0
    signals: list[tuple[int, signal.Signals]] = []

    def fake_monotonic() -> float:
        return clock

    def fake_sleep(seconds: float) -> None:
        nonlocal clock
        clock += seconds

    monkeypatch.setattr(stack_module, "_IS_WINDOWS", False)
    monkeypatch.setattr(stack_module, "_IS_LINUX", True)
    monkeypatch.setattr(stack_module, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(launcher, "_spawn", lambda _service, _command: _ExitedChild())
    monkeypatch.setattr(
        launcher,
        "_healthy",
        lambda _service, *, pid, reuse_check=False: False,
    )
    monkeypatch.setattr(stack_module, "_linux_process_start_ticks", lambda _pid: 1234)
    monkeypatch.setattr(
        stack_module,
        "_linux_process_identity",
        lambda _pid: (1234, observed_command),
    )
    monkeypatch.setattr(stack_module, "_process_group_alive", lambda _pgid: True)
    monkeypatch.setattr(
        "minecraft_ai.stack_lifecycle.os.killpg",
        lambda pid, sent_signal: signals.append((pid, sent_signal)),
        raising=False,
    )
    monkeypatch.setattr("minecraft_ai.stack_lifecycle.time.monotonic", fake_monotonic)
    monkeypatch.setattr("minecraft_ai.stack_lifecycle.time.sleep", fake_sleep)
    expected = (
        "could not establish process identity" if identity_mismatch else "exited with code 17"
    )

    with pytest.raises(StackStartError, match=expected):
        launcher.start()

    failed = StackManifest.load(launcher.manifest_path)
    assert failed.phase == StackPhase.FAILED
    assert len(failed.services) == 1
    provisional = failed.services[0]
    assert provisional.owned is True
    assert provisional.pid == 42
    assert provisional.proc_start_ticks == 1234
    assert provisional.command_digest == stack_module._command_digest(observed_command)
    assert signals == [
        (42, signal.SIGTERM),
        (42, getattr(signal, "SIGKILL", signal.SIGTERM)),
    ]


def test_external_gate_waits_for_readiness_without_claiming_ownership(tmp_path: Path) -> None:
    ready = tmp_path / "world.ready"
    plan = StackPlan(
        profile_id="test-gate",
        services=(
            ServiceSpec(
                service_id="world-ready",
                mode=ServiceMode.EXTERNAL,
                probes=(FileProbe(ready),),
                ready_timeout_s=2.0,
            ),
        ),
    )
    launcher = PortableStackLauncher(plan, runtime_dir=tmp_path / "runtime")
    timer = threading.Timer(0.1, lambda: ready.write_text("hud", encoding="utf-8"))
    timer.start()
    try:
        manifest = launcher.start()
    finally:
        timer.join()
    assert manifest.phase == StackPhase.RUNNING
    assert manifest.services[0].owned is False
    launcher.stop()
    assert ready.read_text(encoding="utf-8") == "hud"


def test_healthy_daemon_endpoint_is_reused_without_running_command(tmp_path: Path) -> None:
    endpoint = tmp_path / "healthy"
    endpoint.write_text("ready", encoding="utf-8")
    started = tmp_path / "should-not-start"
    plan = StackPlan(
        profile_id="test-reuse",
        services=(
            ServiceSpec(
                service_id="model",
                mode=ServiceMode.DAEMON,
                command=_sleeper_command(endpoint, marker=started),
                probes=(ProcessProbe(), FileProbe(endpoint)),
                log_path=tmp_path / "model.log",
            ),
        ),
    )
    launcher = PortableStackLauncher(plan, runtime_dir=tmp_path / "runtime")

    manifest = launcher.start()
    assert manifest.services[0].owned is False
    assert manifest.services[0].pid is None
    assert not started.exists()
    launcher.stop()
    assert endpoint.exists()


def test_restarting_stopped_manifest_does_not_stop_replacement_resource(tmp_path: Path) -> None:
    ready = tmp_path / "resource.ready"
    stops = tmp_path / "stops"
    start_command = (
        sys.executable,
        "-c",
        f"import pathlib; pathlib.Path({str(ready)!r}).write_text('ready')",
    )
    stop_command = (
        sys.executable,
        "-c",
        (
            "import pathlib; "
            f"p=pathlib.Path({str(stops)!r}); "
            "p.write_text(p.read_text()+'x' if p.exists() else 'x'); "
            f"pathlib.Path({str(ready)!r}).unlink(missing_ok=True)"
        ),
    )
    plan = StackPlan(
        profile_id="test-restart",
        services=(
            ServiceSpec(
                service_id="resource",
                mode=ServiceMode.ONESHOT,
                command=start_command,
                stop_command=stop_command,
                probes=(FileProbe(ready),),
                log_path=tmp_path / "resource.log",
            ),
        ),
    )
    launcher = PortableStackLauncher(plan, runtime_dir=tmp_path / "runtime")

    launcher.start()
    launcher.stop()
    assert stops.read_text(encoding="utf-8") == "x"
    ready.write_text("replacement", encoding="utf-8")

    restarted = launcher.start()
    assert restarted.services[0].owned is False
    assert stops.read_text(encoding="utf-8") == "x"
    launcher.stop()
    assert ready.read_text(encoding="utf-8") == "replacement"


def test_stack_plan_rejects_missing_and_cyclic_dependencies() -> None:
    service = ServiceSpec(
        service_id="a",
        mode=ServiceMode.EXTERNAL,
        probes=(FileProbe(Path("healthy")),),
        dependencies=("missing",),
    )
    with pytest.raises(ValueError, match="unknown dependencies"):
        StackPlan(profile_id="invalid", services=(service,))

    a = ServiceSpec(
        service_id="a",
        mode=ServiceMode.EXTERNAL,
        probes=(FileProbe(Path("a")),),
        dependencies=("b",),
    )
    b = ServiceSpec(
        service_id="b",
        mode=ServiceMode.EXTERNAL,
        probes=(FileProbe(Path("b")),),
        dependencies=("a",),
    )
    with pytest.raises(ValueError, match="dependency cycle"):
        StackPlan(profile_id="invalid", services=(a, b))


def test_bedrock_stack_plan_covers_full_runtime_and_external_model_gate(tmp_path: Path) -> None:
    config = RuntimeConfig(
        high_level=ModelConfig(
            enabled=True,
            model_id="planner",
            base_url="http://127.0.0.1:8080/v1",
        ),
        vision_language=ModelConfig(
            enabled=True,
            model_id="vision",
            base_url="http://127.0.0.1:8081/v1",
        ),
    )
    vision = ManagedModelServer(
        command=("llama-server", "--port", "8081"),
        health_url="http://127.0.0.1:8081/health",
    )
    plan = build_bedrock_stack_plan(
        config,
        python_executable="python3",
        vision_server=vision,
        log_dir=tmp_path,
    )
    by_id = {service.service_id: service for service in plan.services}

    assert set(by_id) == {
        "safety-gate",
        "dashboard",
        "bedrock",
        "high-level-model",
        "vision-model",
        "supervisor",
        "world-ready",
        "live-agent",
    }
    assert by_id["high-level-model"].mode == ServiceMode.EXTERNAL
    assert by_id["vision-model"].mode == ServiceMode.DAEMON
    assert isinstance(by_id["vision-model"].probes[1], HttpProbe)
    assert by_id["live-agent"].mode == ServiceMode.ONESHOT
    assert set(by_id["live-agent"].dependencies) == {
        "dashboard",
        "bedrock",
        "high-level-model",
        "vision-model",
        "supervisor",
        "world-ready",
    }
    assert [service.service_id for service in plan.ordered_services()] == [
        "safety-gate",
        "dashboard",
        "bedrock",
        "high-level-model",
        "vision-model",
        "supervisor",
        "world-ready",
        "live-agent",
    ]
