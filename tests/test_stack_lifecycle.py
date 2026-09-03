from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from minecraft_ai.config import ModelConfig, RuntimeConfig
from minecraft_ai.stack_lifecycle import (
    FileProbe,
    HttpProbe,
    ManagedModelServer,
    PortableStackLauncher,
    ProcessProbe,
    ServiceMode,
    ServiceSpec,
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
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(0.02)
    pytest.fail(f"process {pid} remained alive")


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
        assert launcher.manifest_path.stat().st_mode & 0o777 == 0o600
    finally:
        stopped = launcher.stop()
    assert stopped is not None and stopped.phase == StackPhase.STOPPED
    for pid in pids:
        _wait_dead(pid)


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
