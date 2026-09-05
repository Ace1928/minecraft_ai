"""Pre-run cleanup owns resources, never supervisor authority or game inputs."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

import minecraft_ai.runtime as runtime_module
from minecraft_ai.runtime import AgentRuntime


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    item = AgentRuntime(
        perception=SimpleNamespace(close=Mock()),  # type: ignore[arg-type]
        blackboard=SimpleNamespace(),  # type: ignore[arg-type]
        executor=SimpleNamespace(run=object(), close=Mock(), cancel=Mock()),  # type: ignore[arg-type]
        skills=SimpleNamespace(),  # type: ignore[arg-type]
        role=SimpleNamespace(),  # type: ignore[arg-type]
        lease_id="pre-run-test",
        trajectory=SimpleNamespace(close=Mock()),  # type: ignore[arg-type]
        state_db=SimpleNamespace(close=Mock()),  # type: ignore[arg-type]
        telemetry=SimpleNamespace(publish=Mock()),  # type: ignore[arg-type]
    )
    forbidden = Mock(side_effect=AssertionError("pre-run cleanup must not affect control state"))
    monkeypatch.setattr(runtime_module, "send_command", forbidden)
    for name in ("_send_motor", "_failsafe", "_record_terminal_run", "_telemetry_payload"):
        monkeypatch.setattr(item, name, forbidden)
    try:
        yield item
    finally:
        item._pool.shutdown(wait=False, cancel_futures=True)
        item._pool._thread.join(timeout=1.0)
        assert not item._pool._thread.is_alive()
        forbidden.assert_not_called()
        item.executor.cancel.assert_not_called()
        item.state_db.close.assert_not_called()
        item.telemetry.publish.assert_not_called()


def test_default_pre_run_cleanup_drains_actual_idle_pool_without_control_effects(
    runtime: Any,
) -> None:
    resources = (runtime.perception, runtime.executor, runtime.trajectory, runtime.state_db)
    assert runtime.close_before_run()
    assert runtime._stop.is_set()
    assert not runtime._pool._thread.is_alive()
    runtime.perception.close.assert_called_once_with()
    runtime.executor.close.assert_called_once_with()
    assert 0 < runtime.trajectory.close.call_args.kwargs["timeout_s"] <= 2.0
    assert resources == (runtime.perception, runtime.executor, runtime.trajectory, runtime.state_db)
    with pytest.raises(RuntimeError, match="after shutdown"):
        runtime._pool.submit(lambda: None)
    assert runtime.close_before_run(timeout_s=0.0)
    runtime.perception.close.assert_called_once()
    runtime.executor.close.assert_called_once()
    runtime.trajectory.close.assert_called_once()


def test_pre_run_cleanup_handles_absent_trajectory(runtime: Any) -> None:
    runtime.trajectory = None
    assert runtime.close_before_run()
    assert runtime.close_before_run()
    runtime.perception.close.assert_called_once()
    runtime.executor.close.assert_called_once()


def test_run_entry_is_marked_before_even_period_calculation(runtime: Any) -> None:
    runtime.motor_hz = 0.0
    with pytest.raises(ZeroDivisionError):
        runtime.run_forever()
    assert runtime._run_started
    assert not runtime.close_before_run()
    assert runtime._stop.is_set()
    assert not runtime._pool._shutdown
    runtime.perception.close.assert_not_called()
    runtime.executor.close.assert_not_called()
    runtime.trajectory.close.assert_not_called()


def test_blocked_pool_cancels_queue_but_retains_resources_until_drained(runtime: Any) -> None:
    started, release = threading.Event(), threading.Event()

    def blocked() -> None:
        started.set()
        assert release.wait(timeout=2.0)

    running = runtime._pool.submit(blocked)
    try:
        assert started.wait(timeout=1.0)
        queued = runtime._pool.submit(Mock(side_effect=AssertionError("queued work must cancel")))
        before = time.monotonic()
        assert not runtime.close_before_run(timeout_s=0.01)
        assert time.monotonic() - before < 1.0
        assert runtime._stop.is_set() and running.running() and queued.cancelled()
        runtime.perception.close.assert_not_called()
        runtime.executor.close.assert_not_called()
        runtime.trajectory.close.assert_not_called()
    finally:
        release.set()
        running.result(timeout=1.0)
    assert runtime.close_before_run()
    assert not runtime._pool._thread.is_alive()


@pytest.mark.parametrize("failed_resource", ["perception", "executor", "trajectory"])
def test_failed_resource_close_retries_without_reclosing_successes(
    runtime: Any, failed_resource: str,
) -> None:
    names = ("perception", "executor", "trajectory")
    owned = tuple(getattr(runtime, name) for name in names)
    getattr(runtime, failed_resource).close.side_effect = [
        RuntimeError("synthetic close error"), None,
    ]
    assert not runtime.close_before_run()
    failure_index = names.index(failed_resource)
    for index, resource in enumerate(owned):
        assert resource.close.call_count == int(index <= failure_index)
    assert runtime.close_before_run()
    assert runtime.close_before_run()
    for name, resource in zip(names, owned, strict=True):
        assert getattr(runtime, name) is resource
        assert resource.close.call_count == (2 if name == failed_resource else 1)


def test_pool_drain_error_retains_resources_for_retry(
    runtime: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as patch:
        patch.setattr(runtime._pool, "wait_closed", Mock(side_effect=RuntimeError("drain failed")))
        assert not runtime.close_before_run()
    runtime.perception.close.assert_not_called()
    runtime.executor.close.assert_not_called()
    runtime.trajectory.close.assert_not_called()
    assert runtime.close_before_run()


def test_trajectory_receives_only_remaining_cleanup_timeout(
    runtime: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [100.0]
    monkeypatch.setattr(runtime_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    original_wait = runtime._pool.wait_closed

    def drain(timeout_s: float) -> bool:
        assert timeout_s == 2.0
        result = original_wait(timeout_s)
        clock[0] += 0.25
        return result

    monkeypatch.setattr(runtime._pool, "wait_closed", drain)
    runtime.perception.close.side_effect = lambda: clock.__setitem__(0, clock[0] + 0.25)
    runtime.executor.close.side_effect = lambda: clock.__setitem__(0, clock[0] + 0.25)
    assert runtime.close_before_run(timeout_s=2.0)
    runtime.trajectory.close.assert_called_once_with(timeout_s=1.25)


def test_cooperative_close_exhausting_budget_does_not_start_next_resource(
    runtime: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [100.0]
    monkeypatch.setattr(runtime_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    runtime.perception.close.side_effect = lambda: clock.__setitem__(0, clock[0] + 3.0)
    assert not runtime.close_before_run(timeout_s=2.0)
    runtime.perception.close.assert_called_once()
    runtime.executor.close.assert_not_called()
    runtime.trajectory.close.assert_not_called()
    assert runtime.close_before_run(timeout_s=2.0)
    runtime.perception.close.assert_called_once()


@pytest.mark.parametrize("timeout_s", [-1.0, float("inf"), float("nan")])
def test_nonfinite_or_negative_cleanup_timeout_is_rejected(runtime: Any, timeout_s: float) -> None:
    with pytest.raises(ValueError, match="timeout"):
        runtime.close_before_run(timeout_s=timeout_s)


def test_executor_wait_closed_is_bounded_and_requires_actual_worker_exit(runtime: Any) -> None:
    assert not runtime._pool.wait_closed(timeout_s=0.0)
    runtime._pool.shutdown(wait=False, cancel_futures=True)
    assert runtime._pool.wait_closed(timeout_s=1.0)
    assert runtime._pool.wait_closed(timeout_s=0.0)
