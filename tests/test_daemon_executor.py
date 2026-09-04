from __future__ import annotations

import subprocess
import sys
import textwrap
import threading
import time
from types import SimpleNamespace

import pytest

from minecraft_ai.daemon_executor import SingleWorkerDaemonExecutor
from minecraft_ai.runtime import AgentRuntime


def test_blocked_worker_does_not_delay_interpreter_exit() -> None:
    script = textwrap.dedent(
        """
        import threading

        from minecraft_ai.daemon_executor import SingleWorkerDaemonExecutor

        started = threading.Event()
        never_released = threading.Event()

        def block_forever():
            started.set()
            never_released.wait()

        executor = SingleWorkerDaemonExecutor(thread_name="subprocess-daemon-worker")
        executor.submit(block_forever)
        assert started.wait(timeout=1.0)
        executor.shutdown(wait=False, cancel_futures=True)
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=2.0,
    )

    assert completed.returncode == 0, completed.stderr


def test_agent_runtime_uses_daemon_cognition_worker() -> None:
    runtime = AgentRuntime(
        perception=SimpleNamespace(),  # type: ignore[arg-type]
        blackboard=SimpleNamespace(),  # type: ignore[arg-type]
        executor=SimpleNamespace(),  # type: ignore[arg-type]
        skills=SimpleNamespace(),  # type: ignore[arg-type]
        role=SimpleNamespace(),  # type: ignore[arg-type]
        lease_id="test-lease",
    )
    worker = next(
        thread
        for thread in threading.enumerate()
        if thread.name == "minecraft-ai-cognition"
    )

    try:
        assert isinstance(runtime._pool, SingleWorkerDaemonExecutor)
        assert worker.daemon
    finally:
        runtime._pool.shutdown(wait=True, cancel_futures=True)

    assert not worker.is_alive()


def test_shutdown_cancels_queued_work_without_waiting_for_running_call() -> None:
    started = threading.Event()
    release = threading.Event()

    def blocked_call() -> str:
        started.set()
        if not release.wait(timeout=5.0):
            raise TimeoutError("test did not release blocked call")
        return "finished"

    executor = SingleWorkerDaemonExecutor(thread_name="test-daemon-executor-blocked")
    running = executor.submit(blocked_call)
    assert started.wait(timeout=1.0)
    queued = executor.submit(lambda: "must not run")
    worker = next(
        thread
        for thread in threading.enumerate()
        if thread.name == "test-daemon-executor-blocked"
    )

    try:
        started_at = time.perf_counter()
        executor.shutdown(wait=False, cancel_futures=True)

        assert time.perf_counter() - started_at < 1.0
        assert worker.daemon
        assert worker.is_alive()
        assert running.running()
        assert queued.cancelled()
        with pytest.raises(RuntimeError, match="cannot schedule new futures after shutdown"):
            executor.submit(lambda: None)
    finally:
        release.set()
        assert running.result(timeout=1.0) == "finished"
        executor.shutdown(wait=True)

    assert not worker.is_alive()


def test_submitted_calls_preserve_results_arguments_and_exceptions() -> None:
    executor = SingleWorkerDaemonExecutor(thread_name="test-daemon-executor-results")

    def combine(prefix: str, value: int, *, suffix: str) -> str:
        return f"{prefix}{value}{suffix}"

    def fail() -> None:
        raise ValueError("model request failed")

    result = executor.submit(combine, "step-", 7, suffix="-done")
    failure = executor.submit(fail)

    try:
        assert result.result(timeout=1.0) == "step-7-done"
        with pytest.raises(ValueError, match="model request failed"):
            failure.result(timeout=1.0)
    finally:
        executor.shutdown(wait=True)


def test_shutdown_without_cancellation_drains_queued_work() -> None:
    executor = SingleWorkerDaemonExecutor(thread_name="test-daemon-executor-drain")
    first = executor.submit(lambda: 1)
    second = executor.submit(lambda: 2)

    executor.shutdown(wait=True, cancel_futures=False)

    assert first.result() == 1
    assert second.result() == 2
