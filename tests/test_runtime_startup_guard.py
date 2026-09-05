"""Startup rejection must not launch work or skip owned-resource cleanup."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from minecraft_ai.runtime import AgentRuntime
from minecraft_ai.perception import PerceptionBlackboard
from minecraft_ai.perception_service import ActiveVLMWorker, RealtimePerceptionService


@pytest.fixture(autouse=True)
def no_live_pause_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("minecraft_ai.runtime.operator_pause_latched", lambda: False)


def startup_runtime() -> AgentRuntime:
    runtime = object.__new__(AgentRuntime)
    runtime.motor_hz = 20.0
    runtime.lease_id = "startup-lease"
    runtime._stop = threading.Event()
    runtime._lease_thread = None
    runtime._lease_fault = None
    runtime._last_renew_ns = 0
    runtime._lease_heartbeat = Mock()  # type: ignore[method-assign]
    runtime.perception = SimpleNamespace(
        active_vlm=SimpleNamespace(start=Mock()),
        last_capture=object(), capture_once=Mock(), close=Mock(),
    )
    runtime.executor = SimpleNamespace(run=None, close=Mock())
    runtime.trajectory = SimpleNamespace(close=Mock())
    runtime.telemetry = SimpleNamespace(publish=Mock())
    runtime._pool = SimpleNamespace(shutdown=Mock())
    runtime._merge_operator_target = Mock()  # type: ignore[method-assign]
    runtime._start_cognition_if_due = Mock()  # type: ignore[method-assign]
    # The old broken path must terminate too, so the regression cannot hang.
    runtime._warmup_policy = Mock(side_effect=runtime.stop)  # type: ignore[method-assign]
    runtime._telemetry_payload = Mock(return_value={})  # type: ignore[method-assign]
    runtime._flush_pending_skill_stats = Mock()  # type: ignore[method-assign]
    runtime._flush_pending_learning_records = Mock()  # type: ignore[method-assign]
    runtime._failsafe = Mock()  # type: ignore[method-assign]
    return runtime


def assert_closed_without_work(runtime: AgentRuntime) -> None:
    runtime.perception.active_vlm.start.assert_not_called()
    runtime.perception.capture_once.assert_not_called()
    runtime._start_cognition_if_due.assert_not_called()
    runtime._warmup_policy.assert_not_called()
    runtime._lease_heartbeat.assert_not_called()
    runtime.perception.close.assert_called_once()
    runtime.executor.close.assert_called_once()
    runtime.trajectory.close.assert_called_once()
    runtime._pool.shutdown.assert_called_once_with(wait=False, cancel_futures=True)
    assert runtime._stop.is_set()
    assert runtime._lease_thread is None


def test_rejected_initial_lease_never_starts_workers_and_closes_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = startup_runtime()
    commands: list[str] = []

    def send(command: str, **_kwargs: object) -> dict[str, object]:
        commands.append(command)
        if command == "renew":
            raise RuntimeError("lease expired")
        return {}

    monkeypatch.setattr("minecraft_ai.runtime.send_command", send)
    with pytest.raises(RuntimeError, match="lease expired"):
        runtime.run_forever()

    assert commands == ["renew", "disarm"]
    assert runtime._lease_fault == "RuntimeError: lease expired"
    runtime._failsafe.assert_called_once_with("agent-runtime:RuntimeError:lease expired")
    assert_closed_without_work(runtime)


@pytest.mark.parametrize("stop_during_renew", [False, True])
def test_startup_stop_does_not_launch_workers(
    monkeypatch: pytest.MonkeyPatch, stop_during_renew: bool,
) -> None:
    runtime = startup_runtime()
    commands: list[str] = []
    if not stop_during_renew:
        runtime.stop()

    def send(command: str, **_kwargs: object) -> dict[str, object]:
        commands.append(command)
        if command == "renew":
            runtime.stop()
        return {}

    monkeypatch.setattr("minecraft_ai.runtime.send_command", send)
    runtime.run_forever()

    assert commands == (["renew", "disarm"] if stop_during_renew else ["disarm"])
    runtime._failsafe.assert_not_called()
    assert_closed_without_work(runtime)


def test_vlm_start_failure_still_closes_runtime_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = startup_runtime()
    runtime.perception.active_vlm.start.side_effect = RuntimeError("worker start failed")
    monkeypatch.setattr("minecraft_ai.runtime.send_command", lambda *_args, **_kwargs: {})

    with pytest.raises(RuntimeError, match="worker start failed"):
        runtime.run_forever()

    runtime._start_cognition_if_due.assert_not_called()
    runtime._warmup_policy.assert_not_called()
    runtime.perception.close.assert_called_once()
    runtime.executor.close.assert_called_once()
    runtime.trajectory.close.assert_called_once()
    runtime._pool.shutdown.assert_called_once_with(wait=False, cancel_futures=True)
    assert runtime._lease_thread is not None and not runtime._lease_thread.is_alive()
    assert runtime._stop.is_set()


def test_heartbeat_start_failure_does_not_join_unstarted_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = startup_runtime()
    monkeypatch.setattr("minecraft_ai.runtime.send_command", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        "minecraft_ai.runtime.threading.Thread.start",
        Mock(side_effect=RuntimeError("thread start failed")),
    )

    with pytest.raises(RuntimeError, match="thread start failed"):
        runtime.run_forever()

    assert_closed_without_work(runtime)


@pytest.mark.parametrize("stage", ["vlm", "capture", "target", "cognition"])
def test_stop_during_startup_does_not_start_a_later_phase(
    monkeypatch: pytest.MonkeyPatch, stage: str,
) -> None:
    runtime = startup_runtime()
    monkeypatch.setattr("minecraft_ai.runtime.send_command", lambda *_args, **_kwargs: {})
    runtime.perception.last_capture = None
    callbacks = [
        runtime.perception.active_vlm.start,
        runtime.perception.capture_once,
        runtime._merge_operator_target,
        runtime._start_cognition_if_due,
        runtime._warmup_policy,
    ]
    selected = ["vlm", "capture", "target", "cognition"].index(stage)
    callbacks[selected].side_effect = runtime.stop

    runtime.run_forever()

    callbacks[selected].assert_called_once()
    for callback in callbacks[selected + 1:]:
        callback.assert_not_called()
    runtime._failsafe.assert_not_called()
    runtime.perception.close.assert_called_once()
    runtime.executor.close.assert_called_once()
    runtime.trajectory.close.assert_called_once()
    runtime._pool.shutdown.assert_called_once_with(wait=False, cancel_futures=True)
    assert runtime._lease_thread is not None and not runtime._lease_thread.is_alive()


def test_stop_during_warming_telemetry_prevents_capture_and_model_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = startup_runtime()
    monkeypatch.setattr("minecraft_ai.runtime.send_command", lambda *_args, **_kwargs: {})
    runtime.perception.last_capture = None
    runtime.telemetry.publish.side_effect = lambda *_args, **_kwargs: runtime.stop()

    runtime.run_forever()

    runtime.perception.capture_once.assert_not_called()
    runtime._merge_operator_target.assert_not_called()
    runtime._start_cognition_if_due.assert_not_called()
    runtime._warmup_policy.assert_not_called()
    runtime._failsafe.assert_not_called()
    runtime.perception.close.assert_called_once()
    runtime.executor.close.assert_called_once()
    runtime.trajectory.close.assert_called_once()
    runtime._pool.shutdown.assert_called_once_with(wait=False, cancel_futures=True)
    assert runtime.telemetry.publish.call_count == 2  # Warming, then stopped.


@pytest.mark.parametrize("cancellation", ["stop", "pause"])
def test_rejected_renewal_during_cancellation_is_not_a_fault(
    monkeypatch: pytest.MonkeyPatch, cancellation: str,
) -> None:
    runtime = startup_runtime()

    def send(command: str, **_kwargs: object) -> dict[str, object]:
        if command == "renew":
            if cancellation == "stop":
                runtime.stop()
            else:
                monkeypatch.setattr("minecraft_ai.runtime.operator_pause_latched", lambda: True)
            raise RuntimeError("renew rejected")
        return {}

    monkeypatch.setattr("minecraft_ai.runtime.send_command", send)
    runtime.run_forever()

    runtime._failsafe.assert_not_called()
    assert_closed_without_work(runtime)


def test_actual_vlm_thread_start_failure_preserves_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = startup_runtime()
    board = PerceptionBlackboard()
    worker = ActiveVLMWorker(Mock(), board, "bedrock:test")
    capture = SimpleNamespace(capture=Mock(), close=Mock())
    runtime.perception = RealtimePerceptionService(
        capture, board, "bedrock:test", active_vlm=worker, fast_perception=None,
    )
    original_start = threading.Thread.start

    def start(thread: threading.Thread) -> None:
        if thread.name == "minecraft-ai-vlm":
            raise RuntimeError("worker start failed")
        original_start(thread)

    monkeypatch.setattr("minecraft_ai.runtime.threading.Thread.start", start)
    monkeypatch.setattr("minecraft_ai.runtime.send_command", lambda *_args, **_kwargs: {})
    with pytest.raises(RuntimeError, match="worker start failed"):
        runtime.run_forever()

    capture.capture.assert_not_called()
    capture.close.assert_called_once()
    runtime.executor.close.assert_called_once()
    runtime.trajectory.close.assert_called_once()
    runtime._pool.shutdown.assert_called_once_with(wait=False, cancel_futures=True)
    assert worker._thread is None and worker._stop.is_set()
    assert runtime._lease_thread is not None and not runtime._lease_thread.is_alive()
