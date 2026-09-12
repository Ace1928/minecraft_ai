"""Normal runtime retirement orders release, evidence and bounded cleanup."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import minecraft_ai.runtime as runtime_module
from minecraft_ai.builtin_skills import build_bootstrap_skill_library
from minecraft_ai.execution import SkillExecutor
from minecraft_ai.skills import SkillOutcome
from test_runtime_startup_guard import startup_runtime


def prepared(monkeypatch):
    runtime = startup_runtime()
    events = []
    clock = [100_000_000_000]
    monkeypatch.setattr(runtime_module, "time", SimpleNamespace(monotonic_ns=lambda: clock[0]))
    monkeypatch.setattr(runtime_module, "emergency_stop_latched", lambda: False)

    def disarm(command, **kwargs):
        assert command == "disarm" and runtime._stop.is_set()
        assert 0.05 <= kwargs["timeout_s"] <= 1.5
        events.append("release")
        return {"motor_lease_active": False}

    monkeypatch.setattr(runtime_module, "send_command", disarm)
    runtime.executor.policy = SimpleNamespace(
        policy_id="test-policy",
        retire=Mock(side_effect=lambda **kwargs: events.append("policy") or {
            "complete": True, "workers": [],
        }),
    )
    runtime.perception.close.side_effect = lambda **kwargs: events.append("perception") or True
    runtime.trajectory.close.side_effect = lambda **kwargs: events.append("recorder")
    runtime._pool.shutdown.side_effect = lambda **kwargs: events.append("pool")
    runtime._flush_pending_skill_stats.side_effect = lambda **kwargs: events.append("stats")
    runtime._flush_pending_learning_records.side_effect = lambda **kwargs: events.append("learning")
    runtime.telemetry.publish.side_effect = lambda *args, **kwargs: events.append("telemetry")
    return runtime, events, clock


def test_runtime_releases_before_waiting_and_uses_one_cleanup_budget(monkeypatch):
    runtime, events, clock = prepared(monkeypatch)
    runtime._lease_thread = SimpleNamespace(join=Mock(
        side_effect=lambda **kwargs: events.append("lease"),
    ))

    def retire(**kwargs):
        assert kwargs == {"deadline_ns": 115_000_000_000, "inputs_released": True}
        events.append("policy")
        clock[0] = kwargs["deadline_ns"]
        return {"complete": True, "workers": []}

    runtime.executor.policy.retire.side_effect = retire
    runtime._shutdown_runtime()
    assert events == ["release", "pool", "lease", "perception", "policy", "recorder",
                      "stats", "learning", "telemetry"]
    runtime.perception.close.assert_called_once_with(deadline_ns=102_000_000_000)
    runtime.trajectory.close.assert_called_once_with(timeout_s=5.0)
    runtime.executor.close.assert_not_called()  # Never a second legacy stop/save.


@pytest.mark.parametrize("reply", [{}, {"motor_lease_active": True},
                                    {"motor_lease_active": 0}, {"motor_lease_active": None}])
def test_unconfirmed_release_never_admits_checkpoint(monkeypatch, reply):
    runtime, _, _ = prepared(monkeypatch)
    monkeypatch.setattr(runtime_module, "send_command", lambda *args, **kwargs: reply)
    runtime._shutdown_runtime()
    assert runtime.executor.policy.retire.call_args.kwargs["inputs_released"] is False
    assert runtime._shutdown_results["actuator_release"] is False


def test_emergency_does_not_request_checkpoint_even_after_release(monkeypatch):
    runtime, _, _ = prepared(monkeypatch)
    monkeypatch.setattr(runtime_module, "emergency_stop_latched", lambda: True)
    runtime._shutdown_runtime()
    assert runtime.executor.policy.retire.call_args.kwargs["inputs_released"] is False


@pytest.mark.parametrize("failed", ["release", "perception", "policy", "recorder", "stats"])
def test_cleanup_failure_does_not_skip_other_owners_or_leak_private_text(monkeypatch, failed):
    runtime, events, _ = prepared(monkeypatch)
    failure = Mock(side_effect=RuntimeError("/private/native/model/checkpoint.bin"))
    if failed == "release":
        monkeypatch.setattr(runtime_module, "send_command", failure)
    elif failed == "policy":
        runtime.executor.policy.retire = failure
    elif failed == "stats":
        runtime._flush_pending_skill_stats = failure
    else:
        getattr(runtime, "trajectory" if failed == "recorder" else failed).close = failure
    runtime._shutdown_runtime()
    runtime.executor.policy.retire.assert_called_once()
    runtime.trajectory.close.assert_called_once()
    runtime._flush_pending_skill_stats.assert_called_once()
    runtime._flush_pending_learning_records.assert_called_once()
    runtime.telemetry.publish.assert_called_once()
    assert events[-1] == "telemetry"
    assert "/private/" not in repr(runtime._shutdown_results)


def test_stop_deadline_is_not_refreshed_by_later_cleanup(monkeypatch):
    runtime, _, clock = prepared(monkeypatch)
    runtime.stop()
    clock[0] += 12_000_000_000
    runtime.stop()
    runtime._shutdown_runtime()
    assert runtime._shutdown_deadline_ns == 120_000_000_000
    assert runtime.executor.policy.retire.call_args.kwargs["deadline_ns"] == 115_000_000_000
    assert runtime.trajectory.close.call_args.kwargs["timeout_s"] == 8.0


def test_expired_budget_still_attempts_nonblocking_recorder_cleanup(monkeypatch):
    runtime, _, clock = prepared(monkeypatch)
    runtime.stop()
    clock[0] += 30_000_000_000
    forbidden = Mock(side_effect=AssertionError("expired IPC budget"))
    monkeypatch.setattr(runtime_module, "send_command", forbidden)
    runtime._shutdown_runtime()
    forbidden.assert_not_called()
    runtime.trajectory.close.assert_called_once_with(timeout_s=0.0)
    assert runtime.executor.policy.retire.call_args.kwargs["inputs_released"] is False


def test_terminal_cancellation_preserves_run_without_policy_reset_or_recovery(monkeypatch):
    runtime, events, _ = prepared(monkeypatch)
    policy = runtime.executor.policy
    policy.reset = Mock(side_effect=AssertionError("shutdown must not reset worker"))
    executor = SkillExecutor(policy)
    executor.start(build_bootstrap_skill_library().get("craft_wood_planks"),
                   run_id="shutdown-run", context_key="unfinished-task", now_ns=1)
    runtime.executor = executor
    runtime._record_terminal_run = Mock(side_effect=lambda run: events.append("cancelled"))
    runtime._send_motor = Mock(side_effect=AssertionError("no shutdown action"))
    runtime._shutdown_runtime()
    assert events.index("release") < events.index("cancelled") < events.index("policy")
    assert executor.run.outcome == SkillOutcome.CANCELLED
    assert executor.run.run_id == "shutdown-run"
    assert executor.run.context_key == "unfinished-task"
    assert executor.run.failure_reason == "runtime-shutdown"
    assert executor.run.failure_code is None
    policy.reset.assert_not_called()
    runtime._send_motor.assert_not_called()
    runtime._record_terminal_run.assert_called_once_with(executor.run)


def test_terminal_executor_tick_authorizes_no_action_or_recovery():
    policy = SimpleNamespace(policy_id="test-policy",
                             reset=Mock(side_effect=AssertionError("no reset")),
                             status=Mock(side_effect=AssertionError("no native callback")))
    executor = SkillExecutor(policy)
    executor.start(build_bootstrap_skill_library().get("craft_wood_planks"),
                   run_id="terminal", context_key="task", now_ns=1)
    result = executor.cancel_for_shutdown(now_ns=2)
    assert result.action is None and result.recovery_skills == ()
    assert result.run.outcome == SkillOutcome.CANCELLED and result.run.ended_ns == 2
    policy.reset.assert_not_called()
    policy.status.assert_not_called()


def test_invalid_optional_registry_cannot_skip_other_cleanup(monkeypatch):
    runtime, _, _ = prepared(monkeypatch)
    runtime._bound_cognition_requests = None
    runtime._shutdown_runtime()
    assert runtime._shutdown_results["cognition_registry"] == {"error_code": "TypeError"}
    runtime.executor.policy.retire.assert_called_once()
    runtime.trajectory.close.assert_called_once()
    runtime._flush_pending_learning_records.assert_called_once()
