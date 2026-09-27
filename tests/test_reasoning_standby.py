from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from minecraft_ai.agent.daemon_executor import SingleWorkerDaemonExecutor
from minecraft_ai.operator import standby
from minecraft_ai.perception import PerceptionBlackboard
from minecraft_ai.perception_service import RealtimePerceptionService
from minecraft_ai.platforms.bedrock_x11 import CapturedFrame
from minecraft_ai.runtime import AgentRuntime, RuntimeMetrics
from minecraft_ai.skills import SkillOutcome, SkillRun


def runtime():
    return SimpleNamespace(
        executor=SimpleNamespace(run=None),
        perception=SimpleNamespace(active_vlm=None),
        _pending_decision=None,
        _execution_revision=0,
        _cognition_requested=False,
        _reject_bound_cognition=lambda *args: None,
        _release_and_reconcile_inputs=lambda: True,
    )


def test_standby_expires_after_120_seconds_and_requests_fresh_cognition(tmp_path, monkeypatch):
    monkeypatch.setattr(standby, "STANDBY_FILE", tmp_path / "standby.json")
    stamp = [1_000_000_000]
    monkeypatch.setattr(standby.time, "monotonic_ns", lambda: stamp[0])
    standby.set_reasoning_standby(True)
    agent = runtime()
    assert standby.apply_reasoning_standby(agent)
    assert standby.standby_status(agent) == {"state": "standby", "remaining_seconds": 120.0}
    stamp[0] += 120_000_000_000
    assert not standby.apply_reasoning_standby(agent)
    assert standby.standby_status(agent)["state"] == "off"
    assert agent._cognition_requested


def test_detached_running_cognition_reports_draining_until_worker_is_idle(tmp_path, monkeypatch):
    monkeypatch.setattr(standby, "STANDBY_FILE", tmp_path / "standby.json")
    standby.set_reasoning_standby(True)
    entered, finish = threading.Event(), threading.Event()
    pool = SingleWorkerDaemonExecutor()
    agent = runtime()
    agent._pool = pool

    def inference():
        entered.set()
        assert finish.wait(2)

    try:
        future = agent._pending_decision = pool.submit(inference)
        assert entered.wait(2)
        assert standby.apply_reasoning_standby(agent)
        assert agent._pending_decision is None and not future.cancelled()
        assert standby.standby_status(agent)["state"] == "draining"
        finish.set()
        future.result(timeout=2)
        pool.shutdown(wait=True)
        assert standby.standby_status(agent)["state"] == "standby"
    finally:
        finish.set()
        pool.shutdown(wait=True)


def test_standby_preserves_typed_inventory_authority_even_at_expiry(tmp_path, monkeypatch):
    monkeypatch.setattr(standby, "STANDBY_FILE", tmp_path / "standby.json")
    standby.set_reasoning_standby(True)
    agent = runtime()
    owned = SkillRun(
        run_id="inventory-run",
        skill_id="open_inventory",
        started_ns=1,
        parameters={"direction_request_id": "request", "direction_attempt_id": "try"},
    )
    agent.executor.run = owned

    def forbidden_release():
        raise AssertionError("revoked owner")

    agent._release_and_reconcile_inputs = forbidden_release
    assert standby.apply_reasoning_standby(agent)
    assert agent.executor.run is owned and owned.outcome == SkillOutcome.RUNNING
    standby.set_reasoning_standby(False)
    assert not standby.apply_reasoning_standby(agent)
    assert agent.executor.run is owned


def test_pending_vision_work_is_draining_not_false_standby(tmp_path, monkeypatch):
    monkeypatch.setattr(standby, "STANDBY_FILE", tmp_path / "standby.json")
    standby.set_reasoning_standby(True)
    agent = runtime()
    assert standby.apply_reasoning_standby(agent)
    agent.perception.active_vlm = SimpleNamespace(status=lambda: {"pending_requests": 1})
    assert standby.standby_status(agent)["state"] == "draining"


def test_dequeued_vision_work_remains_draining_before_busy_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(standby, "STANDBY_FILE", tmp_path / "standby.json")
    standby.set_reasoning_standby(True)
    agent = runtime()
    assert standby.apply_reasoning_standby(agent)
    agent.perception.active_vlm = SimpleNamespace(
        status=lambda: {"busy": False, "pending_requests": 0},
        available=lambda: False,
    )
    assert standby.standby_status(agent)["state"] == "draining"


def test_runtime_standby_keeps_real_capture_path_without_submitting_model_work(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(standby, "STANDBY_FILE", tmp_path / "standby.json")
    standby.set_reasoning_standby(True)
    captured = []

    def capture():
        item = CapturedFrame(len(captured), time.monotonic_ns(), 64, 64, b"\0" * 64 * 64 * 4)
        captured.append(item)
        return item

    def prohibited(*args, **kwargs):
        raise AssertionError("standby submitted autonomous work")

    agent = runtime()
    agent.blackboard = PerceptionBlackboard()
    agent.perception = RealtimePerceptionService(
        capture_source=SimpleNamespace(capture=capture),
        blackboard=agent.blackboard,
        instance_id="standby-test",
        active_vlm=SimpleNamespace(submit=prohibited),
    )
    agent.state_db = None
    agent.executor.policy = SimpleNamespace()
    agent.metrics = RuntimeMetrics()
    agent._input_release_pending_ns = None
    agent.telemetry = SimpleNamespace(publish=lambda payload: None)
    agent._telemetry_payload = lambda **kwargs: {}
    agent._continue_after_capture = lambda: True
    for name in (
        "_merge_operator_target", "_merge_policy_perception", "_publish_player_chat_facts",
    ):
        setattr(agent, name, getattr(AgentRuntime, name).__get__(agent))
    for name in (
        "_flush_pending_skill_stats", "_flush_pending_learning_records",
        "_flush_pending_operator_status_updates",
    ):
        setattr(agent, name, lambda: None)
    for name in (
        "_planks_retry_requires_wood", "_start_cognition_if_due", "_request_semantics_if_due",
    ):
        setattr(agent, name, prohibited)

    AgentRuntime.tick(agent)
    AgentRuntime.tick(agent)

    assert len(captured) == agent.metrics.frames == 2
    assert agent._reasoning_standby_engaged
    assert agent.perception.last_capture is captured[-1]
