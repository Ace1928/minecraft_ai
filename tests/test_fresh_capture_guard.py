"""Fresh capture vetoes never substitute merged history or grant input authority."""

from __future__ import annotations

import time
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from minecraft_ai.perception import FrameState, PerceptionBlackboard, PerceptionFact
from minecraft_ai.perception.service import CaptureObservation
from minecraft_ai.perception_service import (
    CaptureObservation as CompatibilityCaptureObservation,
    RealtimePerceptionService,
)
from minecraft_ai.platforms.bedrock_x11 import CapturedFrame
from minecraft_ai.runtime import AgentRuntime, RuntimeMetrics
from minecraft_ai.safety import MotorAction
from test_runtime_startup_guard import startup_runtime


@pytest.fixture(autouse=True)
def no_live_supervisor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("minecraft_ai.runtime.operator_pause_latched", lambda: False)
    monkeypatch.setattr("minecraft_ai.runtime.emergency_stop_latched", lambda: False)
    monkeypatch.setattr("minecraft_ai.runtime.send_command", Mock(return_value={}))


def _capture(frame_id: int = 42, captured_ns: int | None = None) -> CapturedFrame:
    return CapturedFrame(
        frame_id=frame_id, captured_ns=captured_ns or time.monotonic_ns(),
        width=2, height=2, bgra=b"\x00\x00\x00\xff" * 4,
    )


def _observation() -> CaptureObservation:
    captured = _capture()
    return CaptureObservation(
        capture=captured,
        frame=FrameState(
            frame_id=7, captured_ns=captured.captured_ns,
            instance_id="test-world", width=2, height=2,
        ),
        fast_facts=(), fast_model_id=None,
    )


def test_observation_retains_exact_capture_and_infer_result_not_merged_history() -> None:
    captured = _capture()
    board = PerceptionBlackboard()
    old = PerceptionFact(
        key="scene.mode", value="world", confidence=0.99,
        observed_ns=captured.captured_ns - 1, source="old-fast", expires_after_ms=30_000,
    )
    board.publish(FrameState(
        frame_id=700, captured_ns=captured.captured_ns - 1,
        instance_id="test-world", width=2, height=2, facts=(old,),
    ))
    facts = (PerceptionFact(
        key="scene.death", value=True, confidence=0.99,
        observed_ns=captured.captured_ns, source="current-fast",
    ),)
    fast = SimpleNamespace(model_id="test-fast-v1", infer=Mock(return_value=facts))
    source = SimpleNamespace(capture=Mock(return_value=captured), close=Mock())
    service = RealtimePerceptionService(source, board, "test-world", fast_perception=fast)
    assert service.last_observation is None

    state = service.capture_once()

    observation = service.last_observation
    assert observation is not None
    assert CompatibilityCaptureObservation is CaptureObservation
    assert observation.capture is captured and observation.frame is state
    assert observation.capture.frame_id == 42 and observation.frame.frame_id == 701
    assert observation.capture.captured_ns == observation.frame.captured_ns
    assert observation.fast_facts is facts and observation.fast_facts[0] is facts[0]
    assert observation.fast_model_id == "test-fast-v1"
    fast.infer.assert_called_once_with(captured)
    assert board.fact("scene.mode", now_ns=captured.captured_ns) is old
    assert old not in observation.fast_facts
    with pytest.raises(FrozenInstanceError):
        observation.fast_model_id = "replacement"


def test_no_fast_perception_publishes_empty_fresh_observation() -> None:
    captured = _capture()
    source = SimpleNamespace(capture=Mock(return_value=captured), close=Mock())
    service = RealtimePerceptionService(
        source, PerceptionBlackboard(), "test-world", fast_perception=None,
    )
    state = service.capture_once()
    observation = service.last_observation
    assert observation is not None
    assert observation.capture is captured and observation.frame is state
    assert observation.fast_facts == () and observation.fast_model_id is None


@pytest.mark.parametrize("failure", ["capture", "infer", "nonmonotonic"])
def test_failed_attempt_clears_old_bundle_before_any_capture_or_inference(failure: str) -> None:
    first = _capture()
    next_frame = _capture(43, first.captured_ns + 1)
    source = SimpleNamespace(capture=Mock(return_value=first), close=Mock())
    fast = SimpleNamespace(model_id="test-fast", infer=Mock(return_value=()))
    service = RealtimePerceptionService(
        source, PerceptionBlackboard(), "test-world", fast_perception=fast,
    )
    service.capture_once()
    assert service.last_observation is not None

    def capture() -> CapturedFrame:
        assert service.last_observation is None
        if failure == "capture":
            raise RuntimeError("capture failed")
        return first if failure == "nonmonotonic" else next_frame

    def infer(_captured: CapturedFrame) -> tuple[PerceptionFact, ...]:
        assert service.last_observation is None
        raise RuntimeError("infer failed")

    source.capture.side_effect = capture
    if failure == "infer":
        fast.infer.side_effect = infer
    with pytest.raises(RuntimeError):
        service.capture_once()
    assert service.last_observation is None


def _runtime() -> AgentRuntime:
    runtime = startup_runtime()
    runtime.metrics = RuntimeMetrics()
    runtime.perception.last_observation = _observation()
    runtime._release_and_reconcile_inputs = Mock(return_value=True)
    return runtime


@pytest.mark.parametrize("missing", [False, True])
def test_default_guard_preserves_existing_path_with_or_without_bundle(missing: bool) -> None:
    runtime = _runtime()
    if missing:
        runtime.perception.last_observation = None
    assert runtime.allow_fresh_capture(runtime.perception.last_observation) is True
    assert runtime._continue_after_capture() is True
    assert not runtime._stop.is_set()
    runtime._release_and_reconcile_inputs.assert_not_called()
    runtime._failsafe.assert_not_called()


@pytest.mark.parametrize("verdict", [False, None, 1, "true", RuntimeError("predicate failed")])
def test_startup_veto_prevents_cognition_warmup_and_still_closes_owned_resources(verdict) -> None:
    runtime = _runtime()
    observation = runtime.perception.last_observation
    runtime.perception.last_capture = None
    runtime.perception.last_observation = None

    def capture() -> FrameState:
        runtime.perception.last_capture = observation.capture
        runtime.perception.last_observation = observation
        return observation.frame

    runtime.perception.capture_once.side_effect = capture
    runtime.allow_fresh_capture = Mock(
        side_effect=verdict if isinstance(verdict, Exception) else None,
        return_value=verdict,
    )
    runtime.run_forever()

    runtime.perception.capture_once.assert_called_once()
    runtime.allow_fresh_capture.assert_called_once_with(observation)
    runtime._merge_operator_target.assert_not_called()
    runtime._start_cognition_if_due.assert_not_called()
    runtime._warmup_policy.assert_not_called()
    runtime._release_and_reconcile_inputs.assert_called_once()
    runtime._failsafe.assert_not_called()
    assert runtime._stop.is_set()
    runtime.perception.close.assert_called_once()
    runtime.executor.close.assert_called_once()
    runtime.trajectory.close.assert_called_once()
    runtime._pool.shutdown.assert_called_once_with(wait=False, cancel_futures=True)
    assert runtime._lease_thread is not None and not runtime._lease_thread.is_alive()


_DOWNSTREAM = (
    "_flush_pending_skill_stats", "_flush_pending_learning_records",
    "_flush_pending_operator_status_updates", "_publish_player_chat_facts",
    "_planks_retry_requires_wood", "_consume_cognition", "_start_cognition_if_due",
    "_reconcile_cognition_perception_probe", "_request_semantics_if_due",
    "_route_observed_scene_recovery", "_advance_headroom_recovery", "_start_skill",
    "_send_motor",
)


def _tick_runtime() -> AgentRuntime:
    runtime = _runtime()
    runtime.blackboard = PerceptionBlackboard()
    runtime.blackboard.publish(runtime.perception.last_observation.frame)
    runtime.perception.capture_once.return_value = runtime.perception.last_observation.frame
    runtime.perception.stale = Mock(return_value=False)
    runtime._merge_policy_perception = Mock()
    runtime.executor.tick = Mock()
    runtime._headroom_scene_is_safe = Mock(return_value=False)
    runtime._explore_keep_alive = Mock(return_value=None)
    for name in _DOWNSTREAM:
        setattr(runtime, name, Mock())
    return runtime


@pytest.mark.parametrize("verdict", [False, None, 1, "true", ValueError("predicate failed")])
def test_tick_veto_prevents_policy_cognition_gui_and_all_successor_work(verdict) -> None:
    runtime = _tick_runtime()
    runtime.allow_fresh_capture = Mock(
        side_effect=verdict if isinstance(verdict, Exception) else None,
        return_value=verdict,
    )
    runtime.tick()
    runtime.allow_fresh_capture.assert_called_once_with(runtime.perception.last_observation)
    for name in _DOWNSTREAM:
        getattr(runtime, name).assert_not_called()
    runtime.executor.tick.assert_not_called()
    runtime._explore_keep_alive.assert_not_called()
    runtime._release_and_reconcile_inputs.assert_called_once()
    runtime._failsafe.assert_not_called()
    assert runtime._stop.is_set()


@pytest.mark.parametrize("release_result", [False, RuntimeError("release failed")])
def test_missing_observation_denial_still_stops_when_release_fails(release_result) -> None:
    runtime = _tick_runtime()
    runtime.perception.last_observation = None
    runtime.allow_fresh_capture = Mock(return_value=False)
    runtime._release_and_reconcile_inputs = Mock(
        side_effect=release_result if isinstance(release_result, Exception) else None,
        return_value=release_result,
    )
    runtime.tick()
    runtime.allow_fresh_capture.assert_called_once_with(None)
    runtime._release_and_reconcile_inputs.assert_called_once()
    runtime._start_cognition_if_due.assert_not_called()
    runtime._route_observed_scene_recovery.assert_not_called()
    runtime.executor.tick.assert_not_called()
    runtime._failsafe.assert_not_called()
    assert runtime._stop.is_set()


@pytest.mark.parametrize("gate", ["stale", "pending-release"])
def test_existing_capture_safety_gates_run_before_optional_predicate(gate: str) -> None:
    runtime = _tick_runtime()
    runtime.allow_fresh_capture = Mock(return_value=True)
    if gate == "stale":
        runtime.perception.stale.return_value = True
    else:
        runtime._input_release_pending_ns = time.monotonic_ns()
    runtime.tick()
    runtime.allow_fresh_capture.assert_not_called()
    runtime._release_and_reconcile_inputs.assert_called_once()
    runtime._consume_cognition.assert_not_called()
    runtime._route_observed_scene_recovery.assert_not_called()
    runtime.executor.tick.assert_not_called()
    runtime._failsafe.assert_not_called()


@pytest.mark.parametrize("during", [False, True])
def test_true_predicate_cannot_revive_a_stopped_runtime(during: bool) -> None:
    runtime = _runtime()

    def allow(_observation: CaptureObservation | None) -> bool:
        runtime.stop()
        return True

    runtime.allow_fresh_capture = Mock(side_effect=allow)
    if not during:
        runtime.stop()
    assert runtime._continue_after_capture() is False
    assert runtime.allow_fresh_capture.call_count == int(during)
    runtime._failsafe.assert_not_called()
    assert runtime._stop.is_set()


def test_allowed_capture_does_not_override_operator_pause_at_input_boundary(monkeypatch) -> None:
    runtime = _runtime()
    runtime.allow_fresh_capture = Mock(return_value=True)
    assert runtime._continue_after_capture() is True
    send = Mock(return_value={})
    monkeypatch.setattr("minecraft_ai.runtime.send_command", send)
    monkeypatch.setattr("minecraft_ai.runtime.operator_pause_latched", lambda: True)
    runtime._send_motor(MotorAction(sequence=0, keys_down=("w",)))
    send.assert_not_called()
    assert runtime._stop.is_set()
    runtime._failsafe.assert_not_called()
