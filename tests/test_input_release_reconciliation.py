"""Pure controller recovery checks: no workers, desktop or transport connections."""

from copy import deepcopy
from io import StringIO
from types import SimpleNamespace
import threading

import pytest

import minecraft_ai.policy_service as policy_module
import minecraft_ai.runtime as runtime_module
from minecraft_ai.action_levels import ActionLevel
from minecraft_ai.config import PolicyConfig
from minecraft_ai.crafting_control import PlankCraftPhase
from minecraft_ai.execution import SkillExecutor
from minecraft_ai.motor import BootstrapMotorPolicy, MotorIntent
from minecraft_ai.perception import PerceptionBlackboard
from minecraft_ai.platforms.bedrock_x11 import CapturedFrame
from minecraft_ai.policy_service import (
    GroundedPolicyRouter,
    LearnedPolicyOutput,
    TemporalPolicyClient,
    _PolicyRequestContext,
)
from minecraft_ai.runtime import AgentRuntime, RuntimeMetrics
from minecraft_ai.safety import MotorAction


@pytest.fixture
def learned(tmp_path):
    # Only configuration path existence is checked; these files are never loaded.
    model = tmp_path / "unused-model"
    model.touch()
    config = PolicyConfig(
        python_path=str(model), source_path=str(tmp_path), model_path=str(model),
        weights_path=str(model), model_sha256="a" * 64, weights_sha256="b" * 64,
        source_commit="c" * 40, model_version="unit-only", license="MIT",
        deadline_ms=5000, threads=1,
    )
    frame = CapturedFrame(frame_id=9, captured_ns=900, width=1, height=1, bgra=b"abcd")
    client = TemporalPolicyClient(config, frame_provider=lambda: frame)
    client._process = SimpleNamespace(stdin=StringIO(), poll=lambda: None)
    return client


def _prediction(*, keys=("w",), buttons=(), dx=0):
    return LearnedPolicyOutput(
        keys=keys, buttons=buttons, mouse_dx=dx, inference_ns=1, model_version="unit-only",
    )


def test_learned_release_represses_same_desired_key_without_worker_reset(learned):
    client = learned
    client._last_sequence = 41
    client._output_action(_prediction(buttons=("left",)), 41)
    client._pending_camera = (90, 50)
    client._world_camera_state.estimated_pitch_units = 17
    worker = client._process
    client.notify_inputs_released()
    client.notify_inputs_released()
    assert not client._held_keys and not client._held_buttons
    assert client._held_until_ns == 0
    assert client._pending_camera == (0, 0)
    assert client._last_sequence == 41
    assert client._world_camera_state.estimated_pitch_units == 17
    assert client._process is worker and worker.stdin.getvalue() == ""
    resumed = client._output_action(_prediction(buttons=("left",)), 42)
    assert resumed.keys_down == ("w",) and resumed.buttons_down == ("left",)


@pytest.mark.parametrize("held_keys", [(), ("w",)])
def test_pending_prediction_is_drained_but_not_applied_after_release(
    learned, monkeypatch, held_keys,
):
    client = learned
    monkeypatch.setattr(policy_module.time, "monotonic_ns", lambda: 1000)
    client._pending_request_id = "old-frame"
    client._pending_deadline_ns = 2_000_000_000
    client._pending_request_context = _PolicyRequestContext("old-frame", {}, None, None)
    client._pending_camera = (99, 88)
    client._held_keys = set(held_keys)
    client._applied_request_context = client._pending_request_context
    client._last_prediction = _prediction(dx=99)
    client.notify_inputs_released()
    client.notify_inputs_released()
    assert client._pending_request_id == "old-frame"
    assert client.metrics.invalidated_requests == 1
    assert client._pending_request_context is None
    assert client._applied_request_context is None
    assert client._last_action_provenance["action_kind"] == "release"
    assert client._last_action_provenance["prediction_id"] is None
    assert client._last_action_provenance["request_id"] is None
    monkeypatch.setattr(client, "_ensure_started", lambda _size: None)
    monkeypatch.setattr(client, "_read_response", lambda _timeout: {
        "type": "prediction", "request_id": "old-frame",
        "output": _prediction(dx=99).model_dump(),
    })
    fresh_requests = []
    monkeypatch.setattr(client, "_submit", lambda frame, *_: fresh_requests.append(frame.frame_id))
    action = client.act(PerceptionBlackboard(), MotorIntent(skill_id="walk", mode="explore"),
                        sequence=1)
    assert fresh_requests == [9]
    assert client.metrics.retired_responses == 1 and client.metrics.responses == 0
    assert action.keys_down == () and action.buttons_down == ()
    assert action.mouse_dx == action.mouse_dy == 0
    assert client._last_action_provenance["action_kind"] == "empty_hold"
    assert client._pending_request_id is None and not client._discard_pending_response
    assert client._process.stdin.getvalue() == ""


def test_bootstrap_release_keeps_intent_rhythm_and_sequence():
    policy = BootstrapMotorPolicy()
    intent = MotorIntent(skill_id="walk", mode="explore")
    assert policy.act(PerceptionBlackboard(), intent, sequence=12).keys_down == ("w",)
    before = policy._tick_count
    policy.notify_inputs_released()
    policy.notify_inputs_released()
    assert policy._last_sequence == 12 and policy._tick_count == before
    assert policy.act(PerceptionBlackboard(), intent, sequence=13).keys_down == ("w",)


def test_router_reconciles_bodies_but_keeps_episode_and_grounding():
    primary, raw, observer = (BootstrapMotorPolicy() for _ in range(3))
    router = GroundedPolicyRouter(primary=primary, grounded=observer, raw_motion=raw, gui=raw)
    router._active = raw
    router._active_route = "raw_motion"
    router._episode_id = "same-option"
    router._episode_level = ActionLevel.RAW
    router._last_sequence = 19
    router._grounding_active = True
    router._grounded_track_id = "retained-track"
    primary._held_keys = {"a"}
    raw._held_keys = {"w"}
    observer._held_buttons = {"left"}
    router.notify_inputs_released()
    assert not primary._held_keys and not raw._held_keys
    assert observer._held_buttons == {"left"}
    assert router._active is raw and router._active_route == "raw_motion"
    assert router._episode_id == "same-option" and router._episode_level == ActionLevel.RAW
    assert router._last_sequence == 19
    assert router._grounding_active and router._grounded_track_id == "retained-track"


@pytest.mark.parametrize("phase", [PlankCraftPhase.LOCATE_RECIPE, PlankCraftPhase.VERIFY_OUTPUT])
def test_executor_preserves_already_sent_crafting_actions_and_deadlines(phase):
    executor = SkillExecutor(BootstrapMotorPolicy())
    crafter = executor._plank_crafter
    crafter._run_id = "craft-once"
    crafter._phase = phase
    crafter._phase_started_ns = 100
    crafter._inventory_toggle_ns = 80
    crafter._interaction_ns = 100
    crafter._baseline_logs = 1
    crafter._last_attempt_ns = 100
    crafter._toggle_attempts = crafter._recipe_attempts = 1
    crafter._last_sequence = 7
    executor._instruction_override = "craft exactly one set"
    executor._inventory_open_sent = True
    before = deepcopy(vars(crafter))
    guard = executor._mining_guard
    guard._held_keys = {"w"}
    guard._held_buttons = {"left"}
    guard._lease = object()
    executor.notify_inputs_released(now_ns=200)
    executor.notify_inputs_released(now_ns=300)
    assert vars(crafter) == before
    assert executor._instruction_override == "craft exactly one set"
    assert executor._inventory_open_sent
    assert not guard.held_keys and not guard.held_buttons and guard._lease is None
    assert guard._last_targeting_change_ns == 200
    # An empty post-release observation cannot replay a completed toggle/click.
    step = crafter.step(PerceptionBlackboard(), run_id="craft-once", sequence=8, now_ns=400)
    assert step.action is None


def _runtime(monkeypatch, *, stale=False):
    runtime = AgentRuntime.__new__(AgentRuntime)
    runtime.lease_id = "unit-lease"
    runtime.executor = SkillExecutor(BootstrapMotorPolicy())
    runtime._input_release_pending_ns = None
    runtime._sequence = 20
    runtime._stop = threading.Event()
    runtime.metrics = RuntimeMetrics()
    runtime.stale_frame_consecutive_limit = 3
    runtime.perception = SimpleNamespace(
        capture_once=lambda: SimpleNamespace(frame_id=9), stale=lambda: stale,
    )
    runtime.telemetry = SimpleNamespace(publish=lambda _payload: None)
    monkeypatch.setattr(runtime, "_merge_operator_target", lambda: None)
    monkeypatch.setattr(runtime, "_merge_policy_perception", lambda: None)
    monkeypatch.setattr(runtime, "_telemetry_payload", lambda **kw: kw)
    monkeypatch.setattr(runtime_module, "operator_pause_latched", lambda: False)
    return runtime


@pytest.mark.parametrize("reply", [None, {}, {"released": False, "lease_active": True},
                                   {"released": True, "lease_active": False},
                                   {"released": 1, "lease_active": True}, TimeoutError()])
def test_unknown_release_stays_pending_and_blocks_actions_until_confirmed(monkeypatch, reply):
    runtime = _runtime(monkeypatch)
    runtime.executor.policy._held_keys = {"w"}
    notifications = []
    notify = runtime.executor.notify_inputs_released
    def record(*, now_ns):
        notifications.append(now_ns)
        notify(now_ns=now_ns)
    monkeypatch.setattr(runtime.executor, "notify_inputs_released", record)
    monkeypatch.setattr(runtime_module.time, "monotonic_ns", lambda: 100)
    sent = []
    def unknown(command, **kwargs):
        sent.append((command, kwargs))
        if isinstance(reply, Exception):
            raise reply
        return reply
    monkeypatch.setattr(runtime_module, "send_command", unknown)
    assert runtime._release_and_reconcile_inputs() is False
    assert runtime._input_release_pending_ns == 100 and notifications == []
    assert runtime.executor.policy._held_keys == {"w"}
    with pytest.raises(RuntimeError, match="acknowledgement is pending"):
        runtime._send_motor(MotorAction(sequence=20, keys_down=("w",)))
    assert [command for command, _ in sent] == ["release-inputs"]
    runtime.tick()  # A fresh capture alone never clears an uncertain release.
    assert [command for command, _ in sent] == ["release-inputs", "release-inputs"]
    assert runtime._input_release_pending_ns == 100 and notifications == []
    monkeypatch.setattr(runtime_module.time, "monotonic_ns", lambda: 200)
    monkeypatch.setattr(runtime_module, "send_command", lambda *_args, **_kw: {
        "released": True, "lease_active": True,
    })
    runtime.tick()  # Confirmed retry still waits for the next post-release image.
    assert runtime._input_release_pending_ns is None
    assert notifications == [100] and runtime._sequence == 20
    assert not runtime.executor.policy._held_keys
    reached = []
    class NextFreshFrame(Exception):
        pass
    def fresh():
        reached.append(True)
        raise NextFreshFrame
    monkeypatch.setattr(runtime, "_flush_pending_skill_stats", fresh)
    with pytest.raises(NextFreshFrame):
        runtime.tick()
    assert reached == [True]


def test_stale_tick_confirms_release_without_advancing_skill_or_sequence(monkeypatch):
    runtime = _runtime(monkeypatch, stale=True)
    runtime.executor.policy._held_keys = {"w"}
    commands = []
    def send(command, **kwargs):
        commands.append((command, kwargs))
        return {"released": True, "lease_active": True}
    monkeypatch.setattr(runtime_module, "send_command", send)
    runtime.tick()
    assert commands == [("release-inputs", {"lease_id": "unit-lease"})]
    assert runtime._input_release_pending_ns is None and runtime._sequence == 20
    assert runtime.metrics.stale_frame_skips == 1 and runtime.metrics.motor_actions == 0
    assert not runtime.executor.policy._held_keys


def test_notification_failure_retains_release_interlock(monkeypatch):
    runtime = _runtime(monkeypatch)
    monkeypatch.setattr(runtime_module, "send_command", lambda *_args, **_kw: {
        "released": True, "lease_active": True,
    })
    runtime.executor.policy = SimpleNamespace(policy_id="unsupported")
    with pytest.raises(RuntimeError, match="cannot reconcile"):
        runtime._release_and_reconcile_inputs()
    assert runtime._input_release_pending_ns is not None


def test_router_missing_notification_fails_before_reconciling_any_body():
    primary = BootstrapMotorPolicy()
    primary._held_keys = {"w"}
    unsupported = SimpleNamespace(policy_id="legacy-without-notification")
    router = GroundedPolicyRouter(
        primary=primary, grounded=BootstrapMotorPolicy(), gui=unsupported,
    )
    with pytest.raises(RuntimeError, match="cannot reconcile"):
        router.notify_inputs_released()
    assert primary._held_keys == {"w"}


def test_headroom_quiescence_uses_same_acknowledged_notification(monkeypatch):
    runtime = _runtime(monkeypatch)
    runtime.executor.policy._held_keys = {"w"}
    monkeypatch.setattr(runtime_module, "send_command", lambda *_args, **_kw: {
        "released": True, "lease_active": True,
    })
    assert runtime._quiesce_headroom_inputs() is True
    assert not runtime.executor.policy._held_keys
    assert runtime._input_release_pending_ns is None and runtime._sequence == 20
