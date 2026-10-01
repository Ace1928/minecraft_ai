"""Exercise the real runtime/executor with clocks, pixels and inputs replaced.

No client, model, network or live database is used. The fake actuator asserts
that positive dispatch retains both actual durable and operator admission.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

import minecraft_ai.runtime as runtime_module
from minecraft_ai.config import RuntimeConfig
from minecraft_ai.control.request_only import RequestOnlyConfig
from minecraft_ai.execution import ExecutionTick, SkillExecutor
from minecraft_ai.builtin_skills import build_bootstrap_skill_library
from minecraft_ai.perception import FrameState, PerceptionBlackboard, PerceptionFact
from minecraft_ai.perception_service import BEDROCK_HUD_SAFETY_SOURCE
from minecraft_ai.roles import get_role
from minecraft_ai.runtime import AgentRuntime
from minecraft_ai.safety import MotorAction
from minecraft_ai.skills import SkillOutcome
from minecraft_ai.social import OperatorMessage
from minecraft_ai.storage import StateDatabase


@pytest.fixture
def rig(monkeypatch, tmp_path):
    clock = SimpleNamespace(mono=10_000_000_000, wall=100_000_000_000)
    monkeypatch.setattr(runtime_module.time, "monotonic_ns", lambda: clock.mono)
    monkeypatch.setattr(runtime_module.time, "time_ns", lambda: clock.wall)
    monkeypatch.setattr(runtime_module, "operator_pause_latched", lambda: False)
    monkeypatch.setattr(runtime_module, "emergency_stop_latched", lambda: False)
    intent = SimpleNamespace(held=False)

    @contextmanager
    def lock(**kwargs):
        assert kwargs["timeout_s"] <= .05 and not intent.held
        intent.held = True
        try:
            yield
        finally:
            intent.held = False

    monkeypatch.setattr(runtime_module, "operator_intent_lock", lock)
    board = PerceptionBlackboard()
    commands = []
    database = StateDatabase(tmp_path / "only-fixture.sqlite")
    policy = SimpleNamespace(
        policy_id="fixture-only", reset=Mock(return_value=MotorAction(sequence=0)),
        notify_inputs_released=Mock(), close=Mock(), warmup=Mock(), request_warm=Mock(),
        status=Mock(return_value={"policy_id": "fixture-only"}),
        observe_accepted_action=Mock(),
        act=Mock(side_effect=lambda board, intent, **kw: MotorAction(
            sequence=kw["sequence"], mouse_dx=15, keys_down=("w", "q", "e", "space", "1"),
            buttons_down=("left", "right"), duration_ms=50,
        )),
    )
    vision = SimpleNamespace(start=Mock(), stop=Mock(), available=Mock(return_value=True),
                             status=Mock(return_value={}))
    perception = SimpleNamespace(
        instance_id="fixture-only", last_capture=None, active_vlm=vision, fast_perception=None,
        stale=Mock(return_value=False), close=Mock(), request_semantics=Mock(),
    )
    unsafe = [False]

    def capture():
        clock.mono += 100_000_000
        clock.wall += 100_000_000
        previous = board.raw_latest()
        frame = FrameState(frame_id=0 if previous is None else previous.frame_id + 1,
                           captured_ns=clock.mono, instance_id="fixture-only", width=32, height=32)
        board.publish(frame)
        board.merge_semantics(instance_id="fixture-only", facts=tuple(
            PerceptionFact(key=key, value=value, confidence=1.0, observed_ns=clock.mono,
                           source=BEDROCK_HUD_SAFETY_SOURCE, expires_after_ms=500)
            for key, value in (("scene.playable", True), ("scene.mode", "world"),
                               ("scene.death", unsafe[0]))
        ))
        perception.last_capture = SimpleNamespace(captured_ns=clock.mono)
        return frame

    perception.capture_once = Mock(side_effect=capture)
    value = AgentRuntime(
        perception=perception, blackboard=board, executor=SkillExecutor(policy),
        skills=build_bootstrap_skill_library(), role=get_role("generalist"), lease_id="fixture",
        state_db=database, operator_request_only=RequestOnlyConfig(),
        high_level=SimpleNamespace(decide=Mock()), telemetry=SimpleNamespace(publish=Mock()),
    )
    value._telemetry_payload = Mock(side_effect=lambda **kw: kw)
    value._flush_pending_skill_stats = Mock()
    value._flush_pending_learning_records = Mock()
    value._flush_pending_operator_status_updates = Mock()
    value._lease_heartbeat = Mock()
    value._merge_operator_target = Mock()
    policy.merge_perception = Mock(return_value=False)
    value._await_post_chat_capture = Mock(return_value=False)
    forbidden = {}
    for name in ("_keepalive_horizon_reorient", "_explore_keep_alive", "_route_observed_scene_recovery",
                 "_start_current_plan_skill", "_deliver_game_chat"):
        forbidden[name] = Mock(side_effect=AssertionError("unrequested work"))
        setattr(value, name, forbidden[name])
    inventory = Mock(side_effect=AssertionError("unrequested inventory direction"))
    monkeypatch.setattr("minecraft_ai.directions.runtime.tick_inventory_direction", inventory)
    release_ok = [True]

    def send(command, **kwargs):
        commands.append((command, kwargs))
        if command == "release-inputs":
            return {"released": release_ok[0], "lease_active": True}
        if command == "motor-action":
            assert intent.held and database.connection.in_transaction
            return {"accepted": True}
        return {}

    monkeypatch.setattr(runtime_module, "send_command", send)
    result = SimpleNamespace(runtime=value, clock=clock, commands=commands, policy=policy,
                             database=database, unsafe=unsafe, release_ok=release_ok,
                             forbidden=forbidden, inventory=inventory, intent=intent)
    try:
        yield result
    finally:
        if value._lease_thread is not None:
            value._lease_thread.join(.5)
            assert not value._lease_thread.is_alive()
        value._pool.shutdown(wait=False, cancel_futures=True)
        assert value._pool.wait_closed(1)
        database.close()
        vision.start.assert_not_called()
        perception.request_semantics.assert_not_called()
        value.high_level.decide.assert_not_called()
        inventory.assert_not_called()
        for callback in forbidden.values():
            callback.assert_not_called()


def enqueue(rig, **changes):
    data = dict(message_id="fresh-fixture", created_ns=rig.clock.wall, text="survey_surroundings",
                kind="correction", execution_budget={"timeout_ms": 1200, "max_skills": 1})
    data.update(changes)
    message = OperatorMessage.model_validate(data)
    rig.database.save_operator_message(message)
    return message


def begin(rig):
    rig.runtime.tick()  # Mandatory release, then a fresh capture.
    enqueue(rig)
    rig.runtime.tick()
    assert rig.runtime._literal_request_fence.state == "running"
    assert rig.runtime.executor.run.skill_id == "survey_surroundings"


@pytest.mark.parametrize("bad", [{"skill_id": "explore_forward"}, {"allow_attack": True},
                                  {"wait_timeout_ms": True}, {"wait_timeout_ms": 180001}])
def test_closed_scope_configuration(bad):
    with pytest.raises(ValidationError):
        RuntimeConfig(operator_request_only=bad)


def test_actual_startup_and_waiting_tick_do_not_start_models_or_autonomy(rig):
    value = rig.runtime
    original_tick = value.tick

    def stop_after_one_actual_tick():
        original_tick()
        value.stop()

    value.tick = stop_after_one_actual_tick
    value._shutdown_runtime = Mock()
    value.run_forever()
    value._warmup_policy()
    value._start_cognition_if_due()
    value._request_semantics_if_due(0)
    assert rig.policy.act.call_count == rig.policy.warmup.call_count == 0
    assert rig.policy.request_warm.call_count == 0
    assert [command for command, _ in rig.commands] == ["renew", "release-inputs"]
    value._shutdown_runtime.assert_called_once()


@pytest.mark.parametrize("changes", [
    {"text": "please look around"}, {"text": "explore_forward"}, {"execution_budget": None},
    {"execution_budget": {"timeout_ms": 12001, "max_skills": 1}},
    {"execution_budget": {"timeout_ms": 1200, "max_skills": 2}},
    {"status": "delivered"}, {"status": "acknowledged"},
    {"created_ns": 1}, {"kind": "question", "execution_budget": None},
    {"direction_request_id": "paid", "execution_budget": None},
])
def test_actual_tick_refuses_unqualified_or_replayed_requests(rig, changes):
    rig.runtime.tick()
    enqueue(rig, **changes)
    for _ in range(3):
        rig.runtime.tick()
    rig.policy.act.assert_not_called()
    rig.policy.request_warm.assert_not_called()
    assert not any(command == "motor-action" for command, _ in rig.commands)


def test_multiple_pending_requests_are_not_guessed(rig):
    rig.runtime.tick()
    enqueue(rig)
    enqueue(rig, message_id="different", text="explore_forward")
    rig.runtime.tick()
    rig.policy.act.assert_not_called()


def test_real_executor_masks_all_presses_and_camera_dispatch_holds_authority(rig):
    begin(rig)
    emitted = [kwargs["action"] for command, kwargs in rig.commands if command == "motor-action"]
    assert len(emitted) == 1 and emitted[0]["mouse_dx"] == 15
    assert emitted[0]["keys_down"] == emitted[0]["buttons_down"] == []
    assert rig.policy.act.call_count == rig.policy.observe_accepted_action.call_count == 1
    assert rig.runtime.metrics.motor_actions == 1
    assert rig.runtime._literal_request_fence.attempt.skills_started == 1


def test_deadline_and_later_queue_never_restore_autonomy_or_replay_owner(rig):
    begin(rig)
    rig.clock.mono += 2_000_000_000
    rig.runtime.tick()
    count = rig.policy.act.call_count
    enqueue(rig, message_id="second-fresh")
    for _ in range(8):
        rig.runtime.tick()
    assert rig.runtime._literal_request_fence.state == "terminal"
    assert rig.policy.act.call_count == count
    assert rig.runtime.executor.run.outcome != SkillOutcome.RUNNING


@pytest.mark.parametrize("acknowledged", [False, True])
def test_expiry_exception_still_releases_and_cannot_restore_autonomy(rig, monkeypatch, acknowledged):
    begin(rig)
    rig.commands.clear()
    rig.release_ok[0] = acknowledged
    calls = []

    def failed_expiry(**kwargs):
        assert kwargs["reason"].startswith("operator.attempt_")
        calls.append(kwargs)
        raise RuntimeError("fixture terminal persistence failure")

    monkeypatch.setattr(rig.runtime.executor, "expire_operator_attempt", failed_expiry)
    rig.clock.mono += 2_000_000_000
    with pytest.raises(RuntimeError, match="terminal persistence failure"):
        rig.runtime.tick()
    assert len(calls) == 1
    assert rig.runtime._literal_request_fence.state == "terminal"
    releases = [command for command, _ in rig.commands if command == "release-inputs"]
    assert len(releases) == (1 if acknowledged else 2)
    assert not any(command == "motor-action" for command, _ in rig.commands)
    assert (rig.runtime._input_release_pending_ns is None) is acknowledged
    rig.release_ok[0] = True
    count = rig.policy.act.call_count
    enqueue(rig, message_id="later-after-persistence-failure")
    for _ in range(4):
        rig.runtime.tick()
    assert rig.policy.act.call_count == count
    assert len(calls) == 1
    assert rig.runtime._literal_request_fence.state == "terminal"


def test_slow_real_policy_return_is_charged_and_cannot_dispatch(rig):
    def late(board, intent, **kwargs):
        rig.clock.mono += 2_000_000_000
        return MotorAction(sequence=kwargs["sequence"], mouse_dx=20)
    rig.policy.act.side_effect = late
    rig.runtime.tick()
    enqueue(rig)
    rig.runtime.tick()
    assert not any(command == "motor-action" for command, _ in rig.commands)
    assert rig.runtime._literal_request_fence.state == "terminal"
    rig.policy.observe_accepted_action.assert_not_called()


@pytest.mark.parametrize("press", [dict(keys_down=("w",)), dict(buttons_down=("left",)),
                                  dict(mouse_dx=20, camera_semantics="cursor"),
                                  dict(cursor_x=.5, cursor_y=.5, camera_semantics="cursor")])
def test_direct_wire_caller_cannot_escape_camera_mask_or_get_acceptance(rig, press):
    begin(rig)
    rig.commands.clear()
    prior = rig.runtime.metrics.motor_actions
    result = ExecutionTick(run=rig.runtime.executor.run,
                           action=MotorAction(sequence=rig.runtime._sequence, **press))
    assert rig.runtime._send_motor(result.action, execution=result) is False
    assert not any(command == "motor-action" for command, _ in rig.commands)
    assert rig.runtime.metrics.motor_actions == prior
    assert rig.runtime._literal_request_fence.state == "terminal"


def test_changed_authority_after_policy_cannot_cross_actual_wire_boundary(rig):
    def changing(board, intent, **kwargs):
        enqueue(rig, message_id="superseding", text="explore_forward")
        return MotorAction(sequence=kwargs["sequence"], mouse_dx=20)
    rig.policy.act.side_effect = changing
    rig.runtime.tick()
    enqueue(rig)
    rig.runtime.tick()
    assert not any(command == "motor-action" for command, _ in rig.commands)
    rig.policy.observe_accepted_action.assert_not_called()
    assert rig.runtime._literal_request_fence.state == "terminal"


def test_waiting_timeout_remains_held_with_a_later_fresh_goal(rig):
    rig.runtime.tick()
    rig.clock.mono += 61_000_000_000
    rig.runtime.tick()
    enqueue(rig)
    rig.runtime.tick()
    assert rig.runtime._literal_request_fence.reason == "no_request_deadline"
    rig.policy.act.assert_not_called()


def test_verified_unsafe_scene_aborts_without_respawn_or_recovery_inputs(rig):
    begin(rig)
    rig.commands.clear()
    rig.unsafe[0] = True
    rig.runtime.tick()
    assert rig.runtime._literal_request_fence.reason == "scene_unavailable"
    assert not any(command == "motor-action" for command, _ in rig.commands)
    assert rig.runtime.executor.run.skill_id == "survey_surroundings"


def test_unacknowledged_release_cannot_start_or_advance_survey(rig):
    rig.release_ok[0] = False
    enqueue(rig)
    for _ in range(3):
        rig.runtime.tick()
    rig.policy.act.assert_not_called()
    assert rig.runtime._input_release_pending_ns is not None


def test_wire_wait_is_charged_again_at_final_boundary(rig, monkeypatch):
    begin(rig)
    rig.commands.clear()
    original = rig.database.admit_operator_revision

    @contextmanager
    def waiting(revision):
        with original(revision) as current:
            rig.clock.mono += 2_000_000_000
            yield current

    monkeypatch.setattr(rig.database, "admit_operator_revision", waiting)
    result = ExecutionTick(run=rig.runtime.executor.run,
                           action=MotorAction(sequence=rig.runtime._sequence, mouse_dx=20))
    assert rig.runtime._send_motor(result.action, execution=result) is False
    assert not any(command == "motor-action" for command, _ in rig.commands)


def test_new_request_only_config_does_not_activate_by_default():
    assert RuntimeConfig().operator_request_only is None


def test_optional_perception_hook_cannot_start_unrequested_work_before_or_after_goal(rig):
    callback = rig.policy.merge_perception
    callback.side_effect = AssertionError("unrequested model/perception callback")
    rig.runtime.tick()
    rig.runtime.tick()
    callback.assert_not_called()
    callback.side_effect = None  # A cache-only fixture is permitted inside the admitted goal.
    enqueue(rig)
    rig.runtime.tick()
    assert callback.call_count == 1
    rig.clock.mono += 2_000_000_000
    callback.side_effect = AssertionError("postterminal model/perception callback")
    rig.runtime.tick()
    rig.runtime.tick()
    assert callback.call_count == 1


def test_refused_actual_skill_start_enters_terminal_hold_without_retry(rig, monkeypatch):
    rig.runtime.tick()
    enqueue(rig)
    original = rig.runtime._start_skill

    def wrong_owner(spec, **kwargs):
        # Exercise the production admission check, not a simulated refusal.
        kwargs["run_id"] = "foreign-owner"
        return original(spec, **kwargs)

    rejected = Mock(side_effect=wrong_owner)
    monkeypatch.setattr(rig.runtime, "_start_skill", rejected)
    rig.runtime.tick()
    assert rig.runtime._literal_request_fence.state == "terminal"
    for _ in range(3):
        rig.runtime.tick()
    rejected.assert_called_once()
    rig.policy.act.assert_not_called()


def test_cursor_pair_invariant_refuses_lone_coordinate():
    with pytest.raises(ValidationError):
        MotorAction(sequence=0, cursor_y=.5, camera_semantics="cursor")


def test_none_mode_retains_existing_cognition_startup_and_legacy_tick_branch(rig, monkeypatch):
    value = rig.runtime
    value.operator_request_only = None  # A distinct fixture for the original default behavior.
    value._literal_request_fence = None
    cognition = Mock()
    warmup = Mock()
    monkeypatch.setattr(value, "_start_cognition_if_due", cognition)
    monkeypatch.setattr(value, "_warmup_policy", warmup)
    monkeypatch.setattr(value, "tick", value.stop)
    monkeypatch.setattr(value, "_shutdown_runtime", Mock())
    value.run_forever()
    cognition.assert_called_once()
    warmup.assert_called_once()
    value.perception.active_vlm.start.assert_called_once()
    value.perception.active_vlm.start.reset_mock()
    # The normal capture branch still merges optional learned perception and
    # reaches inventory-direction dispatch. Stop there, with no actual inputs.
    value._stop.clear()
    monkeypatch.setattr(value, "_expire_current_operator_attempt", Mock(return_value=False))
    monkeypatch.setattr(value, "_publish_player_chat_facts", Mock())
    inventory = Mock(return_value=True)
    monkeypatch.setattr("minecraft_ai.directions.runtime.tick_inventory_direction", inventory)
    monkeypatch.setattr("minecraft_ai.operator.standby.apply_reasoning_standby", Mock(return_value=False))
    AgentRuntime.tick(value)
    rig.policy.merge_perception.assert_called_once()
    inventory.assert_called_once()
