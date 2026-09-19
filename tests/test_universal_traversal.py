from dataclasses import replace

import pytest

from minecraft_ai.builtin_skills import build_bootstrap_skill_library
from minecraft_ai.cognition import CognitionDecision
from minecraft_ai.control.action_envelope import constrain_action
from minecraft_ai.execution import SkillExecutor
from minecraft_ai.motor import BootstrapMotorPolicy
from minecraft_ai.runtime_support.helpers import _verified_obstacle_stall
from minecraft_ai.safety import MotorAction
from minecraft_ai.skills import SkillActionPermissions, SkillOutcome, SkillRun, SkillSpec
from minecraft_ai.trajectory import ActionOrigin
from test_headroom_recovery import _stall_result
from test_agent_core import _runtime_with_completed_decision
from test_mining_control import _ScriptedPolicy, _mining_board
from test_outcome_verifier import _LUMA_A, _LUMA_B, _LUMA_C, _publish_hashes


@pytest.mark.parametrize("skill", ["traverse_level_ground", "explore_forward", "custom_mod_walker"])
def test_verified_stalls_are_model_and_name_independent(skill):
    result = _stall_result(skill_id=skill)
    assert _verified_obstacle_stall(result)
    assert not _verified_obstacle_stall(replace(result, outcome_verification=None))
    assert not _verified_obstacle_stall(replace(
        result, outcome_verification=result.outcome_verification.model_copy(
            update={"run_id": "other"},
        ),
    ))


@pytest.mark.parametrize("permission,keys,buttons", [
    ("allow_movement", ("w", "space"), ()),
    ("allow_jump", ("space",), ()),
    ("allow_attack", (), ("left",)),
    ("allow_use", (), ("right",)),
    ("allow_hotbar", ("1", "9"), ()),
    ("allow_inventory", ("e",), ()),
    ("allow_drop", ("q",), ()),
])
def test_physical_envelope_overrides_noncompliant_policy(permission, keys, buttons):
    proposal = MotorAction(sequence=1, keys_down=keys, buttons_down=buttons, mouse_dx=31)
    result = constrain_action(proposal, {permission: False})
    assert result.keys_down == result.buttons_down == ()
    assert set(keys) <= set(result.keys_up)
    assert set(buttons) <= set(result.buttons_up)
    assert result.mouse_dx == 31
    MotorAction.model_validate(result.model_dump())
    assert constrain_action(proposal, {permission: True}) is proposal


def test_survey_releases_previous_walk_without_inventing_oversize_release():
    spec = build_bootstrap_skill_library().get("survey_surroundings")
    result = constrain_action(
        MotorAction(sequence=1, mouse_dy=12), spec.action_permissions.model_dump(),
        held_keys=("w", "ctrl"), held_buttons=("left",),
    )
    assert result.keys_up == ("ctrl", "w")
    assert result.buttons_up == ("left",)
    MotorAction.model_validate(result.model_dump())


def test_custom_skill_can_use_traversal_verification():
    executor = SkillExecutor(BootstrapMotorPolicy())
    executor.start(SkillSpec(
        skill_id="mod_walker", name="Walker", policy_ref="navigate", outcome_kind="traversal",
    ), run_id="walk", now_ns=1)
    tick = executor.tick(_mining_board(now_ns=1), sequence=1, now_ns=1)
    assert executor._outcome_verifier.active_run_id == "walk"
    verification = executor._observe_traversal_outcome(
        _mining_board(now_ns=2), action=tick.action, now_ns=2,
    )
    assert verification.kind.value == "traversal"


def test_executor_masks_motion_and_records_intervention():
    executor = SkillExecutor(BootstrapMotorPolicy())
    executor.start(SkillSpec(
        skill_id="third_party_survey", name="Survey", policy_ref="navigate",
        action_permissions=SkillActionPermissions(allow_movement=False),
    ), run_id="survey", now_ns=1)
    tick = executor.tick(_mining_board(now_ns=1), sequence=1, now_ns=1)
    assert not tick.action.keys_down
    assert "w" in tick.action.keys_up
    assert tick.action_origin == ActionOrigin.SYNTHETIC


@pytest.mark.parametrize("start_path", ("cognition", "recovery", "plan", "nested"))
@pytest.mark.parametrize("skill_id", (
    "escape_confinement", "backtrack_from_obstacle", "explore_forward", "custom_mod_walker",
))
def test_typed_traversal_completes_from_verified_progress_on_every_start_path(
    monkeypatch, start_path, skill_id,
):
    clock = [1_000_000_000]
    monkeypatch.setattr("time.monotonic_ns", lambda: clock[0])
    runtime = _runtime_with_completed_decision(CognitionDecision(
        skill_id=skill_id, chosen_goal_id="operator:walk",
    ))
    runtime.skills = build_bootstrap_skill_library()
    runtime.skills.register(SkillSpec(
        skill_id="custom_mod_walker", name="Custom walker", outcome_kind="traversal",
    ))
    runtime.executor = SkillExecutor(_ScriptedPolicy(MotorAction(sequence=0, keys_down=("w",))))
    runtime.blackboard = _mining_board(now_ns=clock[0])
    runtime._plan_steps = (skill_id,)
    runtime._plan_goal_id = "operator:walk"
    runtime._plan_index = 0
    runtime._plan_graph = None
    spec = runtime.skills.get(skill_id)
    parent = None
    if start_path == "plan":
        assert runtime._start_current_plan_skill()
    elif start_path == "recovery":
        runtime._start_recovery_skill(spec, SkillRun(
            run_id="failed-parent", skill_id="traverse_level_ground", started_ns=clock[0] - 1,
            outcome=SkillOutcome.FAILED, context_key="operator:walk",
        ))
    else:
        if start_path == "nested":
            parent_spec = SkillSpec(
                skill_id="parent", name="Parent", recovery_skills=(skill_id,),
            )
            runtime.skills.register(parent_spec)
            parent = runtime.executor.start(
                parent_spec, run_id="parent", context_key="operator:walk",
            )
        runtime._consume_cognition()

    run = runtime.executor.run
    assert run is not None and run.skill_id == skill_id
    assert run.context_key == "operator:walk"
    for index, luma in enumerate((_LUMA_A, _LUMA_B, _LUMA_C, _LUMA_B, _LUMA_C, _LUMA_B, _LUMA_C)):
        clock[0] = run.started_ns + index * 250_000_000
        _publish_hashes(runtime.blackboard, clock[0], frame_hash="0" * 16, luma_grid=luma)
        result = runtime.executor.tick(runtime.blackboard, sequence=index, now_ns=clock[0])
        if index < 6:
            assert result.run.outcome == SkillOutcome.RUNNING
    assert result.run.outcome == SkillOutcome.SUCCEEDED
    assert result.outcome_verification.signal.value == "locomotion_progress"
    stats = runtime.skills.record(result.run)
    assert stats.successes == 1 and stats.timeouts == stats.consecutive_failures == 0
    if parent is not None:
        assert runtime.executor.resume_parent() is parent
        assert not runtime.executor._complete_on_locomotion_progress


@pytest.mark.parametrize("evidence", ("attack", "camera", "static", "no_input"))
def test_typed_traversal_never_completes_from_confounded_or_uncommanded_pixels(evidence):
    now = 1_000_000_000
    actions = tuple(MotorAction(
        sequence=index,
        keys_down=() if evidence == "no_input" else ("w",),
        buttons_down=("left",) if evidence == "attack" else (),
        mouse_dx=20 if evidence == "camera" else 0,
    ) for index in range(24))
    executor = SkillExecutor(_ScriptedPolicy(*actions))
    executor.start(build_bootstrap_skill_library().get("escape_confinement"),
                   run_id="escape", now_ns=now)
    for index in range(24):
        current = now + index * 250_000_000
        board = _mining_board(now_ns=current)
        luma = _LUMA_A if evidence == "static" else (_LUMA_B if index % 2 else _LUMA_C)
        _publish_hashes(board, current, frame_hash="0" * 16, luma_grid=luma)
        result = executor.tick(board, sequence=index, now_ns=current)
        assert result.run.outcome != SkillOutcome.SUCCEEDED


def test_traversal_completion_defaults_preserve_explicit_override_and_untyped_contract():
    spec = build_bootstrap_skill_library().get("escape_confinement")
    executor = SkillExecutor(BootstrapMotorPolicy())
    executor.start(spec, run_id="override", complete_on_locomotion_progress=False)
    assert not executor._complete_on_locomotion_progress
    executor.cancel()
    executor.start(spec.model_copy(update={"outcome_kind": None}), run_id="untyped")
    assert not executor._complete_on_locomotion_progress


def test_recovery_keeps_actuator_prohibitions_without_copying_unrelated_parameters():
    runtime = _runtime_with_completed_decision(CognitionDecision())
    runtime.skills = build_bootstrap_skill_library()
    runtime.executor = SkillExecutor(_ScriptedPolicy(MotorAction(
        sequence=0, keys_down=("w", "space"), buttons_down=("left",),
    )))
    run = runtime._start_recovery_skill(runtime.skills.get("escape_confinement"), SkillRun(
        run_id="parent", skill_id="explore_forward", started_ns=1, outcome=SkillOutcome.FAILED,
        context_key="operator:stay", parameters={
            "allow_movement": False, "allow_attack": False, "target": "unrelated",
        },
    ))
    assert run.parameters == {"allow_movement": False, "allow_attack": False}
    tick = runtime.executor.tick(_mining_board(now_ns=run.started_ns),
                                 sequence=0, now_ns=run.started_ns)
    assert not tick.action.keys_down and not tick.action.buttons_down
    assert set(tick.action.keys_up) >= {"w", "space"}
    assert tick.action.buttons_up == ("left",)
