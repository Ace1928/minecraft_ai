from dataclasses import replace

import pytest

from minecraft_ai.builtin_skills import build_bootstrap_skill_library
from minecraft_ai.control.action_envelope import constrain_action
from minecraft_ai.execution import SkillExecutor
from minecraft_ai.motor import BootstrapMotorPolicy
from minecraft_ai.runtime_support.helpers import _verified_obstacle_stall
from minecraft_ai.safety import MotorAction
from minecraft_ai.skills import SkillActionPermissions, SkillSpec
from minecraft_ai.trajectory import ActionOrigin
from test_headroom_recovery import _stall_result
from test_mining_control import _mining_board


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
