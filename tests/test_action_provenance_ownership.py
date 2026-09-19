"""A current execution must not borrow unrelated policy diagnostics as its cause."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_ai.action_levels import ActionLevel
from minecraft_ai.builtin_skills import build_bootstrap_skill_library
from minecraft_ai.execution import ExecutionTick, SkillExecutor
from minecraft_ai.motor import MotorIntent
from minecraft_ai.perception import FrameState, PerceptionBlackboard
from minecraft_ai.platforms.bedrock_x11 import CapturedFrame
from minecraft_ai.runtime_support.helpers import _accepted_action_provenance
from minecraft_ai.safety import MotorAction
from minecraft_ai.skills import SkillRun
from minecraft_ai.trajectory import ActionOrigin, TrajectoryReader, TrajectoryRecorder
from test_trajectory import _manifest
from test_mining_control import _mining_board


class RetainedPolicyStatus:
    policy_id = "fixture:retained-body"

    def __init__(self, intent: MotorIntent) -> None:
        self.calls = 0
        self.causal = {
            "policy_id": self.policy_id, "model_version": "fixture-model",
            "action_kind": "prediction", "request_id": "request-1", "prediction_id": "prediction-1",
            "source_frame_id": 7, "source_captured_ns": 10,
            "condition": intent.model_dump(mode="json"), "action_level": intent.action_level.value,
            "behavior_token": 41, "latent_id": "latent-41",
        }

    def act(self, blackboard, intent, *, sequence):
        self.calls += 1
        return MotorAction(sequence=sequence, buttons_down=("left",), mouse_dx=5)

    def status(self):
        return {
            "policy_id": self.policy_id, "model_version": "fixture-model",
            "active_route": "direct", "last_action_provenance": dict(self.causal),
            "last_prediction": {"behavior_token": 41, "latent_id": "latent-41"},
        }

    def reset(self):
        # Diagnostics need not disappear when a controller no longer uses this body.
        return MotorAction(sequence=1, buttons_up=("left",))


@pytest.mark.parametrize("skill_id", (
    "open_inventory", "close_open_inventory", "craft_wood_planks",
    "respawn_after_death", "dismiss_away_overlay",
))
def test_deterministic_control_never_inherits_stale_prediction(tmp_path, monkeypatch, skill_id):
    policy = RetainedPolicyStatus(MotorIntent(
        skill_id="experiment_interactions", mode="experiment_interactions",
        episode_id="old-run", action_level=ActionLevel.RAW,
    ))
    executor = SkillExecutor(policy)
    executor.start(build_bootstrap_skill_library().get(skill_id), run_id="current-run", now_ns=100)
    frame = CapturedFrame(8, 100, 4, 3, bytes((20, 30, 40, 255)) * 12)
    board = PerceptionBlackboard()
    board.publish(FrameState(
        frame_id=8, captured_ns=100, instance_id="fixture", width=4, height=3,
    ))
    if skill_id == "open_inventory":
        tick = executor._tick_inventory_open(sequence=2)
    elif skill_id == "close_open_inventory":
        tick = executor._tick_inventory_close(sequence=2)
    elif skill_id == "craft_wood_planks":
        executor._plank_crafter = SimpleNamespace(step=lambda *args, **kwargs: SimpleNamespace(
            mode="craft_planks", instruction="craft planks", failure_reason=None,
            verification=None, action=MotorAction(sequence=2, keys_down=("e",), keys_up=("e",)),
        ))
        tick = executor._tick_plank_crafting(board, sequence=2, now_ns=100)
    else:
        name = "death_respawn_control_center" if skill_id == "respawn_after_death" else (
            "away_overlay_click_center"
        )
        monkeypatch.setattr(f"minecraft_ai.control.execution.{name}", lambda capture: (1, 1))
        method = executor._tick_death_respawn if skill_id == "respawn_after_death" else (
            executor._tick_away_dismiss
        )
        tick = method(sequence=2, capture=frame)
    assert policy.calls == 0
    assert tick.action_origin == ActionOrigin.SYNTHETIC
    provenance = _accepted_action_provenance(tick, board, fallback_policy_id=policy.policy_id)
    assert provenance.condition == tick.motor_intent.model_dump(mode="json")
    assert provenance.route_id == "synthetic"
    assert provenance.policy_id == "runtime:synthetic-control"
    assert provenance.model_version is None
    assert provenance.policy_request_id is provenance.prediction_id is None
    assert provenance.source_frame_id is provenance.source_captured_ns is None
    assert provenance.behavior_token is provenance.latent_id is None
    assert policy.causal["condition"]["episode_id"] == "old-run"

    recorder = TrajectoryRecorder(
        manifest=_manifest("ownership-test"), artifact_root=tmp_path / "trajectories",
        state_db_path=tmp_path / "state.sqlite3", min_free_disk_bytes=0,
    )
    try:
        assert recorder.record_accepted(
            action=tick.action, provenance=provenance,
            supervisor_response={"accepted_sequence": 2, "accepted_monotonic_ns": 120},
            frame=frame, blackboard=board.raw_latest(),
            skill_id=tick.run.skill_id, skill_run_id=tick.run.run_id,
        )
    finally:
        manifest = recorder.close()
    assert manifest.accepted_steps == 1 and manifest.dropped_steps == 0
    sample, = TrajectoryReader(tmp_path / "trajectories" / "ownership-test").iter_samples()
    assert sample.step.action == tick.action
    assert sample.step.accepted_ns == 120
    assert sample.step.skill_id == skill_id and sample.step.skill_run_id == "current-run"
    assert sample.step.action_origin == ActionOrigin.SYNTHETIC


@pytest.mark.parametrize("wrong_skill", (True, False))
def test_policy_prediction_with_wrong_execution_identity_is_rejected_not_relabelled(wrong_skill):
    intent = MotorIntent(
        skill_id="old-skill" if wrong_skill else "new-skill", mode="old", episode_id="old-run",
    )
    policy = RetainedPolicyStatus(intent)
    execution = ExecutionTick(
        run=SkillRun(run_id="new-run", skill_id="new-skill", started_ns=100),
        action=MotorAction(sequence=2, keys_down=("w",)),
        motor_intent=MotorIntent(skill_id="new-skill", mode="new", episode_id="new-run"),
        policy_status=policy.status(), action_origin=ActionOrigin.POLICY,
    )
    with pytest.raises(ValueError, match="skill_.*does not match"):
        _accepted_action_provenance(
            execution, PerceptionBlackboard(), fallback_policy_id=policy.policy_id,
        )
    assert policy.causal["condition"] == intent.model_dump(mode="json")


def test_guarded_policy_proposal_keeps_its_actual_condition_and_prediction_identity():
    spec = build_bootstrap_skill_library().get("survey_surroundings")
    original = MotorIntent(
        skill_id=spec.skill_id, mode=spec.policy_ref, episode_id="survey-run",
        action_level=spec.action_level, instruction="the earlier request's exact instruction",
    )
    policy = RetainedPolicyStatus(original)
    executor = SkillExecutor(policy)
    executor.start(spec, run_id="survey-run", now_ns=100)
    board = _mining_board(now_ns=100)
    tick = executor.tick(board, sequence=2, now_ns=100)
    assert policy.calls == 1
    assert tick.policy_proposal.buttons_down == ("left",)
    assert tick.action.buttons_down == () and tick.action.mouse_dx == 5
    assert tick.action_origin == ActionOrigin.SYNTHETIC
    provenance = _accepted_action_provenance(tick, board, fallback_policy_id=policy.policy_id)
    assert provenance.condition == original.model_dump(mode="json")
    assert provenance.condition != tick.motor_intent.model_dump(mode="json")
    assert provenance.policy_id == policy.policy_id
    assert provenance.model_version == "fixture-model"
    assert provenance.policy_request_id == "request-1"
    assert provenance.prediction_id == "prediction-1"
    assert provenance.origin == ActionOrigin.SYNTHETIC  # Not an unmodified policy demonstration.


def test_reset_never_borrows_a_previous_behavior_or_latent_token():
    policy = RetainedPolicyStatus(MotorIntent(
        skill_id="old", mode="old", episode_id="old-run",
    ))
    tick = ExecutionTick(
        run=SkillRun(run_id="current", skill_id="new", started_ns=100),
        action=MotorAction(sequence=4, buttons_up=("left",)),
        policy_status=policy.status(), action_origin=ActionOrigin.RESET,
    )
    provenance = _accepted_action_provenance(
        tick, PerceptionBlackboard(), fallback_policy_id=policy.policy_id,
    )
    assert provenance.condition is None
    assert provenance.policy_request_id is provenance.prediction_id is None
    assert provenance.behavior_token is provenance.latent_id is None


def test_unbound_runtime_action_does_not_claim_a_policy_prediction():
    provenance = _accepted_action_provenance(
        None, PerceptionBlackboard(), fallback_policy_id="unrelated-resident-model",
    )
    assert provenance.origin == ActionOrigin.SYNTHETIC
    assert provenance.policy_id == "runtime:synthetic-control"
    assert provenance.condition is None
    assert provenance.policy_request_id is provenance.prediction_id is None
