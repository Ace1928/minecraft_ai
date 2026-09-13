"""An accepted movement goal must retain executable recovery choices after failure."""

import time
from dataclasses import replace

from minecraft_ai.builtin_skills import build_bootstrap_skill_library
from minecraft_ai.cognition import CognitionContext, HighLevelController
from minecraft_ai.perception import FrameState, PerceptionBlackboard, PerceptionFact
from minecraft_ai.roles import get_role
from minecraft_ai.skills import SkillFailureCode, SkillOutcome, SkillRun, SkillStats
from minecraft_ai.social import OperatorMessage, OperatorMessageKind, OperatorMessageStatus


class UnusedModel:
    model_id = "not-needed-for-physical-recovery"
    calls = 0

    def complete(self, *_args, **_kwargs):
        self.calls += 1
        raise AssertionError("measured recovery must not invoke the language model")


def setup():
    now = time.monotonic_ns()
    board = PerceptionBlackboard()
    board.publish(
        FrameState(
            frame_id=7,
            captured_ns=now,
            instance_id="test-world",
            width=1280,
            height=720,
            facts=tuple(
                PerceptionFact(
                    key=key,
                    value=value,
                    confidence=1,
                    observed_ns=now,
                    source="test",
                    expires_after_ms=10_000,
                )
                for key, value in (
                    ("scene.playable", True),
                    ("scene.mode", "world"),
                    ("scene.ui_overlay", False),
                    ("environment.underwater", False),
                )
            ),
        )
    )
    skills = build_bootstrap_skill_library()
    goal = "operator:walk"
    skills.stats[("explore_forward", goal)] = SkillStats(failures=3, consecutive_failures=3)
    skills.stats[("traverse_visible_obstacle", goal)] = SkillStats(
        failures=3, consecutive_failures=3
    )
    message = OperatorMessage(
        message_id="walk",
        created_ns=1,
        text="explore forward",
        kind=OperatorMessageKind.INSTRUCTION,
        status=OperatorMessageStatus.ACKNOWLEDGED,
    )
    run = SkillRun(
        run_id="blocked-walk",
        skill_id="explore_forward",
        context_key=goal,
        started_ns=now - 2_000_000_000,
        ended_ns=now - 1_000_000_000,
        outcome=SkillOutcome.FAILED,
        failure_code=SkillFailureCode.LOCOMOTION_STALLED,
    )
    context = CognitionContext(
        role=get_role("generalist"),
        goals=(),
        memories=(),
        promises=(),
        wiki=(),
        operator_messages=(message,),
        recent_skill_runs=(run,),
        plan_goal_id=goal,
        current_plan=("explore_forward",),
        plan_index=1,
    )
    return HighLevelController(UnusedModel(), skills), board, context


def test_acknowledged_failed_goal_allows_its_declared_recovery_in_the_grammar():
    controller, board, context = setup()
    bounds = controller._decision_repair_bounds(
        board, context, allowed_skill_ids={"survey_surroundings"}
    )
    assert "survey_surroundings" in bounds.requested_skill_ids
    assert dict(bounds.allowed_skills) == {"survey_surroundings": ()}
    assert bounds.authority_goal_id == "operator:walk"
    assert not controller.skills.get("survey_surroundings").action_permissions.allow_attack


def test_observed_stall_selects_learned_recovery_without_a_second_slow_model_call():
    controller, board, context = setup()
    decision = controller.decide(board, context)
    assert decision.skill_id == "survey_surroundings"
    assert decision.chosen_goal_id == "operator:walk"
    assert not controller.skills.get(decision.skill_id).action_permissions.allow_attack
    assert not decision.request_replan and not decision.ask_perception
    assert decision.model_origin is None
    assert controller.model.calls == 0
    assert controller.status()["fast_recoveries"] == 1


def test_new_directives_corrections_other_goals_and_controller_waits_do_not_expand_authority():
    controller, board, context = setup()
    message = context.operator_messages[0]
    run = context.recent_skill_runs[0]
    for revised in (
        replace(
            context,
            operator_messages=(
                message.model_copy(update={"status": OperatorMessageStatus.QUEUED}),
            ),
        ),
        replace(
            context,
            operator_messages=(
                message.model_copy(update={"kind": OperatorMessageKind.CORRECTION}),
            ),
        ),
        replace(
            context, recent_skill_runs=(run.model_copy(update={"context_key": "another-goal"}),)
        ),
        replace(
            context,
            recent_skill_runs=(
                run.model_copy(update={"failure_code": SkillFailureCode.CONTROLLER_STARVATION}),
            ),
        ),
    ):
        assert controller._observed_locomotion_recovery(board, revised) is None
        assert controller._decision_repair_bounds(board, revised).requested_skill_ids == (
            "explore_forward",
        )


def test_missing_world_evidence_and_a_newer_success_prevent_stale_fast_recovery():
    controller, board, context = setup()
    assert controller._observed_locomotion_recovery(PerceptionBlackboard(), context) is None
    newer = context.recent_skill_runs[0].model_copy(
        update={
            "run_id": "new-success",
            "outcome": SkillOutcome.SUCCEEDED,
            "failure_code": None,
            "ended_ns": time.monotonic_ns(),
            "skill_id": "survey_surroundings",
        }
    )
    revised = replace(context, recent_skill_runs=(*context.recent_skill_runs, newer))
    assert controller._observed_locomotion_recovery(board, revised) is None


def test_explicit_prohibitions_stay_in_force_for_deliberated_recovery():
    controller, board, context = setup()
    message = context.operator_messages[0].model_copy(
        update={"text": "explore forward; do not attack"}
    )
    context = replace(context, operator_messages=(message,))
    assert controller._observed_locomotion_recovery(board, context) is None
    bounds = controller._decision_repair_bounds(
        board, context, allowed_skill_ids={"survey_surroundings"}
    )
    assert dict(bounds.required_action_constraints)["allow_attack"] is False
