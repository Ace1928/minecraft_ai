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
    skills.stats[("escape_confinement", goal)] = SkillStats(
        failures=3, consecutive_failures=3
    )
    skills.stats[("explore_forward", goal)] = SkillStats(failures=3, consecutive_failures=3)
    skills.stats[("traverse_visible_obstacle", goal)] = SkillStats(
        failures=3, consecutive_failures=3
    )
    skills.stats[("backtrack_from_obstacle", goal)] = SkillStats(failures=3, consecutive_failures=3)
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
    decision = controller._observed_locomotion_recovery(board, context)
    assert decision is not None and decision.skill_parameters["allow_attack"] is False
    bounds = controller._decision_repair_bounds(
        board, context, allowed_skill_ids={"survey_surroundings"}
    )
    assert dict(bounds.required_action_constraints)["allow_attack"] is False


def test_confinement_recovery_admits_native_attack_without_inventory_or_use():
    from minecraft_ai.action_levels import ActionLevel

    controller, board, context = setup()
    controller.skills.stats[("escape_confinement", "operator:walk")] = SkillStats()
    selected = controller.decide(board, context)
    assert selected.skill_id == "escape_confinement"
    spec = controller.skills.get(selected.skill_id)
    assert spec.action_level == ActionLevel.LATENT
    assert spec.action_permissions.allow_attack
    assert not spec.action_permissions.allow_use
    assert not spec.action_permissions.allow_inventory
    assert controller.model.calls == 0


def test_literal_skill_command_keeps_the_trained_body_instruction():
    controller, board, context = setup()
    message = context.operator_messages[0].model_copy(update={
        "text": "escape confinement", "status": OperatorMessageStatus.QUEUED,
        "kind": OperatorMessageKind.CORRECTION,
    })
    decision = controller._operator_fast_path_decision(
        board, replace(context, operator_messages=(message,)),
    )
    assert decision is not None and decision.skill_id == "escape_confinement"
    assert decision.instruction == controller.skills.get("escape_confinement").policy_instruction
    assert "Dig forward" in decision.instruction


def test_new_learned_backtracking_method_is_available_after_failed_forward_methods():
    controller, board, context = setup()
    controller.skills.stats[("backtrack_from_obstacle", "operator:walk")] = SkillStats()
    controller.skills.stats[("survey_surroundings", "operator:walk")] = SkillStats(
        failures=3, consecutive_failures=3
    )
    decision = controller.decide(board, context)
    assert decision.skill_id == "backtrack_from_obstacle"
    assert controller.model.calls == 0


def test_unexecutable_plan_can_be_refined_and_old_goal_failures_do_not_poison_new_goal(tmp_path):
    from minecraft_ai.cognition import CognitionDecision
    from minecraft_ai.plan_graph import sanitize_plan_steps
    from minecraft_ai.storage import StateDatabase
    from test_agent_core import _runtime_for_learning

    assert sanitize_plan_steps(("continue_plan_execution", "next_step_if_successful")) == ()
    with StateDatabase(tmp_path / "state.sqlite") as database:
        runtime = _runtime_for_learning(database)
        runtime.skills = build_bootstrap_skill_library()
        runtime._plan_goal_id = "new-goal"
        runtime._plan_steps = ("continue_plan_execution",)
        runtime._adopt_plan_if_revised(
            CognitionDecision(
                chosen_goal_id="new-goal",
                plan_steps=("traverse_level_ground",),
                request_replan=True,
            )
        )
        assert runtime._plan_steps == ("traverse_level_ground",)
        runtime.skills.stats[("gather_nearby_wood", "old-goal")] = SkillStats(
            failures=100,
            consecutive_failures=100,
        )
        assert runtime._progression_goal().goal_id == "progression:gather_nearby_wood"


def test_plan_owned_mobility_proof_releases_old_stall_but_unverified_success_does_not(tmp_path):
    from minecraft_ai.outcome_verifier import (
        OutcomeKind,
        OutcomeSignal,
        OutcomeStatus,
        OutcomeVerification,
    )
    from minecraft_ai.storage import StateDatabase
    from test_agent_core import _runtime_for_learning

    with StateDatabase(tmp_path / "state.sqlite") as database:
        runtime = _runtime_for_learning(database)
        runtime.skills = build_bootstrap_skill_library()
        runtime._traversal_escalation_pending = True
        run = SkillRun(
            run_id="plan-progress",
            skill_id="traverse_visible_obstacle",
            context_key="operator:walk",
            started_ns=10,
            ended_ns=100,
            outcome=SkillOutcome.SUCCEEDED,
        )
        runtime._record_terminal_run(run)
        assert runtime._traversal_escalation_pending
        verified = run.model_copy(update={"run_id": "verified-progress"})
        proof = OutcomeVerification(
            run_id=verified.run_id,
            kind=OutcomeKind.TRAVERSAL,
            status=OutcomeStatus.PROGRESS,
            signal=OutcomeSignal.LOCOMOTION_PROGRESS,
            observed_ns=90,
            confidence=0.95,
            reason="action-bound traversal progress",
        )
        runtime._record_terminal_run(verified, outcome_verification=proof)
        assert not runtime._traversal_escalation_pending
        assert runtime._explore_keep_alive() is not None
        runtime._traversal_escalation_pending = True
        runtime._record_terminal_run(verified, outcome_verification=proof)
        assert runtime._traversal_escalation_pending  # replay cannot clear a newer failure

def test_bounded_null_replan_run_releases_the_escalation_guard(tmp_path):
    from minecraft_ai.storage import StateDatabase
    from test_agent_core import _runtime_for_learning

    with StateDatabase(tmp_path / "state.sqlite") as database:
        runtime = _runtime_for_learning(database)
        runtime.skills = build_bootstrap_skill_library()
        runtime._traversal_escalation_pending = True
        runtime._note_null_replan()
        assert runtime._traversal_escalation_pending  # one null replan still holds
        runtime._note_null_replan()
        assert not runtime._traversal_escalation_pending
        assert runtime._traversal_escalation_release.startswith("bounded_keepalive_resume")
        # The authorized, progress-verified keepalive rotation may resume.
        assert runtime._explore_keep_alive() is not None
        runtime._note_decision_started_skill()
        assert runtime._null_replan_streak == 0
        assert runtime._traversal_escalation_release is None
