from collections import deque
from concurrent.futures import Future
from dataclasses import replace

from minecraft_ai.builtin_skills import build_bootstrap_skill_library
from minecraft_ai.cognition import CognitionDecision
from minecraft_ai.control.execution import SkillExecutor, visible_oak_trunk
from minecraft_ai.motor import BootstrapMotorPolicy
from minecraft_ai.perception import ScreenRegion, Track
from minecraft_ai.skills import SkillFailureCode, SkillOutcome, SkillRun
from test_agent_core import _runtime_with_completed_decision
from test_cognition_recovery_liveness import setup


def test_starved_plan_waits_for_fresh_admission_instead_of_invalidating_planner():
    runtime = _runtime_with_completed_decision(CognitionDecision(skill_id="gather_nearby_wood"))
    runtime.skills = build_bootstrap_skill_library()
    runtime.executor = SkillExecutor(BootstrapMotorPolicy())
    runtime._plan_steps = ("gather_nearby_wood",)
    runtime._plan_index = 0
    runtime._plan_goal_id = "operator:wood"
    runtime._recent_skill_runs = deque((SkillRun(
        run_id="failed", skill_id="gather_nearby_wood", context_key="operator:wood",
        started_ns=1, ended_ns=2, outcome=SkillOutcome.FAILED,
        failure_code=SkillFailureCode.CONTROLLER_STARVATION,
    ),))
    runtime._pending_decision = Future()
    revision = runtime._execution_revision
    assert not runtime._start_current_plan_skill()
    assert runtime._execution_revision == revision
    assert runtime._cognition_requested
    assert runtime.executor.run is None


def test_missing_trunk_is_not_an_executable_gather_option():
    controller, board, context = setup()
    payload = controller._feasible_skill_payloads(
        board, query_text="gather nearby wood", context=context,
    )
    assert "gather_nearby_wood" not in {item["skill_id"] for item in payload}


def test_trunk_search_rejects_leaves_stale_and_bootstrap_tracks():
    _, board, _ = setup()
    frame = board.latest()
    assert frame is not None
    track = Track(track_id="trunk", label="oak_log", confidence=.95,
                  region=ScreenRegion(x=.4, y=.3, width=.2, height=.4),
                  first_seen_ns=frame.captured_ns, last_seen_ns=frame.captured_ns,
                  attributes={"source": "operator"})
    for changes, expected in (({}, True), ({"label": "oak_leaves"}, False),
                              ({"last_seen_ns": frame.captured_ns - 3_000_000_000}, False),
                              ({"attributes": {"source": "bootstrap:guess"}}, False)):
        frame = frame.model_copy(update={
            "frame_id": frame.frame_id + 1, "captured_ns": frame.captured_ns + 1,
        })
        board.publish(frame.model_copy(update={"tracks": (track.model_copy(update=changes),)}))
        assert visible_oak_trunk(board) is expected


def test_starved_wood_search_has_fast_nonattacking_prerequisite():
    controller, board, context = setup()
    controller.skills.stats.clear()
    message = context.operator_messages[0].model_copy(update={
        "text": "gather nearby wood; do not attack",
    })
    run = context.recent_skill_runs[0].model_copy(update={
        "skill_id": "gather_nearby_wood", "failure_code": SkillFailureCode.CONTROLLER_STARVATION,
    })
    context = replace(context, operator_messages=(message,), recent_skill_runs=(run,))
    decision = controller.decide(board, context)
    assert decision.skill_id == "explore_forward"
    assert decision.skill_parameters["allow_attack"] is False
    assert controller.model.calls == 0


def test_plan_continuation_preserves_explicit_action_prohibitions():
    runtime = _runtime_with_completed_decision(CognitionDecision(
        chosen_goal_id="operator:look",
        skill_parameters={"allow_movement": False, "allow_attack": False},
    ))
    runtime.skills = build_bootstrap_skill_library()
    runtime.executor = SkillExecutor(BootstrapMotorPolicy())
    runtime._last_decision = runtime._pending_decision.result()
    runtime._plan_steps = ("explore_forward",)
    runtime._plan_index = 0
    runtime._plan_goal_id = "operator:look"
    assert runtime._start_current_plan_skill()
    assert runtime.executor.parameters["allow_movement"] is False
    assert runtime.executor.parameters["allow_attack"] is False


def test_model_one_option_no_jump_does_not_disable_next_climbing_step(tmp_path):
    from minecraft_ai.storage import StateDatabase
    from minecraft_ai.social import OperatorMessage, OperatorMessageStatus

    with StateDatabase(tmp_path / "state.sqlite3") as database:
        database.save_operator_message(OperatorMessage(
            message_id="climb", created_ns=1, text="Climb the ledges; do not attack.",
            status=OperatorMessageStatus.ACKNOWLEDGED,
        ))
        runtime = _runtime_with_completed_decision(CognitionDecision(
            chosen_goal_id="operator:climb", skill_id="explore_forward",
            skill_parameters={"allow_jump": False, "allow_attack": False},
        ), database=database)
        runtime.skills = build_bootstrap_skill_library()
        runtime.executor = SkillExecutor(BootstrapMotorPolicy())
        runtime._last_decision = runtime._pending_decision.result()
        runtime._plan_steps = ("traverse_visible_obstacle",)
        runtime._plan_index = 0
        runtime._plan_goal_id = "operator:climb"
        assert runtime._start_current_plan_skill()
        assert runtime.executor.policy_parameters["allow_jump"] is True
        assert runtime.executor.policy_parameters["allow_attack"] is False


def test_declared_backtracking_precedes_slow_clearance_inspection():
    from test_headroom_recovery import _stall_result

    runtime = _runtime_with_completed_decision(CognitionDecision())
    runtime.skills = build_bootstrap_skill_library()
    _, runtime.blackboard, _ = setup()
    result = replace(_stall_result(), recovery_skills=("backtrack_from_obstacle",))
    assert not runtime._route_headroom_terminal(result)
    assert getattr(runtime, "_headroom_recovery", None) is None


def test_literal_backtracking_uses_existing_zero_model_latency_path():
    from minecraft_ai.social import OperatorMessageStatus

    controller, board, context = setup()
    message = context.operator_messages[0].model_copy(update={
        "text": "backtrack_from_obstacle", "status": OperatorMessageStatus.QUEUED,
    })
    decision = controller.decide(board, replace(context, operator_messages=(message,)))
    assert decision.skill_id == "backtrack_from_obstacle"
    assert controller.model.calls == 0
