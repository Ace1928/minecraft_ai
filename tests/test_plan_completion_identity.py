from __future__ import annotations

from pathlib import Path

import pytest

from minecraft_ai.builtin_skills import build_bootstrap_skill_library
from minecraft_ai.cognition import CognitionDecision
from minecraft_ai.plan_graph import PlanNodeState, plan_graph_from_steps
from minecraft_ai.skills import SkillOutcome, SkillRun
from minecraft_ai.storage import StateDatabase
from test_agent_core import _runtime_for_learning


def _runtime(database: StateDatabase, steps: tuple[str, ...]):
    runtime = _runtime_for_learning(database)
    runtime.skills = build_bootstrap_skill_library()
    decision = CognitionDecision(chosen_goal_id="test:progress", plan_steps=steps)
    runtime._adopt_plan_if_revised(decision)
    runtime._last_decision = decision
    return runtime


def _run(skill_id: str, run_id: str = "completed") -> SkillRun:
    return SkillRun(
        run_id=run_id, skill_id=skill_id, context_key="test:progress",
        started_ns=1, ended_ns=2, outcome=SkillOutcome.SUCCEEDED,
    )


def test_unrelated_world_successes_cannot_diverge_or_rewind_bound_plan(tmp_path: Path) -> None:
    with StateDatabase(tmp_path / "state.sqlite3") as database:
        runtime = _runtime(database, (
            "gather_nearby_wood", "traverse_level_ground", "gather_nearby_wood",
        ))
        graph = runtime._plan_graph
        assert graph is not None
        graph.mark_running("gather_nearby_wood")
        original = graph.model_dump()

        for index in range(2):
            runtime._record_terminal_run(_run("explore_forward", f"wander-{index}"))
            assert runtime._plan_index == graph.cursor == 0
            assert graph.model_dump() == original
            assert runtime._plan_step_completed_ns == 0

        completed = _run("gather_nearby_wood", "gathered")
        runtime._record_terminal_run(completed)
        assert runtime._plan_index == graph.cursor == 1
        assert graph.nodes[graph.order[0]].state is PlanNodeState.SUCCEEDED
        completed_ns = runtime._plan_step_completed_ns
        assert completed_ns > 0
        runtime._record_terminal_run(completed)
        assert runtime._plan_index == graph.cursor == 1
        assert runtime._plan_step_completed_ns == completed_ns
        assert runtime.metrics.skill_successes == 3


@pytest.mark.parametrize(("step", "skill_id"), (
    ("Find a nearby oak trunk", "explore_forward"),
    ("check the inventory", "open_inventory"),
    ("close the inventory after inspection", "close_open_inventory"),
    ("click the Play button", "activate_visible_gui_control"),
))
def test_unbound_legacy_prose_advances_graph_and_projection_together(
    tmp_path: Path, step: str, skill_id: str,
) -> None:
    with StateDatabase(tmp_path / "state.sqlite3") as database:
        runtime = _runtime(database, (step, "gather_nearby_wood"))
        graph = runtime._plan_graph
        assert graph is not None and graph.current() is not None
        assert graph.current().skill_id is None

        runtime._record_terminal_run(_run(skill_id))

        assert runtime._plan_index == graph.cursor == 1
        assert graph.nodes[graph.order[0]].state is PlanNodeState.SUCCEEDED


@pytest.mark.parametrize("blocker", (
    "different-goal", "failed", "timed-out", "cancelled", "plan-neutral", "no-advance",
))
@pytest.mark.parametrize("step", ("gather_nearby_wood", "Find a nearby oak trunk"))
def test_plan_completion_preserves_existing_terminal_gates(
    tmp_path: Path, blocker: str, step: str,
) -> None:
    with StateDatabase(tmp_path / "state.sqlite3") as database:
        runtime = _runtime(database, (step, "traverse_level_ground"))
        graph = runtime._plan_graph
        assert graph is not None
        original = graph.model_dump()
        run = _run("gather_nearby_wood")
        if blocker == "different-goal":
            runtime._last_decision = CognitionDecision(chosen_goal_id="operator:other")
        elif blocker == "plan-neutral":
            runtime._plan_neutral_recovery_runs = {run.run_id}
        else:
            run = run.model_copy(update={"outcome": {
                "failed": SkillOutcome.FAILED,
                "timed-out": SkillOutcome.TIMED_OUT,
                "cancelled": SkillOutcome.CANCELLED,
            }.get(blocker, SkillOutcome.SUCCEEDED)})

        runtime._record_terminal_run(run, advance_plan=blocker != "no-advance")

        assert runtime._plan_index == graph.cursor == 0
        assert graph.model_dump() == original
        assert runtime._plan_step_completed_ns == 0


def test_gui_success_cannot_consume_unbound_world_objective(tmp_path: Path) -> None:
    with StateDatabase(tmp_path / "state.sqlite3") as database:
        runtime = _runtime(database, ("Find a nearby oak trunk",))
        graph = runtime._plan_graph
        assert graph is not None
        runtime._record_terminal_run(_run("open_inventory"))
        assert runtime._plan_index == graph.cursor == 0
        assert graph.nodes[graph.order[0]].state is PlanNodeState.READY


def test_graph_free_legacy_world_success_remains_permissive(tmp_path: Path) -> None:
    with StateDatabase(tmp_path / "state.sqlite3") as database:
        runtime = _runtime(database, ("gather_nearby_wood",))
        runtime._plan_graph = None
        runtime._record_terminal_run(_run("explore_forward"))
        assert runtime._plan_index == 1


def test_unbound_graph_completion_requires_explicit_caller_allowance() -> None:
    graph = plan_graph_from_steps(
        ("Find a nearby oak trunk",), goal_id="test:progress",
        skills=build_bootstrap_skill_library(),
    )
    original = graph.model_dump()
    assert not graph.mark_succeeded("explore_forward")
    assert graph.model_dump() == original


def test_unbound_allowance_does_not_override_exact_skill_identity() -> None:
    graph = plan_graph_from_steps(
        ("gather_nearby_wood",), goal_id="test:progress",
        skills=build_bootstrap_skill_library(),
    )
    original = graph.model_dump()
    assert not graph.mark_succeeded("explore_forward", allow_unbound=True)
    assert graph.model_dump() == original
