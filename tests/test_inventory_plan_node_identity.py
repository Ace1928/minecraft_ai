from __future__ import annotations

from pathlib import Path

import pytest

from minecraft_ai.cognition import CognitionDecision
from minecraft_ai.runtime_support.helpers import _plan_step_requests_inventory_transition
from minecraft_ai.skills import SkillOutcome, SkillRun
from minecraft_ai.storage import StateDatabase
from test_agent_core import _runtime_for_learning


@pytest.mark.parametrize("step", (
    "close_open_inventory", " CLOSE_OPEN_INVENTORY\n", "Close_Open_Inventory",
    "close inventory", "close the inventory after inspection", "exit inventory",
))
def test_canonical_close_node_and_existing_prose_match(step: str) -> None:
    assert _plan_step_requests_inventory_transition("close_open_inventory", step)


@pytest.mark.parametrize("step", (
    "", "open_inventory", "craft_wood_planks", "activate_visible_gui_control",
    "do not close_open_inventory", "do not close inventory",
    "close_open_inventory later", "close_open_inventory_extra", "close-open-inventory",
))
def test_close_node_match_does_not_invent_aliases_or_ignore_negation(step: str) -> None:
    assert not _plan_step_requests_inventory_transition("close_open_inventory", step)


@pytest.mark.parametrize("skill_id", ("open_inventory", "activate_visible_gui_control"))
def test_canonical_close_node_does_not_match_other_gui_skills(skill_id: str) -> None:
    assert not _plan_step_requests_inventory_transition(skill_id, "close_open_inventory")


def _close_run(*, outcome: SkillOutcome = SkillOutcome.SUCCEEDED) -> SkillRun:
    return SkillRun(
        run_id="observed-close", skill_id="close_open_inventory",
        context_key="scene-recovery", started_ns=3, ended_ns=4, outcome=outcome,
    )


def test_recorded_open_then_close_progresses_exact_nodes_once(tmp_path: Path) -> None:
    with StateDatabase(tmp_path / "state.sqlite3") as database:
        runtime = _runtime_for_learning(database)
        runtime._plan_steps = ("open_inventory", "close_open_inventory", "gather_nearby_wood")
        runtime._plan_goal_id = "operator:inventory-check"
        runtime._last_decision = CognitionDecision(chosen_goal_id=runtime._plan_goal_id)
        opened = SkillRun(
            run_id="observed-open", skill_id="open_inventory",
            context_key=runtime._plan_goal_id, started_ns=1, ended_ns=2,
            outcome=SkillOutcome.SUCCEEDED,
        )

        runtime._record_terminal_run(opened)
        assert runtime._plan_index == 1
        closed = _close_run()
        runtime._record_terminal_run(closed)
        completed_ns = runtime._plan_step_completed_ns

        assert runtime._plan_index == 2
        # Existing observed-transition progress does not relabel who executed it.
        assert runtime._recent_skill_runs[0] == closed
        assert closed.context_key == "scene-recovery"
        runtime._record_terminal_run(closed)
        assert runtime._plan_index == 2
        assert runtime._plan_step_completed_ns == completed_ns
        assert runtime.metrics.skill_successes == 2
        assert runtime.skills.stats[(closed.skill_id, closed.context_key)].successes == 1


@pytest.mark.parametrize("blocker", (
    "different-goal", "failed", "timed-out", "cancelled", "plan-neutral", "no-advance",
))
def test_exact_close_node_preserves_existing_progress_gates(
    tmp_path: Path, blocker: str,
) -> None:
    with StateDatabase(tmp_path / "state.sqlite3") as database:
        runtime = _runtime_for_learning(database)
        runtime._plan_steps = ("close_open_inventory",)
        runtime._plan_goal_id = "operator:inventory-check"
        runtime._last_decision = CognitionDecision(
            chosen_goal_id="other-goal" if blocker == "different-goal" else runtime._plan_goal_id,
        )
        outcome = {
            "failed": SkillOutcome.FAILED,
            "timed-out": SkillOutcome.TIMED_OUT,
            "cancelled": SkillOutcome.CANCELLED,
        }.get(blocker, SkillOutcome.SUCCEEDED)
        closed = _close_run(outcome=outcome)
        if blocker == "plan-neutral":
            runtime._plan_neutral_recovery_runs = {closed.run_id}

        runtime._record_terminal_run(closed, advance_plan=blocker != "no-advance")

        assert runtime._plan_index == 0
        assert runtime._plan_step_completed_ns == 0
        assert runtime._recent_skill_runs[0].context_key == "scene-recovery"
