from __future__ import annotations

from minecraft_ai.cognition import CognitionDecision
from minecraft_ai.experience_graph import ExperienceGraph
from minecraft_ai.plan_graph import (
    PlanNodeState,
    plan_graph_from_steps,
    progression_skill_for_capabilities,
    sanitize_plan_steps,
)
from minecraft_ai.skills import (
    SkillLibrary,
    SkillOutcome,
    SkillRun,
    SkillSpec,
    SkillStage,
)


def test_cognition_decision_rejects_null_plan_nodes() -> None:
    decision = CognitionDecision(
        plan_steps=("null", "None", "n/a", "gather_nearby_wood", "gather_nearby_wood"),
    )
    assert decision.plan_steps == ("gather_nearby_wood",)


def test_sanitize_plan_steps_drops_sentinels_and_duplicate_copies() -> None:
    assert sanitize_plan_steps(("null", "none", "n/a", "", "  ")) == ()
    assert sanitize_plan_steps(
        (
            "gather_nearby_wood",
            "gather_nearby_wood",
            "gather nearby wood",
            "open_inventory",
            "null",
        )
    ) == ("gather_nearby_wood", "open_inventory")


def test_plan_graph_blocks_failed_method_and_exposes_sibling() -> None:
    skills = SkillLibrary()
    skills.register(
        SkillSpec(
            skill_id="gather_nearby_wood",
            name="Gather",
            expected_effects=("block_broken",),
            recovery_skills=("reacquire_target",),
        )
    )
    skills.register(
        SkillSpec(
            skill_id="mine_visible_block",
            name="Mine",
            expected_effects=("block_broken",),
        )
    )
    skills.register(
        SkillSpec(skill_id="reacquire_target", name="Reacquire", stage=SkillStage.TRUSTED)
    )
    graph = plan_graph_from_steps(
        ("gather_nearby_wood", "gather_nearby_wood", "open_inventory"),
        goal_id="progress",
        skills=skills,
    )
    assert graph.sequential_labels() == ("gather_nearby_wood", "open_inventory")
    current = graph.current()
    assert current is not None and current.skill_id == "gather_nearby_wood"
    graph.block_current_method("gather_nearby_wood", reason="timeout", skills=skills)
    sibling = graph.current()
    assert sibling is not None
    assert sibling.skill_id in {"mine_visible_block", "reacquire_target"}
    assert sibling.state is PlanNodeState.READY


def test_hierarchical_shrinkage_uses_global_prior_not_worst_streak() -> None:
    library = SkillLibrary()
    library.register(SkillSpec(skill_id="mine_visible_block", name="Mine"))
    library.record(
        SkillRun(
            run_id="ravine",
            skill_id="mine_visible_block",
            context_key="night-ravine",
            started_ns=1,
            ended_ns=2,
            outcome=SkillOutcome.FAILED,
        )
    )
    library.record(
        SkillRun(
            run_id="ravine-2",
            skill_id="mine_visible_block",
            context_key="night-ravine",
            started_ns=3,
            ended_ns=4,
            outcome=SkillOutcome.FAILED,
        )
    )
    library.record(
        SkillRun(
            run_id="surface",
            skill_id="mine_visible_block",
            context_key="day-surface",
            started_ns=5,
            ended_ns=6,
            outcome=SkillOutcome.SUCCEEDED,
        )
    )
    combined = library.combined_stats("mine_visible_block")
    assert combined is not None
    assert combined.consecutive_failures == 0
    assert library.contextual_failure_streak("mine_visible_block", "night-ravine") == 2
    surface = library.hierarchical_success_probability(
        "mine_visible_block", "day-surface"
    )
    ravine = library.hierarchical_success_probability(
        "mine_visible_block", "night-ravine"
    )
    assert surface > ravine
    assert 0.0 < surface < 1.0


def test_experience_graph_does_not_invent_mechanics() -> None:
    skills = SkillLibrary()
    spec = SkillSpec(skill_id="smelt", name="Smelt", expected_effects=("ingot",))
    skills.register(spec)
    graph = ExperienceGraph()
    graph.observe(
        SkillRun(
            run_id="1",
            skill_id="smelt",
            started_ns=1,
            ended_ns=2,
            outcome=SkillOutcome.SUCCEEDED,
        ),
        spec,
    )
    belief = graph.beliefs[("smelt", "default", "ingot")]
    assert belief.successes == 1
    assert graph.success_probability("smelt", "default", skills, effect="ingot") > 0.5


def test_progression_skill_follows_inventory_capabilities() -> None:
    available = {
        "gather_nearby_wood",
        "craft_wood_planks",
        "craft_crafting_table",
        "mine_visible_block",
        "explore_forward",
    }
    assert (
        progression_skill_for_capabilities({}, available_skill_ids=available)
        == "gather_nearby_wood"
    )
    assert (
        progression_skill_for_capabilities(
            {"oak_log": 4}, available_skill_ids=available
        )
        == "craft_wood_planks"
    )
    assert (
        progression_skill_for_capabilities(
            {},
            available_skill_ids=available,
            recently_failed_skill_ids={"gather_nearby_wood"},
        )
        is None
    )
