from __future__ import annotations

from minecraft_ai.action_levels import ActionLevel
from minecraft_ai.skill_editor import SkillLifecycleManager, SkillPromotionEvidence
from minecraft_ai.skills import (
    SkillCondition,
    SkillLibrary,
    SkillOutcome,
    SkillRun,
    SkillStage,
)


def test_skill_editor_creation_and_editing() -> None:
    library = SkillLibrary()
    editor = SkillLifecycleManager(library)

    spec = editor.create_skill(
        skill_id="custom_harvest",
        name="Custom Harvest",
        description="Harvest crops nearby",
        policy_ref="mine",
        preconditions=[SkillCondition(key="crop.ripe", operator="truthy")],
        success_conditions=[SkillCondition(key="crop.harvested", operator="truthy")],
    )

    assert spec.skill_id == "custom_harvest"
    assert spec.stage == SkillStage.CANDIDATE
    assert library.get("custom_harvest") == spec

    edited = editor.edit_skill("custom_harvest", description="Updated description")
    assert edited.version == 2
    assert edited.description == "Updated description"


def test_skill_promotion_requires_explicit_benchmark_evidence() -> None:
    library = SkillLibrary()
    editor = SkillLifecycleManager(library)

    editor.create_skill(
        skill_id="test_skill",
        name="Test Skill",
        stage=SkillStage.CANDIDATE,
    )

    # Runtime observations alone never trigger promotion.
    for i in range(5):
        run = SkillRun(
            run_id=f"r_{i}",
            skill_id="test_skill",
            started_ns=100,
            ended_ns=200,
            outcome=SkillOutcome.SUCCEEDED,
        )
        editor.record(run)

    assert library.get("test_skill").stage == SkillStage.CANDIDATE

    promoted = editor.evaluate_and_promote(
        "test_skill",
        SkillPromotionEvidence(
            benchmark_run_id="benchmark-1",
            sample_count=20,
            successes=18,
            context_count=4,
        ),
    )
    assert promoted is not None
    assert promoted.stage == SkillStage.EXPERIMENTAL


def test_synthesize_recovery_variant() -> None:
    library = SkillLibrary()
    editor = SkillLifecycleManager(library)

    editor.create_skill(
        skill_id="mine_coal",
        name="Mine Coal",
        policy_ref="mine",
        action_level=ActionLevel.GROUNDED,
    )

    variant = editor.draft_recovery_candidate("mine_coal", "failure:lava_near")
    assert variant.skill_id == "mine_coal_recovery_candidate"
    assert variant.stage == SkillStage.CANDIDATE
    assert variant.action_level == ActionLevel.GROUNDED
    assert any(cond.key == "lava_near" for cond in variant.failure_conditions)
    assert library.get("mine_coal_recovery_candidate") == variant
