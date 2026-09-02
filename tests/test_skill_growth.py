from __future__ import annotations

import tempfile
from pathlib import Path

from minecraft_ai.skill_editor import SkillEditor
from minecraft_ai.skills import (
    SkillCondition,
    SkillLibrary,
    SkillOutcome,
    SkillRun,
    SkillStage,
)
from minecraft_ai.storage import StateDatabase


def test_skill_editor_creation_and_editing() -> None:
    library = SkillLibrary()
    editor = SkillEditor(library)

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


def test_skill_auto_promotion() -> None:
    library = SkillLibrary()
    editor = SkillEditor(library)

    editor.create_skill(
        skill_id="test_skill",
        name="Test Skill",
        stage=SkillStage.CANDIDATE,
    )

    # Record successful runs to trigger promotion
    for i in range(5):
        run = SkillRun(
            run_id=f"r_{i}",
            skill_id="test_skill",
            started_ns=100,
            ended_ns=200,
            outcome=SkillOutcome.SUCCEEDED,
        )
        editor.record_and_evaluate(run)

    promoted = library.get("test_skill")
    assert promoted.stage in {SkillStage.EXPERIMENTAL, SkillStage.TRUSTED, SkillStage.PREFERRED}


def test_synthesize_recovery_variant() -> None:
    library = SkillLibrary()
    editor = SkillEditor(library)

    editor.create_skill(
        skill_id="mine_coal",
        name="Mine Coal",
        policy_ref="mine",
    )

    variant = editor.synthesize_recovery_variant("mine_coal", "failure:lava_near")
    assert variant.skill_id == "mine_coal_adapted"
    assert any(cond.key == "lava_near" for cond in variant.failure_conditions)
    assert library.get("mine_coal_adapted") == variant
