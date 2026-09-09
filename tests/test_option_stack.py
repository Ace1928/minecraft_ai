from __future__ import annotations

from minecraft_ai.control.execution import SkillExecutor
from minecraft_ai.motor import MotorIntent
from minecraft_ai.perception import PerceptionBlackboard
from minecraft_ai.safety import MotorAction
from minecraft_ai.skills import SkillOutcome, SkillSpec


class _HoldPolicy:
    policy_id = "hold"

    def act(
        self,
        blackboard: PerceptionBlackboard,
        intent: MotorIntent,
        *,
        sequence: int,
    ) -> MotorAction:
        return MotorAction(sequence=sequence, keys_down=("w",))

    def reset(self) -> MotorAction:
        return MotorAction(sequence=0)


def test_nested_option_resumes_parent_after_child_succeeds() -> None:
    executor = SkillExecutor(_HoldPolicy())  # type: ignore[arg-type]
    parent = SkillSpec(
        skill_id="gather_nearby_wood",
        name="Gather",
        max_duration_ms=90_000,
        recovery_skills=("reacquire_target",),
    )
    child = SkillSpec(
        skill_id="reacquire_target",
        name="Reacquire",
        max_duration_ms=5_000,
        success_conditions=(),
    )
    parent_run = executor.start(parent, run_id="parent")
    assert parent_run.outcome is SkillOutcome.RUNNING
    child_run = executor.push_child(child, run_id="child")
    assert child_run.skill_id == "reacquire_target"
    assert executor.parent_run() is not None
    assert executor.option_depth == 2
    restored = executor.resume_parent()
    assert restored.run_id == "parent"
    assert restored.outcome is SkillOutcome.RUNNING
    assert executor.run is not None
    assert executor.run.skill_id == "gather_nearby_wood"
    assert executor.parent_run() is None
