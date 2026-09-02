from __future__ import annotations

import time
from dataclasses import dataclass

from .motor import MotorIntent, MotorPolicy
from .perception import PerceptionBlackboard
from .safety import MotorAction
from .skills import SkillCondition, SkillOutcome, SkillRun, SkillSpec


@dataclass(frozen=True)
class ExecutionTick:
    run: SkillRun
    action: MotorAction | None
    recovery_skills: tuple[str, ...] = ()


class SkillExecutor:
    """Evaluate semantic skill contracts against live perception every tick."""

    def __init__(self, policy: MotorPolicy) -> None:
        self.policy = policy
        self._spec: SkillSpec | None = None
        self._run: SkillRun | None = None
        self._parameters: dict[str, str | int | float | bool] = {}

    @property
    def run(self) -> SkillRun | None:
        return self._run

    def start(
        self,
        spec: SkillSpec,
        *,
        run_id: str,
        context_key: str = "default",
        parameters: dict[str, str | int | float | bool] | None = None,
        now_ns: int | None = None,
    ) -> SkillRun:
        if self._run is not None and self._run.outcome == SkillOutcome.RUNNING:
            raise RuntimeError("a skill is already running")
        started = time.monotonic_ns() if now_ns is None else now_ns
        self._spec = spec
        self._parameters = dict(parameters or {})
        self._run = SkillRun(
            run_id=run_id,
            skill_id=spec.skill_id,
            started_ns=started,
            context_key=context_key,
        )
        return self._run

    def tick(
        self,
        blackboard: PerceptionBlackboard,
        *,
        sequence: int,
        now_ns: int | None = None,
    ) -> ExecutionTick:
        if self._spec is None or self._run is None:
            raise RuntimeError("no skill is running")
        if self._run.outcome != SkillOutcome.RUNNING:
            return ExecutionTick(run=self._run, action=None)
        now = time.monotonic_ns() if now_ns is None else now_ns

        if now - self._run.started_ns > self._spec.max_duration_ms * 1_000_000:
            return self._finish(SkillOutcome.TIMED_OUT, now, "skill-timeout")
        failed = _first_matching(self._spec.failure_conditions, blackboard, now_ns=now)
        if failed is not None:
            return self._finish(
                SkillOutcome.FAILED,
                now,
                f"failure-condition:{failed.key}",
                recover=True,
            )
        if self._spec.success_conditions and _all_matching(
            self._spec.success_conditions, blackboard, now_ns=now
        ):
            return self._finish(SkillOutcome.SUCCEEDED, now, None)
        if self._spec.preconditions and not _all_matching(
            self._spec.preconditions, blackboard, now_ns=now
        ):
            return self._finish(
                SkillOutcome.FAILED,
                now,
                "precondition-lost",
                recover=True,
            )

        intent = MotorIntent(
            skill_id=self._spec.skill_id,
            mode=self._spec.policy_ref or self._spec.skill_id,
            parameters=self._parameters,
        )
        action = self.policy.act(blackboard, intent, sequence=sequence)
        return ExecutionTick(run=self._run, action=action)

    def cancel(self, *, now_ns: int | None = None) -> ExecutionTick:
        if self._run is None:
            raise RuntimeError("no skill is running")
        now = time.monotonic_ns() if now_ns is None else now_ns
        return self._finish(SkillOutcome.CANCELLED, now, "cancelled")

    def _finish(
        self,
        outcome: SkillOutcome,
        ended_ns: int,
        reason: str | None,
        *,
        recover: bool = False,
    ) -> ExecutionTick:
        if self._run is None or self._spec is None:
            raise RuntimeError("no skill is running")
        current = self._run
        self._run = current.model_copy(
            update={
                "ended_ns": ended_ns,
                "outcome": outcome,
                "failure_reason": reason,
            }
        )
        release = self.policy.reset()
        recovery = self._spec.recovery_skills if recover else ()
        return ExecutionTick(run=self._run, action=release, recovery_skills=recovery)


def _all_matching(
    conditions: tuple[SkillCondition, ...],
    blackboard: PerceptionBlackboard,
    now_ns: int | None = None,
) -> bool:
    return all(_matches(condition, blackboard, now_ns=now_ns) for condition in conditions)


def _first_matching(
    conditions: tuple[SkillCondition, ...],
    blackboard: PerceptionBlackboard,
    now_ns: int | None = None,
) -> SkillCondition | None:
    for condition in conditions:
        if _matches(condition, blackboard, now_ns=now_ns):
            return condition
    return None


def _matches(
    condition: SkillCondition,
    blackboard: PerceptionBlackboard,
    now_ns: int | None = None,
) -> bool:
    try:
        fact = blackboard.fact(
            condition.key,
            min_confidence=condition.min_confidence,
            now_ns=now_ns,
        )
    except TypeError:
        fact = blackboard.fact(condition.key, min_confidence=condition.min_confidence)
    if condition.operator == "exists":
        return fact is not None
    if fact is None:
        return False
    value = fact.value
    expected = condition.value
    if condition.operator == "truthy":
        return bool(value)
    if condition.operator == "falsy":
        return not bool(value)
    if condition.operator == "eq":
        return value == expected
    if condition.operator == "neq":
        return value != expected
    if not isinstance(value, (int, float)) or not isinstance(expected, (int, float)):
        return False
    if condition.operator == "gte":
        return float(value) >= float(expected)
    if condition.operator == "lte":
        return float(value) <= float(expected)
    if condition.operator == "gt":
        return float(value) > float(expected)
    if condition.operator == "lt":
        return float(value) < float(expected)
    return False
