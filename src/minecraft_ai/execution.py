from __future__ import annotations

import time
from dataclasses import dataclass

from .motor import MotorIntent, MotorPolicy
from .perception import PerceptionBlackboard
from .safety import MotorAction
from .skills import SkillActionPermissions, SkillCondition, SkillOutcome, SkillRun, SkillSpec


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
        self._initiated = False

    @property
    def run(self) -> SkillRun | None:
        return self._run

    @property
    def instruction(self) -> str | None:
        if self._spec is None or self._run is None:
            return None
        return _skill_instruction(self._spec, self._parameters)

    @property
    def parameters(self) -> dict[str, str | int | float | bool]:
        """Return the active option bindings without exposing mutable executor state."""
        return dict(self._parameters)

    @property
    def policy_parameters(self) -> dict[str, str | int | float | bool]:
        """Return option bindings intersected with the skill's action contract."""
        if self._spec is None or self._run is None:
            return {}
        return _policy_parameters(self._spec.action_permissions, self._parameters)

    def close(self) -> None:
        close = getattr(self.policy, "close", None)
        if callable(close):
            close()

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
        self._initiated = False
        self._run = SkillRun(
            run_id=run_id,
            skill_id=spec.skill_id,
            started_ns=started,
            context_key=context_key,
            parameters=self._parameters,
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
        if self._spec.success_conditions and conditions_satisfied(
            self._spec.success_conditions, blackboard, now_ns=now
        ):
            return self._finish(SkillOutcome.SUCCEEDED, now, None)
        if not self._initiated:
            if not initiation_satisfied(self._spec, blackboard, now_ns=now):
                return self._finish(
                    SkillOutcome.FAILED,
                    now,
                    "initiation-precondition-unsatisfied",
                    recover=True,
                )
            self._initiated = True
        if self._spec.invariants and not conditions_satisfied(
            self._spec.invariants, blackboard, now_ns=now
        ):
            return self._finish(
                SkillOutcome.FAILED,
                now,
                "invariant-lost",
                recover=True,
            )

        intent = MotorIntent(
            skill_id=self._spec.skill_id,
            mode=self._spec.policy_ref or self._spec.skill_id,
            instruction=_policy_instruction(self._spec),
            target_label=_target_label(self._parameters),
            parameters=self.policy_parameters,
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


def _skill_instruction(
    spec: SkillSpec,
    parameters: dict[str, str | int | float | bool],
) -> str:
    """Render the semantic option contract for a goal-conditioned motor policy."""
    instruction = spec.description.strip() or spec.name.strip()
    if not parameters:
        return instruction
    rendered = ", ".join(f"{key}={parameters[key]}" for key in sorted(parameters))
    return f"{instruction}. Parameters: {rendered}"


def _policy_instruction(spec: SkillSpec) -> str:
    """Return the concise command used to condition a learned motor policy.

    Planner-facing descriptions deliberately retain the complete option contract.
    Visuomotor policies are conditioned with the short command distribution used
    by their published training/evaluation interface instead of receiving prose
    intended for an LLM or an operator.
    """
    return spec.policy_instruction or spec.description.strip() or spec.name.strip()


def _target_label(parameters: dict[str, str | int | float | bool]) -> str | None:
    target = parameters.get("target")
    if not isinstance(target, str) or not target.strip():
        return None
    return target.strip()


def _policy_parameters(
    permissions: SkillActionPermissions,
    parameters: dict[str, str | int | float | bool],
) -> dict[str, str | int | float | bool]:
    """Intersect planner/operator bindings with an option's learned-action envelope."""
    merged = dict(parameters)
    for name in ("allow_attack", "allow_use", "allow_jump"):
        skill_allows = bool(getattr(permissions, name))
        runtime_allows = parameters.get(name) is not False
        merged[name] = skill_allows and runtime_allows
    return merged


def conditions_satisfied(
    conditions: tuple[SkillCondition, ...],
    blackboard: PerceptionBlackboard,
    now_ns: int | None = None,
) -> bool:
    """Evaluate a complete semantic option condition set against fresh observations."""
    return all(_matches(condition, blackboard, now_ns=now_ns) for condition in conditions)


def initiation_satisfied(
    spec: SkillSpec,
    blackboard: PerceptionBlackboard,
    now_ns: int | None = None,
) -> bool:
    """Evaluate OR-of-AND initiation groups for a learned option contract."""
    groups = tuple(
        group
        for group in (spec.preconditions, *spec.initiation_alternatives)
        if group
    )
    if not groups:
        return True
    return any(conditions_satisfied(group, blackboard, now_ns=now_ns) for group in groups)


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
