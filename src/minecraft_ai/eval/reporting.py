from __future__ import annotations

import uuid
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr,
    model_validator,
)

from .tasks import BenchmarkSuite, MetricCriterion, MetricOperator

MetricValue = StrictFloat | StrictInt | StrictBool | StrictStr
ReportSummary = dict[str, int | float | bool | str | None]


class EvaluationStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    UNSCORED = "unscored"
    ERROR = "error"


class CriterionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    criterion: MetricCriterion
    observed: MetricValue | None = None
    passed: StrictBool | None = None


class BenchmarkTaskResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    task_id: str = Field(min_length=1, pattern=r"\S")
    repetition: int = Field(default=0, ge=0, strict=True)
    status: EvaluationStatus
    trajectory_id: str = Field(min_length=1, pattern=r"\S")
    criteria: tuple[CriterionResult, ...]
    metrics: dict[str, MetricValue]
    evidence_sources: tuple[str, ...]
    error: str | None = None


def _evaluate_criterion(
    criterion: MetricCriterion,
    metrics: dict[str, float | int | bool | str],
) -> CriterionResult:
    observed = metrics.get(criterion.metric)
    if observed is None:
        return CriterionResult(criterion=criterion)
    expected = criterion.value
    passed: bool
    if criterion.operator == MetricOperator.TRUTHY:
        passed = bool(observed)
    elif criterion.operator == MetricOperator.EQ:
        passed = observed == expected
    elif criterion.operator == MetricOperator.GTE:
        passed = _number(observed) >= _number(expected)
    elif criterion.operator == MetricOperator.LTE:
        passed = _number(observed) <= _number(expected)
    else:  # pragma: no cover - enum exhaustiveness guard
        raise ValueError(f"unsupported metric operator: {criterion.operator}")
    return CriterionResult(criterion=criterion, observed=observed, passed=passed)


def _number(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"metric comparison requires a number, got {value!r}")


def criteria_status(criteria: tuple[CriterionResult, ...]) -> EvaluationStatus:
    """Score criteria once, preserving the explicit missing-outcome state."""
    if not criteria:
        raise ValueError("benchmark task must define at least one criterion")
    if any(result.passed is None for result in criteria):
        return EvaluationStatus.UNSCORED
    return (
        EvaluationStatus.PASSED
        if all(result.passed for result in criteria)
        else EvaluationStatus.FAILED
    )


def summarize_results(
    suite: BenchmarkSuite, results: tuple[BenchmarkTaskResult, ...],
) -> ReportSummary:
    """Validate result/contract consistency and derive canonical admission counts."""
    tasks = {task.task_id: task for task in suite.tasks}
    repetitions: set[tuple[str, int]] = set()
    trajectories: set[tuple[str, str]] = set()
    counts = {status: 0 for status in EvaluationStatus}
    scored_repetitions = {task_id: 0 for task_id in tasks}
    protected_failures = 0
    for result in results:
        task = tasks.get(result.task_id)
        if task is None:
            raise ValueError(f"result references unknown suite task: {result.task_id}")
        repetition = (result.task_id, result.repetition)
        trajectory = (result.task_id, result.trajectory_id)
        if repetition in repetitions:
            raise ValueError(f"duplicate task repetition: {repetition}")
        if trajectory in trajectories:
            raise ValueError(f"reused trajectory cannot count as another repetition: {trajectory}")
        repetitions.add(repetition)
        trajectories.add(trajectory)
        if result.status == EvaluationStatus.ERROR:
            if not result.error or not result.error.strip():
                raise ValueError("error result must include an error reason")
            if result.criteria or result.metrics or result.evidence_sources:
                raise ValueError("error result cannot also claim scored evidence")
        else:
            if result.error is not None:
                raise ValueError("non-error result includes an error reason")
            if not result.evidence_sources or any(
                not source.strip() for source in result.evidence_sources
            ):
                raise ValueError("scored/unscored result must identify evidence sources")
            expected = tuple(
                _evaluate_criterion(criterion, result.metrics) for criterion in task.criteria
            )
            if result.criteria != expected:
                raise ValueError(f"criteria disagree with suite or metrics: {result.task_id}")
            if result.status != criteria_status(expected):
                raise ValueError(f"status disagrees with evaluated criteria: {result.task_id}")
        counts[result.status] += 1
        if result.status in {EvaluationStatus.PASSED, EvaluationStatus.FAILED}:
            scored_repetitions[result.task_id] += 1
        protected_failures += result.status == EvaluationStatus.FAILED and task.protected

    passed = counts[EvaluationStatus.PASSED]
    failed = counts[EvaluationStatus.FAILED]
    errors = counts[EvaluationStatus.ERROR]
    scored = passed + failed
    tasks_meeting_minimum = sum(
        scored_repetitions[task.task_id] >= task.minimum_repetitions for task in suite.tasks
    )
    coverage_complete = bool(suite.tasks) and tasks_meeting_minimum == len(suite.tasks)
    return {
        "tasks": len(results),
        "suite_tasks": len(suite.tasks),
        "scored": scored,
        "unique_tasks_scored": sum(count > 0 for count in scored_repetitions.values()),
        "tasks_meeting_minimum_repetitions": tasks_meeting_minimum,
        "minimum_scored_results_required": sum(task.minimum_repetitions for task in suite.tasks),
        "passed": passed,
        "failed": failed,
        "unscored": counts[EvaluationStatus.UNSCORED],
        "errors": errors,
        "success_rate": passed / scored if scored else None,
        "protected_failures": protected_failures,
        "coverage_complete": coverage_complete,
        "promotion_eligible": bool(coverage_complete and protected_failures == 0 and errors == 0),
    }


class BenchmarkReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal[2] = 2
    benchmark_run_id: str = Field(min_length=1, pattern=r"\S")
    suite_id: str = Field(min_length=1, pattern=r"\S")
    suite: BenchmarkSuite
    created_ns: int = Field(gt=0, strict=True)
    git_commit: str | None = None
    results: tuple[BenchmarkTaskResult, ...]
    summary: ReportSummary

    @model_validator(mode="after")
    def consistent_evidence(self) -> BenchmarkReport:
        """Treat serialized summaries as cross-checks, never admission authority."""
        if self.suite_id != self.suite.suite_id:
            raise ValueError("report suite_id disagrees with embedded suite contract")
        expected = summarize_results(self.suite, self.results)
        if self.summary.keys() != expected.keys() or any(
            type(self.summary[key]) is not type(value) or self.summary[key] != value
            for key, value in expected.items()
        ):
            raise ValueError("report summary disagrees with canonical result admission")
        return self

    def write(self, path: Path) -> Path:
        """Atomically write a contract-validated benchmark report."""
        # Frozen models can still contain mutated metric/summary dictionaries.
        report = type(self).model_validate(self.model_dump())
        path.parent.mkdir(parents=True, exist_ok=True)
        staged = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        staged.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        staged.replace(path)
        return path
