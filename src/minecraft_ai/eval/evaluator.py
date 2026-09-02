from __future__ import annotations

import json
import time
import uuid
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ..trajectory import TrajectoryReader
from .metrics import TraceMetrics, trace_metrics
from .tasks import BenchmarkSuite, BenchmarkTask, MetricCriterion, MetricOperator


class EvaluationStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    UNSCORED = "unscored"
    ERROR = "error"


class EvaluationEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str
    metrics: dict[str, float | int | bool | str] = Field(default_factory=dict)
    artifact_refs: tuple[str, ...] = ()


class CriterionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    criterion: MetricCriterion
    observed: float | int | bool | str | None = None
    passed: bool | None = None


class BenchmarkTaskResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    repetition: int = Field(default=0, ge=0)
    status: EvaluationStatus
    trajectory_id: str
    criteria: tuple[CriterionResult, ...]
    metrics: dict[str, float | int | bool | str]
    evidence_sources: tuple[str, ...]
    error: str | None = None


class BenchmarkReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    benchmark_run_id: str
    suite_id: str
    created_ns: int
    git_commit: str | None = None
    results: tuple[BenchmarkTaskResult, ...]
    summary: dict[str, int | float | bool | str | None]

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        staged = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        staged.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        staged.replace(path)
        return path


class BenchmarkRunner:
    def __init__(self, suite: BenchmarkSuite) -> None:
        self.suite = suite

    def evaluate_trajectory(
        self,
        trajectory: Path,
        *,
        task_ids: tuple[str, ...],
        evidence: EvaluationEvidence | None = None,
        repetition: int = 0,
        git_commit: str | None = None,
    ) -> BenchmarkReport:
        reader = TrajectoryReader(trajectory)
        validation = reader.validate()
        if not validation.valid:
            results = tuple(
                BenchmarkTaskResult(
                    task_id=task_id,
                    repetition=repetition,
                    status=EvaluationStatus.ERROR,
                    trajectory_id=reader.manifest.trajectory_id,
                    criteria=(),
                    metrics={},
                    evidence_sources=(),
                    error="; ".join(validation.errors),
                )
                for task_id in task_ids
            )
        else:
            metrics = trace_metrics(reader.iter_samples())
            if evidence is not None:
                metrics = metrics.merged(evidence.metrics)
            results = tuple(
                self._evaluate_task(
                    self.suite.task(task_id),
                    reader.manifest.trajectory_id,
                    metrics,
                    evidence,
                    repetition,
                )
                for task_id in task_ids
            )
        return self._report(results, git_commit=git_commit)

    def evaluate_many(
        self,
        trajectories: tuple[Path, ...],
        *,
        evidence_by_trajectory: dict[str, EvaluationEvidence] | None = None,
        git_commit: str | None = None,
    ) -> BenchmarkReport:
        results: list[BenchmarkTaskResult] = []
        repetitions: dict[str, int] = {}
        evidence_map = evidence_by_trajectory or {}
        for path in trajectories:
            reader = TrajectoryReader(path)
            task_id = reader.manifest.task_id
            if task_id is None:
                continue
            repetition = repetitions.get(task_id, 0)
            repetitions[task_id] = repetition + 1
            report = self.evaluate_trajectory(
                path,
                task_ids=(task_id,),
                evidence=evidence_map.get(reader.manifest.trajectory_id),
                repetition=repetition,
                git_commit=git_commit,
            )
            results.extend(report.results)
        return self._report(tuple(results), git_commit=git_commit)

    def _evaluate_task(
        self,
        task: BenchmarkTask,
        trajectory_id: str,
        metrics: TraceMetrics,
        evidence: EvaluationEvidence | None,
        repetition: int,
    ) -> BenchmarkTaskResult:
        criterion_results = tuple(
            _evaluate_criterion(criterion, metrics.values) for criterion in task.criteria
        )
        missing = any(result.passed is None for result in criterion_results)
        status = (
            EvaluationStatus.UNSCORED
            if missing
            else (
                EvaluationStatus.PASSED
                if all(result.passed for result in criterion_results)
                else EvaluationStatus.FAILED
            )
        )
        sources = ["trajectory:supervisor-accepted-actions"]
        if evidence is not None:
            sources.append(evidence.source)
        return BenchmarkTaskResult(
            task_id=task.task_id,
            repetition=repetition,
            status=status,
            trajectory_id=trajectory_id,
            criteria=criterion_results,
            metrics=metrics.values,
            evidence_sources=tuple(sources),
        )

    def _report(
        self,
        results: tuple[BenchmarkTaskResult, ...],
        *,
        git_commit: str | None,
    ) -> BenchmarkReport:
        passed = sum(result.status == EvaluationStatus.PASSED for result in results)
        failed = sum(result.status == EvaluationStatus.FAILED for result in results)
        unscored = sum(result.status == EvaluationStatus.UNSCORED for result in results)
        errors = sum(result.status == EvaluationStatus.ERROR for result in results)
        scored = passed + failed
        scored_repetitions = {
            task.task_id: sum(
                result.task_id == task.task_id
                and result.status in {EvaluationStatus.PASSED, EvaluationStatus.FAILED}
                for result in results
            )
            for task in self.suite.tasks
        }
        tasks_meeting_minimum = sum(
            scored_repetitions[task.task_id] >= task.minimum_repetitions
            for task in self.suite.tasks
        )
        coverage_complete = tasks_meeting_minimum == len(self.suite.tasks)
        protected_failures = sum(
            result.status == EvaluationStatus.FAILED
            and self.suite.task(result.task_id).protected
            for result in results
        )
        return BenchmarkReport(
            benchmark_run_id=f"benchmark-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-"
            f"{uuid.uuid4().hex[:12]}",
            suite_id=self.suite.suite_id,
            created_ns=time.time_ns(),
            git_commit=git_commit,
            results=results,
            summary={
                "tasks": len(results),
                "suite_tasks": len(self.suite.tasks),
                "scored": scored,
                "unique_tasks_scored": sum(count > 0 for count in scored_repetitions.values()),
                "tasks_meeting_minimum_repetitions": tasks_meeting_minimum,
                "minimum_scored_results_required": sum(
                    task.minimum_repetitions for task in self.suite.tasks
                ),
                "passed": passed,
                "failed": failed,
                "unscored": unscored,
                "errors": errors,
                "success_rate": passed / scored if scored else None,
                "protected_failures": protected_failures,
                "coverage_complete": coverage_complete,
                "promotion_eligible": bool(
                    coverage_complete and protected_failures == 0 and errors == 0
                ),
            },
        )


def load_evidence(path: Path) -> EvaluationEvidence:
    return EvaluationEvidence.model_validate(json.loads(path.read_text(encoding="utf-8")))


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
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"metric comparison requires a number, got {value!r}")
