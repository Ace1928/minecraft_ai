from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..trajectory import TrajectoryReader
from .metrics import TraceMetrics, trace_metrics, validate_external_metrics
from .reporting import (
    BenchmarkReport as BenchmarkReport,
    BenchmarkTaskResult as BenchmarkTaskResult,
    CriterionResult as CriterionResult,
    EvaluationStatus as EvaluationStatus,
    _evaluate_criterion as _evaluate_criterion,
    _number as _number,
    criteria_status,
    summarize_results,
)
from .tasks import BenchmarkSuite, BenchmarkTask


class EvaluationEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str = Field(min_length=1)
    metrics: dict[str, float | int | bool | str] = Field(default_factory=dict)
    artifact_refs: tuple[str, ...] = ()

    @field_validator("metrics")
    @classmethod
    def independent_metrics(
        cls, values: dict[str, float | int | bool | str],
    ) -> dict[str, float | int | bool | str]:
        """Keep evaluator outcomes separate from trajectory-owned namespaces."""
        validate_external_metrics(values)
        return values


class BenchmarkRunner:
    def __init__(self, suite: BenchmarkSuite) -> None:
        self.suite = BenchmarkSuite.model_validate(suite.model_dump())

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
        status = criteria_status(criterion_results)
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
        return BenchmarkReport(
            benchmark_run_id=f"benchmark-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-"
            f"{uuid.uuid4().hex[:12]}",
            suite_id=self.suite.suite_id,
            suite=self.suite,
            created_ns=time.time_ns(),
            git_commit=git_commit,
            results=results,
            summary=summarize_results(self.suite, results),
        )


def load_evidence(path: Path) -> EvaluationEvidence:
    return EvaluationEvidence.model_validate(json.loads(path.read_text(encoding="utf-8")))
