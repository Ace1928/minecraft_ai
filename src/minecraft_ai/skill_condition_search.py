from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .datasets import DatasetSourceType
from .eval import (
    BenchmarkRunner,
    EvaluationStatus,
    bedrock_baseline_suite,
    load_evidence,
)
from .eval.metrics import percentile, wilson_interval
from .trajectory import ActionOrigin, TrajectoryReader


# No integrated policy currently accepts a latent identifier as an inference
# condition. ``TrajectoryStep.latent_id`` is an observed output/provenance hook,
# not an input selector. Add a provider here only when its runtime adapter has a
# tested latent-conditioning input contract.
_LATENT_CONDITIONING_PROVIDERS: frozenset[str] = frozenset()


class SearchModelReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str = Field(min_length=1, max_length=128)
    model_version: str = Field(min_length=1, max_length=256)
    model_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    weights_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_commit: str | None = Field(default=None, pattern=r"^[a-f0-9]{40}$")


class ConditionTrial(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    initial_pose_seed: int = Field(ge=0)
    trajectory: Path
    evidence: Path | None = None


class SkillConditionCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    instruction: str = Field(min_length=1, max_length=1024)
    condition_scale: float = Field(ge=0.0, le=12.0)
    latent_id: str | None = Field(default=None, min_length=1, max_length=256)
    trials: tuple[ConditionTrial, ...]

    @model_validator(mode="after")
    def validate_pose_identity(self) -> SkillConditionCandidate:
        seeds = tuple(trial.initial_pose_seed for trial in self.trials)
        if not seeds:
            raise ValueError("a condition candidate requires at least one trial")
        if len(seeds) != len(set(seeds)):
            raise ValueError("initial_pose_seed must be unique within a candidate")
        return self


class SkillConditionSearchSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    skill_id: str = Field(min_length=1, max_length=256)
    task_id: str = Field(min_length=1, max_length=256)
    bedrock_version: str = Field(min_length=1, max_length=128)
    evaluator_git_commit: str = Field(pattern=r"^[a-f0-9]{40}$")
    model: SearchModelReference
    candidates: tuple[SkillConditionCandidate, ...]
    minimum_pose_seeds: int = Field(default=5, ge=2, le=1000)

    @model_validator(mode="after")
    def validate_search_space(self) -> SkillConditionSearchSpec:
        if self.model.provider != "minestudio-steve1":
            raise ValueError("skill condition search currently supports STEVE-1 only")
        if len(self.candidates) < 2:
            raise ValueError("condition search requires at least two candidates")
        identities = {
            (candidate.instruction.strip(), candidate.condition_scale, candidate.latent_id)
            for candidate in self.candidates
        }
        if len(identities) != len(self.candidates):
            raise ValueError("condition candidates must be unique")
        if any(candidate.latent_id is not None for candidate in self.candidates):
            if self.model.provider not in _LATENT_CONDITIONING_PROVIDERS:
                raise ValueError(
                    f"provider {self.model.provider!r} has no latent-conditioning input contract"
                )
        return self


class TrialArtifactResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    initial_pose_seed: int
    trajectory_ref: str
    trajectory_id: str
    trajectory_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    trajectory_shard_sha256: tuple[str, ...]
    dataset_source_type: DatasetSourceType
    evidence_ref: str | None = None
    evidence_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    status: EvaluationStatus
    completion_time_s: float | None = None
    metrics: dict[str, float | int | bool | str]
    evidence_sources: tuple[str, ...]


class CandidateAggregate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    instruction: str
    condition_scale: float
    latent_id: str | None = None
    initial_pose_seeds: tuple[int, ...]
    trials: tuple[TrialArtifactResult, ...]
    scored_trials: int
    passed: int
    failed: int
    unscored: int
    errors: int
    success_rate: float | None
    success_wilson_95: tuple[float, float] | None
    successful_completion_time_s_p50: float | None
    successful_completion_time_s_p95: float | None


class ConditionComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_pose_seeds: int
    paired_initial_pose_seeds: tuple[int, ...]
    identical_pose_coverage: bool
    all_trials_scored: bool
    multi_pose_evidence: bool
    comparison_ready: bool
    ranked_candidate_ids: tuple[str, ...]
    leading_candidate_id: str | None


class SkillConditionSearchManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    artifact_type: str = "steve-skill-condition-search"
    suite_id: str
    suite_version: int
    skill_id: str
    task_id: str
    bedrock_version: str
    evaluator_git_commit: str
    model: SearchModelReference
    latent_conditioning_supported: bool
    candidates: tuple[CandidateAggregate, ...]
    comparison: ConditionComparison
    promotion_gate: dict[str, bool | int | str]


class ContentAddressedManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    manifest: SkillConditionSearchManifest


def load_search_spec(path: Path) -> SkillConditionSearchSpec:
    return SkillConditionSearchSpec.model_validate_json(path.read_text(encoding="utf-8"))


def evaluate_skill_conditions(spec: SkillConditionSearchSpec) -> SkillConditionSearchManifest:
    suite = bedrock_baseline_suite()
    task = suite.task(spec.task_id)
    runner = BenchmarkRunner(suite)
    aggregates = tuple(
        _evaluate_candidate(spec, candidate, runner=runner) for candidate in spec.candidates
    )
    pose_sets = tuple(set(candidate.initial_pose_seeds) for candidate in aggregates)
    paired = set.intersection(*pose_sets)
    required_pose_seeds = max(spec.minimum_pose_seeds, task.minimum_repetitions)
    identical_pose_coverage = all(seeds == pose_sets[0] for seeds in pose_sets[1:])
    all_trials_scored = all(
        candidate.unscored == 0 and candidate.errors == 0 for candidate in aggregates
    )
    multi_pose_evidence = len(paired) >= required_pose_seeds
    comparison_ready = identical_pose_coverage and all_trials_scored and multi_pose_evidence
    ranked = tuple(
        candidate.candidate_id for candidate in sorted(aggregates, key=_candidate_rank_key)
    )
    comparison = ConditionComparison(
        minimum_pose_seeds=required_pose_seeds,
        paired_initial_pose_seeds=tuple(sorted(paired)),
        identical_pose_coverage=identical_pose_coverage,
        all_trials_scored=all_trials_scored,
        multi_pose_evidence=multi_pose_evidence,
        comparison_ready=comparison_ready,
        ranked_candidate_ids=ranked,
        leading_candidate_id=ranked[0] if comparison_ready else None,
    )
    return SkillConditionSearchManifest(
        suite_id=suite.suite_id,
        suite_version=suite.version,
        skill_id=spec.skill_id,
        task_id=task.task_id,
        bedrock_version=spec.bedrock_version,
        evaluator_git_commit=spec.evaluator_git_commit,
        model=spec.model,
        latent_conditioning_supported=(spec.model.provider in _LATENT_CONDITIONING_PROVIDERS),
        candidates=aggregates,
        comparison=comparison,
        promotion_gate={
            "minimum_distinct_initial_pose_seeds": required_pose_seeds,
            "multi_pose_empirical_evidence_required": True,
            "comparison_evidence_ready": comparison_ready,
            "automatic_promotion": False,
            "promotion_performed": False,
            "required_next_action": "manual-review-and-separate-policy-promotion",
        },
    )


def write_content_addressed_manifest(
    manifest: SkillConditionSearchManifest,
    output_dir: Path,
) -> ContentAddressedManifest:
    payload = _canonical_bytes(manifest)
    digest = hashlib.sha256(payload).hexdigest()
    destination = output_dir.expanduser().resolve() / f"{digest}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != payload:
            raise RuntimeError(f"content-address collision at {destination}")
    else:
        staged = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        staged.write_bytes(payload)
        staged.replace(destination)
    return ContentAddressedManifest(path=destination, sha256=digest, manifest=manifest)


def content_address(manifest: SkillConditionSearchManifest) -> str:
    return hashlib.sha256(_canonical_bytes(manifest)).hexdigest()


def _evaluate_candidate(
    spec: SkillConditionSearchSpec,
    candidate: SkillConditionCandidate,
    *,
    runner: BenchmarkRunner,
) -> CandidateAggregate:
    candidate_id = _candidate_id(spec, candidate)
    trials = tuple(
        _evaluate_trial(spec, candidate, trial, runner=runner) for trial in candidate.trials
    )
    trajectory_ids = tuple(trial.trajectory_id for trial in trials)
    if len(trajectory_ids) != len(set(trajectory_ids)):
        raise ValueError("each initial pose requires a distinct trajectory artifact")
    passed = sum(trial.status == EvaluationStatus.PASSED for trial in trials)
    failed = sum(trial.status == EvaluationStatus.FAILED for trial in trials)
    unscored = sum(trial.status == EvaluationStatus.UNSCORED for trial in trials)
    errors = sum(trial.status == EvaluationStatus.ERROR for trial in trials)
    scored = passed + failed
    durations = sorted(
        trial.completion_time_s
        for trial in trials
        if trial.status == EvaluationStatus.PASSED and trial.completion_time_s is not None
    )
    return CandidateAggregate(
        candidate_id=candidate_id,
        instruction=candidate.instruction.strip(),
        condition_scale=candidate.condition_scale,
        latent_id=candidate.latent_id,
        initial_pose_seeds=tuple(sorted(trial.initial_pose_seed for trial in trials)),
        trials=trials,
        scored_trials=scored,
        passed=passed,
        failed=failed,
        unscored=unscored,
        errors=errors,
        success_rate=passed / scored if scored else None,
        success_wilson_95=wilson_interval(passed, scored) if scored else None,
        successful_completion_time_s_p50=(percentile(durations, 0.50) if durations else None),
        successful_completion_time_s_p95=(percentile(durations, 0.95) if durations else None),
    )


def _evaluate_trial(
    spec: SkillConditionSearchSpec,
    candidate: SkillConditionCandidate,
    trial: ConditionTrial,
    *,
    runner: BenchmarkRunner,
) -> TrialArtifactResult:
    reader = TrajectoryReader(trial.trajectory)
    _validate_trial_contract(spec, candidate, reader)
    evidence = None if trial.evidence is None else load_evidence(trial.evidence)
    report = runner.evaluate_trajectory(
        reader.directory,
        task_ids=(spec.task_id,),
        evidence=evidence,
        repetition=trial.initial_pose_seed,
        git_commit=spec.evaluator_git_commit,
    )
    result = report.results[0]
    duration = result.metrics.get("trace.duration_s")
    completion_time_s = (
        float(duration)
        if isinstance(duration, (int, float)) and not isinstance(duration, bool)
        else None
    )
    return TrialArtifactResult(
        initial_pose_seed=trial.initial_pose_seed,
        trajectory_ref=str(reader.directory),
        trajectory_id=reader.manifest.trajectory_id,
        trajectory_manifest_sha256=_sha256(reader.manifest_path),
        trajectory_shard_sha256=tuple(shard.sha256 for shard in reader.manifest.shards),
        dataset_source_type=reader.manifest.source.source_type,
        evidence_ref=None if trial.evidence is None else str(trial.evidence.resolve()),
        evidence_sha256=None if trial.evidence is None else _sha256(trial.evidence),
        status=result.status,
        completion_time_s=completion_time_s,
        metrics=result.metrics,
        evidence_sources=result.evidence_sources,
    )


def _validate_trial_contract(
    spec: SkillConditionSearchSpec,
    candidate: SkillConditionCandidate,
    reader: TrajectoryReader,
) -> None:
    validation = reader.validate()
    if not validation.valid:
        raise ValueError(
            f"trajectory {reader.manifest.trajectory_id} is invalid: "
            + "; ".join(validation.errors)
        )
    manifest = reader.manifest
    if (
        manifest.source.edition.casefold() != "bedrock"
        or manifest.source.source_type != DatasetSourceType.BEDROCK_AGENT
    ):
        raise ValueError(f"trajectory {manifest.trajectory_id} is not Bedrock-native")
    if not manifest.shards:
        raise ValueError(
            f"trajectory {manifest.trajectory_id} lacks content-addressed shard manifests"
        )
    if manifest.game_version != spec.bedrock_version:
        raise ValueError(
            f"trajectory {manifest.trajectory_id} Bedrock version {manifest.game_version!r} "
            f"does not match search version {spec.bedrock_version!r}"
        )
    if spec.bedrock_version not in manifest.source.game_versions:
        raise ValueError(
            f"trajectory {manifest.trajectory_id} source provenance omits Bedrock version "
            f"{spec.bedrock_version!r}"
        )
    if manifest.task_id != spec.task_id:
        raise ValueError(
            f"trajectory {manifest.trajectory_id} task {manifest.task_id!r} "
            f"does not match {spec.task_id!r}"
        )
    matched = 0
    expected_instruction = candidate.instruction.strip()
    for sample in reader.iter_samples():
        step = sample.step
        if step.action_origin != ActionOrigin.POLICY or step.skill_id != spec.skill_id:
            continue
        condition = step.condition
        if condition is None:
            raise ValueError(
                f"trajectory {manifest.trajectory_id} has unconditioned policy actions "
                f"for skill {spec.skill_id!r}"
            )
        if condition.get("instruction") != expected_instruction:
            raise ValueError(
                f"trajectory {manifest.trajectory_id} instruction does not match candidate"
            )
        scale = condition.get("condition_scale")
        if isinstance(scale, bool) or not isinstance(scale, (int, float)):
            raise ValueError(
                f"trajectory {manifest.trajectory_id} lacks an explicit condition_scale"
            )
        if float(scale) != candidate.condition_scale:
            raise ValueError(
                f"trajectory {manifest.trajectory_id} condition_scale does not match candidate"
            )
        if step.model_version != spec.model.model_version:
            raise ValueError(
                f"trajectory {manifest.trajectory_id} model version does not match search model"
            )
        policy_id = step.policy_id or ""
        if f":{spec.model.provider}:" not in policy_id:
            raise ValueError(
                f"trajectory {manifest.trajectory_id} provider does not match search model"
            )
        matched += 1
    if matched == 0:
        raise ValueError(
            f"trajectory {manifest.trajectory_id} contains no accepted conditioned actions "
            f"for skill {spec.skill_id!r}"
        )


def _candidate_id(
    spec: SkillConditionSearchSpec,
    candidate: SkillConditionCandidate,
) -> str:
    identity = {
        "skill_id": spec.skill_id,
        "provider": spec.model.provider,
        "model_version": spec.model.model_version,
        "instruction": candidate.instruction.strip(),
        "condition_scale": candidate.condition_scale,
        "latent_id": candidate.latent_id,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _candidate_rank_key(candidate: CandidateAggregate) -> tuple[float, float, str]:
    success = -1.0 if candidate.success_rate is None else candidate.success_rate
    completion = candidate.successful_completion_time_s_p50
    return (-success, float("inf") if completion is None else completion, candidate.candidate_id)


def _canonical_bytes(manifest: SkillConditionSearchManifest) -> bytes:
    payload = manifest.model_dump(mode="json")
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
