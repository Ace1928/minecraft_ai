from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from minecraft_ai.datasets.hindsight import HindsightDatasetManifest, HindsightExample


class SelfImitationExample(HindsightExample):
    """One unique accepted action selected by empirical return for self-imitation."""

    source_example_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    empirical_return: float = Field(ge=0.0, le=1.0)
    group_baseline: float = Field(ge=0.0, le=1.0)
    advantage: float = Field(ge=-1.0, le=1.0)
    awr_weight: float = Field(gt=0.0, le=64.0)
    training_weight: float = Field(gt=0.0, le=64.0)


class SelfImitationSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    data_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    examples: int = Field(ge=1)
    training_allowed: bool
    redistribution_allowed: bool


class SelfImitationDatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    artifact_type: str = "bedrock-advantage-weighted-self-imitation-dataset"
    data_filename: str
    data_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    examples: int = Field(ge=1)
    source_examples: int = Field(ge=1)
    unique_actions: int = Field(ge=1)
    successful_examples: int = Field(ge=0)
    partial_progress_examples: int = Field(ge=0)
    beta: float = Field(gt=0.0)
    maximum_awr_weight: float = Field(ge=1.0, le=64.0)
    elite_fraction: float = Field(gt=0.0, le=1.0)
    grouping: tuple[str, ...] = ("skill_id", "action_level", "route_id")
    sources: tuple[SelfImitationSource, ...]
    training_allowed: bool
    redistribution_allowed: bool
    automatic_promotion: bool = False


class SelfImitationDatasetArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    data_path: Path
    manifest_path: Path
    data_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    manifest: SelfImitationDatasetManifest


def write_self_imitation_dataset(
    hindsight_manifests: Sequence[Path],
    *,
    output_dir: Path,
    beta: float = 0.25,
    maximum_awr_weight: float = 4.0,
    elite_fraction: float = 0.5,
) -> SelfImitationDatasetArtifact:
    """Rank verified hindsight actions and write a deterministic elite/AWR artifact.

    A failed run may relabel the same physical action for several milestones. The
    VPT body is not goal-conditioned, so retaining every relabel would silently
    multiply that actuator target. This pass keeps the strongest empirical label
    per accepted action before ranking within a skill/action-level/route cohort.
    """

    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be positive and finite")
    if (
        not math.isfinite(maximum_awr_weight)
        or not 1 <= maximum_awr_weight <= 64
    ):
        raise ValueError("maximum_awr_weight must be finite and between 1 and 64")
    if not math.isfinite(elite_fraction) or not 0 < elite_fraction <= 1:
        raise ValueError("elite_fraction must be finite and in (0, 1]")
    selected = tuple(Path(path).expanduser().resolve() for path in hindsight_manifests)
    if not selected:
        raise ValueError("at least one hindsight manifest is required")

    examples: list[HindsightExample] = []
    sources: list[SelfImitationSource] = []
    for manifest_path in selected:
        manifest_raw = manifest_path.read_bytes()
        manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
        if manifest_path.name != f"{manifest_sha256}.manifest.json":
            raise ValueError(f"hindsight manifest identity mismatch: {manifest_path}")
        source_manifest = HindsightDatasetManifest.model_validate_json(manifest_raw)
        if not source_manifest.training_allowed:
            raise ValueError(f"hindsight dataset is not licensed for training: {manifest_path}")
        data_path = manifest_path.parent / source_manifest.data_filename
        if not data_path.is_file() or _file_sha256(data_path) != source_manifest.data_sha256:
            raise ValueError(f"hindsight data integrity mismatch: {data_path}")
        loaded = tuple(_iter_examples(data_path))
        if len(loaded) != source_manifest.examples:
            raise ValueError(f"hindsight example count mismatch: {data_path}")
        examples.extend(loaded)
        sources.append(
            SelfImitationSource(
                manifest_sha256=manifest_sha256,
                data_sha256=source_manifest.data_sha256,
                examples=source_manifest.examples,
                training_allowed=source_manifest.training_allowed,
                redistribution_allowed=source_manifest.redistribution_allowed,
            )
        )

    unique = _deduplicate_actions(examples)
    ranked = _rank_examples(
        unique,
        beta=beta,
        maximum_awr_weight=maximum_awr_weight,
        elite_fraction=elite_fraction,
    )
    if not ranked:
        raise ValueError("hindsight datasets contain no eligible self-imitation actions")

    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    staged = destination / f".self-imitation-{os.getpid()}.jsonl.tmp"
    digest = hashlib.sha256()
    try:
        with staged.open("wb") as handle:
            for example in ranked:
                encoded = _canonical_line(example)
                handle.write(encoded)
                digest.update(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        data_sha256 = digest.hexdigest()
        data_path = destination / f"{data_sha256}.jsonl"
        if data_path.exists():
            if _file_sha256(data_path) != data_sha256:
                raise RuntimeError(f"content-address collision at {data_path}")
            staged.unlink()
        else:
            staged.replace(data_path)
        output_manifest = SelfImitationDatasetManifest(
            data_filename=data_path.name,
            data_sha256=data_sha256,
            examples=len(ranked),
            source_examples=len(examples),
            unique_actions=len(unique),
            successful_examples=sum(example.success for example in ranked),
            partial_progress_examples=sum(not example.success for example in ranked),
            beta=beta,
            maximum_awr_weight=maximum_awr_weight,
            elite_fraction=elite_fraction,
            sources=tuple(sources),
            training_allowed=all(source.training_allowed for source in sources),
            redistribution_allowed=all(source.redistribution_allowed for source in sources),
        )
        manifest_bytes = _canonical_document(output_manifest)
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        manifest_path = destination / f"{manifest_sha256}.manifest.json"
        _atomic_content_write(manifest_path, manifest_bytes)
        return SelfImitationDatasetArtifact(
            data_path=data_path,
            manifest_path=manifest_path,
            data_sha256=data_sha256,
            manifest_sha256=manifest_sha256,
            manifest=output_manifest,
        )
    finally:
        staged.unlink(missing_ok=True)


def _deduplicate_actions(examples: Iterable[HindsightExample]) -> tuple[HindsightExample, ...]:
    selected: dict[tuple[str, int, str], HindsightExample] = {}
    for example in examples:
        key = (example.trajectory_id, example.step_index, example.skill_run_id)
        previous = selected.get(key)
        if previous is None or _selection_key(example) > _selection_key(previous):
            selected[key] = example
    return tuple(selected[key] for key in sorted(selected))


def _selection_key(example: HindsightExample) -> tuple[float, bool, float, str]:
    return (
        _empirical_return(example),
        example.success,
        example.sample_weight,
        example.example_id,
    )


def _empirical_return(example: HindsightExample) -> float:
    confidence = 1.0 if example.evidence_confidence is None else example.evidence_confidence
    return 1.0 if example.success else example.progress * confidence


def _rank_examples(
    examples: Sequence[HindsightExample],
    *,
    beta: float,
    maximum_awr_weight: float,
    elite_fraction: float,
) -> tuple[SelfImitationExample, ...]:
    groups: dict[tuple[str, str, str], list[HindsightExample]] = defaultdict(list)
    for example in examples:
        groups[(example.skill_id, example.action_level.value, example.route_id)].append(example)
    output: list[SelfImitationExample] = []
    for group in sorted(groups):
        candidates = groups[group]
        baseline = sum(_empirical_return(example) for example in candidates) / len(candidates)
        ordered = sorted(
            candidates,
            key=lambda item: (-_empirical_return(item), item.example_id),
        )
        elite_count = max(1, math.ceil(len(ordered) * elite_fraction))
        admitted = {
            example.example_id
            for example in ordered[:elite_count]
        } | {example.example_id for example in ordered if example.success}
        for example in ordered:
            if example.example_id not in admitted:
                continue
            empirical_return = _empirical_return(example)
            advantage = empirical_return - baseline
            awr_weight = min(maximum_awr_weight, math.exp(advantage / beta))
            output.append(
                SelfImitationExample(
                    **example.model_dump(mode="python"),
                    source_example_id=example.example_id,
                    empirical_return=empirical_return,
                    group_baseline=baseline,
                    advantage=advantage,
                    awr_weight=awr_weight,
                    training_weight=example.sample_weight * awr_weight,
                )
            )
    return tuple(
        sorted(output, key=lambda item: (item.trajectory_id, item.step_index, item.example_id))
    )


def _iter_examples(path: Path) -> Iterable[HindsightExample]:
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                raise ValueError(f"blank hindsight record at {path}:{line_number}")
            try:
                yield HindsightExample.model_validate_json(raw)
            except ValueError as exc:
                raise ValueError(f"invalid hindsight record at {path}:{line_number}") from exc


def _canonical_line(model: BaseModel) -> bytes:
    return _canonical_json(model.model_dump(mode="json")) + b"\n"


def _canonical_document(model: BaseModel) -> bytes:
    return _canonical_json(model.model_dump(mode="json")) + b"\n"


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_content_write(path: Path, value: bytes) -> None:
    if path.exists():
        if path.read_bytes() != value:
            raise RuntimeError(f"content-address collision at {path}")
        return
    staged = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        with staged.open("xb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        staged.replace(path)
    finally:
        staged.unlink(missing_ok=True)


__all__ = (
    "SelfImitationDatasetArtifact",
    "SelfImitationDatasetManifest",
    "SelfImitationExample",
    "SelfImitationSource",
    "write_self_imitation_dataset",
)
