from __future__ import annotations

import hashlib
import json
import os
import tarfile
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from pydantic import BaseModel, ConfigDict, Field

from minecraft_ai.datasets.schema import ActionLevel, DatasetSourceType
from minecraft_ai.episodes import RuntimeEvent, RuntimeEventKind
from minecraft_ai.perception import FrameState
from minecraft_ai.safety import MotorAction
from minecraft_ai.storage import StateDatabase
from minecraft_ai.trajectory import ActionOrigin, TrajectoryReader, TrajectoryStep


_TERMINAL_KINDS = {
    RuntimeEventKind.SKILL_SUCCEEDED,
    RuntimeEventKind.SKILL_FAILED,
    RuntimeEventKind.SKILL_TIMED_OUT,
    RuntimeEventKind.SKILL_CANCELLED,
}


class HindsightLabelSource(StrEnum):
    TERMINAL_SUCCESS = "terminal_success"
    OBSERVED_TRANSITION = "observed_transition"


class HindsightExample(BaseModel):
    """One accepted policy action with an empirically supported goal label."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    example_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    trajectory_id: str
    step_index: int = Field(ge=0)
    captured_ns: int
    frame_ref: str
    frame_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    visual_embedding_ref: str | None = None
    history_step_indices: tuple[int, ...]
    previous_action: MotorAction | None = None
    action: MotorAction
    action_level: ActionLevel
    behavior_token: int | None = None
    skill_run_id: str
    skill_id: str
    goal_id: str | None = None
    plan_node_id: str | None = None
    intended_goal: str
    achieved_goal: str
    terminal_outcome: str
    success: bool
    progress: float = Field(ge=0.0, le=1.0)
    sample_weight: float = Field(gt=0.0, le=1.0)
    label_source: HindsightLabelSource
    evidence_event_ids: tuple[str, ...] = ()
    evidence_fact_key: str | None = None
    evidence_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    policy_id: str
    model_version: str | None = None
    route_id: str
    condition_id: str | None = None
    condition: dict[str, object] | None = None
    latent_id: str | None = None
    target_track_id: str | None = None


class HindsightTrajectorySource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    trajectory_id: str
    manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    shard_sha256: tuple[str, ...]
    source_type: DatasetSourceType
    license: str
    redistribution_allowed: bool
    training_allowed: bool
    game_version: str
    accepted_steps: int = Field(ge=0)


class HindsightDatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    artifact_type: str = "bedrock-hindsight-action-dataset"
    data_filename: str
    data_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    examples: int = Field(ge=0)
    successful_examples: int = Field(ge=0)
    partial_progress_examples: int = Field(ge=0)
    history_steps: int = Field(ge=1)
    partial_window_steps: int = Field(ge=1)
    sources: tuple[HindsightTrajectorySource, ...]
    training_allowed: bool
    redistribution_allowed: bool
    automatic_promotion: bool = False


class HindsightDatasetArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    data_path: Path
    manifest_path: Path
    data_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    manifest: HindsightDatasetManifest


@dataclass(frozen=True, slots=True)
class _ObservedStep:
    step: TrajectoryStep
    facts: dict[str, tuple[str | int | float | bool, float]]
    gui_mode: str


@dataclass(frozen=True, slots=True)
class _Milestone:
    fact_key: str
    achieved_goal: str
    progress: float
    missing_counts_as_false: bool = True


_MILESTONES = (
    _Milestone("target.visible", "find requested target", 0.25),
    _Milestone("target.centered", "center requested target", 0.45),
    _Milestone("target.near", "approach requested target", 0.65),
    _Milestone("target.broken", "break requested target", 0.9),
    _Milestone("interaction.changed", "complete requested interaction", 0.75),
    _Milestone("placement.changed", "place requested block", 0.8),
    _Milestone(
        "scene.playable",
        "return to playable world",
        0.75,
        missing_counts_as_false=False,
    ),
    _Milestone("player.gui.inventory", "open inventory", 0.7),
)


def iter_hindsight_examples(
    trajectory_paths: Sequence[Path],
    *,
    state_db_path: Path,
    history_steps: int = 128,
    partial_window_steps: int = 128,
) -> Iterator[HindsightExample]:
    """Relabel accepted Bedrock policy actions from verified outcomes and observations."""

    if history_steps < 1:
        raise ValueError("history_steps must be positive")
    if partial_window_steps < 1:
        raise ValueError("partial_window_steps must be positive")
    with StateDatabase(state_db_path) as database:
        for path in trajectory_paths:
            reader = TrajectoryReader(path)
            _validate_source(reader)
            events = database.load_runtime_events_for_trajectory(reader.manifest.trajectory_id)
            events_by_run = _events_by_run(events)
            terminal_by_run = _terminal_events(events_by_run)
            for run_id, observed_steps in _iter_terminal_run_steps(
                _iter_observed_steps(reader),
                terminal_by_run=terminal_by_run,
            ):
                event = terminal_by_run[run_id]
                run_events = events_by_run.get(run_id, ())
                yield from _examples_for_run(
                    observed_steps,
                    event,
                    run_events=run_events,
                    history_steps=history_steps,
                    partial_window_steps=partial_window_steps,
                )


def _iter_terminal_run_steps(
    observed_steps: Iterable[_ObservedStep],
    *,
    terminal_by_run: dict[str, RuntimeEvent],
) -> Iterator[tuple[str, tuple[_ObservedStep, ...]]]:
    """Yield terminal skill episodes while supporting suspended option parents.

    A hierarchical parent legitimately disappears while a child option owns
    accepted actions, then resumes under the same run ID. Terminal event time
    is the trustworthy episode boundary; adjacency in the flat action stream
    is not. Retain only terminal runs that are concurrently open and flush a
    run once a later action crosses its terminal timestamp.
    """

    pending: dict[str, list[_ObservedStep]] = {}
    yielded: set[str] = set()
    for observed in observed_steps:
        run_id = observed.step.skill_run_id
        captured_ns = observed.step.captured_ns
        ready = sorted(
            (
                candidate
                for candidate, steps in pending.items()
                if steps
                and candidate != run_id
                and terminal_by_run[candidate].observed_ns <= captured_ns
            ),
            key=lambda candidate: (
                terminal_by_run[candidate].observed_ns,
                candidate,
            ),
        )
        for candidate in ready:
            yield candidate, tuple(pending.pop(candidate))
            yielded.add(candidate)
        # Supervisor-only release/safety actions have no skill owner and must
        # not split or become part of a learned option.
        if run_id is None:
            continue
        if run_id in yielded:
            raise ValueError(f"skill run {run_id!r} has actions after its terminal event")
        if run_id in terminal_by_run:
            pending.setdefault(run_id, []).append(observed)
    for run_id in sorted(
        pending,
        key=lambda candidate: (terminal_by_run[candidate].observed_ns, candidate),
    ):
        if pending[run_id]:
            yield run_id, tuple(pending[run_id])


def write_hindsight_dataset(
    trajectory_paths: Sequence[Path],
    *,
    state_db_path: Path,
    output_dir: Path,
    history_steps: int = 128,
    partial_window_steps: int = 128,
) -> HindsightDatasetArtifact:
    """Write deterministic JSONL examples and a content-addressed provenance manifest."""

    selected_paths = tuple(Path(path).expanduser().resolve() for path in trajectory_paths)
    if not selected_paths:
        raise ValueError("at least one trajectory is required")
    readers = tuple(TrajectoryReader(path) for path in selected_paths)
    sources = tuple(_source_record(reader) for reader in readers)
    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    staged = destination / f".hindsight-{os.getpid()}.jsonl.tmp"
    digest = hashlib.sha256()
    count = 0
    successful = 0
    partial = 0
    try:
        with staged.open("wb") as handle:
            for example in iter_hindsight_examples(
                selected_paths,
                state_db_path=state_db_path,
                history_steps=history_steps,
                partial_window_steps=partial_window_steps,
            ):
                encoded = _canonical_line(example)
                handle.write(encoded)
                digest.update(encoded)
                count += 1
                successful += int(example.success)
                partial += int(not example.success)
            handle.flush()
            os.fsync(handle.fileno())
        if count == 0:
            raise ValueError("selected trajectories contain no empirically labelled policy actions")
        data_sha256 = digest.hexdigest()
        data_path = destination / f"{data_sha256}.jsonl"
        if data_path.exists():
            if _file_sha256(data_path) != data_sha256:
                raise RuntimeError(f"content-address collision at {data_path}")
            staged.unlink()
        else:
            staged.replace(data_path)
        manifest = HindsightDatasetManifest(
            data_filename=data_path.name,
            data_sha256=data_sha256,
            examples=count,
            successful_examples=successful,
            partial_progress_examples=partial,
            history_steps=history_steps,
            partial_window_steps=partial_window_steps,
            sources=sources,
            training_allowed=all(source.training_allowed for source in sources),
            redistribution_allowed=all(source.redistribution_allowed for source in sources),
        )
        manifest_bytes = _canonical_document(manifest)
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        manifest_path = destination / f"{manifest_sha256}.manifest.json"
        _atomic_content_write(manifest_path, manifest_bytes)
        return HindsightDatasetArtifact(
            data_path=data_path,
            manifest_path=manifest_path,
            data_sha256=data_sha256,
            manifest_sha256=manifest_sha256,
            manifest=manifest,
        )
    finally:
        staged.unlink(missing_ok=True)


def discover_trajectory_paths(root: Path, *, maximum: int | None = None) -> tuple[Path, ...]:
    """Find sealed, non-empty trajectory directories newest first."""

    if maximum is not None and maximum < 1:
        raise ValueError("maximum must be positive")
    selected_root = root.expanduser().resolve()
    candidates: list[tuple[int, Path]] = []
    if not selected_root.is_dir():
        return ()
    for manifest_path in selected_root.glob("*/manifest.json"):
        reader = TrajectoryReader(manifest_path)
        manifest = reader.manifest
        if manifest.ended_ns is None or manifest.accepted_steps < 1:
            continue
        candidates.append((manifest.ended_ns, reader.directory))
    ordered = tuple(path for _, path in sorted(candidates, reverse=True))
    return ordered if maximum is None else ordered[:maximum]


def _examples_for_run(
    observed_steps: Sequence[_ObservedStep],
    terminal: RuntimeEvent,
    *,
    run_events: Sequence[RuntimeEvent],
    history_steps: int,
    partial_window_steps: int,
) -> Iterator[HindsightExample]:
    policy_steps = tuple(
        observed
        for observed in observed_steps
        if observed.step.action_origin == ActionOrigin.POLICY
        and observed.step.policy_id is not None
        and observed.step.route_id is not None
        and observed.step.skill_id is not None
        and observed.step.skill_run_id is not None
    )
    if not policy_steps:
        return
    intended_goal = _intended_goal(policy_steps[0].step)
    event_ids = tuple(event.event_id for event in run_events)
    if terminal.kind == RuntimeEventKind.SKILL_SUCCEEDED:
        for position, observed in enumerate(policy_steps):
            yield _example(
                observed.step,
                history=_history(policy_steps, position, history_steps),
                intended_goal=intended_goal,
                achieved_goal=intended_goal,
                terminal=terminal,
                success=True,
                progress=1.0,
                sample_weight=1.0,
                label_source=HindsightLabelSource.TERMINAL_SUCCESS,
                evidence_event_ids=event_ids,
            )
        return

    previous_values: dict[str, bool] = {}
    emitted_goals: set[str] = set()
    for observed in observed_steps:
        values = dict(observed.facts)
        values["player.gui.inventory"] = (
            observed.gui_mode.casefold() in {"inventory", "inventory_screen"},
            1.0,
        )
        for milestone in _MILESTONES:
            current = _fact_true(values.get(milestone.fact_key))
            previously_seen = milestone.fact_key in previous_values
            previous = previous_values.get(milestone.fact_key, False)
            if (
                current
                and not previous
                and milestone.achieved_goal not in emitted_goals
                and (milestone.missing_counts_as_false or previously_seen)
            ):
                emitted_goals.add(milestone.achieved_goal)
                eligible = tuple(
                    step
                    for step in policy_steps
                    if step.step.step_index <= observed.step.step_index
                )[-partial_window_steps:]
                for position, policy_step in enumerate(eligible):
                    yield _example(
                        policy_step.step,
                        history=_history(eligible, position, history_steps),
                        intended_goal=intended_goal,
                        achieved_goal=milestone.achieved_goal,
                        terminal=terminal,
                        success=False,
                        progress=milestone.progress,
                        sample_weight=0.6,
                        label_source=HindsightLabelSource.OBSERVED_TRANSITION,
                        evidence_event_ids=event_ids,
                        evidence_fact_key=milestone.fact_key,
                        evidence_confidence=values[milestone.fact_key][1],
                    )
            previous_values[milestone.fact_key] = current


def _example(
    step: TrajectoryStep,
    *,
    history: tuple[int, ...],
    intended_goal: str,
    achieved_goal: str,
    terminal: RuntimeEvent,
    success: bool,
    progress: float,
    sample_weight: float,
    label_source: HindsightLabelSource,
    evidence_event_ids: tuple[str, ...],
    evidence_fact_key: str | None = None,
    evidence_confidence: float | None = None,
) -> HindsightExample:
    assert step.policy_id is not None
    assert step.route_id is not None
    assert step.skill_run_id is not None
    assert step.skill_id is not None
    identity = {
        "trajectory_id": step.trajectory_id,
        "step_index": step.step_index,
        "skill_run_id": step.skill_run_id,
        "achieved_goal": achieved_goal,
        "label_source": label_source.value,
        "evidence_fact_key": evidence_fact_key,
        "terminal_event_id": terminal.event_id,
    }
    example_id = hashlib.sha256(_canonical_json(identity)).hexdigest()
    return HindsightExample(
        example_id=example_id,
        trajectory_id=step.trajectory_id,
        step_index=step.step_index,
        captured_ns=step.captured_ns,
        frame_ref=step.frame_ref,
        frame_hash=step.frame_hash,
        visual_embedding_ref=step.visual_embedding_ref,
        history_step_indices=history,
        previous_action=step.previous_action,
        action=step.action,
        action_level=step.action_level,
        behavior_token=step.behavior_token,
        skill_run_id=step.skill_run_id,
        skill_id=step.skill_id,
        goal_id=step.goal_id,
        plan_node_id=step.plan_node_id,
        intended_goal=intended_goal,
        achieved_goal=achieved_goal,
        terminal_outcome=terminal.kind.value.removeprefix("skill_"),
        success=success,
        progress=progress,
        sample_weight=sample_weight,
        label_source=label_source,
        evidence_event_ids=evidence_event_ids,
        evidence_fact_key=evidence_fact_key,
        evidence_confidence=evidence_confidence,
        policy_id=step.policy_id,
        model_version=step.model_version,
        route_id=step.route_id,
        condition_id=step.condition_id,
        condition=step.condition,
        latent_id=step.latent_id,
        target_track_id=step.target_track_id,
    )


def _history(
    steps: Sequence[_ObservedStep],
    position: int,
    maximum: int,
) -> tuple[int, ...]:
    start = max(0, position - maximum)
    return tuple(item.step.step_index for item in steps[start:position])


def _intended_goal(step: TrajectoryStep) -> str:
    condition = step.condition
    if condition is not None:
        instruction = condition.get("instruction")
        if isinstance(instruction, str) and instruction.strip():
            return instruction.strip()
    assert step.skill_id is not None
    return step.skill_id.replace("_", " ")


def _events_by_run(events: Iterable[RuntimeEvent]) -> dict[str, tuple[RuntimeEvent, ...]]:
    collected: dict[str, list[RuntimeEvent]] = defaultdict(list)
    for event in events:
        run_id = _event_text(event, "run_id")
        if run_id is not None:
            collected[run_id].append(event)
    return {run_id: tuple(items) for run_id, items in collected.items()}


def _terminal_events(
    events_by_run: dict[str, tuple[RuntimeEvent, ...]],
) -> dict[str, RuntimeEvent]:
    selected: dict[str, RuntimeEvent] = {}
    for run_id, events in events_by_run.items():
        terminals = tuple(event for event in events if event.kind in _TERMINAL_KINDS)
        if len(terminals) > 1:
            raise ValueError(f"skill run {run_id!r} has multiple terminal events")
        if terminals:
            selected[run_id] = terminals[0]
    return selected


def _event_text(event: RuntimeEvent, key: str) -> str | None:
    value = event.payload.get(key)
    return value if isinstance(value, str) and value else None


def _fact_true(value: tuple[str | int | float | bool, float] | None) -> bool:
    if value is None:
        return False
    raw = value[0]
    return raw is True or (isinstance(raw, str) and raw.casefold() == "true")


def _iter_observed_steps(reader: TrajectoryReader) -> Iterator[_ObservedStep]:
    expected_index = 0
    for shard in reader.manifest.shards:
        shard_path = reader.directory / shard.filename
        if not shard_path.is_file():
            raise FileNotFoundError(f"trajectory shard does not exist: {shard.filename}")
        if shard_path.stat().st_size != shard.bytes or _file_sha256(shard_path) != shard.sha256:
            raise ValueError(f"trajectory shard integrity mismatch: {shard.filename}")
        with tarfile.open(shard_path, mode="r") as archive:
            members = {member.name: member for member in archive.getmembers() if member.isfile()}
            step_names = sorted(name for name in members if name.endswith(".step.json"))
            for step_name in step_names:
                key = step_name.removesuffix(".step.json")
                blackboard_name = f"{key}.blackboard.json"
                if blackboard_name not in members:
                    raise ValueError(f"trajectory sample {key} has no blackboard")
                step = TrajectoryStep.model_validate_json(
                    _read_member(archive, members[step_name])
                )
                if step.step_index != expected_index or int(key) != expected_index:
                    raise ValueError(
                        f"non-contiguous step index: expected {expected_index}, got "
                        f"{step.step_index}"
                    )
                if step.trajectory_id != reader.manifest.trajectory_id:
                    raise ValueError(f"trajectory sample {key} has wrong trajectory identity")
                frame_member = urlparse(step.frame_ref).fragment
                if not frame_member or frame_member not in members:
                    raise ValueError(f"trajectory sample {key} has an invalid frame reference")
                blackboard_raw = _read_member(archive, members[blackboard_name])
                expected_hash = _reference_sha256(step.blackboard_snapshot_ref)
                if (
                    expected_hash is None
                    or hashlib.sha256(blackboard_raw).hexdigest() != expected_hash
                ):
                    raise ValueError(f"trajectory sample {key} blackboard hash mismatch")
                blackboard = FrameState.model_validate_json(blackboard_raw)
                if blackboard.captured_ns != step.captured_ns:
                    raise ValueError(f"trajectory sample {key} blackboard timestamp mismatch")
                facts = {
                    fact.key: (fact.value, fact.confidence)
                    for fact in blackboard.facts
                }
                yield _ObservedStep(
                    step=step,
                    facts=facts,
                    gui_mode=blackboard.player.gui_mode,
                )
                expected_index += 1
    if expected_index != reader.manifest.accepted_steps:
        raise ValueError(
            f"trajectory accepted_steps={reader.manifest.accepted_steps}, read={expected_index}"
        )


def _validate_source(reader: TrajectoryReader) -> None:
    manifest = reader.manifest
    if manifest.ended_ns is None:
        raise ValueError(f"trajectory {manifest.trajectory_id} is not sealed")
    if not manifest.shards:
        raise ValueError(f"trajectory {manifest.trajectory_id} has no sealed shard manifests")
    if manifest.source.edition.casefold() != "bedrock":
        raise ValueError(f"trajectory {manifest.trajectory_id} is not Bedrock-native")
    if not manifest.source.training_allowed:
        raise ValueError(f"trajectory {manifest.trajectory_id} is not licensed for training")


def _source_record(reader: TrajectoryReader) -> HindsightTrajectorySource:
    _validate_source(reader)
    manifest = reader.manifest
    return HindsightTrajectorySource(
        trajectory_id=manifest.trajectory_id,
        manifest_sha256=_file_sha256(reader.manifest_path),
        shard_sha256=tuple(shard.sha256 for shard in manifest.shards),
        source_type=manifest.source.source_type,
        license=manifest.source.license,
        redistribution_allowed=manifest.source.redistribution_allowed,
        training_allowed=manifest.source.training_allowed,
        game_version=manifest.game_version,
        accepted_steps=manifest.accepted_steps,
    )


def _reference_sha256(reference: str) -> str | None:
    values = parse_qs(urlparse(reference).query).get("sha256")
    return None if not values else values[0]


def _read_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    handle = archive.extractfile(member)
    if handle is None:
        raise ValueError(f"cannot read trajectory member {member.name}")
    return handle.read()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _canonical_line(model: BaseModel) -> bytes:
    return _canonical_json(model.model_dump(mode="json")) + b"\n"


def _canonical_document(model: BaseModel) -> bytes:
    return _canonical_line(model)


def _atomic_content_write(destination: Path, payload: bytes) -> None:
    if destination.exists():
        if destination.read_bytes() != payload:
            raise RuntimeError(f"content-address collision at {destination}")
        return
    staged = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with staged.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        staged.replace(destination)
    finally:
        staged.unlink(missing_ok=True)
