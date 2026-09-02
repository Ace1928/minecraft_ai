from __future__ import annotations

import hashlib
import json
import queue
import tarfile
import threading
import time
import uuid
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from collections.abc import Callable, Iterator

from pydantic import BaseModel, ConfigDict, Field

from .datasets.schema import ActionLevel, DatasetValidationReport, TrajectoryManifest
from .datasets.shards import TrajectoryShardWriter
from .perception import FrameState
from .platforms.bedrock_x11 import CapturedFrame
from .safety import MotorAction


class TrajectoryStep(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    trajectory_id: str
    step_index: int = Field(ge=0)
    captured_ns: int
    accepted_ns: int | None = None
    frame_ref: str
    frame_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    visual_embedding_ref: str | None = None
    previous_action: MotorAction | None = None
    action: MotorAction
    action_level: ActionLevel
    behavior_token: int | None = None
    skill_run_id: str | None = None
    skill_id: str | None = None
    goal_id: str | None = None
    plan_node_id: str | None = None
    blackboard_snapshot_ref: str
    place_event_id: str | None = None
    reward_signals: dict[str, float] = Field(default_factory=dict)
    event_ids: tuple[str, ...] = ()
    correction_of_step: int | None = Field(default=None, ge=0)


@dataclass(frozen=True)
class AcceptedTrajectorySample:
    step: TrajectoryStep
    frame: CapturedFrame
    blackboard_json: bytes


@dataclass(frozen=True)
class ReplayTrajectorySample:
    step: TrajectoryStep
    frame: CapturedFrame
    blackboard: FrameState


class TrajectoryReader:
    """Stream and verify a portable frame/action-aligned trajectory."""

    def __init__(self, path: Path) -> None:
        selected = path.expanduser().resolve()
        self.manifest_path = selected / "manifest.json" if selected.is_dir() else selected
        self.directory = self.manifest_path.parent
        self.manifest = TrajectoryManifest.model_validate_json(
            self.manifest_path.read_text(encoding="utf-8")
        )

    def iter_samples(self) -> Iterator[ReplayTrajectorySample]:
        yield from self._iter_samples(self._shard_paths())

    def _iter_samples(
        self,
        shard_paths: list[Path],
    ) -> Iterator[ReplayTrajectorySample]:
        expected_index = 0
        previous_action: MotorAction | None = None
        for shard_path in shard_paths:
            with tarfile.open(shard_path, mode="r") as archive:
                members = {
                    member.name: member
                    for member in archive.getmembers()
                    if member.isfile()
                }
                step_names = sorted(name for name in members if name.endswith(".step.json"))
                for step_name in step_names:
                    key = step_name.removesuffix(".step.json")
                    required = {
                        f"{key}.step.json",
                        f"{key}.frame.json",
                        f"{key}.frame.bgra.zlib",
                        f"{key}.blackboard.json",
                    }
                    missing = sorted(required - members.keys())
                    if missing:
                        raise ValueError(
                            f"shard {shard_path.name} sample {key} missing {', '.join(missing)}"
                        )
                    step = TrajectoryStep.model_validate_json(
                        self._read_member(archive, members[f"{key}.step.json"])
                    )
                    if step.trajectory_id != self.manifest.trajectory_id:
                        raise ValueError(f"sample {key} trajectory identity mismatch")
                    if step.step_index != expected_index or int(key) != expected_index:
                        raise ValueError(
                            f"non-contiguous step index: expected {expected_index}, got "
                            f"{step.step_index}"
                        )
                    if step.previous_action != previous_action:
                        raise ValueError(f"step {expected_index} previous_action is not aligned")
                    header_raw = self._read_member(archive, members[f"{key}.frame.json"])
                    header = json.loads(header_raw)
                    if not isinstance(header, dict) or header.get("codec") != "zlib":
                        raise ValueError(f"step {expected_index} has unsupported frame header")
                    compressed = self._read_member(
                        archive, members[f"{key}.frame.bgra.zlib"]
                    )
                    pixels = zlib.decompress(compressed)
                    width = int(header["width"])
                    height = int(header["height"])
                    if len(pixels) != width * height * 4 or len(pixels) != int(
                        header["raw_bytes"]
                    ):
                        raise ValueError(f"step {expected_index} frame byte count mismatch")
                    if hashlib.sha256(pixels).hexdigest() != step.frame_hash:
                        raise ValueError(f"step {expected_index} frame hash mismatch")
                    blackboard_raw = self._read_member(
                        archive, members[f"{key}.blackboard.json"]
                    )
                    blackboard = FrameState.model_validate_json(blackboard_raw)
                    if (
                        blackboard.captured_ns != step.captured_ns
                        or blackboard.width != width
                        or blackboard.height != height
                    ):
                        raise ValueError(f"step {expected_index} blackboard/frame misalignment")
                    expected_blackboard_hash = _reference_sha256(
                        step.blackboard_snapshot_ref
                    )
                    if expected_blackboard_hash is not None and hashlib.sha256(
                        blackboard_raw
                    ).hexdigest() != expected_blackboard_hash:
                        raise ValueError(f"step {expected_index} blackboard hash mismatch")
                    yield ReplayTrajectorySample(
                        step=step,
                        frame=CapturedFrame(
                            frame_id=blackboard.frame_id,
                            captured_ns=step.captured_ns,
                            width=width,
                            height=height,
                            bgra=pixels,
                        ),
                        blackboard=blackboard,
                    )
                    previous_action = step.action
                    expected_index += 1

    def validate(
        self,
        *,
        on_sample: Callable[[ReplayTrajectorySample], None] | None = None,
    ) -> DatasetValidationReport:
        errors: list[str] = []
        warnings: list[str] = []
        count = 0
        first_ns: int | None = None
        last_ns: int | None = None
        latencies_ms: list[float] = []
        shard_paths = self._shard_paths()
        try:
            self._validate_shard_artifacts(shard_paths)
            for sample in self._iter_samples(shard_paths):
                count += 1
                first_ns = sample.step.captured_ns if first_ns is None else first_ns
                last_ns = sample.step.captured_ns
                if on_sample is not None:
                    on_sample(sample)
                if sample.step.accepted_ns is not None:
                    latencies_ms.append(
                        (sample.step.accepted_ns - sample.step.captured_ns) / 1_000_000.0
                    )
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        if self.manifest.ended_ns is None:
            warnings.append("trajectory is still open; only sealed shards were inspected")
        elif count != self.manifest.accepted_steps:
            errors.append(
                f"manifest accepted_steps={self.manifest.accepted_steps}, replayed={count}"
            )
        if self.manifest.dropped_steps:
            warnings.append(f"recorder dropped {self.manifest.dropped_steps} accepted samples")
        return DatasetValidationReport(
            trajectory_id=self.manifest.trajectory_id,
            valid=not errors,
            step_count=count,
            shard_count=len(shard_paths),
            first_captured_ns=first_ns,
            last_captured_ns=last_ns,
            frame_action_latency_ms_p50=_percentile(latencies_ms, 0.50),
            frame_action_latency_ms_p95=_percentile(latencies_ms, 0.95),
            errors=tuple(errors),
            warnings=tuple(warnings),
        )

    def _shard_paths(self) -> list[Path]:
        if self.manifest.shards:
            return [self.directory / item.filename for item in self.manifest.shards]
        if self.manifest.shard_ids:
            return [self.directory / f"{shard_id}.tar" for shard_id in self.manifest.shard_ids]
        # Open trajectories do not publish the final shard list until close;
        # sealed immutable shards remain safe to replay while recording.
        return sorted(self.directory.glob(f"{self.manifest.trajectory_id}-shard-*.tar"))

    def _validate_shard_artifacts(self, paths: list[Path]) -> None:
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"trajectory shard does not exist: {path.name}")
        if not self.manifest.shards:
            return
        by_name = {item.filename: item for item in self.manifest.shards}
        for path in paths:
            expected = by_name[path.name]
            if path.stat().st_size != expected.bytes:
                raise ValueError(f"shard size mismatch: {path.name}")
            if _file_sha256(path) != expected.sha256:
                raise ValueError(f"shard hash mismatch: {path.name}")

    @staticmethod
    def _read_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
        handle = archive.extractfile(member)
        if handle is None:
            raise ValueError(f"cannot read trajectory member {member.name}")
        return handle.read()


@dataclass
class TrajectoryRecorder:
    manifest: TrajectoryManifest
    artifact_root: Path
    state_db_path: Path
    shard_steps: int = 256
    queue_size: int = 512
    _queue: queue.Queue[AcceptedTrajectorySample | None] = field(init=False)
    _thread: threading.Thread = field(init=False)
    _step_index: int = field(default=0, init=False)
    _previous_action: MotorAction | None = field(default=None, init=False)
    _accepted_steps: int = field(default=0, init=False)
    _dropped_steps: int = field(default=0, init=False)
    _closed: bool = field(default=False, init=False)
    _worker_error: BaseException | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._queue = queue.Queue(maxsize=self.queue_size)
        self._thread = threading.Thread(
            target=self._run_writer,
            name="minecraft-ai-trajectory-writer",
            daemon=True,
        )
        self._thread.start()

    def record_accepted(
        self,
        *,
        action: MotorAction,
        supervisor_response: dict[str, object],
        frame: CapturedFrame,
        blackboard: FrameState,
        action_level: ActionLevel = ActionLevel.RAW,
        skill_run_id: str | None = None,
        skill_id: str | None = None,
        goal_id: str | None = None,
        plan_node_id: str | None = None,
        reward_signals: dict[str, float] | None = None,
        event_ids: tuple[str, ...] = (),
        correction_of_step: int | None = None,
    ) -> bool:
        if self._closed:
            raise RuntimeError("trajectory recorder is closed")
        if self._worker_error is not None:
            raise RuntimeError("trajectory writer failed") from self._worker_error
        accepted = supervisor_response.get("accepted_sequence")
        if not isinstance(accepted, int) or accepted != action.sequence:
            return False
        accepted_ns = supervisor_response.get("accepted_monotonic_ns")
        if accepted_ns is not None and (
            not isinstance(accepted_ns, int) or accepted_ns < frame.captured_ns
        ):
            raise ValueError("supervisor acceptance timestamp precedes captured frame")
        frame_hash = hashlib.sha256(frame.bgra).hexdigest()
        blackboard_json = blackboard.model_dump_json().encode()
        blackboard_hash = hashlib.sha256(blackboard_json).hexdigest()
        sample_key = f"{self.manifest.trajectory_id}/{self._step_index:012d}"
        shard_id = f"{self.manifest.trajectory_id}-shard-{self._step_index // self.shard_steps:06d}"
        step = TrajectoryStep(
            trajectory_id=self.manifest.trajectory_id,
            step_index=self._step_index,
            captured_ns=frame.captured_ns,
            accepted_ns=accepted_ns,
            frame_ref=f"wds://{self.manifest.trajectory_id}/{shard_id}.tar#{sample_key}.frame.bgra.zlib",
            frame_hash=frame_hash,
            previous_action=self._previous_action,
            action=action,
            action_level=action_level,
            skill_run_id=skill_run_id,
            skill_id=skill_id,
            goal_id=goal_id,
            plan_node_id=plan_node_id,
            blackboard_snapshot_ref=(
                f"wds://{self.manifest.trajectory_id}/{shard_id}.tar#"
                f"{sample_key}.blackboard.json?sha256={blackboard_hash}"
            ),
            reward_signals={} if reward_signals is None else reward_signals,
            event_ids=event_ids,
            correction_of_step=correction_of_step,
        )
        sample = AcceptedTrajectorySample(step=step, frame=frame, blackboard_json=blackboard_json)
        try:
            self._queue.put_nowait(sample)
        except queue.Full:
            self._dropped_steps += 1
            return False
        self._step_index += 1
        self._accepted_steps += 1
        self._previous_action = action
        return True

    def close(self, *, timeout_s: float = 15.0) -> TrajectoryManifest:
        if not self._closed:
            self._closed = True
            self._queue.put(None)
            self._thread.join(timeout=timeout_s)
        if self._thread.is_alive():
            raise TimeoutError("trajectory writer did not flush before timeout")
        if self._worker_error is not None:
            raise RuntimeError("trajectory writer failed") from self._worker_error
        manifest_path = self.artifact_root / self.manifest.trajectory_id / "manifest.json"
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        return TrajectoryManifest.model_validate(raw)

    def _run_writer(self) -> None:
        try:
            writer = TrajectoryShardWriter(
                manifest=self.manifest,
                artifact_root=self.artifact_root,
                state_db_path=self.state_db_path,
                max_steps=self.shard_steps,
            )
            try:
                while True:
                    sample = self._queue.get()
                    if sample is None:
                        break
                    writer.append(sample)
            finally:
                writer.close(
                    accepted_steps=self._accepted_steps,
                    dropped_steps=self._dropped_steps,
                )
        except BaseException as exc:
            self._worker_error = exc


def new_trajectory_id(prefix: str = "trajectory") -> str:
    return f"{prefix}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:12]}"


def _reference_sha256(reference: str) -> str | None:
    values = parse_qs(urlparse(reference).query).get("sha256")
    return None if not values else values[0]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
