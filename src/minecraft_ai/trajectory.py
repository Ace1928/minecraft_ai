from __future__ import annotations

import hashlib
import io
import json
import logging
import queue
import shutil
import tarfile
import threading
import time
import uuid
import zlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .datasets.schema import ActionLevel, DatasetValidationReport, TrajectoryManifest
from .datasets.shards import TrajectoryShardWriter
from .motor import MotorIntent
from .perception import FrameState
from .platforms.bedrock_x11 import CapturedFrame
from .safety import MotorAction


_LOG = logging.getLogger(__name__)
_GIB = 1024**3


class TrajectoryDiskSpaceError(RuntimeError):
    """Recording cannot continue without consuming the configured disk reserve."""


@dataclass(frozen=True)
class EncodedTrajectoryFrame:
    """Compact frame payload retained by the asynchronous shard queue."""

    member_suffix: str
    header_json: bytes
    payload: bytes
    width: int
    height: int


class ActionOrigin(StrEnum):
    """The mechanism that emitted an action later accepted by the supervisor."""

    POLICY = "policy"
    RESET = "reset"
    HUMAN = "human"
    SYNTHETIC = "synthetic"
    LEGACY = "legacy"


class ActionProvenance(BaseModel):
    """Causal policy/route/condition identity captured before actuation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_id: str = Field(min_length=1, max_length=512)
    model_version: str | None = Field(default=None, min_length=1, max_length=256)
    route_id: str = Field(min_length=1, max_length=128)
    policy_action_kind: str | None = Field(default=None, min_length=1, max_length=128)
    policy_request_id: str | None = Field(default=None, min_length=1, max_length=256)
    prediction_id: str | None = Field(default=None, min_length=1, max_length=256)
    action_level: ActionLevel
    origin: ActionOrigin
    condition_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    condition: dict[str, Any] | None = None
    behavior_token: int | None = Field(default=None, ge=0)
    latent_id: str | None = Field(default=None, min_length=1, max_length=256)
    target_track_id: str | None = Field(default=None, min_length=1, max_length=512)

    @model_validator(mode="after")
    def validate_condition_identity(self) -> ActionProvenance:
        if self.condition is None:
            return self
        _validate_condition_links(
            self.condition,
            action_level=self.action_level,
            target_track_id=self.target_track_id,
        )
        expected = motor_condition_id(
            self.condition,
            route_id=self.route_id,
            target_track_id=self.target_track_id,
        )
        if self.condition_id != expected:
            raise ValueError("condition_id does not identify the serialized motor condition")
        return self


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
    dropped_steps_before: int = Field(default=0, ge=0)
    action: MotorAction
    action_level: ActionLevel
    behavior_token: int | None = None
    latent_id: str | None = Field(default=None, min_length=1, max_length=256)
    action_origin: ActionOrigin = ActionOrigin.LEGACY
    policy_id: str | None = Field(default=None, min_length=1, max_length=512)
    model_version: str | None = Field(default=None, min_length=1, max_length=256)
    route_id: str | None = Field(default=None, min_length=1, max_length=128)
    policy_action_kind: str | None = Field(default=None, min_length=1, max_length=128)
    policy_request_id: str | None = Field(default=None, min_length=1, max_length=256)
    prediction_id: str | None = Field(default=None, min_length=1, max_length=256)
    condition_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    condition: dict[str, Any] | None = None
    target_track_id: str | None = Field(default=None, min_length=1, max_length=512)
    skill_run_id: str | None = None
    skill_id: str | None = None
    goal_id: str | None = None
    plan_node_id: str | None = None
    blackboard_snapshot_ref: str
    place_event_id: str | None = None
    reward_signals: dict[str, float] = Field(default_factory=dict)
    event_ids: tuple[str, ...] = ()
    correction_of_step: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_provenance(self) -> TrajectoryStep:
        if self.dropped_steps_before and self.previous_action is None:
            raise ValueError("dropped_steps_before requires the physical previous_action")
        if self.condition is None:
            return self
        if self.route_id is None:
            raise ValueError("serialized motor condition requires route_id")
        _validate_condition_links(
            self.condition,
            action_level=self.action_level,
            target_track_id=self.target_track_id,
        )
        expected = motor_condition_id(
            self.condition,
            route_id=self.route_id,
            target_track_id=self.target_track_id,
        )
        if self.condition_id != expected:
            raise ValueError("trajectory condition_id does not match serialized condition")
        return self


@dataclass(frozen=True)
class AcceptedTrajectorySample:
    step: TrajectoryStep
    frame: EncodedTrajectoryFrame
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
                    member.name: member for member in archive.getmembers() if member.isfile()
                }
                step_names = sorted(name for name in members if name.endswith(".step.json"))
                for step_name in step_names:
                    key = step_name.removesuffix(".step.json")
                    required = {
                        f"{key}.step.json",
                        f"{key}.frame.json",
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
                    if (
                        step.dropped_steps_before == 0
                        and step.previous_action != previous_action
                    ):
                        raise ValueError(f"step {expected_index} previous_action is not aligned")
                    header_raw = self._read_member(archive, members[f"{key}.frame.json"])
                    header = json.loads(header_raw)
                    if not isinstance(header, dict):
                        raise ValueError(f"step {expected_index} has unsupported frame header")
                    width, height, pixels = self._decode_frame(
                        archive,
                        members,
                        key=key,
                        step_index=expected_index,
                        step=step,
                        header=header,
                    )
                    blackboard_raw = self._read_member(archive, members[f"{key}.blackboard.json"])
                    blackboard = FrameState.model_validate_json(blackboard_raw)
                    if header.get("codec") == "jpeg":
                        _validate_compact_blackboard_evidence(blackboard, expected_index)
                    if (
                        blackboard.captured_ns != step.captured_ns
                        or blackboard.width != width
                        or blackboard.height != height
                    ):
                        raise ValueError(f"step {expected_index} blackboard/frame misalignment")
                    expected_blackboard_hash = _reference_sha256(step.blackboard_snapshot_ref)
                    if (
                        expected_blackboard_hash is not None
                        and hashlib.sha256(blackboard_raw).hexdigest() != expected_blackboard_hash
                    ):
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

    def _decode_frame(
        self,
        archive: tarfile.TarFile,
        members: dict[str, tarfile.TarInfo],
        *,
        key: str,
        step_index: int,
        step: TrajectoryStep,
        header: dict[str, Any],
    ) -> tuple[int, int, bytes]:
        codec = header.get("codec")
        width = int(header["width"])
        height = int(header["height"])
        if width < 1 or height < 1:
            raise ValueError(f"step {step_index} has invalid frame dimensions")
        if codec == "zlib":
            # Version-one shards stored full-resolution BGRA bytes. Keep this
            # branch indefinitely so accumulated demonstrations stay usable.
            member_name = f"{key}.frame.bgra.zlib"
            member = members.get(member_name)
            if member is None:
                raise ValueError(f"shard sample {key} missing {member_name}")
            compressed = self._read_member(archive, member)
            pixels = zlib.decompress(compressed)
            if len(pixels) != width * height * 4 or len(pixels) != int(header["raw_bytes"]):
                raise ValueError(f"step {step_index} frame byte count mismatch")
            if hashlib.sha256(pixels).hexdigest() != step.frame_hash:
                raise ValueError(f"step {step_index} frame hash mismatch")
            return width, height, pixels
        if codec != "jpeg" or header.get("pixel_format") != "RGB":
            raise ValueError(f"step {step_index} has unsupported frame header")
        member_name = f"{key}.frame.jpg"
        member = members.get(member_name)
        if member is None:
            raise ValueError(f"shard sample {key} missing {member_name}")
        encoded = self._read_member(archive, member)
        if len(encoded) != int(header.get("encoded_bytes", -1)):
            raise ValueError(f"step {step_index} encoded frame byte count mismatch")
        encoded_hash = hashlib.sha256(encoded).hexdigest()
        if encoded_hash != step.frame_hash or header.get("encoded_sha256") != encoded_hash:
            raise ValueError(f"step {step_index} frame hash mismatch")
        with Image.open(io.BytesIO(encoded)) as image:
            if image.format != "JPEG" or image.size != (width, height):
                raise ValueError(f"step {step_index} encoded frame metadata mismatch")
            pixels = image.convert("RGBA").tobytes("raw", "BGRA")
        if len(pixels) != width * height * 4:
            raise ValueError(f"step {step_index} frame byte count mismatch")
        return width, height, pixels

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
    frame_max_width: int = 256
    frame_jpeg_quality: int = 80
    min_free_disk_bytes: int = 5 * _GIB
    _queue: queue.Queue[AcceptedTrajectorySample | None] = field(init=False)
    _writer: TrajectoryShardWriter = field(init=False)
    _thread: threading.Thread = field(init=False)
    _step_index: int = field(default=0, init=False)
    _previous_action: MotorAction | None = field(default=None, init=False)
    _pending_dropped_steps: int = field(default=0, init=False)
    _written_steps: int = field(default=0, init=False)
    _dropped_steps: int = field(default=0, init=False)
    _closed: bool = field(default=False, init=False)
    _worker_error: BaseException | None = field(default=None, init=False)
    _recording_disabled_reason: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.frame_max_width < 16:
            raise ValueError("frame_max_width must be at least 16")
        if not 40 <= self.frame_jpeg_quality <= 95:
            raise ValueError("frame_jpeg_quality must be in 40..95")
        if self.min_free_disk_bytes < 0:
            raise ValueError("min_free_disk_bytes cannot be negative")
        require_trajectory_disk_reserve(
            self.artifact_root,
            minimum_free_bytes=self.min_free_disk_bytes,
        )
        self._queue = queue.Queue(maxsize=self.queue_size)
        # Register the trajectory synchronously before the recorder becomes
        # visible to AgentRuntime. Terminal gameplay events carry this foreign
        # key immediately; asynchronous registration allowed a fast first
        # option to race the writer thread and terminate play with an FK error.
        # Setup failures still fail open at the agent-process boundary.
        self._writer = TrajectoryShardWriter(
            manifest=self.manifest,
            artifact_root=self.artifact_root,
            state_db_path=self.state_db_path,
            max_steps=self.shard_steps,
            minimum_free_bytes=self.min_free_disk_bytes,
        )
        self._thread = threading.Thread(
            target=self._run_writer,
            name="minecraft-ai-trajectory-writer",
            daemon=True,
        )
        self._thread.start()

    def status(self) -> dict[str, object]:
        """Return a non-blocking snapshot of continuous trajectory recording."""

        worker_error = self._worker_error
        disabled_reason = self._recording_disabled_reason
        if worker_error is not None:
            disabled_reason = f"{type(worker_error).__name__}: {worker_error}"
        elif self._closed and disabled_reason is None:
            disabled_reason = "recorder-closed"
        return {
            "enabled": not self._closed
            and worker_error is None
            and disabled_reason is None,
            "disabled_reason": disabled_reason,
            "written_steps": self._written_steps,
            "dropped_steps": self._dropped_steps,
            "queued_samples": self._queue.qsize(),
            "queue_capacity": self.queue_size,
        }

    def disable(self, reason: str) -> None:
        """Fail open for gameplay while retaining recorder diagnostics."""

        detail = reason.strip()
        if not detail:
            raise ValueError("trajectory disable reason cannot be empty")
        if self._recording_disabled_reason is None:
            self._recording_disabled_reason = detail

    def record_accepted(
        self,
        *,
        action: MotorAction,
        provenance: ActionProvenance,
        supervisor_response: dict[str, object],
        frame: CapturedFrame,
        blackboard: FrameState,
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
        accepted = supervisor_response.get("accepted_sequence")
        if not isinstance(accepted, int) or accepted != action.sequence:
            return False
        if self._recording_disabled_reason is not None:
            self._dropped_steps += 1
            return False
        if self._worker_error is not None:
            raise RuntimeError("trajectory writer failed") from self._worker_error
        accepted_ns = supervisor_response.get("accepted_monotonic_ns")
        if accepted_ns is not None and (
            not isinstance(accepted_ns, int) or accepted_ns < frame.captured_ns
        ):
            raise ValueError("supervisor acceptance timestamp precedes captured frame")
        encoded_frame = _encode_compact_frame(
            frame,
            max_width=self.frame_max_width,
            jpeg_quality=self.frame_jpeg_quality,
        )
        frame_hash = hashlib.sha256(encoded_frame.payload).hexdigest()
        compact_blackboard = _compact_blackboard_without_exact_evidence(
            blackboard,
            width=encoded_frame.width,
            height=encoded_frame.height,
        )
        blackboard_json = compact_blackboard.model_dump_json().encode()
        blackboard_hash = hashlib.sha256(blackboard_json).hexdigest()
        condition_skill_id = _condition_text(provenance.condition, "skill_id")
        condition_run_id = _condition_text(provenance.condition, "episode_id")
        if skill_id is not None and condition_skill_id not in {None, skill_id}:
            raise ValueError("skill_id does not match the accepted action condition")
        if skill_run_id is not None and condition_run_id not in {None, skill_run_id}:
            raise ValueError("skill_run_id does not match the accepted action condition")
        accepted_skill_id = skill_id if skill_id is not None else condition_skill_id
        accepted_skill_run_id = skill_run_id if skill_run_id is not None else condition_run_id
        sample_key = f"{self._step_index:012d}"
        shard_id = f"{self.manifest.trajectory_id}-shard-{self._step_index // self.shard_steps:06d}"
        step = TrajectoryStep(
            trajectory_id=self.manifest.trajectory_id,
            step_index=self._step_index,
            captured_ns=frame.captured_ns,
            accepted_ns=accepted_ns,
            frame_ref=f"wds://{self.manifest.trajectory_id}/{shard_id}.tar#{sample_key}.frame.jpg",
            frame_hash=frame_hash,
            previous_action=self._previous_action,
            dropped_steps_before=self._pending_dropped_steps,
            action=action,
            action_level=provenance.action_level,
            behavior_token=provenance.behavior_token,
            latent_id=provenance.latent_id,
            action_origin=provenance.origin,
            policy_id=provenance.policy_id,
            model_version=provenance.model_version,
            route_id=provenance.route_id,
            policy_action_kind=provenance.policy_action_kind,
            policy_request_id=provenance.policy_request_id,
            prediction_id=provenance.prediction_id,
            condition_id=provenance.condition_id,
            condition=provenance.condition,
            target_track_id=provenance.target_track_id,
            skill_run_id=accepted_skill_run_id,
            skill_id=accepted_skill_id,
            goal_id=goal_id,
            plan_node_id=plan_node_id,
            blackboard_snapshot_ref=(
                f"wds://{self.manifest.trajectory_id}/{shard_id}.tar"
                f"?sha256={blackboard_hash}#{sample_key}.blackboard.json"
            ),
            reward_signals={} if reward_signals is None else reward_signals,
            event_ids=event_ids,
            correction_of_step=correction_of_step,
        )
        sample = AcceptedTrajectorySample(
            step=step,
            frame=encoded_frame,
            blackboard_json=blackboard_json,
        )
        try:
            self._queue.put_nowait(sample)
        except queue.Full:
            self._dropped_steps += 1
            self._pending_dropped_steps += 1
            self._previous_action = action
            return False
        self._step_index += 1
        self._pending_dropped_steps = 0
        self._previous_action = action
        return True

    def close(self, *, timeout_s: float = 15.0) -> TrajectoryManifest:
        if not self._closed:
            self._closed = True
            deadline = time.monotonic() + max(0.0, timeout_s)
            while True:
                if self._worker_error is not None:
                    raise RuntimeError("trajectory writer failed") from self._worker_error
                if not self._thread.is_alive():
                    raise RuntimeError("trajectory writer exited before close")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("trajectory writer queue did not accept close signal")
                try:
                    self._queue.put(None, timeout=min(0.1, remaining))
                except queue.Full:
                    continue
                break
            self._thread.join(timeout=max(0.0, deadline - time.monotonic()))
        elif self._thread.is_alive():
            self._thread.join(timeout=max(0.0, timeout_s))
        if self._thread.is_alive():
            raise TimeoutError("trajectory writer did not flush before timeout")
        if self._worker_error is not None:
            raise RuntimeError("trajectory writer failed") from self._worker_error
        manifest_path = self.artifact_root / self.manifest.trajectory_id / "manifest.json"
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        return TrajectoryManifest.model_validate(raw)

    def _run_writer(self) -> None:
        try:
            writer = self._writer
            try:
                while True:
                    sample = self._queue.get()
                    if sample is None:
                        break
                    if self._recording_disabled_reason is not None:
                        self._dropped_steps += 1
                        continue
                    try:
                        writer.append(sample)
                    except TrajectoryDiskSpaceError as exc:
                        self._recording_disabled_reason = str(exc)
                        self._dropped_steps += 1
                        _LOG.error(
                            "trajectory recording stopped to preserve disk reserve: %s",
                            exc,
                        )
                    else:
                        self._written_steps += 1
            finally:
                writer.close(
                    accepted_steps=self._written_steps,
                    dropped_steps=self._dropped_steps,
                )
        except BaseException as exc:
            self._worker_error = exc


def trajectory_disk_free_bytes(path: Path) -> int:
    """Return free bytes on the filesystem that will contain ``path``."""

    selected = path.expanduser().resolve()
    while not selected.exists():
        parent = selected.parent
        if parent == selected:
            raise FileNotFoundError(f"cannot resolve a filesystem for {path}")
        selected = parent
    return shutil.disk_usage(selected).free


def require_trajectory_disk_reserve(
    path: Path,
    *,
    minimum_free_bytes: int,
    incoming_bytes: int = 0,
) -> None:
    """Refuse a write that would consume the operator's free-space reserve."""

    required = minimum_free_bytes + max(0, incoming_bytes)
    free = trajectory_disk_free_bytes(path)
    if free < required:
        raise TrajectoryDiskSpaceError(
            f"{free / _GIB:.2f} GiB free; "
            f"{minimum_free_bytes / _GIB:.2f} GiB reserve plus "
            f"{max(0, incoming_bytes) / (1024**2):.2f} MiB pending is required"
        )


def _encode_compact_frame(
    frame: CapturedFrame,
    *,
    max_width: int,
    jpeg_quality: int,
) -> EncodedTrajectoryFrame:
    if frame.width < 1 or frame.height < 1:
        raise ValueError("captured frame dimensions must be positive")
    expected_bytes = frame.width * frame.height * 4
    if len(frame.bgra) != expected_bytes:
        raise ValueError(
            f"captured frame has {len(frame.bgra)} bytes; expected {expected_bytes}"
        )
    width = min(frame.width, max_width)
    height = max(1, round(frame.height * width / frame.width))
    image = Image.frombytes(
        "RGBA",
        (frame.width, frame.height),
        frame.bgra,
        "raw",
        "BGRA",
    )
    if image.size != (width, height):
        image = image.resize((width, height), resample=Image.Resampling.BILINEAR)
    encoded = io.BytesIO()
    image.convert("RGB").save(
        encoded,
        format="JPEG",
        quality=jpeg_quality,
        subsampling=2,
        optimize=False,
        progressive=False,
    )
    payload = encoded.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    header = {
        "schema_version": 2,
        "codec": "jpeg",
        "pixel_format": "RGB",
        "width": width,
        "height": height,
        "source_width": frame.width,
        "source_height": frame.height,
        "encoded_bytes": len(payload),
        "encoded_sha256": digest,
    }
    return EncodedTrajectoryFrame(
        member_suffix="frame.jpg",
        header_json=json.dumps(header, sort_keys=True, separators=(",", ":")).encode(),
        payload=payload,
        width=width,
        height=height,
    )


def _compact_blackboard_without_exact_evidence(
    blackboard: FrameState,
    *,
    width: int,
    height: int,
) -> FrameState:
    """Keep semantics but remove exact-pixel claims from lossy/downscaled frames."""

    return blackboard.model_copy(
        update={
            "width": width,
            "height": height,
            "facts": tuple(
                fact.model_copy(update={"evidence_refs": ()}) for fact in blackboard.facts
            ),
            "tracks": tuple(
                track.model_copy(update={"evidence_refs": ()}) for track in blackboard.tracks
            ),
            "chat": tuple(
                line.model_copy(update={"evidence_refs": ()}) for line in blackboard.chat
            ),
            "evidence": (),
        }
    )


def _validate_compact_blackboard_evidence(
    blackboard: FrameState,
    step_index: int,
) -> None:
    """Reject exact-pixel evidence that cannot be reproduced from lossy JPEG."""

    referenced = any(fact.evidence_refs for fact in blackboard.facts)
    referenced = referenced or any(track.evidence_refs for track in blackboard.tracks)
    referenced = referenced or any(line.evidence_refs for line in blackboard.chat)
    if blackboard.evidence or referenced:
        raise ValueError(
            f"step {step_index} compact frame claims unavailable exact pixel evidence"
        )


def new_trajectory_id(prefix: str = "trajectory") -> str:
    return f"{prefix}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:12]}"


def motor_condition_id(
    condition: MotorIntent | dict[str, Any],
    *,
    route_id: str,
    target_track_id: str | None,
) -> str:
    """Content identity for the complete stable condition routed to a motor policy."""

    serialized = (
        condition.model_dump(mode="json") if isinstance(condition, MotorIntent) else condition
    )
    payload = {
        "condition": serialized,
        "route_id": route_id,
        "target_track_id": target_track_id,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _condition_text(condition: dict[str, Any] | None, key: str) -> str | None:
    if condition is None:
        return None
    value = condition.get(key)
    return value if isinstance(value, str) and value else None


def _validate_condition_links(
    condition: dict[str, Any],
    *,
    action_level: ActionLevel,
    target_track_id: str | None,
) -> None:
    reported_level = condition.get("action_level")
    if isinstance(reported_level, str) and reported_level != action_level.value:
        raise ValueError("action_level does not match the serialized motor condition")
    target = condition.get("target_track")
    reported_target = target.get("track_id") if isinstance(target, dict) else None
    if isinstance(reported_target, str) and reported_target != target_track_id:
        raise ValueError("target_track_id does not match the serialized motor condition")


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
