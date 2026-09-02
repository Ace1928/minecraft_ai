from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .datasets.schema import ActionLevel, TrajectoryManifest
from .datasets.shards import TrajectoryShardWriter
from .perception import FrameState
from .platforms.bedrock_x11 import CapturedFrame
from .safety import MotorAction


class TrajectoryStep(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    trajectory_id: str
    step_index: int = Field(ge=0)
    captured_ns: int
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
        frame_hash = hashlib.sha256(frame.bgra).hexdigest()
        blackboard_json = blackboard.model_dump_json().encode()
        blackboard_hash = hashlib.sha256(blackboard_json).hexdigest()
        sample_key = f"{self.manifest.trajectory_id}/{self._step_index:012d}"
        shard_id = f"{self.manifest.trajectory_id}-shard-{self._step_index // self.shard_steps:06d}"
        step = TrajectoryStep(
            trajectory_id=self.manifest.trajectory_id,
            step_index=self._step_index,
            captured_ns=frame.captured_ns,
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
