from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .schema import TrajectoryManifest, TrajectoryShardManifest

if TYPE_CHECKING:
    from ..trajectory import AcceptedTrajectorySample


@dataclass(frozen=True)
class ShardArtifact:
    shard_id: str
    trajectory_id: str
    path: Path
    sha256: str
    first_step_index: int
    last_step_index: int
    step_count: int
    bytes: int


@dataclass
class TrajectoryShardWriter:
    manifest: TrajectoryManifest
    artifact_root: Path
    state_db_path: Path
    max_steps: int = 256
    minimum_free_bytes: int = 0
    _directory: Path = field(init=False)
    _tar: tarfile.TarFile | None = field(default=None, init=False)
    _staged_path: Path | None = field(default=None, init=False)
    _final_path: Path | None = field(default=None, init=False)
    _shard_id: str | None = field(default=None, init=False)
    _shard_index: int = field(default=0, init=False)
    _samples: list[AcceptedTrajectorySample] = field(default_factory=list, init=False)
    _artifacts: list[ShardArtifact] = field(default_factory=list, init=False)
    _write_failure: Exception | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        self._directory = self.artifact_root / self.manifest.trajectory_id
        self._directory.mkdir(parents=True, exist_ok=True)
        self._write_manifest(self.manifest)
        from ..storage import StateDatabase

        with StateDatabase(self.state_db_path) as database:
            database.save_trajectory_manifest(self.manifest)

    def append(self, sample: AcceptedTrajectorySample) -> None:
        if self._write_failure is not None:
            message = "trajectory shard writer is poisoned by a prior write failure"
            raise RuntimeError(message) from self._write_failure
        if sample.step.trajectory_id != self.manifest.trajectory_id:
            raise ValueError("sample trajectory identity does not match writer")
        if len(self._samples) >= self.max_steps:
            self._seal_shard()
        key = f"{sample.step.step_index:012d}"
        payloads = (
            (f"{key}.step.json", sample.step.model_dump_json().encode()),
            (f"{key}.frame.json", sample.frame.header_json),
            (f"{key}.{sample.frame.member_suffix}", sample.frame.payload),
            (f"{key}.blackboard.json", sample.blackboard_json),
        )
        # Tar headers, block rounding, and the end marker are small but real.
        # Reserving an extra 64 KiB makes the pre-write guard conservative.
        incoming_bytes = sum(len(payload) for _, payload in payloads) + 64 * 1024
        from ..trajectory import require_trajectory_disk_reserve

        require_trajectory_disk_reserve(
            self._directory,
            minimum_free_bytes=self.minimum_free_bytes,
            incoming_bytes=incoming_bytes,
        )
        try:
            if self._tar is None:
                self._open_shard()
            assert self._tar is not None
            for name, payload in payloads:
                self._add_bytes(name, payload)
        except Exception as exc:
            self._write_failure = exc
            self._discard_open_shard()
            raise
        self._samples.append(sample)

    def close(self, *, accepted_steps: int, dropped_steps: int) -> None:
        if self._write_failure is not None:
            self._discard_open_shard()
            raise RuntimeError(
                "cannot finalize trajectory manifest after a shard write failure"
            ) from self._write_failure
        if self._tar is not None:
            self._seal_shard()
        completed = self.manifest.model_copy(
            update={
                "ended_ns": time.time_ns(),
                "accepted_steps": accepted_steps,
                "dropped_steps": dropped_steps,
                "shard_ids": tuple(artifact.shard_id for artifact in self._artifacts),
                "shards": tuple(
                    TrajectoryShardManifest(
                        shard_id=artifact.shard_id,
                        filename=artifact.path.name,
                        sha256=artifact.sha256,
                        first_step_index=artifact.first_step_index,
                        last_step_index=artifact.last_step_index,
                        step_count=artifact.step_count,
                        bytes=artifact.bytes,
                    )
                    for artifact in self._artifacts
                ),
            }
        )
        self._write_manifest(completed)
        from ..storage import StateDatabase

        with StateDatabase(self.state_db_path) as database:
            database.save_trajectory_manifest(completed)

    def _open_shard(self) -> None:
        shard_id = f"{self.manifest.trajectory_id}-shard-{self._shard_index:06d}"
        final_path = self._directory / f"{shard_id}.tar"
        staged_path = final_path.with_name(f".{final_path.name}.{os.getpid()}.tmp")
        self._tar = tarfile.open(staged_path, mode="w")
        self._staged_path = staged_path
        self._final_path = final_path
        self._shard_id = shard_id
        self._samples = []

    def _seal_shard(self) -> None:
        assert self._tar is not None
        assert self._staged_path is not None
        assert self._final_path is not None
        assert self._shard_id is not None
        if not self._samples:
            self._tar.close()
            self._staged_path.unlink(missing_ok=True)
            self._reset_open_shard()
            return
        renamed = False
        try:
            self._tar.close()
            self._staged_path.replace(self._final_path)
            renamed = True
            digest = hashlib.sha256(self._final_path.read_bytes()).hexdigest()
            artifact = ShardArtifact(
                shard_id=self._shard_id,
                trajectory_id=self.manifest.trajectory_id,
                path=self._final_path,
                sha256=digest,
                first_step_index=self._samples[0].step.step_index,
                last_step_index=self._samples[-1].step.step_index,
                step_count=len(self._samples),
                bytes=self._final_path.stat().st_size,
            )
            from ..storage import StateDatabase

            with StateDatabase(self.state_db_path) as database:
                # Publish the shard and every contained frame index in one short
                # transaction. Per-frame commits caused a burst of 257 competing
                # writer acquisitions and could starve the realtime operator path.
                with database.connection:
                    database.save_trajectory_shard(
                        shard_id=artifact.shard_id,
                        trajectory_id=artifact.trajectory_id,
                        path=str(artifact.path),
                        sha256=artifact.sha256,
                        first_step_index=artifact.first_step_index,
                        last_step_index=artifact.last_step_index,
                        step_count=artifact.step_count,
                        bytes_count=artifact.bytes,
                        commit=False,
                    )
                    for sample in self._samples:
                        database.save_trajectory_step_index(
                            trajectory_id=self.manifest.trajectory_id,
                            step_index=sample.step.step_index,
                            captured_ns=sample.step.captured_ns,
                            accepted_ns=sample.step.accepted_ns,
                            shard_id=artifact.shard_id,
                            sample_key=f"{sample.step.step_index:012d}",
                            frame_hash=sample.step.frame_hash,
                            action_json=sample.step.action.model_dump_json(),
                            action_level=sample.step.action_level.value,
                            action_origin=sample.step.action_origin.value,
                            policy_id=sample.step.policy_id,
                            model_version=sample.step.model_version,
                            route_id=sample.step.route_id,
                            policy_action_kind=sample.step.policy_action_kind,
                            policy_request_id=sample.step.policy_request_id,
                            prediction_id=sample.step.prediction_id,
                            condition_id=sample.step.condition_id,
                            condition_json=(
                                None
                                if sample.step.condition is None
                                else json.dumps(
                                    sample.step.condition,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                )
                            ),
                            behavior_token=sample.step.behavior_token,
                            latent_id=sample.step.latent_id,
                            target_track_id=sample.step.target_track_id,
                            skill_run_id=sample.step.skill_run_id,
                            skill_id=sample.step.skill_id,
                            goal_id=sample.step.goal_id,
                            plan_node_id=sample.step.plan_node_id,
                            correction_of_step=sample.step.correction_of_step,
                            commit=False,
                        )
        except Exception as exc:
            self._write_failure = exc
            try:
                self._tar.close()
            except (OSError, tarfile.TarError):
                pass
            self._staged_path.unlink(missing_ok=True)
            if renamed:
                self._final_path.unlink(missing_ok=True)
            self._reset_open_shard()
            raise
        self._artifacts.append(artifact)
        self._shard_index += 1
        self._reset_open_shard()

    def _reset_open_shard(self) -> None:
        self._tar = None
        self._staged_path = None
        self._final_path = None
        self._shard_id = None
        self._samples = []

    def _discard_open_shard(self) -> None:
        archive = self._tar
        staged_path = self._staged_path
        try:
            if archive is not None:
                archive.close()
        finally:
            try:
                if staged_path is not None:
                    staged_path.unlink(missing_ok=True)
            finally:
                self._reset_open_shard()

    def _add_bytes(self, name: str, payload: bytes) -> None:
        assert self._tar is not None
        info = tarfile.TarInfo(name=name)
        info.size = len(payload)
        info.mtime = int(time.time())
        info.mode = 0o600
        self._tar.addfile(info, io.BytesIO(payload))

    def _write_manifest(self, manifest: TrajectoryManifest) -> None:
        path = self._directory / "manifest.json"
        staged = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        staged.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        staged.replace(path)
