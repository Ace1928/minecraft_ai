from __future__ import annotations

import hashlib
import tarfile
import time
from pathlib import Path

from minecraft_ai.datasets import (
    DatasetSource,
    DatasetSourceType,
    TrajectoryManifest,
)
from minecraft_ai.perception import FrameState
from minecraft_ai.platforms.bedrock_x11 import CapturedFrame
from minecraft_ai.safety import MotorAction
from minecraft_ai.storage import StateDatabase
from minecraft_ai.trajectory import TrajectoryReader, TrajectoryRecorder


def _manifest(trajectory_id: str) -> TrajectoryManifest:
    return TrajectoryManifest(
        trajectory_id=trajectory_id,
        source=DatasetSource(
            source_id="minecraft-ai:test",
            source_type=DatasetSourceType.SYNTHETIC,
            license="CC0-1.0",
            redistribution_allowed=True,
            training_allowed=True,
            edition="bedrock",
            game_versions=("1.test",),
        ),
        role="generalist",
        label="deterministic-fixture",
        game_version="1.test",
        platform="pytest",
        launcher_profile="fixture",
        resolution=(4, 3),
        started_ns=time.time_ns(),
    )


def test_records_only_supervisor_accepted_actions_into_aligned_shards(tmp_path: Path) -> None:
    trajectory_id = "trajectory-test"
    db_path = tmp_path / "state.sqlite3"
    recorder = TrajectoryRecorder(
        manifest=_manifest(trajectory_id),
        artifact_root=tmp_path / "trajectories",
        state_db_path=db_path,
        shard_steps=2,
        queue_size=32,
    )
    for index in range(3):
        captured_ns = time.monotonic_ns() + index
        pixels = bytes((index, 2, 3, 255)) * 12
        frame = CapturedFrame(
            frame_id=index,
            captured_ns=captured_ns,
            width=4,
            height=3,
            bgra=pixels,
        )
        blackboard = FrameState(
            frame_id=index,
            captured_ns=captured_ns,
            instance_id="bedrock:test",
            width=4,
            height=3,
        )
        action = MotorAction(sequence=index, keys_down=("w",) if index == 0 else ())
        assert recorder.record_accepted(
            action=action,
            supervisor_response={
                "accepted_sequence": index,
                "accepted_monotonic_ns": captured_ns + 5_000_000,
            },
            frame=frame,
            blackboard=blackboard,
            skill_id="explore_forward",
        )
    rejected = MotorAction(sequence=99, buttons_down=("left",))
    assert not recorder.record_accepted(
        action=rejected,
        supervisor_response={"accepted_sequence": 98},
        frame=frame,
        blackboard=blackboard,
    )

    manifest = recorder.close()

    assert manifest.accepted_steps == 3
    assert manifest.dropped_steps == 0
    assert len(manifest.shard_ids) == 2
    shard_paths = sorted((tmp_path / "trajectories" / trajectory_id).glob("*.tar"))
    assert len(shard_paths) == 2
    with tarfile.open(shard_paths[0]) as shard:
        names = set(shard.getnames())
    assert "000000000000.step.json" in names
    assert "000000000000.frame.bgra.zlib" in names
    assert "000000000001.blackboard.json" in names

    with StateDatabase(db_path) as database:
        steps = database.connection.execute(
            "SELECT step_index, frame_hash FROM trajectory_steps_index "
            "WHERE trajectory_id=? ORDER BY step_index",
            (trajectory_id,),
        ).fetchall()
        shards = database.connection.execute(
            "SELECT sha256, path FROM trajectory_shards WHERE trajectory_id=? ORDER BY shard_id",
            (trajectory_id,),
        ).fetchall()
    assert [row[0] for row in steps] == [0, 1, 2]
    assert steps[0][1] == hashlib.sha256(bytes((0, 2, 3, 255)) * 12).hexdigest()
    assert all(
        hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for digest, path in shards
    )

    reader = TrajectoryReader(tmp_path / "trajectories" / trajectory_id)
    replay = list(reader.iter_samples())
    report = reader.validate()

    assert [sample.step.action.sequence for sample in replay] == [0, 1, 2]
    assert replay[1].step.previous_action == replay[0].step.action
    assert replay[2].frame.bgra == bytes((2, 2, 3, 255)) * 12
    assert replay[0].blackboard.frame_id == 0
    assert report.valid
    assert report.step_count == 3
    assert report.shard_count == 2
    assert report.frame_action_latency_ms_p50 == 5.0
    assert report.frame_action_latency_ms_p95 == 5.0


def test_replay_validation_rejects_frame_corruption(tmp_path: Path) -> None:
    trajectory_id = "trajectory-corrupt"
    recorder = TrajectoryRecorder(
        manifest=_manifest(trajectory_id),
        artifact_root=tmp_path / "trajectories",
        state_db_path=tmp_path / "state.sqlite3",
        shard_steps=16,
    )
    captured_ns = time.monotonic_ns()
    frame = CapturedFrame(
        frame_id=0,
        captured_ns=captured_ns,
        width=4,
        height=3,
        bgra=bytes((1, 2, 3, 255)) * 12,
    )
    blackboard = FrameState(
        frame_id=0,
        captured_ns=captured_ns,
        instance_id="bedrock:test",
        width=4,
        height=3,
    )
    assert recorder.record_accepted(
        action=MotorAction(sequence=0, keys_down=("w",)),
        supervisor_response={
            "accepted_sequence": 0,
            "accepted_monotonic_ns": captured_ns + 1,
        },
        frame=frame,
        blackboard=blackboard,
    )
    manifest = recorder.close()
    shard = tmp_path / "trajectories" / trajectory_id / f"{manifest.shard_ids[0]}.tar"
    with shard.open("ab") as handle:
        handle.write(b"corruption")

    report = TrajectoryReader(shard.parent).validate()

    assert not report.valid
    assert "shard size mismatch" in report.errors[0]
