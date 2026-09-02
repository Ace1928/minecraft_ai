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
from minecraft_ai.trajectory import TrajectoryRecorder


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
            supervisor_response={"accepted_sequence": index},
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
