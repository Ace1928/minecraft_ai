from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from minecraft_ai.datasets import (
    ActionLevel,
    DatasetSource,
    DatasetSourceType,
    TrajectoryManifest,
    TrajectoryShardWriter,
)
from minecraft_ai.storage import StateDatabase
from minecraft_ai.safety import MotorAction
from minecraft_ai.trajectory import (
    AcceptedTrajectorySample,
    EncodedTrajectoryFrame,
    TrajectoryStep,
)


def _manifest(trajectory_id: str) -> TrajectoryManifest:
    return TrajectoryManifest(
        trajectory_id=trajectory_id,
        source=DatasetSource(
            source_id="minecraft-ai:shard-atomicity-test",
            source_type=DatasetSourceType.SYNTHETIC,
            license="CC0-1.0",
            redistribution_allowed=True,
            training_allowed=True,
            edition="bedrock",
            game_versions=("1.test",),
        ),
        role="generalist",
        label="injected-write-failure",
        game_version="1.test",
        platform="pytest",
        launcher_profile="fixture",
        resolution=(4, 3),
        started_ns=1,
    )


def _sample(trajectory_id: str, step_index: int) -> AcceptedTrajectorySample:
    key = f"{step_index:012d}"
    return AcceptedTrajectorySample(
        step=TrajectoryStep(
            trajectory_id=trajectory_id,
            step_index=step_index,
            captured_ns=100 + step_index,
            frame_ref=f"wds://{trajectory_id}/pending.tar#{key}.frame.jpg",
            frame_hash=f"{step_index:064x}",
            action=MotorAction(sequence=step_index),
            action_level=ActionLevel.RAW,
            blackboard_snapshot_ref=(
                f"wds://{trajectory_id}/pending.tar#{key}.blackboard.json"
            ),
        ),
        frame=EncodedTrajectoryFrame(
            member_suffix="frame.jpg",
            header_json=b'{"codec":"jpeg","width":4,"height":3}',
            payload=b"synthetic-jpeg-payload",
            width=4,
            height=3,
        ),
        blackboard_json=b'{"frame_id":0}',
    )


def test_mid_sample_failure_discards_staged_shard_and_refuses_manifest_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory_id = "trajectory-mid-sample-failure"
    artifact_root = tmp_path / "trajectories"
    state_db_path = tmp_path / "state.sqlite3"
    writer = TrajectoryShardWriter(
        manifest=_manifest(trajectory_id),
        artifact_root=artifact_root,
        state_db_path=state_db_path,
        max_steps=16,
    )
    writer.append(_sample(trajectory_id, 0))

    original_add_bytes = writer._add_bytes
    writes = 0

    def fail_during_second_sample(name: str, payload: bytes) -> None:
        nonlocal writes
        writes += 1
        if writes == 3:
            raise OSError("injected tar member write failure")
        original_add_bytes(name, payload)

    monkeypatch.setattr(writer, "_add_bytes", fail_during_second_sample)

    with pytest.raises(OSError, match="injected tar member write failure"):
        writer.append(_sample(trajectory_id, 1))

    directory = artifact_root / trajectory_id
    assert list(directory.glob("*.tar")) == []
    assert list(directory.glob(".*.tmp")) == []
    with pytest.raises(RuntimeError, match="cannot finalize trajectory manifest"):
        writer.close(accepted_steps=2, dropped_steps=0)

    persisted = TrajectoryManifest.model_validate_json(
        (directory / "manifest.json").read_text(encoding="utf-8")
    )
    assert persisted.ended_ns is None
    assert persisted.accepted_steps == 0
    assert persisted.shards == ()

    with sqlite3.connect(state_db_path) as connection:
        row = connection.execute(
            "SELECT ended_ns, payload FROM trajectories WHERE trajectory_id=?",
            (trajectory_id,),
        ).fetchone()
        shard_count = connection.execute(
            "SELECT COUNT(*) FROM trajectory_shards WHERE trajectory_id=?",
            (trajectory_id,),
        ).fetchone()
    assert row is not None and row[0] is None
    assert TrajectoryManifest.model_validate(json.loads(row[1])).ended_ns is None
    assert shard_count == (0,)


def test_post_rename_database_failure_removes_unpublished_shard_and_poisons_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory_id = "trajectory-index-publication-failure"
    artifact_root = tmp_path / "trajectories"
    state_db_path = tmp_path / "state.sqlite3"
    writer = TrajectoryShardWriter(
        manifest=_manifest(trajectory_id),
        artifact_root=artifact_root,
        state_db_path=state_db_path,
        max_steps=1,
    )
    writer.append(_sample(trajectory_id, 0))

    def fail_publication(self: StateDatabase, **_kwargs: object) -> None:
        raise sqlite3.OperationalError("injected shard index failure")

    monkeypatch.setattr(StateDatabase, "save_trajectory_shard", fail_publication)

    with pytest.raises(sqlite3.OperationalError, match="injected shard index failure"):
        writer.append(_sample(trajectory_id, 1))

    directory = artifact_root / trajectory_id
    assert list(directory.glob("*.tar")) == []
    assert list(directory.glob(".*.tmp")) == []
    with pytest.raises(RuntimeError, match="poisoned"):
        writer.append(_sample(trajectory_id, 1))
    with pytest.raises(RuntimeError, match="cannot finalize trajectory manifest"):
        writer.close(accepted_steps=1, dropped_steps=0)

    persisted = TrajectoryManifest.model_validate_json(
        (directory / "manifest.json").read_text(encoding="utf-8")
    )
    assert persisted.ended_ns is None
    assert persisted.shards == ()
    with sqlite3.connect(state_db_path) as connection:
        shard_count = connection.execute(
            "SELECT COUNT(*) FROM trajectory_shards WHERE trajectory_id=?",
            (trajectory_id,),
        ).fetchone()
        step_count = connection.execute(
            "SELECT COUNT(*) FROM trajectory_steps_index WHERE trajectory_id=?",
            (trajectory_id,),
        ).fetchone()
    assert shard_count == (0,)
    assert step_count == (0,)
