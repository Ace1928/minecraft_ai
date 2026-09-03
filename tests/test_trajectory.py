from __future__ import annotations

import hashlib
import json
import tarfile
import time
from pathlib import Path

import pytest

from minecraft_ai.datasets import (
    ActionLevel,
    DatasetSource,
    DatasetSourceType,
    TrajectoryManifest,
)
from minecraft_ai.motor import MotorIntent
from minecraft_ai.perception import FrameState
from minecraft_ai.platforms.bedrock_x11 import CapturedFrame
from minecraft_ai.safety import MotorAction
from minecraft_ai.storage import StateDatabase
from minecraft_ai.trajectory import (
    ActionOrigin,
    ActionProvenance,
    TrajectoryReader,
    TrajectoryRecorder,
    TrajectoryStep,
    motor_condition_id,
)


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


def _learned_provenance() -> ActionProvenance:
    condition = MotorIntent(
        skill_id="mine_log",
        mode="mine",
        episode_id="skill-run-42",
        action_level=ActionLevel.GROUNDED,
        instruction="mine log",
        condition_scale=5.5,
        target_label="oak_log",
    )
    route_id = "semantic"
    target_track_id = "operator:oak-log"
    return ActionProvenance(
        policy_id="learned:minestudio-steve1:steve1-1x",
        model_version="steve1-1x",
        route_id=route_id,
        policy_action_kind="prediction",
        policy_request_id="request-42",
        prediction_id="prediction-42",
        action_level=ActionLevel.GROUNDED,
        origin=ActionOrigin.POLICY,
        condition_id=motor_condition_id(
            condition,
            route_id=route_id,
            target_track_id=target_track_id,
        ),
        condition=condition.model_dump(mode="json"),
        behavior_token=41,
        latent_id="z_041",
        target_track_id=target_track_id,
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
            provenance=_learned_provenance(),
            supervisor_response={
                "accepted_sequence": index,
                "accepted_monotonic_ns": captured_ns + 5_000_000,
            },
            frame=frame,
            blackboard=blackboard,
            skill_run_id="skill-run-42",
            skill_id="mine_log",
        )
    rejected = MotorAction(sequence=99, buttons_down=("left",))
    assert not recorder.record_accepted(
        action=rejected,
        provenance=_learned_provenance(),
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
            "SELECT step_index, frame_hash, action_origin, policy_id, model_version, "
            "route_id, policy_action_kind, policy_request_id, prediction_id, "
            "condition_id, condition_json, behavior_token, latent_id, "
            "target_track_id, skill_run_id, skill_id FROM trajectory_steps_index "
            "WHERE trajectory_id=? ORDER BY step_index",
            (trajectory_id,),
        ).fetchall()
        shards = database.connection.execute(
            "SELECT sha256, path FROM trajectory_shards WHERE trajectory_id=? ORDER BY shard_id",
            (trajectory_id,),
        ).fetchall()
    assert [row[0] for row in steps] == [0, 1, 2]
    assert steps[0][1] == hashlib.sha256(bytes((0, 2, 3, 255)) * 12).hexdigest()
    expected = _learned_provenance()
    assert steps[0][2:] == (
        ActionOrigin.POLICY.value,
        expected.policy_id,
        expected.model_version,
        expected.route_id,
        expected.policy_action_kind,
        expected.policy_request_id,
        expected.prediction_id,
        expected.condition_id,
        (
            json.dumps(expected.condition, sort_keys=True, separators=(",", ":"))
            if expected.condition is not None
            else None
        ),
        expected.behavior_token,
        expected.latent_id,
        expected.target_track_id,
        "skill-run-42",
        "mine_log",
    )
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
    assert replay[0].step.action_origin == ActionOrigin.POLICY
    assert replay[0].step.policy_id == expected.policy_id
    assert replay[0].step.model_version == expected.model_version
    assert replay[0].step.route_id == expected.route_id
    assert replay[0].step.policy_action_kind == "prediction"
    assert replay[0].step.policy_request_id == "request-42"
    assert replay[0].step.prediction_id == "prediction-42"
    assert replay[0].step.condition_id == expected.condition_id
    assert replay[0].step.condition == expected.condition
    assert replay[0].step.behavior_token == 41
    assert replay[0].step.latent_id == "z_041"
    assert replay[0].step.target_track_id == "operator:oak-log"
    assert replay[0].step.skill_run_id == "skill-run-42"
    assert replay[0].step.skill_id == "mine_log"
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
        provenance=ActionProvenance(
            policy_id="synthetic:test",
            route_id="synthetic",
            action_level=ActionLevel.RAW,
            origin=ActionOrigin.SYNTHETIC,
        ),
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


def test_legacy_step_without_provenance_remains_replay_compatible() -> None:
    legacy = TrajectoryStep.model_validate(
        {
            "trajectory_id": "legacy",
            "step_index": 0,
            "captured_ns": 1,
            "frame_ref": "wds://legacy/shard.tar#0.frame",
            "frame_hash": "0" * 64,
            "action": {"sequence": 0},
            "action_level": "raw",
            "blackboard_snapshot_ref": "wds://legacy/shard.tar#0.blackboard",
        }
    )

    assert legacy.action_origin == ActionOrigin.LEGACY
    assert legacy.policy_id is None
    assert legacy.route_id is None
    assert legacy.condition_id is None


def test_condition_identity_rejects_mismatched_serialized_condition() -> None:
    condition = MotorIntent(
        skill_id="mine_log",
        mode="mine",
        action_level=ActionLevel.GROUNDED,
    )

    with pytest.raises(ValueError, match="condition_id"):
        ActionProvenance(
            policy_id="learned:test",
            route_id="semantic",
            action_level=ActionLevel.GROUNDED,
            origin=ActionOrigin.POLICY,
            condition_id="0" * 64,
            condition=condition.model_dump(mode="json"),
        )
