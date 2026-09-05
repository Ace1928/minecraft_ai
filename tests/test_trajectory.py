from __future__ import annotations

import hashlib
import io
import json
import queue
import random
import tarfile
import threading
import time
import zlib
from pathlib import Path
from urllib.parse import urlparse

import pytest

from minecraft_ai.datasets import (
    ActionLevel,
    DatasetSource,
    DatasetSourceType,
    TrajectoryManifest,
    TrajectoryShardWriter,
)
from minecraft_ai.motor import MotorIntent
from minecraft_ai.perception import (
    ChatLine,
    EvidenceRegion,
    FrameState,
    PerceptionEvidence,
    PerceptionFact,
    ScreenRegion,
    Track,
)
from minecraft_ai.platforms.bedrock_x11 import CapturedFrame
from minecraft_ai.safety import MotorAction
from minecraft_ai.storage import StateDatabase
from minecraft_ai.trajectory import (
    AcceptedTrajectorySample,
    ActionOrigin,
    ActionProvenance,
    TrajectoryDiskSpaceError,
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
        min_free_disk_bytes=0,
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
        first_step_file = shard.extractfile("000000000000.step.json")
        assert first_step_file is not None
        first_step = TrajectoryStep.model_validate_json(first_step_file.read())
        first_frame = shard.extractfile("000000000000.frame.jpg")
        assert first_frame is not None
        first_frame_payload = first_frame.read()
        first_header_file = shard.extractfile("000000000000.frame.json")
        assert first_header_file is not None
        first_header = json.loads(first_header_file.read())
    assert "000000000000.step.json" in names
    assert "000000000000.frame.jpg" in names
    assert "000000000001.blackboard.json" in names
    assert urlparse(first_step.frame_ref).fragment in names
    assert urlparse(first_step.blackboard_snapshot_ref).fragment in names
    assert first_header["codec"] == "jpeg"
    assert first_header["encoded_sha256"] == hashlib.sha256(first_frame_payload).hexdigest()

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
    assert steps[0][1] == hashlib.sha256(first_frame_payload).hexdigest()
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
    assert replay[2].frame.width == 4
    assert replay[2].frame.height == 3
    assert len(replay[2].frame.bgra) == 4 * 3 * 4
    assert replay[2].frame.bgra[3::4] == bytes((255,)) * 12
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


def test_source_frame_roundtrip_preserves_distinct_action_time_observation(tmp_path: Path) -> None:
    trajectory_id = "distinct-source-and-action-frames"
    source_ns = 1_000_000_000
    action_ns = 1_300_000_000
    accepted_ns = action_ns + 7_000_000
    provenance = ActionProvenance.model_validate(
        {
            **_learned_provenance().model_dump(),
            "source_frame_id": 11,
            "source_captured_ns": source_ns,
        }
    )
    frame = CapturedFrame(
        frame_id=29,
        captured_ns=action_ns,
        width=4,
        height=3,
        bgra=bytes((7, 80, 150, 255)) * 12,
    )
    # Blackboard identity is deliberately neither the source nor action frame.
    blackboard = FrameState(
        frame_id=777,
        captured_ns=action_ns,
        instance_id="bedrock:test",
        width=4,
        height=3,
    )
    recorder = TrajectoryRecorder(
        manifest=_manifest(trajectory_id),
        artifact_root=tmp_path / "trajectories",
        state_db_path=tmp_path / "state.sqlite3",
        min_free_disk_bytes=0,
    )
    try:
        assert recorder.record_accepted(
            action=MotorAction(sequence=42, keys_down=("w",)),
            provenance=provenance,
            supervisor_response={
                "accepted_sequence": 42,
                "accepted_monotonic_ns": accepted_ns,
            },
            frame=frame,
            blackboard=blackboard,
        )
    finally:
        recorder.close()

    directory = tmp_path / "trajectories" / trajectory_id
    shard_path = next(directory.glob("*.tar"))
    with tarfile.open(shard_path) as shard:
        names = set(shard.getnames())
        step_file = shard.extractfile("000000000000.step.json")
        frame_file = shard.extractfile("000000000000.frame.jpg")
        assert step_file is not None and frame_file is not None
        step_payload = json.loads(step_file.read())
        frame_payload = frame_file.read()
    assert names == {
        "000000000000.step.json",
        "000000000000.frame.json",
        "000000000000.frame.jpg",
        "000000000000.blackboard.json",
    }
    assert step_payload["source_frame_id"] == 11
    assert step_payload["source_captured_ns"] == source_ns
    assert step_payload["frame_id"] == 29
    assert step_payload["captured_ns"] == action_ns
    assert step_payload["accepted_ns"] == accepted_ns
    assert step_payload["frame_hash"] == hashlib.sha256(frame_payload).hexdigest()
    assert urlparse(step_payload["frame_ref"]).fragment == "000000000000.frame.jpg"

    reader = TrajectoryReader(directory)
    replay = list(reader.iter_samples())
    assert len(replay) == 1
    sample = replay[0]
    assert sample.frame.frame_id == sample.step.frame_id == 29
    assert sample.frame.captured_ns == sample.step.captured_ns == action_ns
    assert sample.blackboard.frame_id == 777
    assert sample.step.source_frame_id == 11
    assert sample.step.source_captured_ns == source_ns
    assert sample.step.policy_request_id == provenance.policy_request_id
    assert sample.step.prediction_id == provenance.prediction_id
    assert sample.step.skill_run_id == "skill-run-42"
    report = reader.validate()
    assert report.valid
    assert report.frame_action_latency_ms_p50 == 7.0


@pytest.mark.parametrize("origin", [ActionOrigin.LEGACY, ActionOrigin.HUMAN, ActionOrigin.RESET])
def test_source_free_provenance_does_not_invent_inference_frame(origin: ActionOrigin) -> None:
    provenance = ActionProvenance(
        policy_id="source-free:test",
        route_id="direct",
        action_level=ActionLevel.RAW,
        origin=origin,
    )
    serialized = json.loads(provenance.model_dump_json())
    assert serialized["source_frame_id"] is None
    assert serialized["source_captured_ns"] is None
    restored = ActionProvenance.model_validate_json(provenance.model_dump_json())
    assert restored.source_frame_id is restored.source_captured_ns is None


def _source_identity_payload(
    model_type: type[ActionProvenance] | type[TrajectoryStep],
) -> dict[str, object]:
    if model_type is ActionProvenance:
        return _learned_provenance().model_dump()
    return {
        "trajectory_id": "identity-validation",
        "step_index": 0,
        "captured_ns": 20,
        "frame_ref": "wds://identity-validation/shard.tar#0.frame",
        "frame_hash": "0" * 64,
        "action": {"sequence": 0},
        "action_level": "raw",
        "blackboard_snapshot_ref": "wds://identity-validation/shard.tar#0.blackboard",
    }


@pytest.mark.parametrize("model_type", [ActionProvenance, TrajectoryStep])
@pytest.mark.parametrize("field", ["source_frame_id", "source_captured_ns"])
@pytest.mark.parametrize("invalid", [-1, True, "11", 1.5])
def test_source_identity_requires_nonnegative_strict_integers(model_type, field, invalid) -> None:
    payload = {
        **_source_identity_payload(model_type),
        "source_frame_id": 11,
        "source_captured_ns": 10,
        field: invalid,
    }
    with pytest.raises(ValueError, match=field):
        model_type.model_validate(payload)


@pytest.mark.parametrize("model_type", [ActionProvenance, TrajectoryStep])
@pytest.mark.parametrize("field", ["source_frame_id", "source_captured_ns"])
def test_source_identity_rejects_partial_pairs(model_type, field) -> None:
    payload = _source_identity_payload(model_type)
    payload[field] = 11
    with pytest.raises(ValueError):
        model_type.model_validate(payload)


@pytest.mark.parametrize("model_type", [ActionProvenance, TrajectoryStep])
def test_source_identity_accepts_zero_pair_and_serializes_fields(model_type) -> None:
    payload = {
        **_source_identity_payload(model_type),
        "source_frame_id": 0,
        "source_captured_ns": 0,
    }
    parsed = model_type.model_validate_json(json.dumps(payload))
    assert parsed.source_frame_id == parsed.source_captured_ns == 0


@pytest.mark.parametrize("invalid", [-1, True, "29", 1.5])
def test_action_time_frame_identity_requires_nonnegative_strict_integer(invalid) -> None:
    with pytest.raises(ValueError, match="frame_id"):
        TrajectoryStep.model_validate(
            {**_source_identity_payload(TrajectoryStep), "frame_id": invalid}
        )


def test_recorder_registers_trajectory_before_returning(tmp_path: Path) -> None:
    trajectory_id = "trajectory-synchronous-registration"
    db_path = tmp_path / "state.sqlite3"

    recorder = TrajectoryRecorder(
        manifest=_manifest(trajectory_id),
        artifact_root=tmp_path / "trajectories",
        state_db_path=db_path,
        min_free_disk_bytes=0,
    )

    with StateDatabase(db_path) as database:
        registered = database.connection.execute(
            "SELECT trajectory_id FROM trajectories WHERE trajectory_id=?",
            (trajectory_id,),
        ).fetchone()
    assert registered == (trajectory_id,)
    recorder.close()


def test_queue_drop_preserves_physical_previous_action_and_marks_sequence_break(
    tmp_path: Path,
) -> None:
    trajectory_id = "trajectory-queue-discontinuity"

    class _ScriptedQueue:
        def __init__(self) -> None:
            self.calls = 0
            self.samples: list[AcceptedTrajectorySample] = []

        def put_nowait(self, sample: AcceptedTrajectorySample) -> None:
            self.calls += 1
            if self.calls == 2:
                raise queue.Full
            self.samples.append(sample)

    scripted_queue = _ScriptedQueue()
    recorder = object.__new__(TrajectoryRecorder)
    recorder.manifest = _manifest(trajectory_id)
    recorder.shard_steps = 16
    recorder.frame_max_width = 256
    recorder.frame_jpeg_quality = 80
    recorder._queue = scripted_queue  # type: ignore[assignment]
    recorder._step_index = 0
    recorder._previous_action = None
    recorder._pending_dropped_steps = 0
    recorder._dropped_steps = 0
    recorder._closed = False
    recorder._worker_error = None
    recorder._recording_disabled_reason = None

    for sequence, expected_recorded in ((0, True), (1, False), (2, True)):
        captured_ns = time.monotonic_ns() + sequence
        assert (
            recorder.record_accepted(
                action=MotorAction(sequence=sequence, mouse_dx=sequence),
                provenance=_learned_provenance(),
                supervisor_response={"accepted_sequence": sequence},
                frame=CapturedFrame(
                    frame_id=sequence,
                    captured_ns=captured_ns,
                    width=4,
                    height=3,
                    bgra=bytes((sequence, 2, 3, 255)) * 12,
                ),
                blackboard=FrameState(
                    frame_id=sequence,
                    captured_ns=captured_ns,
                    instance_id="bedrock:test",
                    width=4,
                    height=3,
                ),
            )
            is expected_recorded
        )

    assert [sample.step.action.sequence for sample in scripted_queue.samples] == [0, 2]
    resumed = scripted_queue.samples[1].step
    assert resumed.step_index == 1
    assert resumed.dropped_steps_before == 1
    assert resumed.previous_action == MotorAction(sequence=1, mouse_dx=1)

    artifact_root = tmp_path / "trajectories"
    writer = TrajectoryShardWriter(
        manifest=recorder.manifest,
        artifact_root=artifact_root,
        state_db_path=tmp_path / "state.sqlite3",
        max_steps=16,
    )
    for sample in scripted_queue.samples:
        writer.append(sample)
    writer.close(accepted_steps=2, dropped_steps=1)

    reader = TrajectoryReader(artifact_root / trajectory_id)
    replay = tuple(reader.iter_samples())
    assert replay[1].step.previous_action == MotorAction(sequence=1, mouse_dx=1)
    assert replay[1].step.dropped_steps_before == 1
    assert reader.validate().valid


def test_replay_validation_rejects_frame_corruption(tmp_path: Path) -> None:
    trajectory_id = "trajectory-corrupt"
    recorder = TrajectoryRecorder(
        manifest=_manifest(trajectory_id),
        artifact_root=tmp_path / "trajectories",
        state_db_path=tmp_path / "state.sqlite3",
        shard_steps=16,
        min_free_disk_bytes=0,
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


def test_replay_validation_rejects_tampered_blackboard_member(tmp_path: Path) -> None:
    trajectory_id = "trajectory-blackboard-corrupt"
    directory = tmp_path / "trajectories" / trajectory_id
    recorder = TrajectoryRecorder(
        manifest=_manifest(trajectory_id),
        artifact_root=tmp_path / "trajectories",
        state_db_path=tmp_path / "state.sqlite3",
        min_free_disk_bytes=0,
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
        action=MotorAction(sequence=0),
        provenance=ActionProvenance(
            policy_id="synthetic:test",
            route_id="synthetic",
            action_level=ActionLevel.RAW,
            origin=ActionOrigin.SYNTHETIC,
        ),
        supervisor_response={"accepted_sequence": 0},
        frame=frame,
        blackboard=blackboard,
    )
    manifest = recorder.close()
    shard_path = directory / f"{manifest.shard_ids[0]}.tar"
    with tarfile.open(shard_path) as shard:
        payloads = {
            member.name: shard.extractfile(member).read()
            for member in shard.getmembers()
            if member.isfile() and shard.extractfile(member) is not None
        }
    payloads["000000000000.blackboard.json"] = blackboard.model_copy(
        update={"instance_id": "tampered"}
    ).model_dump_json().encode()
    with tarfile.open(shard_path, mode="w") as shard:
        for name, payload in payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            shard.addfile(info, io.BytesIO(payload))
    # Use an open manifest so the member-level blackboard hash is the first
    # integrity boundary under test rather than the sealed whole-shard hash.
    (directory / "manifest.json").write_text(
        _manifest(trajectory_id).model_dump_json(),
        encoding="utf-8",
    )

    report = TrajectoryReader(directory).validate()

    assert not report.valid
    assert "blackboard hash mismatch" in report.errors[0]


def test_compact_frames_are_downscaled_and_materially_smaller(tmp_path: Path) -> None:
    trajectory_id = "trajectory-compact"
    width, height = 640, 360
    pixels = random.Random(1928).randbytes(width * height * 4)
    captured_ns = time.monotonic_ns()
    recorder = TrajectoryRecorder(
        manifest=_manifest(trajectory_id),
        artifact_root=tmp_path / "trajectories",
        state_db_path=tmp_path / "state.sqlite3",
        shard_steps=16,
        frame_max_width=128,
        frame_jpeg_quality=80,
        min_free_disk_bytes=0,
    )
    assert recorder.record_accepted(
        action=MotorAction(sequence=0),
        provenance=ActionProvenance(
            policy_id="synthetic:test",
            route_id="synthetic",
            action_level=ActionLevel.RAW,
            origin=ActionOrigin.SYNTHETIC,
        ),
        supervisor_response={"accepted_sequence": 0},
        frame=CapturedFrame(
            frame_id=0,
            captured_ns=captured_ns,
            width=width,
            height=height,
            bgra=pixels,
        ),
        blackboard=FrameState(
            frame_id=0,
            captured_ns=captured_ns,
            instance_id="bedrock:test",
            width=width,
            height=height,
        ),
    )
    manifest = recorder.close()
    shard_path = tmp_path / "trajectories" / trajectory_id / f"{manifest.shard_ids[0]}.tar"
    with tarfile.open(shard_path) as shard:
        header_file = shard.extractfile("000000000000.frame.json")
        payload_file = shard.extractfile("000000000000.frame.jpg")
        assert header_file is not None and payload_file is not None
        header = json.loads(header_file.read())
        payload = payload_file.read()

    assert (header["source_width"], header["source_height"]) == (width, height)
    assert (header["width"], header["height"]) == (128, 72)
    assert len(payload) * 20 < len(zlib.compress(pixels, level=1))
    replay = next(TrajectoryReader(shard_path.parent).iter_samples())
    assert (replay.frame.width, replay.frame.height) == (128, 72)
    assert (replay.blackboard.width, replay.blackboard.height) == (128, 72)


def test_compact_frame_keeps_semantics_without_unverifiable_pixel_claims(
    tmp_path: Path,
) -> None:
    trajectory_id = "trajectory-compact-evidence"
    captured_ns = time.monotonic_ns()
    evidence = PerceptionEvidence(
        evidence_id="visible-tree",
        frame_id=0,
        captured_ns=captured_ns,
        region_kind=EvidenceRegion.WORLD,
        region=ScreenRegion(x=0.25, y=0.25, width=0.5, height=0.5),
        pixel_sha256=hashlib.sha256(b"exact-original-crop").hexdigest(),
        crop_width=2,
        crop_height=2,
    )
    blackboard = FrameState(
        frame_id=0,
        captured_ns=captured_ns,
        instance_id="bedrock:test",
        width=32,
        height=18,
        facts=(
            PerceptionFact(
                key="scene.tree_visible",
                value=True,
                confidence=0.9,
                observed_ns=captured_ns,
                source="grounded:test",
                evidence_refs=(evidence.evidence_id,),
            ),
        ),
        tracks=(
            Track(
                track_id="tree-1",
                label="tree",
                confidence=0.9,
                region=evidence.region,
                first_seen_ns=captured_ns,
                last_seen_ns=captured_ns,
                evidence_refs=(evidence.evidence_id,),
            ),
        ),
        chat=(
            ChatLine(
                text="hello",
                observed_ns=captured_ns,
                evidence_refs=(evidence.evidence_id,),
            ),
        ),
        evidence=(evidence,),
    )
    recorder = TrajectoryRecorder(
        manifest=_manifest(trajectory_id),
        artifact_root=tmp_path / "trajectories",
        state_db_path=tmp_path / "state.sqlite3",
        min_free_disk_bytes=0,
    )
    assert recorder.record_accepted(
        action=MotorAction(sequence=0),
        provenance=ActionProvenance(
            policy_id="synthetic:test",
            route_id="synthetic",
            action_level=ActionLevel.RAW,
            origin=ActionOrigin.SYNTHETIC,
        ),
        supervisor_response={"accepted_sequence": 0},
        frame=CapturedFrame(
            frame_id=0,
            captured_ns=captured_ns,
            width=32,
            height=18,
            bgra=bytes((1, 2, 3, 255)) * (32 * 18),
        ),
        blackboard=blackboard,
    )
    recorder.close()

    replay = next(
        TrajectoryReader(tmp_path / "trajectories" / trajectory_id).iter_samples()
    )
    assert replay.blackboard.evidence == ()
    assert replay.blackboard.facts[0].key == "scene.tree_visible"
    assert replay.blackboard.facts[0].evidence_refs == ()
    assert replay.blackboard.tracks[0].evidence_refs == ()
    assert replay.blackboard.chat[0].evidence_refs == ()


def test_low_disk_guard_seals_without_writing_the_queued_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory_id = "trajectory-low-disk"
    recorder = TrajectoryRecorder(
        manifest=_manifest(trajectory_id),
        artifact_root=tmp_path / "trajectories",
        state_db_path=tmp_path / "state.sqlite3",
        min_free_disk_bytes=0,
    )
    monkeypatch.setattr("minecraft_ai.trajectory.trajectory_disk_free_bytes", lambda _path: 0)
    captured_ns = time.monotonic_ns()
    assert recorder.record_accepted(
        action=MotorAction(sequence=0),
        provenance=ActionProvenance(
            policy_id="synthetic:test",
            route_id="synthetic",
            action_level=ActionLevel.RAW,
            origin=ActionOrigin.SYNTHETIC,
        ),
        supervisor_response={"accepted_sequence": 0},
        frame=CapturedFrame(
            frame_id=0,
            captured_ns=captured_ns,
            width=4,
            height=3,
            bgra=bytes((1, 2, 3, 255)) * 12,
        ),
        blackboard=FrameState(
            frame_id=0,
            captured_ns=captured_ns,
            instance_id="bedrock:test",
            width=4,
            height=3,
        ),
    )

    manifest = recorder.close()

    assert manifest.accepted_steps == 0
    assert manifest.dropped_steps == 1
    assert manifest.shards == ()


def test_initial_low_disk_guard_fails_before_starting_a_writer(tmp_path: Path) -> None:
    with pytest.raises(TrajectoryDiskSpaceError, match="reserve"):
        TrajectoryRecorder(
            manifest=_manifest("trajectory-no-space"),
            artifact_root=tmp_path / "trajectories",
            state_db_path=tmp_path / "state.sqlite3",
            min_free_disk_bytes=10**30,
        )


def test_close_does_not_block_when_writer_died_with_a_full_queue() -> None:
    recorder = object.__new__(TrajectoryRecorder)
    recorder._closed = False
    recorder._queue = queue.Queue(maxsize=1)
    recorder._queue.put_nowait(None)
    recorder._thread = threading.Thread()
    recorder._worker_error = RuntimeError("writer-boom")

    with pytest.raises(RuntimeError, match="trajectory writer failed"):
        recorder.close(timeout_s=0.01)


def test_recorder_status_surfaces_runtime_disable_and_drops(tmp_path: Path) -> None:
    recorder = TrajectoryRecorder(
        manifest=_manifest("trajectory-status"),
        artifact_root=tmp_path / "trajectories",
        state_db_path=tmp_path / "state.sqlite3",
        min_free_disk_bytes=0,
    )

    assert recorder.status() == {
        "enabled": True,
        "disabled_reason": None,
        "written_steps": 0,
        "dropped_steps": 0,
        "queued_samples": 0,
        "queue_capacity": 512,
    }
    recorder._recording_disabled_reason = "disk reserve reached"
    recorder._dropped_steps = 3
    assert recorder.status()["enabled"] is False
    assert recorder.status()["disabled_reason"] == "disk reserve reached"
    assert recorder.status()["dropped_steps"] == 3
    recorder.close()


def test_reader_validates_legacy_full_resolution_zlib_shard(tmp_path: Path) -> None:
    trajectory_id = "legacy-zlib"
    directory = tmp_path / trajectory_id
    directory.mkdir()
    captured_ns = time.monotonic_ns()
    pixels = bytes((7, 8, 9, 255)) * 12
    step = TrajectoryStep(
        trajectory_id=trajectory_id,
        step_index=0,
        captured_ns=captured_ns,
        frame_ref="wds://legacy-zlib/shard.tar#000000000000.frame.bgra.zlib",
        frame_hash=hashlib.sha256(pixels).hexdigest(),
        action=MotorAction(sequence=0),
        action_level=ActionLevel.RAW,
        blackboard_snapshot_ref="wds://legacy-zlib/shard.tar#000000000000.blackboard.json",
    )
    blackboard = FrameState(
        frame_id=73,
        captured_ns=captured_ns,
        instance_id="bedrock:test",
        width=4,
        height=3,
    )
    payloads = {
        "000000000000.step.json": step.model_dump_json(
            exclude={"frame_id", "source_frame_id", "source_captured_ns"}
        ).encode(),
        "000000000000.frame.json": json.dumps(
            {
                "codec": "zlib",
                "pixel_format": "BGRA",
                "width": 4,
                "height": 3,
                "raw_bytes": len(pixels),
            }
        ).encode(),
        "000000000000.frame.bgra.zlib": zlib.compress(pixels, level=1),
        "000000000000.blackboard.json": blackboard.model_dump_json().encode(),
    }
    shard_path = directory / f"{trajectory_id}-shard-000000.tar"
    with tarfile.open(shard_path, mode="w") as shard:
        for name, payload in payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            shard.addfile(info, io.BytesIO(payload))
    (directory / "manifest.json").write_text(
        _manifest(trajectory_id).model_dump_json(),
        encoding="utf-8",
    )

    replay = tuple(TrajectoryReader(directory).iter_samples())

    assert len(replay) == 1
    assert replay[0].frame.bgra == pixels
    assert replay[0].step.frame_id is None
    assert replay[0].frame.frame_id == replay[0].blackboard.frame_id == 73
    assert replay[0].step.source_frame_id is replay[0].step.source_captured_ns is None
    assert TrajectoryReader(directory).validate().valid


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
    assert legacy.frame_id is None
    assert legacy.source_frame_id is legacy.source_captured_ns is None


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
