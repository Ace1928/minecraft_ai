from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from minecraft_ai.datasets import ActionLevel, DatasetSource, DatasetSourceType, TrajectoryManifest
from minecraft_ai.datasets.hindsight import (
    HindsightLabelSource,
    iter_hindsight_examples,
    write_hindsight_dataset,
)
from minecraft_ai.episodes import RuntimeEvent, RuntimeEventKind
from minecraft_ai.motor import MotorIntent
from minecraft_ai.perception import FrameState, PerceptionFact
from minecraft_ai.platforms.bedrock_x11 import CapturedFrame
from minecraft_ai.safety import MotorAction
from minecraft_ai.storage import StateDatabase
from minecraft_ai.trajectory import (
    ActionOrigin,
    ActionProvenance,
    TrajectoryRecorder,
    motor_condition_id,
)


def _manifest(*, training_allowed: bool = True) -> TrajectoryManifest:
    return TrajectoryManifest(
        trajectory_id="bedrock-hindsight-fixture",
        source=DatasetSource(
            source_id="minecraft-ai:hindsight-fixture",
            source_type=DatasetSourceType.BEDROCK_AGENT,
            license="CC0-1.0",
            redistribution_allowed=True,
            training_allowed=training_allowed,
            edition="bedrock",
            game_versions=("1.test",),
        ),
        role="generalist",
        label="hindsight-fixture",
        game_version="1.test",
        platform="pytest",
        launcher_profile="fixture",
        resolution=(16, 12),
        started_ns=time.time_ns(),
    )


def _provenance(run_id: str, skill_id: str, instruction: str) -> ActionProvenance:
    condition = MotorIntent(
        skill_id=skill_id,
        mode="policy",
        episode_id=run_id,
        action_level=ActionLevel.MOTION,
        instruction=instruction,
    )
    route_id = "raw-motion"
    return ActionProvenance(
        policy_id="learned:openai-vpt:foundation-1x",
        model_version="foundation-1x",
        route_id=route_id,
        policy_action_kind="prediction",
        action_level=ActionLevel.MOTION,
        origin=ActionOrigin.POLICY,
        condition_id=motor_condition_id(
            condition,
            route_id=route_id,
            target_track_id=None,
        ),
        condition=condition.model_dump(mode="json"),
    )


def _record_fixture(tmp_path: Path, *, training_allowed: bool = True) -> tuple[Path, Path]:
    db_path = tmp_path / "state.sqlite3"
    root = tmp_path / "trajectories"
    recorder = TrajectoryRecorder(
        manifest=_manifest(training_allowed=training_allowed),
        artifact_root=root,
        state_db_path=db_path,
        shard_steps=3,
        frame_max_width=16,
        queue_size=32,
        min_free_disk_bytes=0,
    )
    cases = (
        ("success-run", "move_forward", "move forward", ()),
        ("success-run", "move_forward", "move forward", ()),
        (
            "failed-run",
            "mine_log",
            "mine the visible log",
            (("target.visible", False, 0.99),),
        ),
        (
            "failed-run",
            "mine_log",
            "mine the visible log",
            (("target.visible", True, 0.97),),
        ),
        (
            "failed-run",
            "mine_log",
            "mine the visible log",
            (("target.visible", True, 0.96), ("target.centered", True, 0.94)),
        ),
    )
    base_ns = time.monotonic_ns()
    for index, (run_id, skill_id, instruction, fact_values) in enumerate(cases):
        captured_ns = base_ns + index * 1_000_000
        frame = CapturedFrame(
            frame_id=index,
            captured_ns=captured_ns,
            width=16,
            height=12,
            bgra=bytes((10 + index, 20, 30, 255)) * (16 * 12),
        )
        facts = tuple(
            PerceptionFact(
                key=key,
                value=value,
                confidence=confidence,
                observed_ns=captured_ns,
                source="fixture",
            )
            for key, value, confidence in fact_values
        )
        blackboard = FrameState(
            frame_id=index,
            captured_ns=captured_ns,
            instance_id="bedrock:test",
            width=16,
            height=12,
            facts=facts,
        )
        action = MotorAction(sequence=index, keys_down=("w",))
        assert recorder.record_accepted(
            action=action,
            provenance=_provenance(run_id, skill_id, instruction),
            supervisor_response={
                "accepted_sequence": index,
                "accepted_monotonic_ns": captured_ns + 1,
            },
            frame=frame,
            blackboard=blackboard,
            skill_run_id=run_id,
            skill_id=skill_id,
        )
    recorder.close()
    with StateDatabase(db_path) as database:
        database.save_runtime_event(
            RuntimeEvent(
                event_id="skill-run:success-run:terminal",
                kind=RuntimeEventKind.SKILL_SUCCEEDED,
                observed_ns=base_ns + 2_000_000,
                trajectory_id="bedrock-hindsight-fixture",
                payload={
                    "run_id": "success-run",
                    "skill_id": "move_forward",
                    "outcome": "succeeded",
                },
            )
        )
        database.save_runtime_event(
            RuntimeEvent(
                event_id="skill-run:failed-run:terminal",
                kind=RuntimeEventKind.SKILL_FAILED,
                observed_ns=base_ns + 6_000_000,
                trajectory_id="bedrock-hindsight-fixture",
                payload={
                    "run_id": "failed-run",
                    "skill_id": "mine_log",
                    "outcome": "failed",
                },
            )
        )
    return root / "bedrock-hindsight-fixture", db_path


def test_hindsight_relabels_verified_success_and_failed_run_progress(tmp_path: Path) -> None:
    trajectory, db_path = _record_fixture(tmp_path)

    examples = tuple(
        iter_hindsight_examples(
            (trajectory,),
            state_db_path=db_path,
            history_steps=2,
            partial_window_steps=3,
        )
    )

    successful = tuple(example for example in examples if example.success)
    partial = tuple(example for example in examples if not example.success)
    assert len(successful) == 2
    assert {example.achieved_goal for example in successful} == {"move forward"}
    assert successful[0].history_step_indices == ()
    assert successful[1].history_step_indices == (0,)
    assert all(
        example.label_source == HindsightLabelSource.TERMINAL_SUCCESS
        for example in successful
    )
    assert len(partial) == 5
    assert {example.achieved_goal for example in partial} == {
        "find requested target",
        "center requested target",
    }
    assert {example.evidence_fact_key for example in partial} == {
        "target.visible",
        "target.centered",
    }
    assert all(example.intended_goal == "mine the visible log" for example in partial)
    assert all(example.policy_id == "learned:openai-vpt:foundation-1x" for example in examples)


def test_hindsight_artifact_is_content_addressed_and_reproducible(tmp_path: Path) -> None:
    trajectory, db_path = _record_fixture(tmp_path)
    output = tmp_path / "datasets"

    first = write_hindsight_dataset(
        (trajectory,),
        state_db_path=db_path,
        output_dir=output,
        history_steps=4,
        partial_window_steps=3,
    )
    second = write_hindsight_dataset(
        (trajectory,),
        state_db_path=db_path,
        output_dir=output,
        history_steps=4,
        partial_window_steps=3,
    )

    assert first.data_sha256 == second.data_sha256
    assert first.manifest_sha256 == second.manifest_sha256
    assert first.manifest.examples == 7
    assert first.manifest.successful_examples == 2
    assert first.manifest.partial_progress_examples == 5
    assert first.manifest.training_allowed is True
    assert first.manifest.automatic_promotion is False
    lines = first.data_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 7
    assert all(len(json.loads(line)["example_id"]) == 64 for line in lines)


def test_hindsight_rejects_source_without_training_permission(tmp_path: Path) -> None:
    trajectory, db_path = _record_fixture(tmp_path, training_allowed=False)

    with pytest.raises(ValueError, match="not licensed for training"):
        tuple(iter_hindsight_examples((trajectory,), state_db_path=db_path))
