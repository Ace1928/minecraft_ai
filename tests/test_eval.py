from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

from minecraft_ai.datasets import ActionLevel, DatasetSource, DatasetSourceType, TrajectoryManifest
from minecraft_ai.eval import (
    BenchmarkCategory,
    BenchmarkRunner,
    EvaluationEvidence,
    EvaluationStatus,
    bedrock_baseline_suite,
    compare_reports,
)
from minecraft_ai.eval.bedrock_worlds import BEDROCK_WORLD_CONTRACTS
from minecraft_ai.eval.metrics import TraceMetricAccumulator
from minecraft_ai.perception import FrameState
from minecraft_ai.platforms.bedrock_x11 import CapturedFrame
from minecraft_ai.safety import MotorAction
from minecraft_ai.storage import StateDatabase
from minecraft_ai.trajectory import ActionOrigin, ActionProvenance, TrajectoryRecorder


def _record_jump_trajectory(tmp_path: Path) -> Path:
    trajectory_id = "benchmark-jump"
    manifest = TrajectoryManifest(
        trajectory_id=trajectory_id,
        source=DatasetSource(
            source_id="minecraft-ai:benchmark-test",
            source_type=DatasetSourceType.SYNTHETIC,
            license="CC0-1.0",
            redistribution_allowed=True,
            training_allowed=True,
            edition="bedrock",
            game_versions=("1.test",),
        ),
        role="generalist",
        label="jump-range-fixture",
        task_id="a_jump_obstacle",
        game_version="1.test",
        platform="pytest",
        launcher_profile="fixture",
        resolution=(4, 3),
        started_ns=time.time_ns(),
    )
    recorder = TrajectoryRecorder(
        manifest=manifest,
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
        action=MotorAction(sequence=0, keys_down=("w", "ctrl", "space")),
        provenance=ActionProvenance(
            policy_id="synthetic:benchmark-fixture",
            route_id="synthetic",
            action_level=ActionLevel.RAW,
            origin=ActionOrigin.SYNTHETIC,
        ),
        supervisor_response={
            "accepted_sequence": 0,
            "accepted_monotonic_ns": captured_ns + 4_000_000,
        },
        frame=frame,
        blackboard=blackboard,
    )
    recorder.close()
    return tmp_path / "trajectories" / trajectory_id


def test_frozen_suite_is_broad_and_world_contracts_are_closed() -> None:
    suite = bedrock_baseline_suite()
    contract_ids = {contract.world_fixture_id for contract in BEDROCK_WORLD_CONTRACTS}

    assert len(suite.tasks) >= 20
    assert {task.category for task in suite.tasks} == set(BenchmarkCategory)
    assert len({task.task_id for task in suite.tasks}) == len(suite.tasks)
    assert {task.world_fixture_id for task in suite.tasks} <= contract_ids
    assert all(task.criteria for task in suite.tasks)


def test_benchmark_requires_outcome_evidence_and_persists_report(tmp_path: Path) -> None:
    trajectory = _record_jump_trajectory(tmp_path)
    runner = BenchmarkRunner(bedrock_baseline_suite())

    unscored = runner.evaluate_trajectory(
        trajectory,
        task_ids=("a_jump_obstacle",),
        git_commit="test-commit",
    )
    assert unscored.results[0].status == EvaluationStatus.UNSCORED
    assert unscored.summary["scored"] == 0

    report = runner.evaluate_trajectory(
        trajectory,
        task_ids=("a_jump_obstacle",),
        evidence=EvaluationEvidence(
            source="controlled-world:test",
            metrics={"event.destination_reached": 1},
            artifact_refs=("fixture://movement-range/trial-1",),
        ),
        git_commit="test-commit",
    )
    assert report.results[0].status == EvaluationStatus.PASSED
    assert report.results[0].metrics["action.jump_presses"] == 1
    assert report.summary["promotion_eligible"] is False

    with StateDatabase(tmp_path / "state.sqlite3") as database:
        database.save_benchmark_report(report)
        loaded = database.load_benchmark_report_payload(report.benchmark_run_id)
        stored_results = database.connection.execute(
            "SELECT task_id, status FROM benchmark_task_results WHERE benchmark_run_id=?",
            (report.benchmark_run_id,),
        ).fetchall()
    assert loaded["suite_id"] == "bedrock-m1-baseline-v1"
    assert stored_results == [("a_jump_obstacle", "passed")]


def test_comparison_refuses_small_sample_promotion_evidence() -> None:
    comparison = compare_reports(
        {
            "benchmark_run_id": "baseline",
            "summary": {"passed": 1, "scored": 1},
        },
        {
            "benchmark_run_id": "candidate",
            "summary": {"passed": 1, "scored": 1},
        },
    )

    assert comparison["promotion_evidence_sufficient"] is False


def test_trace_metrics_separate_jump_edges_holds_and_world_pitch_drift() -> None:
    accumulator = TraceMetricAccumulator()
    actions = (
        MotorAction(
            sequence=0,
            keys_down=("space", "w"),
            mouse_dy=7,
        ),
        MotorAction(sequence=1, mouse_dy=4),
        MotorAction(sequence=2, keys_up=("space",), mouse_dy=-2),
        MotorAction(
            sequence=3,
            mouse_dy=99,
            camera_semantics="cursor",
        ),
    )
    for index, action in enumerate(actions):
        step = SimpleNamespace(
            action=action,
            step_index=index,
            accepted_ns=None,
            captured_ns=index,
            frame_hash=str(index),
        )
        accumulator.add(SimpleNamespace(step=step))

    values = accumulator.finish().values

    assert values["action.jump_presses"] == 1
    assert values["action.jump_held_steps"] == 2
    assert values["action.forward_held_steps"] == 4
    assert values["camera.world_updates"] == 3
    assert values["camera.world_pitch_net_units"] == 9
    assert values["camera.world_pitch_down_units"] == 11
    assert values["camera.world_pitch_up_units"] == 2
    assert values["camera.world_pitch_down_max_streak"] == 2
    assert values["camera.world_pitch_up_max_streak"] == 1
