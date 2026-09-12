from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import minecraft_ai.cli as cli
from minecraft_ai.datasets import ActionLevel, DatasetSource, DatasetSourceType, TrajectoryManifest
from minecraft_ai.eval import (
    BenchmarkCategory,
    BenchmarkRunner,
    BenchmarkSuite,
    BenchmarkTaskResult,
    EvaluationEvidence,
    EvaluationStatus,
    bedrock_baseline_suite,
    compare_reports,
)
from minecraft_ai.eval.bedrock_worlds import BEDROCK_WORLD_CONTRACTS
from minecraft_ai.eval.metrics import TraceMetricAccumulator, TraceMetrics
from minecraft_ai.perception import FrameState
from minecraft_ai.platforms.bedrock_x11 import CapturedFrame
from minecraft_ai.safety import MotorAction
from minecraft_ai.storage import StateDatabase
from minecraft_ai.trajectory import ActionOrigin, ActionProvenance, TrajectoryRecorder


def _record_jump_trajectory(
    tmp_path: Path, *, noop: bool = False, trajectory_id: str = "benchmark-jump",
    task_id: str = "a_jump_obstacle",
) -> Path:
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
        task_id=task_id,
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
        action=MotorAction(sequence=0, keys_down=() if noop else ("w", "ctrl", "space")),
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
    assert comparison["same_suite"] is False
    assert comparison["baseline_errors"]


def _complete_contract_report(suite: BenchmarkSuite | None = None):
    """Synthetic scoring fixture for adversarial serialized-report mutations."""
    suite = bedrock_baseline_suite() if suite is None else suite
    runner = BenchmarkRunner(suite)
    results = []
    for task in suite.tasks:
        metrics = TraceMetricAccumulator().finish().values
        metrics["trace.steps"] = 1
        metrics.update({criterion.metric: criterion.value for criterion in task.criteria})
        for repetition in range(task.minimum_repetitions):
            results.append(runner._evaluate_task(
                task, f"fixture-{task.task_id}-{repetition}", TraceMetrics(values=metrics),
                EvaluationEvidence(source="synthetic:report-contract-test"), repetition,
            ))
    return runner._report(tuple(results), git_commit="contract-test")


@pytest.fixture
def complete_report():
    return _complete_contract_report()


def test_complete_builtin_contract_reports_compare(complete_report) -> None:
    payload = complete_report.model_dump(mode="json")
    candidate = deepcopy(payload)
    candidate["benchmark_run_id"] = "candidate"
    comparison = compare_reports(payload, candidate)
    assert comparison["baseline_valid"] and comparison["candidate_valid"]
    assert comparison["same_suite"] and comparison["promotion_evidence_sufficient"]
    assert comparison["baseline_success_rate"] == 1.0
    assert comparison["success_rate_delta"] == 0.0


def test_complete_recorded_custom_suite_requires_independent_contract(tmp_path: Path) -> None:
    baseline = bedrock_baseline_suite()
    suite = BenchmarkSuite(suite_id="recorded-regression-v1", version=1, tasks=tuple(
        baseline.task(task).model_copy(update={"minimum_repetitions": 2})
        for task in ("a_move_forward", "a_jump_obstacle")
    ))
    paths = []
    evidence = {}
    for task in suite.tasks:
        for repetition in range(task.minimum_repetitions):
            identity = f"{task.task_id}-{repetition}"
            paths.append(_record_jump_trajectory(
                tmp_path, trajectory_id=identity, task_id=task.task_id,
            ))
            evidence[identity] = EvaluationEvidence(
                source="controlled-fixture:test", metrics={"event.destination_reached": 1},
            )
    runner = BenchmarkRunner(suite)
    report = runner.evaluate_many(tuple(paths), evidence_by_trajectory=evidence)
    assert report.summary["scored"] == 4
    assert report.summary["promotion_eligible"] is True
    path = report.write(tmp_path / "report.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert compare_reports(payload, payload, suite=suite)["promotion_evidence_sufficient"]
    assert not compare_reports(payload, payload)["promotion_evidence_sufficient"]
    with pytest.raises(ValueError, match="reused trajectory"):
        runner.evaluate_many((paths[0], paths[0]), evidence_by_trajectory=evidence)


@pytest.mark.parametrize("payload", [
    {}, {"summary": {"promotion_eligible": True}},
    {"schema_version": 1, "summary": {"promotion_eligible": True}},
    {"schema_version": 2, "summary": {"promotion_eligible": True}},
])
def test_malformed_or_historical_reports_never_promote(payload) -> None:
    comparison = compare_reports(payload, payload)
    assert comparison["promotion_evidence_sufficient"] is False
    assert comparison["same_suite"] is False
    assert comparison["baseline_valid"] is False
    assert comparison["baseline_errors"]
    assert comparison["baseline_success_rate"] is None


@pytest.mark.parametrize("field", ["suite", "suite_id", "benchmark_run_id", "results", "created_ns"])
def test_report_requires_complete_identity_and_results(complete_report, field: str) -> None:
    baseline = complete_report.model_dump(mode="json")
    candidate = deepcopy(baseline)
    del candidate[field]
    comparison = compare_reports(baseline, candidate)
    assert comparison["candidate_valid"] is False
    assert comparison["candidate_errors"]
    assert comparison["promotion_evidence_sufficient"] is False


@pytest.mark.parametrize("field,value", [
    ("suite_id", ""), ("suite_id", "different-suite"), ("benchmark_run_id", " "),
    ("created_ns", True), ("created_ns", -1), ("results", []),
    ("schema_version", 99), ("summary", {"promotion_eligible": True}),
])
def test_invalid_report_fields_cannot_reuse_valid_summary(complete_report, field, value) -> None:
    baseline = complete_report.model_dump(mode="json")
    candidate = deepcopy(baseline)
    candidate[field] = value
    comparison = compare_reports(baseline, candidate)
    assert comparison["candidate_valid"] is False
    assert comparison["promotion_evidence_sufficient"] is False


@pytest.mark.parametrize("key", [
    "tasks", "suite_tasks", "scored", "unique_tasks_scored", "tasks_meeting_minimum_repetitions",
    "minimum_scored_results_required", "passed", "failed", "unscored", "errors", "success_rate",
    "protected_failures", "coverage_complete", "promotion_eligible",
])
def test_every_summary_field_is_cross_checked(complete_report, key: str) -> None:
    baseline = complete_report.model_dump(mode="json")
    candidate = deepcopy(baseline)
    value = candidate["summary"][key]
    candidate["summary"][key] = not value if isinstance(value, bool) else value + 1
    comparison = compare_reports(baseline, candidate)
    assert comparison["candidate_valid"] is False
    assert "canonical result admission" in str(comparison["candidate_errors"])
    assert comparison["promotion_evidence_sufficient"] is False


@pytest.mark.parametrize("change", ["missing", "extra", "coerced"])
def test_summary_requires_exact_fields_and_scalar_types(complete_report, change: str) -> None:
    payload = complete_report.model_dump(mode="json")
    if change == "missing":
        del payload["summary"]["protected_failures"]
    elif change == "extra":
        payload["summary"]["admitted"] = True
    else:
        payload["summary"]["protected_failures"] = False
    assert not compare_reports(payload, payload)["promotion_evidence_sufficient"]


@pytest.mark.parametrize("change", [
    "status", "observed", "passed", "threshold", "metrics", "unknown-task", "duplicate",
    "reused-trajectory", "empty-criteria", "error", "nan",
])
def test_result_tampering_is_not_repaired_from_a_fabricated_summary(
    complete_report, change: str,
) -> None:
    payload = complete_report.model_dump(mode="json")
    result = payload["results"][0]
    if change == "status":
        result["status"] = "failed"
    elif change in {"observed", "passed"}:
        result["criteria"][0][change] = 0 if change == "observed" else False
    elif change == "threshold":
        result["criteria"][0]["criterion"]["value"] = 0
    elif change in {"metrics", "nan"}:
        result["metrics"]["action.camera_updates"] = 0 if change == "metrics" else float("nan")
    elif change == "unknown-task":
        result["task_id"] = "invented"
    elif change == "duplicate":
        payload["results"][1] = deepcopy(result)
    elif change == "reused-trajectory":
        payload["results"][1]["trajectory_id"] = result["trajectory_id"]
    elif change == "empty-criteria":
        result["criteria"] = []
    elif change == "error":
        result["error"] = "unacknowledged failure"
    comparison = compare_reports(payload, payload)
    assert comparison["candidate_valid"] is False
    assert comparison["candidate_errors"]
    assert comparison["promotion_evidence_sufficient"] is False


@pytest.mark.parametrize("change", ["version", "minimum", "protected", "criteria", "world"])
def test_same_suite_id_does_not_authorize_changed_suite_definition(change: str) -> None:
    suite = bedrock_baseline_suite().model_dump(mode="json")
    if change == "version":
        suite["version"] += 1
    elif change == "minimum":
        suite["tasks"][0]["minimum_repetitions"] = 1
    elif change == "protected":
        suite["tasks"][0]["protected"] = False
    elif change == "criteria":
        suite["tasks"][0]["criteria"][0]["value"] = 0
    else:
        suite["tasks"][0]["world_fixture_id"] = "unqualified-world"
    payload = _complete_contract_report(BenchmarkSuite.model_validate(suite)).model_dump(mode="json")
    # Even two internally consistent reports sharing the same altered definition
    # must not replace the independently trusted frozen suite.
    comparison = compare_reports(payload, payload)
    assert comparison["candidate_valid"] is False
    assert "expected frozen contract" in str(comparison["candidate_errors"])
    assert comparison["promotion_evidence_sufficient"] is False


@pytest.mark.parametrize("change", ["empty", "duplicate-task", "no-criteria", "no-threshold"])
def test_invalid_suite_contracts_are_rejected(change: str) -> None:
    suite = bedrock_baseline_suite().model_dump(mode="json")
    if change == "empty":
        suite["tasks"] = []
    elif change == "duplicate-task":
        suite["tasks"].append(deepcopy(suite["tasks"][0]))
    elif change == "no-criteria":
        suite["tasks"][0]["criteria"] = []
    else:
        suite["tasks"][0]["criteria"][0]["value"] = None
    with pytest.raises(ValueError):
        BenchmarkSuite.model_validate(suite)


@pytest.mark.parametrize("kind", ["empty", "incomplete", "protected-failure", "error", "unscored"])
def test_canonical_admission_preserves_negative_results(complete_report, kind: str) -> None:
    runner = BenchmarkRunner(complete_report.suite)
    results = list(complete_report.results)
    if kind == "empty":
        results = []
    elif kind == "incomplete":
        results.pop()
    elif kind == "error":
        first = results[0]
        results[0] = BenchmarkTaskResult(
            task_id=first.task_id, repetition=first.repetition, trajectory_id=first.trajectory_id,
            status=EvaluationStatus.ERROR, criteria=(), metrics={}, evidence_sources=(),
            error="trajectory checksum mismatch",
        )
    else:
        first = results[0]
        metrics = dict(first.metrics)
        if kind == "unscored":
            del metrics["event.target_centered"]
        else:
            metrics["action.camera_updates"] = 0
        results[0] = runner._evaluate_task(
            runner.suite.task(first.task_id), first.trajectory_id, TraceMetrics(values=metrics),
            None, first.repetition,
        )
    report = runner._report(tuple(results), git_commit="test")
    assert report.summary["promotion_eligible"] is False
    if kind == "protected-failure":
        assert report.summary["protected_failures"] == 1
        assert report.summary["coverage_complete"] is True
    if kind == "error":
        assert report.summary["errors"] == 1
    if kind == "unscored":
        assert report.summary["unscored"] == 1
    comparison = compare_reports(
        complete_report.model_dump(mode="json"), report.model_dump(mode="json"),
    )
    assert comparison["candidate_valid"] is True
    assert comparison["promotion_evidence_sufficient"] is False


def test_mutated_report_is_revalidated_before_writing(tmp_path: Path, complete_report) -> None:
    complete_report.summary["passed"] = 0
    destination = tmp_path / "new-directory" / "report.json"
    with pytest.raises(ValueError, match="canonical result admission"):
        complete_report.write(destination)
    assert not destination.parent.exists()


def test_cli_comparison_preserves_historical_invalidity(tmp_path: Path) -> None:
    report = tmp_path / "historical.json"
    report.write_text('{"summary": {"promotion_eligible": true}}', encoding="utf-8")
    result = CliRunner().invoke(cli.app, ["eval", "compare", str(report), str(report)])
    assert result.exit_code == 0, result.output
    comparison = json.loads(result.output)
    assert comparison["promotion_evidence_sufficient"] is False
    assert "historical reports must be re-evaluated" in str(comparison["baseline_errors"])


def test_external_outcome_cannot_turn_noop_into_measured_movement(tmp_path: Path) -> None:
    trajectory = _record_jump_trajectory(tmp_path, noop=True)
    runner = BenchmarkRunner(bedrock_baseline_suite())
    evidence = EvaluationEvidence(
        source="controlled-world:test", metrics={"event.destination_reached": 1},
    )
    report = runner.evaluate_trajectory(
        trajectory, task_ids=("a_move_forward",), evidence=evidence,
    )
    assert report.results[0].status == EvaluationStatus.FAILED
    assert report.results[0].metrics["action.forward_presses"] == 0
    assert report.results[0].criteria[0].observed == 0
    assert report.results[0].criteria[1].passed is True

    # Frozen Pydantic models still contain mutable dictionaries: the merge boundary
    # must enforce the same rule even when construction-time validation is bypassed.
    evidence.metrics["action.forward_presses"] = 1
    with pytest.raises(ValueError, match="invalid evaluation evidence.*action.forward_presses"):
        runner.evaluate_trajectory(trajectory, task_ids=("a_move_forward",), evidence=evidence)


@pytest.mark.parametrize("key", [
    "action.forward_presses", "trace.duration_s", "latency.frame_to_accept_p95_ms",
    "safety.sequence_violations", "camera.world_pitch_net_units", "action.future_metric",
    "trace", "latency",
])
def test_evidence_rejects_computed_namespaces_even_when_measurement_is_absent(key: str) -> None:
    with pytest.raises(ValueError, match="invalid evaluation evidence"):
        EvaluationEvidence(source="test", metrics={key: 1})
    with pytest.raises(ValueError, match="invalid evaluation evidence"):
        TraceMetrics().merged({key: 1})


def test_external_merge_rejects_exact_collision_outside_reserved_namespaces() -> None:
    measured = TraceMetrics(values={"independent.outcome": 0})
    with pytest.raises(ValueError, match="computed metric names are reserved"):
        measured.merged({"independent.outcome": 1})
    assert measured.values == {"independent.outcome": 0}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_external_outcomes_must_be_finite(value: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        EvaluationEvidence(source="test", metrics={"event.destination_reached": value})


def test_independent_event_reward_and_custom_outcomes_remain_supported(tmp_path: Path) -> None:
    trajectory = _record_jump_trajectory(tmp_path)
    evidence = EvaluationEvidence(source="controlled-world:test", metrics={
        "event.item_crafted": 1, "reward.inventory_planks_delta": 4, "world.fixture": "crafting",
    })
    report = BenchmarkRunner(bedrock_baseline_suite()).evaluate_trajectory(
        trajectory, task_ids=("b_craft_planks", "a_move_forward"), evidence=evidence,
    )
    assert [result.status for result in report.results] == [
        EvaluationStatus.PASSED, EvaluationStatus.UNSCORED,
    ]
    assert report.results[0].metrics["world.fixture"] == "crafting"


@pytest.mark.parametrize("command", ["eval", "benchmark"])
def test_cli_reports_invalid_evidence_before_report_or_database_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str,
) -> None:
    trajectory = _record_jump_trajectory(tmp_path)
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({
        "source": "test", "metrics": {"action.forward_presses": 1},
    }), encoding="utf-8")
    monkeypatch.setattr(cli, "StateDatabase", lambda *_: pytest.fail("invalid report persisted"))
    output = tmp_path / "report.json"
    args = (
        ["eval", "run", "--trajectory", str(trajectory), "--task", "a_move_forward",
         "--evidence", str(evidence)]
        if command == "eval" else
        ["benchmark", "report", "--trajectory-root", str(trajectory.parent),
         "--evidence-dir", str(tmp_path)]
    )
    result = CliRunner().invoke(cli.app, [*args, "--output", str(output)])
    assert result.exit_code == 2, result.output
    assert "invalid evaluation evidence" in result.output
    assert not output.exists()


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
