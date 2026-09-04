from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from minecraft_ai.action_levels import ActionLevel
from minecraft_ai.datasets import DatasetSource, DatasetSourceType, TrajectoryManifest
from minecraft_ai.eval import EvaluationEvidence
from minecraft_ai.motor import MotorIntent
from minecraft_ai.perception import FrameState
from minecraft_ai.platforms.bedrock_x11 import CapturedFrame
from minecraft_ai.safety import MotorAction
from minecraft_ai.skill_condition_search import (
    ConditionTrial,
    SearchModelReference,
    SkillConditionCandidate,
    SkillConditionSearchSpec,
    content_address,
    evaluate_skill_conditions,
    write_content_addressed_manifest,
)
from minecraft_ai.trajectory import (
    ActionOrigin,
    ActionProvenance,
    TrajectoryRecorder,
    motor_condition_id,
)


def _record_trial(
    root: Path,
    *,
    trajectory_id: str,
    skill_id: str,
    instruction: str,
    condition_scale: float,
    destination_reached: bool,
) -> tuple[Path, Path]:
    manifest = TrajectoryManifest(
        trajectory_id=trajectory_id,
        source=DatasetSource(
            source_id="minecraft-ai:condition-search-test",
            source_type=DatasetSourceType.BEDROCK_AGENT,
            license="CC0-1.0",
            redistribution_allowed=True,
            training_allowed=True,
            edition="bedrock",
            game_versions=("1.test",),
        ),
        role="generalist",
        label="condition-search-fixture",
        task_id="a_jump_obstacle",
        game_version="1.test",
        platform="pytest",
        launcher_profile="fixture",
        resolution=(4, 3),
        started_ns=time.time_ns(),
    )
    trajectory_root = root / "trajectories"
    recorder = TrajectoryRecorder(
        manifest=manifest,
        artifact_root=trajectory_root,
        state_db_path=root / "state.sqlite3",
        shard_steps=16,
        min_free_disk_bytes=0,
    )
    intent = MotorIntent(
        skill_id=skill_id,
        mode="traverse",
        episode_id=f"episode:{trajectory_id}",
        action_level=ActionLevel.LATENT,
        instruction=instruction,
        condition_scale=condition_scale,
    )
    condition = intent.model_dump(mode="json")
    provenance = ActionProvenance(
        policy_id="learned:minestudio-steve1:official-v1",
        model_version="official-v1",
        route_id="direct",
        policy_action_kind="prediction",
        action_level=ActionLevel.LATENT,
        origin=ActionOrigin.POLICY,
        condition_id=motor_condition_id(
            condition,
            route_id="direct",
            target_track_id=None,
        ),
        condition=condition,
    )
    captured_ns = time.monotonic_ns()
    for index in range(2):
        frame_ns = captured_ns + index * 1_000_000_000
        frame = CapturedFrame(
            frame_id=index,
            captured_ns=frame_ns,
            width=4,
            height=3,
            bgra=bytes((index, 2, 3, 255)) * 12,
        )
        blackboard = FrameState(
            frame_id=index,
            captured_ns=frame_ns,
            instance_id="bedrock:test",
            width=4,
            height=3,
        )
        action = MotorAction(
            sequence=index,
            keys_down=("space", "w") if index == 0 else (),
        )
        assert recorder.record_accepted(
            action=action,
            provenance=provenance,
            supervisor_response={
                "accepted_sequence": index,
                "accepted_monotonic_ns": frame_ns + 1_000_000,
            },
            frame=frame,
            blackboard=blackboard,
            skill_run_id=intent.episode_id,
            skill_id=skill_id,
        )
    recorder.close()
    evidence_path = root / f"{trajectory_id}.evidence.json"
    evidence_path.write_text(
        EvaluationEvidence(
            source=f"controlled-world:{trajectory_id}",
            metrics={"event.destination_reached": int(destination_reached)},
            artifact_refs=(f"fixture://movement-range/{trajectory_id}",),
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    return trajectory_root / trajectory_id, evidence_path


def _spec(tmp_path: Path, *, minimum_pose_seeds: int = 2) -> SkillConditionSearchSpec:
    skill_id = "jump_obstacle"
    candidates: list[SkillConditionCandidate] = []
    for candidate_index, (instruction, scale) in enumerate(
        (("jump forward", 4.0), ("run and jump over the obstacle", 6.0))
    ):
        trials = []
        for pose_seed in (11, 29, 41, 53, 67):
            trajectory, evidence = _record_trial(
                tmp_path / f"candidate-{candidate_index}-pose-{pose_seed}",
                trajectory_id=f"candidate-{candidate_index}-pose-{pose_seed}",
                skill_id=skill_id,
                instruction=instruction,
                condition_scale=scale,
                destination_reached=(candidate_index == 0 or pose_seed in {11, 29, 41}),
            )
            trials.append(
                ConditionTrial(
                    initial_pose_seed=pose_seed,
                    trajectory=trajectory,
                    evidence=evidence,
                )
            )
        candidates.append(
            SkillConditionCandidate(
                instruction=instruction,
                condition_scale=scale,
                trials=tuple(trials),
            )
        )
    return SkillConditionSearchSpec(
        skill_id=skill_id,
        task_id="a_jump_obstacle",
        bedrock_version="1.test",
        evaluator_git_commit="d" * 40,
        model=SearchModelReference(
            provider="minestudio-steve1",
            model_version="official-v1",
            model_sha256="a" * 64,
            weights_sha256="b" * 64,
            source_commit="c" * 40,
        ),
        candidates=tuple(candidates),
        minimum_pose_seeds=minimum_pose_seeds,
    )


def test_condition_search_is_paired_multi_pose_and_never_promotes(tmp_path: Path) -> None:
    manifest = evaluate_skill_conditions(_spec(tmp_path))

    assert manifest.comparison.comparison_ready is True
    assert manifest.comparison.paired_initial_pose_seeds == (11, 29, 41, 53, 67)
    assert manifest.comparison.leading_candidate_id == manifest.candidates[0].candidate_id
    assert manifest.candidates[0].success_rate == 1.0
    assert manifest.candidates[1].success_rate == 0.6
    assert manifest.candidates[0].successful_completion_time_s_p50 == 1.0
    assert manifest.promotion_gate["automatic_promotion"] is False
    assert manifest.promotion_gate["promotion_performed"] is False
    assert manifest.latent_conditioning_supported is False


def test_condition_search_manifest_is_deterministically_content_addressed(
    tmp_path: Path,
) -> None:
    manifest = evaluate_skill_conditions(_spec(tmp_path))

    first = write_content_addressed_manifest(manifest, tmp_path / "results")
    second = write_content_addressed_manifest(manifest, tmp_path / "results")

    assert first.path == second.path
    assert first.sha256 == second.sha256 == content_address(manifest)
    assert first.path.stem == hashlib.sha256(first.path.read_bytes()).hexdigest()
    payload = json.loads(first.path.read_text(encoding="utf-8"))
    assert payload["model"]["weights_sha256"] == "b" * 64
    assert payload["bedrock_version"] == "1.test"


def test_condition_search_script_dry_run_evaluates_but_writes_nothing(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    spec_path = tmp_path / "search.json"
    spec_path.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    output_dir = tmp_path / "results"

    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[1] / "scripts/search_skill_conditions.py"),
            "--spec",
            str(spec_path),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    output = json.loads(completed.stdout)
    assert output["dry_run"] is True
    assert output["manifest_path"] is None
    assert output["comparison_ready"] is True
    assert output["automatic_promotion"] is False
    assert not output_dir.exists()


def test_condition_search_refuses_candidate_trajectory_provenance_mismatch(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    wrong = spec.candidates[0].model_copy(update={"instruction": "look around"})
    mismatched = spec.model_copy(update={"candidates": (wrong, spec.candidates[1])})

    with pytest.raises(ValueError, match="instruction does not match candidate"):
        evaluate_skill_conditions(mismatched)


def test_condition_search_honors_benchmark_pose_minimum_before_comparison(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path, minimum_pose_seeds=2)
    incomplete = spec.model_copy(
        update={
            "candidates": tuple(
                candidate.model_copy(update={"trials": candidate.trials[:4]})
                for candidate in spec.candidates
            )
        }
    )

    manifest = evaluate_skill_conditions(incomplete)

    assert manifest.comparison.multi_pose_evidence is False
    assert manifest.comparison.minimum_pose_seeds == 5
    assert manifest.comparison.comparison_ready is False
    assert manifest.comparison.leading_candidate_id is None
    assert manifest.promotion_gate["comparison_evidence_ready"] is False


def test_steve_latent_condition_is_rejected_without_runtime_input_contract() -> None:
    with pytest.raises(ValueError, match="no latent-conditioning input contract"):
        SkillConditionSearchSpec(
            skill_id="jump_obstacle",
            task_id="a_jump_obstacle",
            bedrock_version="1.test",
            evaluator_git_commit="d" * 40,
            model=SearchModelReference(
                provider="minestudio-steve1",
                model_version="official-v1",
                model_sha256="a" * 64,
                weights_sha256="b" * 64,
            ),
            candidates=(
                SkillConditionCandidate(
                    instruction="jump forward",
                    condition_scale=4.0,
                    latent_id="z_041",
                    trials=(ConditionTrial(initial_pose_seed=1, trajectory=Path("unused")),),
                ),
                SkillConditionCandidate(
                    instruction="jump over obstacle",
                    condition_scale=6.0,
                    trials=(ConditionTrial(initial_pose_seed=2, trajectory=Path("unused-2")),),
                ),
            ),
            minimum_pose_seeds=2,
        )
