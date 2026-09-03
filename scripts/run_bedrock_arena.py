#!/usr/bin/env python3
"""Run the six initial frozen tasks through the real Bedrock control boundary."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from minecraft_ai.agent_lifecycle import agent_alive
from minecraft_ai.config import RuntimeConfig, app_paths, load_config
from minecraft_ai.eval.arena import ARENA_TASK_IDS, BedrockArenaRunner
from minecraft_ai.eval.evaluator import EvaluationEvidence, load_evidence
from minecraft_ai.eval.tasks import BenchmarkTask
from minecraft_ai.execution import SkillExecutor
from minecraft_ai.perception import PerceptionBlackboard
from minecraft_ai.perception_service import RealtimePerceptionService, bedrock_survival_hud_present
from minecraft_ai.platforms import create_bedrock_capture, discover_bedrock_linux_install
from minecraft_ai.platforms.bedrock_session import BedrockSession, bedrock_session_alive
from minecraft_ai.policy_service import GroundedPolicyRouter, TemporalPolicyClient
from minecraft_ai.safety import SupervisorState
from minecraft_ai.storage import StateDatabase
from minecraft_ai.supervisor import send_command


class _FixtureDriver:
    def __init__(self, executable: Path, *, timeout_s: float) -> None:
        selected = executable.expanduser().resolve()
        if not selected.is_file() or not os.access(selected, os.X_OK):
            raise ValueError(f"fixture driver is not executable: {selected}")
        self.executable = selected
        self.timeout_s = timeout_s

    def prepare(self, task: BenchmarkTask, repetition: int) -> None:
        task_id = task.task_id
        fixture_id = task.world_fixture_id
        self._run("prepare", task_id, fixture_id, repetition)

    def evidence(
        self,
        task: BenchmarkTask,
        repetition: int,
        trajectory: Path,
    ) -> EvaluationEvidence:
        task_id = task.task_id
        fixture_id = task.world_fixture_id
        result = self._run(
            "evaluate",
            task_id,
            fixture_id,
            repetition,
            trajectory=trajectory,
        )
        try:
            return EvaluationEvidence.model_validate_json(result.stdout)
        except ValueError as exc:
            raise RuntimeError("fixture driver returned invalid EvaluationEvidence JSON") from exc

    def _run(
        self,
        phase: str,
        task_id: str,
        fixture_id: str,
        repetition: int,
        *,
        trajectory: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        command = [
            str(self.executable),
            phase,
            "--task",
            task_id,
            "--fixture",
            fixture_id,
            "--repetition",
            str(repetition),
        ]
        if trajectory is not None:
            command.extend(("--trajectory", str(trajectory)))
        return subprocess.run(
            command,
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            timeout=self.timeout_s,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=ARENA_TASK_IDS, default="a_move_forward")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--fixture-driver",
        type=Path,
        help=(
            "Executable implementing prepare/evaluate for the declared fixture; evaluator "
            "stdout must be EvaluationEvidence JSON."
        ),
    )
    parser.add_argument(
        "--fixture-timeout-s",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--prepared",
        action="store_true",
        help="Run one attempt in a fixture the operator has already prepared.",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        help="Post-hoc evidence for one operator-prepared attempt.",
    )
    return parser


def _policy(
    config: RuntimeConfig,
    perception: RealtimePerceptionService,
) -> GroundedPolicyRouter:
    if not config.policy.enabled:
        raise RuntimeError("the Bedrock arena requires an enabled learned semantic policy")
    grounded_config = config.grounded_policy
    if grounded_config is None or not grounded_config.enabled:
        raise RuntimeError("the six-task arena requires an enabled ROCKET grounding observer")
    primary = TemporalPolicyClient(config.policy, frame_provider=lambda: perception.last_capture)
    grounded = TemporalPolicyClient(grounded_config, frame_provider=lambda: perception.last_capture)
    raw_motion = (
        None
        if config.raw_motion_policy is None or not config.raw_motion_policy.enabled
        else TemporalPolicyClient(
            config.raw_motion_policy,
            frame_provider=lambda: perception.last_capture,
        )
    )
    gui = (
        None
        if config.gui_policy is None or not config.gui_policy.enabled
        else TemporalPolicyClient(
            config.gui_policy,
            frame_provider=lambda: perception.last_capture,
        )
    )
    return GroundedPolicyRouter(primary, grounded, raw_motion=raw_motion, gui=gui)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be positive")
    if args.fixture_timeout_s <= 0:
        raise SystemExit("--fixture-timeout-s must be positive")
    if args.fixture_driver is None:
        if not args.prepared:
            raise SystemExit("provide --fixture-driver or explicitly acknowledge --prepared")
        if args.repetitions != 1:
            raise SystemExit("operator-prepared mode permits exactly one attempt")
    elif args.prepared or args.evidence is not None:
        raise SystemExit("--fixture-driver cannot be combined with --prepared/--evidence")
    if args.evidence is not None and not args.evidence.is_file():
        raise SystemExit(f"evidence file does not exist: {args.evidence}")
    if agent_alive():
        raise SystemExit("stop the realtime agent before granting the arena motor authority")

    config = load_config(args.config)
    paths = app_paths()
    install = discover_bedrock_linux_install()
    if install is None or install.selected_build is None:
        raise SystemExit("arena requires the exact active BedrockOnLinux build identity")
    game_version = install.selected_build.version
    session = BedrockSession.load()
    if not bedrock_session_alive(session):
        raise SystemExit("managed Bedrock session is not alive")
    window_id = session.find_window()
    if window_id is None:
        raise SystemExit("managed Bedrock window is unavailable")
    binding = session.host_monitor_binding()
    allow_host = binding is not None
    capture = create_bedrock_capture(
        session.display,
        window_id,
        allow_host=allow_host,
        host_monitor_binding=binding,
    )
    blackboard = PerceptionBlackboard()
    instance_id = f"bedrock:{game_version}:x11:{window_id}"
    perception = RealtimePerceptionService(
        capture_source=capture,
        blackboard=blackboard,
        instance_id=instance_id,
        target_hz=config.motor_hz,
        stale_frame_ms=config.stale_frame_ms,
    )
    executor: SkillExecutor | None = None
    database: StateDatabase | None = None
    armed = False
    lease_id: str | None = None
    try:
        perception.capture_once()
        frame = perception.last_capture
        if frame is None or not bedrock_survival_hud_present(frame):
            raise RuntimeError("arena requires a complete in-world survival HUD")
        policy = _policy(config, perception)
        executor = SkillExecutor(policy)
        policy.warmup()

        status = send_command("status")
        if status.get("state") != SupervisorState.SAFE_IDLE.value:
            raise RuntimeError("arena requires the supervisor to be SAFE_IDLE")
        attached = send_command(
            "attach-bedrock-x11",
            display=session.display,
            window_id=window_id,
            allow_host=allow_host,
            host_monitor_binding=None if binding is None else binding.payload(),
        )
        camera = attached.get("world_camera")
        if not isinstance(camera, dict) or camera.get("origin_calibrated") is not True:
            raise RuntimeError(
                "arena requires the existing measured camera origin; run the normal live "
                "calibration path first"
            )
        pitch = camera.get("estimated_pitch_units")
        if isinstance(pitch, int) and not isinstance(pitch, bool):
            policy.restore_world_camera_state(estimated_pitch_units=pitch)
        armed_response = send_command("arm", target_instance=instance_id)
        lease = armed_response.get("lease")
        if not isinstance(lease, dict) or not isinstance(lease.get("lease_id"), str):
            raise RuntimeError("supervisor did not issue an arena motor lease")
        lease_id = str(lease["lease_id"])
        send_command("activate")
        armed = True

        database = StateDatabase(paths.state_db)
        fixture_driver = (
            None
            if args.fixture_driver is None
            else _FixtureDriver(args.fixture_driver, timeout_s=args.fixture_timeout_s)
        )
        static_evidence = None if args.evidence is None else load_evidence(args.evidence)
        runner = BedrockArenaRunner(
            perception=perception,
            blackboard=blackboard,
            executor=executor,
            lease_id=lease_id,
            trajectory_root=paths.data_dir / "trajectories",
            state_db_path=paths.state_db,
            game_version=game_version,
            instance_id=instance_id,
            prepare_attempt=(
                (lambda task, repetition: None)
                if fixture_driver is None
                else fixture_driver.prepare
            ),
            collect_evidence=(
                (lambda task, repetition, trajectory: static_evidence)
                if fixture_driver is None
                else fixture_driver.evidence
            ),
            supervisor_command=send_command,
            target_loader=database.load_operator_target,
            motor_hz=config.motor_hz,
            lease_renew_ms=config.lease_renew_ms,
            shard_steps=config.trajectory.shard_steps,
            queue_size=config.trajectory.queue_size,
        )
        result = runner.run(args.task, repetitions=args.repetitions)
        destination = args.output or (
            paths.data_dir / "benchmarks" / f"{result.report.benchmark_run_id}.json"
        )
        result.report.write(destination)
        database.save_benchmark_report(result.report)
        print(
            json.dumps(
                {
                    "task_id": result.task_id,
                    "attempts": [
                        {
                            "repetition": attempt.repetition,
                            "trajectory_id": attempt.trajectory_id,
                            "trajectory_path": str(attempt.trajectory_path),
                            "skill_outcomes": [run.outcome.value for run in attempt.skill_runs],
                        }
                        for attempt in result.attempts
                    ],
                    "report_path": str(destination),
                    "report": result.report.model_dump(mode="json"),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    finally:
        if armed and lease_id is not None:
            try:
                send_command("release-inputs", lease_id=lease_id)
            except Exception:
                pass
            try:
                send_command("disarm")
            except Exception:
                pass
        if database is not None:
            database.close()
        if executor is not None:
            executor.close()
        perception.close()


if __name__ == "__main__":
    raise SystemExit(main())
