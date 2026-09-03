from __future__ import annotations

import argparse
import platform
import signal
import sys
import time
from pathlib import Path

from .builtin_skills import build_bootstrap_skill_library
from .cognition import HighLevelController
from .config import app_paths, load_config
from .datasets import DatasetSource, DatasetSourceType, TrajectoryManifest
from .execution import SkillExecutor
from .models import OpenAICompatibleLocalModel
from .motor import BootstrapMotorPolicy, MotorPolicy
from .perception import PerceptionBlackboard
from .perception_service import ActiveVLMWorker, RealtimePerceptionService
from .platforms.bedrock_x11 import IsolatedX11Capture
from .policy_service import GroundedPolicyRouter, TemporalPolicyClient
from .roles import get_role
from .runtime import AgentRuntime
from .storage import StateDatabase
from .supervisor import send_command
from .trajectory import TrajectoryRecorder, new_trajectory_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minecraft AI realtime agent process")
    parser.add_argument("--lease-id", required=True)
    parser.add_argument("--display", required=True)
    parser.add_argument("--window-id", required=True, type=int)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--role", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--allow-host-capture", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = None if args.config is None else Path(args.config)
    config = load_config(config_path)
    if args.role is not None:
        config = config.model_copy(update={"role": args.role})
    role = get_role(config.role)

    paths = app_paths()
    database = StateDatabase(paths.state_db)
    try:
        persisted = database.load_skills()
        bootstrap = build_bootstrap_skill_library()
        for skill_id, spec in bootstrap.specs.items():
            existing = persisted.specs.get(skill_id)
            if existing is None or existing.version < spec.version:
                persisted.specs[skill_id] = spec
                database.save_skill(spec)
        skills = persisted
        memories = database.load_memories()
        social = database.load_social()

        blackboard = PerceptionBlackboard()
        capture = IsolatedX11Capture(
            args.display,
            args.window_id,
            allow_host=bool(args.allow_host_capture),
        )
        capture_probe = capture.capture()

        high_level: HighLevelController | None = None
        if config.high_level.enabled:
            if not config.high_level.model_id:
                raise RuntimeError("high-level model is enabled but model_id is empty")
            high_model = OpenAICompatibleLocalModel(
                model_id=config.high_level.model_id,
                base_url=config.high_level.base_url,
                api_key=config.high_level.api_key,
                timeout_s=config.high_level.timeout_s,
            )
            high_level = HighLevelController(high_model, skills)

        active_vlm: ActiveVLMWorker | None = None
        if config.vision_language.enabled:
            if not config.vision_language.model_id:
                raise RuntimeError("vision-language model is enabled but model_id is empty")
            vlm_model = OpenAICompatibleLocalModel(
                model_id=config.vision_language.model_id,
                base_url=config.vision_language.base_url,
                api_key=config.vision_language.api_key,
                timeout_s=config.vision_language.timeout_s,
            )
            active_vlm = ActiveVLMWorker(vlm_model, blackboard, args.instance_id)

        perception = RealtimePerceptionService(
            capture_source=capture,
            blackboard=blackboard,
            instance_id=args.instance_id,
            target_hz=config.motor_hz,
            stale_frame_ms=config.stale_frame_ms,
            active_vlm=active_vlm,
        )
        policy: MotorPolicy
        if config.policy.enabled:
            primary_policy = TemporalPolicyClient(
                config.policy,
                frame_provider=lambda: perception.last_capture,
            )
            if config.grounded_policy is not None and config.grounded_policy.enabled:
                grounded_policy = TemporalPolicyClient(
                    config.grounded_policy,
                    frame_provider=lambda: perception.last_capture,
                )
                gui_policy = (
                    None
                    if config.gui_policy is None or not config.gui_policy.enabled
                    else TemporalPolicyClient(
                        config.gui_policy,
                        frame_provider=lambda: perception.last_capture,
                    )
                )
                policy = GroundedPolicyRouter(
                    primary_policy,
                    grounded_policy,
                    gui=gui_policy,
                )
            else:
                policy = primary_policy
        else:
            policy = BootstrapMotorPolicy()
        supervisor_status = send_command("status")
        camera_status = supervisor_status.get("world_camera")
        if isinstance(camera_status, dict):
            estimated_pitch = camera_status.get("estimated_pitch_units")
            restore_camera = getattr(policy, "restore_world_camera_state", None)
            if isinstance(estimated_pitch, int) and callable(restore_camera):
                restore_camera(estimated_pitch_units=estimated_pitch)
        executor = SkillExecutor(policy)
        trajectory: TrajectoryRecorder | None = None
        if config.trajectory.enabled:
            trajectory_id = new_trajectory_id("bedrock-agent")
            manifest = TrajectoryManifest(
                trajectory_id=trajectory_id,
                source=DatasetSource(
                    source_id=f"minecraft-ai:{trajectory_id}",
                    source_type=DatasetSourceType.BEDROCK_AGENT,
                    license="operator-owned-gameplay",
                    redistribution_allowed=False,
                    training_allowed=True,
                    edition="bedrock",
                    game_versions=(args.instance_id.split(":", 2)[1],),
                ),
                role=role.role_id,
                label="autonomous-play",
                game_version=args.instance_id.split(":", 2)[1],
                platform=platform.platform(),
                launcher_profile="bedrock-on-linux/winegdk",
                resolution=(capture_probe.width, capture_probe.height),
                started_ns=time.time_ns(),
            )
            trajectory = TrajectoryRecorder(
                manifest=manifest,
                artifact_root=paths.data_dir / "trajectories",
                state_db_path=paths.state_db,
                shard_steps=config.trajectory.shard_steps,
                queue_size=config.trajectory.queue_size,
            )
        # Startup and migrations may wait for the operator console, but the
        # realtime loop must never spend seconds blocked behind a UI write.
        database.set_busy_timeout_ms(100)
        runtime = AgentRuntime(
            perception=perception,
            blackboard=blackboard,
            executor=executor,
            skills=skills,
            role=role,
            lease_id=args.lease_id,
            high_level=high_level,
            memories=memories,
            social=social,
            state_db=database,
            motor_hz=config.motor_hz,
            cognition_hz=config.cognition_hz,
            semantic_hz=config.semantic_hz,
            lease_renew_ms=config.lease_renew_ms,
            stale_frame_consecutive_limit=config.stale_frame_consecutive_limit,
            trajectory=trajectory,
        )

        def _stop(_signum: int, _frame: object) -> None:
            runtime.stop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, _stop)
        runtime.run_forever()
        return 0
    finally:
        database.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
