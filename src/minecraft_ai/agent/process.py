from __future__ import annotations

import argparse
import platform
import sys
import time
from collections.abc import Callable
from pathlib import Path

from minecraft_ai.builtin_skills import build_bootstrap_skill_library
from minecraft_ai.cognition import HighLevelController
from minecraft_ai.config import RuntimeConfig, app_paths, load_config
from minecraft_ai.datasets import DatasetSource, DatasetSourceType, TrajectoryManifest
from minecraft_ai.execution import SkillExecutor
from minecraft_ai.models import OpenAICompatibleLocalModel
from minecraft_ai.motor import BootstrapMotorPolicy, MotorPolicy
from minecraft_ai.perception import PerceptionBlackboard
from minecraft_ai.perception_service import ActiveVLMWorker, RealtimePerceptionService
from minecraft_ai.platforms import create_bedrock_capture
from minecraft_ai.platforms.bedrock_session import BedrockSession
from minecraft_ai.platforms.bedrock_x11 import CapturedFrame, IsolationError
from minecraft_ai.platforms.capture_source import BedrockCaptureSource
from minecraft_ai.policy_service import GroundedPolicyRouter, TemporalPolicyClient
from minecraft_ai.roles import get_role
from minecraft_ai.runtime_factory import RuntimeStartupCleanupIncomplete, run_agent_runtime
from minecraft_ai.storage import StateDatabase
from minecraft_ai.supervisor import send_command
from minecraft_ai.trajectory import TrajectoryRecorder, new_trajectory_id


def build_motor_policy(
    config: RuntimeConfig,
    *,
    frame_provider: Callable[[], CapturedFrame | None],
) -> MotorPolicy:
    """Assemble the configured body experts independently of one another.

    RAW/MOTION and GUI specialists no longer require a GROUNDED observer.
    """
    if not config.policy.enabled:
        return BootstrapMotorPolicy()
    primary = TemporalPolicyClient(config.policy, frame_provider=frame_provider)
    grounded = (
        None
        if config.grounded_policy is None or not config.grounded_policy.enabled
        else TemporalPolicyClient(config.grounded_policy, frame_provider=frame_provider)
    )
    gui = (
        None
        if config.gui_policy is None or not config.gui_policy.enabled
        else TemporalPolicyClient(config.gui_policy, frame_provider=frame_provider)
    )
    raw_motion = (
        None
        if config.raw_motion_policy is None or not config.raw_motion_policy.enabled
        else TemporalPolicyClient(
            config.raw_motion_policy, frame_provider=frame_provider
        )
    )
    if grounded is None and gui is None and raw_motion is None:
        return primary
    return GroundedPolicyRouter(
        primary,
        grounded=grounded,
        gui=gui,
        raw_motion=raw_motion,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minecraft AI realtime agent process")
    parser.add_argument("--lease-id", required=True)
    parser.add_argument("--display", required=True)
    parser.add_argument("--window-id", required=True, type=int)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--role", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--allow-host-capture", action="store_true")
    parser.add_argument(
        "--capture-source",
        type=BedrockCaptureSource,
        default=BedrockCaptureSource.PIPEWIRE,
        choices=tuple(BedrockCaptureSource),
    )
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
    close_database = True
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
        host_binding = None
        if bool(args.allow_host_capture):
            try:
                session = BedrockSession.load()
                host_binding = session.host_monitor_binding()
            except (OSError, ValueError, TypeError, KeyError) as exc:
                raise IsolationError(
                    "host capture requires a valid managed Bedrock session"
                ) from exc
            if host_binding is None:
                raise IsolationError("host capture requires a persisted exact host-monitor binding")
        capture = create_bedrock_capture(
            args.display,
            args.window_id,
            allow_host=bool(args.allow_host_capture),
            host_monitor_binding=host_binding,
            source=args.capture_source,
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
                max_tokens=config.high_level.max_tokens,
                thinking_budget_tokens=config.high_level.thinking_budget_tokens,
                reasoning_format=config.high_level.reasoning_format,
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
                max_tokens=config.vision_language.max_tokens,
                thinking_budget_tokens=config.vision_language.thinking_budget_tokens,
                reasoning_format=config.vision_language.reasoning_format,
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
        policy = build_motor_policy(config, frame_provider=lambda: perception.last_capture)
        supervisor_status = send_command("status")
        camera_status = supervisor_status.get("world_camera")
        if isinstance(camera_status, dict):
            estimated_pitch = camera_status.get("estimated_pitch_units")
            restore_camera = getattr(policy, "restore_world_camera_state", None)
            if isinstance(estimated_pitch, int) and callable(restore_camera):
                restore_camera(estimated_pitch_units=estimated_pitch)
        executor = SkillExecutor(policy)
        executor.configure_mining_timing(
            max_hold_ms=config.mining_max_hold_ms,
            acquisition_ms=config.mining_acquisition_timeout_ms,
        )
        if config.mining_rule_snapshot is not None:
            from minecraft_ai.control.mining_knowledge import MiningKnowledge, MiningRuleSnapshot
            snapshot = MiningRuleSnapshot.load(Path(config.mining_rule_snapshot).expanduser())
            executor.configure_mining_knowledge(MiningKnowledge(memories, snapshot.scope, snapshot))
        trajectory: TrajectoryRecorder | None = None
        trajectory_disabled_reason: str | None = (
            None if config.trajectory.enabled else "disabled-by-configuration"
        )
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
            try:
                trajectory = TrajectoryRecorder(
                    manifest=manifest,
                    artifact_root=paths.data_dir / "trajectories",
                    state_db_path=paths.state_db,
                    shard_steps=config.trajectory.shard_steps,
                    queue_size=config.trajectory.queue_size,
                    frame_max_width=config.trajectory.frame_max_width,
                    frame_jpeg_quality=config.trajectory.frame_jpeg_quality,
                    min_free_disk_bytes=int(config.trajectory.min_free_disk_gib * 1024**3),
                )
            except Exception as exc:
                # Gameplay remains available when storage is tight. The shard
                # writer applies the same reserve before every later sample;
                # other recorder setup faults must also fail open for play.
                print(f"trajectory recording disabled: {exc}", file=sys.stderr, flush=True)
                trajectory_disabled_reason = f"{type(exc).__name__}: {exc}"
        # Startup and migrations may wait for the operator console, but the
        # realtime loop must never spend seconds blocked behind a UI write.
        database.set_busy_timeout_ms(100)
        runtime_kwargs = dict(
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
            mining_ruleset_id=config.mining_ruleset_id,
            motor_hz=config.motor_hz,
            cognition_hz=config.cognition_hz,
            semantic_hz=config.semantic_hz,
            lease_renew_ms=config.lease_renew_ms,
            stale_frame_consecutive_limit=config.stale_frame_consecutive_limit,
            trajectory=trajectory,
            trajectory_disabled_reason=trajectory_disabled_reason,
        )

        run_agent_runtime(runtime_kwargs, factory_config=config.runtime_factory)
        return 0
    except RuntimeStartupCleanupIncomplete:
        close_database = False
        raise
    finally:
        if close_database:
            database.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
