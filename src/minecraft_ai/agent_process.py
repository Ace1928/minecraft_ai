from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

from .builtin_skills import build_bootstrap_skill_library
from .cognition import HighLevelController
from .config import app_paths, load_config
from .execution import SkillExecutor
from .models import OpenAICompatibleLocalModel
from .motor import HeuristicMotorPolicy
from .perception import PerceptionBlackboard
from .perception_service import ActiveVLMWorker, RealtimePerceptionService
from .platforms.bedrock_x11 import IsolatedX11Capture
from .roles import get_role
from .runtime import AgentRuntime
from .storage import StateDatabase


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minecraft AI realtime agent process")
    parser.add_argument("--lease-id", required=True)
    parser.add_argument("--display", required=True)
    parser.add_argument("--window-id", required=True, type=int)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--role", default=None)
    parser.add_argument("--config", default=None)
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
        capture = IsolatedX11Capture(args.display, args.window_id)

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
        executor = SkillExecutor(HeuristicMotorPolicy())
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
