from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from platformdirs import user_config_dir, user_data_dir
from pydantic import BaseModel, ConfigDict, Field


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    model_id: str = ""
    base_url: str = "http://127.0.0.1:8080/v1"
    api_key: str = "local"
    timeout_s: float = Field(default=60.0, gt=0.0, le=600.0)


class TrajectoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    shard_steps: int = Field(default=256, ge=16, le=4096)
    queue_size: int = Field(default=512, ge=32, le=8192)


class PolicyConfig(BaseModel):
    """Configuration for the isolated learned visuomotor policy process."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    provider: str = Field(
        pattern="^(openai-vpt|minestudio-steve1|minestudio-rocket2)$",
        default="openai-vpt",
    )
    python_path: str = ""
    source_path: str = ""
    model_path: str = ""
    weights_path: str = ""
    model_sha256: str = ""
    weights_sha256: str = ""
    model_version: str = ""
    source_commit: str = ""
    license: str = ""
    research_only: bool = False
    device: str = "cpu"
    threads: int = Field(default=4, ge=1, le=64)
    startup_timeout_s: float = Field(default=60.0, ge=5.0, le=600.0)
    deadline_ms: int = Field(default=48, ge=10, le=5000)
    action_hold_ms: int = Field(default=50, ge=50, le=250)
    stochastic: bool = True
    condition_scale: float = Field(default=4.0, ge=0.0, le=12.0)
    deterministic_condition: bool = True
    # STEVE-1 already carries a trained MineCLIP encoder. Reuse it at a low
    # event-driven cadence for scene-mode belief instead of adding a second
    # heavyweight VLM to the realtime loop. Zero disables periodic probes;
    # inventory/crafting intents still request an event-time verification.
    scene_probe_interval: int = Field(default=0, ge=0, le=10_000)
    # Optional evidence-gated Bedrock scene head. It runs on the exact frame
    # already consumed by STEVE, replacing brittle zero-shot scene prompts
    # while retaining MineCLIP as the compatibility fallback.
    scene_model_path: str = ""
    scene_model_sha256: str = ""
    scene_model_version: str = ""
    scene_min_confidence: float = Field(default=0.80, ge=0.5, le=1.0)
    # MineRL/VPT camera actions are degrees. At Minecraft Java's default 0.5
    # mouse sensitivity one relative count is 0.15 degrees. Bedrock has a
    # different sensitivity curve, so live deployments must replace this
    # default with empirically measured per-axis counts-per-degree calibration.
    # `camera_scale` remains the yaw/legacy scale so existing configs stay
    # readable; pitch may differ under WineGDK and must be measured separately.
    camera_scale: float = Field(default=20.0 / 3.0, ge=0.0, le=100.0)
    camera_pitch_scale: float | None = Field(default=None, gt=0.0, le=100.0)
    # When Minecraft releases the pointer for a GUI, X11 relative motion is in
    # screen pixels rather than world-camera degrees. Keep that independently
    # calibrated so a correct GUI prediction cannot inherit world sensitivity.
    gui_camera_scale: float = Field(default=1.0, ge=0.0, le=20.0)
    camera_max_step: int = Field(default=12, ge=0, le=100)
    # This is expressed in calibrated actuator counts, not model camera bins.
    # A Bedrock view needs roughly 90 degrees of authority in either direction;
    # low-sensitivity deployments can therefore require more than 2,000 counts.
    camera_pitch_limit: int = Field(default=300, ge=0, le=5000)
    camera_recovery_release: int = Field(default=100, ge=0, le=5000)
    seed: int = 1928

    @property
    def effective_camera_pitch_scale(self) -> float:
        return self.camera_scale if self.camera_pitch_scale is None else self.camera_pitch_scale


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str = "generalist"
    motor_hz: float = Field(default=20.0, ge=5.0, le=60.0)
    cognition_hz: float = Field(default=0.5, gt=0.0, le=10.0)
    # Zero is an intentional event-only mode: cognition and tactical events may
    # still submit active-perception questions, but no periodic VLM narration is
    # generated. This is the appropriate mode for a capable, slower local VLM.
    semantic_hz: float = Field(default=2.0, ge=0.0, le=20.0)
    stale_frame_ms: int = Field(default=500, ge=100, le=5000)
    stale_frame_consecutive_limit: int = Field(default=3, ge=1, le=20)
    lease_renew_ms: int = Field(default=500, ge=100, le=2000)
    high_level: ModelConfig = Field(default_factory=ModelConfig)
    vision_language: ModelConfig = Field(default_factory=ModelConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    grounded_policy: PolicyConfig | None = None
    gui_policy: PolicyConfig | None = None
    trajectory: TrajectoryConfig = Field(default_factory=TrajectoryConfig)
    online_wiki: bool = True


@dataclass(frozen=True)
class AppPaths:
    config_dir: Path
    data_dir: Path
    config_file: Path
    state_db: Path
    wiki_cache: Path
    knowledge_dir: Path


def app_paths() -> AppPaths:
    config_dir = Path(user_config_dir("minecraft-ai"))
    data_dir = Path(user_data_dir("minecraft-ai"))
    return AppPaths(
        config_dir=config_dir,
        data_dir=data_dir,
        config_file=config_dir / "config.yaml",
        state_db=data_dir / "state.sqlite3",
        wiki_cache=data_dir / "wiki-cache",
        knowledge_dir=data_dir / "knowledge",
    )


def default_config() -> RuntimeConfig:
    return RuntimeConfig()


def load_config(path: Path | None = None) -> RuntimeConfig:
    selected = app_paths().config_file if path is None else path
    if not selected.exists():
        return default_config()
    raw = yaml.safe_load(selected.read_text(encoding="utf-8"))
    if raw is None:
        return default_config()
    if not isinstance(raw, dict):
        raise ValueError(f"configuration root must be a mapping: {selected}")
    return RuntimeConfig.model_validate(raw)


def save_config(config: RuntimeConfig, path: Path | None = None) -> Path:
    selected = app_paths().config_file if path is None else path
    selected.parent.mkdir(parents=True, exist_ok=True)
    payload = config.model_dump(mode="json")
    staged = selected.with_name(f".{selected.name}.tmp")
    staged.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    try:
        staged.chmod(0o600)
    except OSError:
        pass
    staged.replace(selected)
    return selected


def ensure_default_config() -> Path:
    paths = app_paths()
    if paths.config_file.exists():
        return paths.config_file
    return save_config(default_config(), paths.config_file)
