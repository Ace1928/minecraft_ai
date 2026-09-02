from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ActionLevel(StrEnum):
    RAW = "raw"
    MOTION = "motion"
    LATENT = "latent"
    GROUNDED = "grounded"
    GUI = "gui"
    SKILL = "skill"


class DatasetSourceType(StrEnum):
    BEDROCK_AGENT = "bedrock_agent"
    BEDROCK_HUMAN = "bedrock_human"
    IMPORTED = "imported"
    SYNTHETIC = "synthetic"


class DatasetSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(min_length=1, max_length=256)
    source_type: DatasetSourceType
    license: str = Field(min_length=1, max_length=256)
    redistribution_allowed: bool
    training_allowed: bool
    edition: str
    game_versions: tuple[str, ...]
    provenance_url: str | None = None
    checksum: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")


class TrajectoryManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    trajectory_id: str
    source: DatasetSource
    role: str
    label: str
    task_id: str | None = None
    game_version: str
    platform: str
    launcher_profile: str
    resolution: tuple[int, int]
    fov: float | None = None
    mouse_sensitivity: float | None = None
    started_ns: int
    ended_ns: int | None = None
    accepted_steps: int = Field(default=0, ge=0)
    dropped_steps: int = Field(default=0, ge=0)
    shard_ids: tuple[str, ...] = ()
