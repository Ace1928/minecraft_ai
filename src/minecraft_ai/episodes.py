from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class EpisodeOutcome(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class Episode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_id: str
    trajectory_id: str
    first_step_index: int = Field(ge=0)
    last_step_index: int | None = Field(default=None, ge=0)
    started_ns: int
    ended_ns: int | None = None
    goal_id: str | None = None
    skill_run_id: str | None = None
    outcome: EpisodeOutcome = EpisodeOutcome.RUNNING
    reward_total: float = 0.0
    event_ids: tuple[str, ...] = ()
    summary: str = ""


class RuntimeEventKind(StrEnum):
    DAMAGE_TAKEN = "damage_taken"
    DEATH = "death"
    INVENTORY_CHANGED = "inventory_changed"
    GUI_CHANGED = "gui_changed"
    BLOCK_BROKEN = "block_broken"
    TARGET_LOST = "target_lost"
    WORLD_MODEL_SURPRISE = "world_model_surprise"
    HUMAN_CORRECTION = "human_correction"
    SKILL_SUCCEEDED = "skill_succeeded"
    SKILL_FAILED = "skill_failed"


class RuntimeEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str
    kind: RuntimeEventKind
    observed_ns: int
    trajectory_id: str | None = None
    step_index: int | None = Field(default=None, ge=0)
    payload: dict[str, str | int | float | bool] = Field(default_factory=dict)
