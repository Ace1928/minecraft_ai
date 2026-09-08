from __future__ import annotations

import concurrent.futures
import time
from dataclasses import dataclass, field
from enum import Enum

from minecraft_ai.cognition import (
    CognitionDecision,
)
from minecraft_ai.crafting_control import PlankCraftPhase
from minecraft_ai.model_requests import RequestBinding
from minecraft_ai.perception import (
    PerceptionFact,
)

class SkillStartSource(str, Enum):
    """Why execution created a new run, not an attribution of model credit."""

    COGNITION = "cognition"
    RECOVERY = "recovery"
    CONTINUATION = "continuation"
    KEEPALIVE = "keepalive"
    BOOTSTRAP = "bootstrap"

@dataclass(frozen=True, slots=True)
class SkillDecisionOrigin:
    """Exact accepted model decision which started this run, if one exists."""

    request: RequestBinding
    attempt_id: str
    source_decision_sha256: str
    final_decision_sha256: str

@dataclass(frozen=True)
class _CraftSemanticProbe:
    run_id: str
    phase: PlankCraftPhase
    terminal_count_before: int

@dataclass(frozen=True)
class _CognitionPerceptionProbe:
    """One stable-scene observation requested by a perception-only decision."""

    query_id: str | None
    requested_keys: tuple[str, ...]
    frame_id: int
    execution_revision: int
    terminal_count_before: int | None
    settle_dhash: str | None
    settle_deadline_ns: int
    grounding_deadline_ns: int | None = None
    handoff_deadline_ns: int | None = None
    query_source: str | None = None
    retained_facts: tuple[PerceptionFact, ...] = ()
    cognition_future: concurrent.futures.Future[CognitionDecision] | None = None
    trigger_run_id: str | None = None
    trigger_decision: CognitionDecision | None = None
    # Explicit decision description only; never inferred target evidence or authority.
    target_description: str | None = None

@dataclass
class _GatherAcquisitionContinuation:
    """One volatile, evidence-bound three-log gather transaction."""

    context_key: str
    parameters: dict[str, str | int | float | bool]
    instruction: str | None
    active_run_id: str
    last_exact_count: int
    resource_acquired_events: int = 0

@dataclass
class _HeadroomRecovery:
    """One fail-closed clear-and-retry transaction after a verified traversal stall."""

    context_key: str
    traversal_parameters: dict[str, str | int | float | bool]
    deadline_ns: int
    origin_skill_id: str = "traverse_visible_obstacle"
    origin_run_id: str | None = None
    phase: str = "reorient"
    reoriented_frame_id: int | None = None
    reorientation_moved: bool = False
    pre_reorient_dhash: str | None = None
    settle_deadline_ns: int | None = None
    settle_frame_id: int | None = None
    settle_crosshair_dhash: str | None = None
    settle_rgb_grid: str | None = None
    settle_stable_successors: int = 0
    query_id: str | None = None
    query_started_ns: int = 0
    query_frame_dhash: str | None = None
    query_crosshair_dhash: str | None = None
    query_frame_id: int | None = None
    query_captured_ns: int | None = None
    query_frame_width: int | None = None
    query_frame_height: int | None = None
    query_pixel_sha256: str | None = None
    query_rgb_grid: str | None = None
    query_source: str | None = None
    mining_run_id: str | None = None
    retry_run_id: str | None = None
    target_track_id: str | None = None

@dataclass(frozen=True)
class _HeadroomTarget:
    kind: str
    source: str
    evidence_id: str
    confidence: float
    observed_ns: int

@dataclass
class RuntimeMetrics:
    frames: int = 0
    motor_actions: int = 0
    cognition_calls: int = 0
    semantic_requests: int = 0
    operator_responses: int = 0
    game_chat_messages: int = 0
    skill_successes: int = 0
    skill_failures: int = 0
    skill_failed_outcomes: int = 0
    skill_timeouts: int = 0
    skill_cancellations: int = 0
    started_ns: int = field(default_factory=time.monotonic_ns)
    last_capture_ms: float = 0.0
    last_motor_ms: float = 0.0
    stale_frame_skips: int = 0
    consecutive_stale_frames: int = 0
    storage_contentions: int = 0
    last_storage_error: str | None = None

