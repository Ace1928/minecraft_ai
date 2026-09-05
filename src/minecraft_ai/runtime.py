from __future__ import annotations

import concurrent.futures
import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterable
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path

from .cognition import (
    BootstrapCognitionPolicy,
    CognitionContext,
    CognitionDecision,
    HighLevelController,
    planks_retry_requires_wood,
)
from .action_levels import ActionLevel
from .curriculum import CurriculumCandidate, CurriculumScheduler, role_standing_goals
from .crafting_control import PlankCraftPhase
from .daemon_executor import SingleWorkerDaemonExecutor
from .episodes import RuntimeEvent, RuntimeEventKind
from .emergency import emergency_stop_latched
from .execution import ExecutionTick, SkillExecutor, initiation_satisfied
from .grounded_perception import (
    CROSSHAIR_BLOCK_FAST_SOURCE,
    crosshair_block_crop_dimensions,
    crosshair_block_pixel_sha256,
    crosshair_block_region,
    crosshair_block_rgb_grid,
    crosshair_block_rgb_grid_distance,
    crosshair_block_visually_equivalent,
    resolve_grounded_output_keys,
)
from .memory import MemoryKind, MemoryRecord, MemoryStore
from .mining_control import (
    is_hand_safe_soft_block,
    normalize_block_kind,
)
from .models import local_model_inference_available
from .motor import MotorIntent
from .outcome_verifier import OutcomeKind, OutcomeSignal, OutcomeStatus, OutcomeVerification
from .perception import (
    ActivePerceptionQuery,
    EvidenceRegion,
    PerceptionBlackboard,
    PerceptionFact,
    PerceptionQueryMode,
    ScreenRegion,
    Track,
)
from .perception_service import (
    BEDROCK_HUD_SAFETY_SOURCE,
    BEDROCK_HOTBAR_LOG_COUNT_SOURCE,
    RealtimePerceptionService,
    crosshair_block_dhash,
    frame_dhash,
    perceptual_hash_distance,
)
from .planning import Goal
from .platforms.bedrock_x11 import CapturedFrame
from .roles import RoleProfile
from .safety import MotorAction
from .skills import (
    SkillLibrary,
    SkillFailureCode,
    SkillOutcome,
    SkillRun,
    SkillSpec,
    SkillStats,
)
from .social import (
    OperatorMessage,
    OperatorMessageKind,
    OperatorMessageStatus,
    SocialState,
)
from .telemetry import TelemetryPublisher
from .trajectory import ActionOrigin, ActionProvenance, TrajectoryRecorder, motor_condition_id
from .storage import StateDatabase
from .supervisor import operator_pause_latched, send_command


_EXPLORE_KEEPALIVE_CONTEXT = "explore-keepalive"
_BOUNDED_KEEPALIVE_SKILL_IDS = frozenset(
    {"explore_forward", "traverse_level_ground", "traverse_visible_obstacle"}
)
_ATOMIC_SKILL_IDS = frozenset(
    {
        "open_inventory",
        "close_open_inventory",
        "collect_recent_drop",
        "respawn_after_death",
    }
)
_WOOD_INVENTORY_AUDIT_SKILLS = frozenset({"craft_wood_planks", "open_inventory"})
_RECORDED_RUN_ID_LIMIT = 4_096
_COGNITION_RETRY_BASE_NS = 2_000_000_000
_COGNITION_RETRY_MAX_NS = 30_000_000_000
_COGNITION_PERCEPTION_SETTLE_TIMEOUT_NS = 2_000_000_000
_COGNITION_PERCEPTION_GROUNDING_TIMEOUT_NS = 180_000_000_000
_COGNITION_PERCEPTION_HANDOFF_TIMEOUT_NS = 90_000_000_000
_COGNITION_PERCEPTION_ACTION_GRACE_NS = 2_000_000_000
_OPERATOR_FOLLOWUP_DELAY_NS = 250_000_000
_CRAFT_SEMANTIC_LATENCY_MARGIN = 1.25
_CRAFT_SEMANTIC_MAX_REQUIRED_BUDGET_MS = 60_000
_PLANKS_NO_LOGS_REASON = "crafting-no-logs-observed-in-inventory"
_PLANKS_RETRY_CLEAR_MEMORY = "working:planks-retry-positive-log-evidence"
# Positive Bedrock pitch looks down. Recovery aims at a small absolute
# downward pitch from the calibrated horizon instead of adding a fixed nudge
# to whatever extreme pose the learned controller left behind.
_HEADROOM_REORIENT_TARGET_PITCH_UNITS = 96
_HEADROOM_REORIENT_TARGET_TOLERANCE_UNITS = 32
_HEADROOM_REORIENT_MAX_ABS_DY = 96
_HEADROOM_MIN_TIMEOUT_S = 60.0
_HEADROOM_TIMEOUT_MULTIPLIER = 5.0
_HEADROOM_TIMEOUT_MARGIN_S = 5.0
_HEADROOM_TRANSACTION_MAX_S = 180.0
_HEADROOM_SETTLE_TIMEOUT_NS = 2_000_000_000
_HEADROOM_STABLE_SUCCESSOR_FRAMES = 2
_GATHER_ACQUISITIONS_REQUIRED = 3


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


def _headroom_deadline_ns(active_vlm: object, *, now_ns: int) -> int:
    """Bound recovery without imposing a competing timeout on the VLM call.

    The local inference lane is serialized and grounded inspection permits one
    schema repair. The configured timeout sizes a conservative wait budget, but
    a hard transaction cap ensures a busy or hung worker cannot wedge the agent.
    A late worker result is harmless because query ownership is checked again.
    """

    model = getattr(active_vlm, "model", None)
    configured = getattr(model, "timeout_s", _HEADROOM_MIN_TIMEOUT_S)
    timeout_s = (
        float(configured)
        if isinstance(configured, (int, float))
        and not isinstance(configured, bool)
        and configured > 0
        else _HEADROOM_MIN_TIMEOUT_S
    )
    timeout_s = max(_HEADROOM_MIN_TIMEOUT_S, timeout_s)
    budget_s = min(
        timeout_s * _HEADROOM_TIMEOUT_MULTIPLIER + _HEADROOM_TIMEOUT_MARGIN_S,
        _HEADROOM_TRANSACTION_MAX_S,
    )
    return now_ns + int(budget_s * 1_000_000_000)


def _headroom_reorient_mouse_dy(current_pitch_units: int) -> int:
    """Return one bounded delta toward the calibrated near-ground pose."""

    delta = _HEADROOM_REORIENT_TARGET_PITCH_UNITS - current_pitch_units
    if abs(delta) <= _HEADROOM_REORIENT_TARGET_TOLERANCE_UNITS:
        return 0
    return max(
        -_HEADROOM_REORIENT_MAX_ABS_DY,
        min(_HEADROOM_REORIENT_MAX_ABS_DY, delta),
    )


def _restore_policy_world_camera(policy: object, *, pitch_units: int) -> None:
    """Synchronize learned routes after a runtime-owned physical camera action."""

    restore = getattr(policy, "restore_world_camera_state", None)
    if not callable(restore):
        return
    restore(estimated_pitch_units=pitch_units)


def _expected_keepalive_expiry(run: SkillRun) -> bool:
    return (
        run.outcome == SkillOutcome.TIMED_OUT
        and run.context_key == _EXPLORE_KEEPALIVE_CONTEXT
        and run.skill_id in _BOUNDED_KEEPALIVE_SKILL_IDS
    )


def _plan_step_requests_inventory_transition(skill_id: str, step: str) -> bool:
    """Require an explicit GUI plan node before its transition can consume progress."""

    normalized = " ".join(step.casefold().replace("_", " ").replace("-", " ").split())
    if skill_id == "open_inventory":
        prefixes = {
            "open inventory", "open the inventory", "inspect inventory",
            "inspect the inventory", "check inventory", "check the inventory",
            "audit inventory", "audit the inventory", "view inventory", "view the inventory",
        }
        return any(
            normalized == prefix or normalized.startswith(f"{prefix} ")
            for prefix in prefixes
        )
    if skill_id == "close_open_inventory":
        prefixes = {
            "close inventory", "close the inventory", "exit inventory",
            "exit the inventory", "leave inventory", "leave the inventory",
        }
        return any(
            normalized == prefix or normalized.startswith(f"{prefix} ")
            for prefix in prefixes
        )
    if skill_id == "activate_visible_gui_control":
        words = set(normalized.split())
        return bool(
            words.intersection({"activate", "choose", "click", "press", "select"})
            and words.intersection(
                {"button", "control", "gui", "menu", "play", "server", "tab", "world"}
            )
        )
    return True


def _verified_log_break(verification: OutcomeVerification | None) -> bool:
    """Return whether exact bound mining evidence identifies a vanilla log."""

    if (
        verification is None
        or verification.kind != OutcomeKind.MINING
        or verification.status != OutcomeStatus.SUCCEEDED
        or verification.signal != OutcomeSignal.BLOCK_BROKEN
        or not isinstance(verification.target_kind, str)
    ):
        return False
    target = verification.target_kind.casefold().removeprefix("minecraft:")
    return target == "log" or target.endswith("_log")


def _verified_oak_log_break(verification: OutcomeVerification | None) -> bool:
    """Accept only the oak species covered by the deterministic count observer."""

    if not _verified_log_break(verification) or verification is None:
        return False
    target = verification.target_kind
    return bool(
        isinstance(target, str)
        and target.casefold().removeprefix("minecraft:") == "oak_log"
    )


def _exact_frozen_log_count(fact: PerceptionFact | None, run: SkillRun) -> int | None:
    """Validate a canonical pre-attack count without expiring its frozen snapshot."""

    if (
        fact is None
        or fact.key != "inventory.hotbar.logs"
        or fact.source != BEDROCK_HOTBAR_LOG_COUNT_SOURCE
        or fact.confidence < 0.99
        or not isinstance(fact.value, int)
        or isinstance(fact.value, bool)
        or fact.value < 0
        or run.ended_ns is None
        or fact.observed_ns > run.ended_ns
    ):
        return None
    return fact.value


def _verified_gather_acquisition(
    result: ExecutionTick,
    continuation: _GatherAcquisitionContinuation,
    *,
    exact_count: int | None,
) -> bool:
    """Require this continuation's active collection run and exact next count."""

    verification = result.outcome_verification
    return bool(
        continuation.resource_acquired_events < _GATHER_ACQUISITIONS_REQUIRED
        and continuation.active_run_id == result.run.run_id
        and continuation.context_key == result.run.context_key
        and result.run.skill_id == "collect_recent_drop"
        and result.run.outcome == SkillOutcome.SUCCEEDED
        and verification is not None
        and verification.run_id == result.run.run_id
        and verification.kind == OutcomeKind.RESOURCE_ACQUISITION
        and verification.status == OutcomeStatus.SUCCEEDED
        and verification.signal == OutcomeSignal.RESOURCE_ACQUIRED
        and verification.target_kind == "log"
        and "inventory.hotbar.logs" in verification.evidence_keys
        and isinstance(exact_count, int)
        and not isinstance(exact_count, bool)
        and exact_count == continuation.last_exact_count + 1
    )


def _verified_obstacle_stall(result: ExecutionTick) -> bool:
    """Accept only an exact action-bound traversal stall from an eligible option."""

    verification = result.outcome_verification
    return bool(
        result.run.skill_id in {"gather_nearby_wood", "traverse_visible_obstacle"}
        and result.run.outcome == SkillOutcome.FAILED
        and result.run.failure_code == SkillFailureCode.LOCOMOTION_STALLED
        and verification is not None
        and verification.run_id == result.run.run_id
        and verification.kind == OutcomeKind.TRAVERSAL
        and verification.status == OutcomeStatus.STALLED
        and verification.signal == OutcomeSignal.LOCOMOTION_STALLED
    )


def _verified_block_break(result: ExecutionTick) -> bool:
    verification = result.outcome_verification
    return bool(
        result.run.skill_id == "mine_visible_block"
        and result.run.outcome == SkillOutcome.SUCCEEDED
        and verification is not None
        and verification.run_id == result.run.run_id
        and verification.kind == OutcomeKind.MINING
        and verification.status == OutcomeStatus.SUCCEEDED
        and verification.signal == OutcomeSignal.BLOCK_BROKEN
    )


def _verified_traversal_progress(result: ExecutionTick) -> bool:
    """Accept only a transaction-owned retry's exact locomotion progress proof."""

    verification = result.outcome_verification
    return bool(
        result.run.skill_id == "traverse_visible_obstacle"
        and result.run.outcome == SkillOutcome.SUCCEEDED
        and verification is not None
        and verification.run_id == result.run.run_id
        and verification.kind == OutcomeKind.TRAVERSAL
        and verification.status == OutcomeStatus.PROGRESS
        and verification.signal == OutcomeSignal.LOCOMOTION_PROGRESS
    )


def _verified_headroom_retry(
    result: ExecutionTick,
    recovery: _HeadroomRecovery | None,
) -> bool:
    """Bind verified traversal progress to this transaction's sole retry run."""

    return bool(
        recovery is not None
        and recovery.phase == "retry"
        and recovery.retry_run_id == result.run.run_id
        and recovery.context_key == result.run.context_key
        and _verified_traversal_progress(result)
    )


def _headroom_retry_advances_plan(
    result: ExecutionTick,
    recovery: _HeadroomRecovery | None,
    *,
    plan_steps: tuple[str, ...],
    plan_index: int,
    plan_goal_id: str | None,
) -> bool:
    """Consume a plan step only when that active plan owned the recovered run."""

    if not _verified_headroom_retry(result, recovery) or recovery is None:
        return False
    if recovery.origin_skill_id != "traverse_visible_obstacle":
        # Clearing terrain while gathering restores mobility; it does not
        # prove that any log was acquired or complete the gather plan node.
        return False
    if recovery.context_key == _EXPLORE_KEEPALIVE_CONTEXT:
        return False
    expected_context = plan_goal_id or "default"
    return bool(
        0 <= plan_index < len(plan_steps)
        and recovery.context_key == expected_context
    )


def _headroom_clear_target(
    blackboard: PerceptionBlackboard,
    recovery: _HeadroomRecovery,
    *,
    now_ns: int,
    current_frame: CapturedFrame | None,
) -> _HeadroomTarget | None:
    """Resolve one current, query-owned, hand-safe center classification."""

    query_id = recovery.query_id
    requested_crop_hash = recovery.query_crosshair_dhash
    frame_id = recovery.query_frame_id
    captured_ns = recovery.query_captured_ns
    if (
        query_id is None
        or requested_crop_hash is None
        or recovery.query_frame_dhash is None
        or frame_id is None
        or captured_ns is None
        or recovery.query_frame_width is None
        or recovery.query_frame_height is None
        or recovery.query_pixel_sha256 is None
        or recovery.query_rgb_grid is None
        or recovery.query_source is None
    ):
        return None
    block = blackboard.fact("recovery.crosshair.block", min_confidence=0.70, now_ns=now_ns)
    crop_hash = blackboard.fact(
        "recovery.crosshair.observation_dhash", min_confidence=1.0, now_ns=now_ns
    )
    source_hash = blackboard.fact(
        "recovery.crosshair.frame_dhash", min_confidence=1.0, now_ns=now_ns
    )
    current_hash = blackboard.fact(
        "frame.crosshair_block_dhash", min_confidence=1.0, now_ns=now_ns
    )
    current_grid = blackboard.fact(
        "frame.crosshair_block_rgb_grid", min_confidence=1.0, now_ns=now_ns
    )
    facts = (block, crop_hash, source_hash)
    if any(fact is None for fact in facts) or current_hash is None or current_grid is None:
        return None
    assert block is not None and crop_hash is not None and source_hash is not None
    source = block.source
    observed_ns = block.observed_ns
    evidence_id = f"frame-{frame_id}:crosshair-block"
    if (
        source != recovery.query_source
        or any(fact is None or fact.source != source for fact in facts)
        or any(fact is None or fact.observed_ns != observed_ns for fact in facts)
        or observed_ns <= recovery.query_started_ns
        or observed_ns > now_ns
        or block.evidence_refs != (evidence_id,)
        or not isinstance(block.value, str)
        or crop_hash.value != requested_crop_hash
        or source_hash.value != recovery.query_frame_dhash
        or not isinstance(current_hash.value, str)
        or not isinstance(current_grid.value, str)
    ):
        return None
    normalized_kind = normalize_block_kind(block.value)
    if not is_hand_safe_soft_block(normalized_kind):
        return None
    latest = blackboard.latest()
    raw_latest = blackboard.raw_latest()
    if (
        latest is None
        or raw_latest is None
        or current_frame is None
        or current_frame.captured_ns != raw_latest.captured_ns
        or current_hash.source != CROSSHAIR_BLOCK_FAST_SOURCE
        or current_grid.source != CROSSHAIR_BLOCK_FAST_SOURCE
        or not raw_latest.captured_ns <= current_hash.observed_ns <= now_ns
        or not raw_latest.captured_ns <= current_grid.observed_ns <= now_ns
    ):
        return None
    evidence = tuple(item for item in latest.evidence if item.evidence_id == evidence_id)
    expected_region = crosshair_block_region(
        recovery.query_frame_width,
        recovery.query_frame_height,
    )
    expected_crop_width, expected_crop_height = crosshair_block_crop_dimensions(
        recovery.query_frame_width,
        recovery.query_frame_height,
    )
    if (
        len(evidence) != 1
        or evidence[0].frame_id != frame_id
        or evidence[0].captured_ns != captured_ns
        or evidence[0].region_kind != EvidenceRegion.WORLD
        or evidence[0].region != expected_region
        or evidence[0].crop_width != expected_crop_width
        or evidence[0].crop_height != expected_crop_height
        or evidence[0].pixel_sha256 != recovery.query_pixel_sha256
    ):
        return None
    if (
        current_frame.width != recovery.query_frame_width
        or current_frame.height != recovery.query_frame_height
    ):
        return None
    if not crosshair_block_visually_equivalent(
        requested_crop_hash,
        current_hash.value,
        recovery.query_rgb_grid,
        current_grid.value,
    ):
        return None
    return _HeadroomTarget(
        kind=normalized_kind,
        source=source,
        evidence_id=evidence_id,
        confidence=block.confidence,
        observed_ns=observed_ns,
    )


def _semantic_deadline_ms(semantic_hz: float) -> int:
    """Bound request lifetime independently from a slower query cadence."""
    if semantic_hz <= 0:
        raise ValueError("periodic semantic frequency must be positive")
    return min(10_000, max(250, int(1000 / semantic_hz)))


def _accepted_action_provenance(
    execution: ExecutionTick | None,
    blackboard: PerceptionBlackboard,
    *,
    fallback_policy_id: str,
) -> ActionProvenance:
    """Resolve the exact route snapshot that produced a supervisor-bound action."""

    status = {} if execution is None else execution.policy_status
    is_reset = execution is not None and execution.action_origin == ActionOrigin.RESET
    route_value = "reset" if is_reset else status.get("active_route", "direct")
    route_id = route_value if isinstance(route_value, str) and route_value else "direct"
    component_key = "primary" if route_id == "semantic" else route_id
    component = status.get(component_key)
    selected = component if isinstance(component, dict) else status
    causal = selected.get("last_action_provenance")
    causal_fields = causal if isinstance(causal, dict) and not is_reset else {}
    policy_value = causal_fields.get("policy_id", selected.get("policy_id"))
    policy_id = (
        policy_value if isinstance(policy_value, str) and policy_value else fallback_policy_id
    )
    version_value = selected.get("model_version")
    model_version = version_value if isinstance(version_value, str) and version_value else None
    prediction = selected.get("last_prediction")
    prediction_fields = prediction if isinstance(prediction, dict) else {}
    behavior_value = causal_fields.get(
        "behavior_token",
        prediction_fields.get("behavior_token"),
    )
    behavior_token = (
        behavior_value
        if isinstance(behavior_value, int)
        and not isinstance(behavior_value, bool)
        and behavior_value >= 0
        else None
    )
    latent_value = causal_fields.get("latent_id", prediction_fields.get("latent_id"))
    latent_id = latent_value if isinstance(latent_value, str) and latent_value else None
    action_kind_value = causal_fields.get("action_kind")
    policy_action_kind = (
        action_kind_value
        if isinstance(action_kind_value, str) and action_kind_value
        else ("reset" if is_reset else "direct")
    )
    request_value = causal_fields.get("request_id")
    policy_request_id = request_value if isinstance(request_value, str) and request_value else None
    prediction_value = causal_fields.get("prediction_id")
    prediction_id = (
        prediction_value if isinstance(prediction_value, str) and prediction_value else None
    )
    intent = None if execution is None else execution.motor_intent
    causal_condition = causal_fields.get("condition")
    if is_reset or (causal and causal_condition is None):
        condition = None
    elif isinstance(causal_condition, dict):
        condition = causal_condition
    else:
        condition = None if intent is None else intent.model_dump(mode="json")
    causal_target = causal_fields.get("target_track_id")
    if is_reset:
        target_track_id = None
    elif isinstance(causal, dict):
        target_track_id = (
            causal_target if isinstance(causal_target, str) and causal_target else None
        )
    else:
        target_track_id = _condition_target_track_id(intent, blackboard)
    causal_version = causal_fields.get("model_version")
    if isinstance(causal_version, str) and causal_version:
        model_version = causal_version
    condition_id = (
        None
        if condition is None
        else motor_condition_id(
            condition,
            route_id=route_id,
            target_track_id=target_track_id,
        )
    )
    action_level = _reported_action_level(execution, status, causal_fields)
    return ActionProvenance(
        policy_id=policy_id,
        model_version=model_version,
        route_id=route_id,
        policy_action_kind=policy_action_kind,
        policy_request_id=policy_request_id,
        prediction_id=prediction_id,
        action_level=action_level,
        origin=(ActionOrigin.POLICY if execution is None else execution.action_origin),
        condition_id=condition_id,
        condition=condition,
        behavior_token=behavior_token,
        latent_id=latent_id,
        target_track_id=target_track_id,
    )


def _trajectory_outcome_annotations(
    execution: ExecutionTick | None,
) -> tuple[dict[str, float], tuple[str, ...]]:
    if execution is None or execution.outcome_verification is None:
        return {}, ()
    verification = execution.outcome_verification
    if (
        execution.run.outcome != SkillOutcome.SUCCEEDED
        or verification.run_id != execution.run.run_id
        or verification.status != OutcomeStatus.SUCCEEDED
        or verification.signal not in {
            OutcomeSignal.BLOCK_BROKEN, OutcomeSignal.RESOURCE_ACQUIRED
        }
    ):
        return {}, ()
    event_suffix = verification.signal.value.replace("_", "-")
    return (
        {verification.signal.value: verification.confidence},
        (f"skill-run:{execution.run.run_id}:{event_suffix}",),
    )


def _reported_action_level(
    execution: ExecutionTick | None,
    status: dict[str, object],
    causal_fields: dict[str, object],
) -> ActionLevel:
    """Prefer the condition that causally produced an asynchronous action."""

    causal_level = causal_fields.get("action_level")
    if not isinstance(causal_level, str):
        causal_condition = causal_fields.get("condition")
        if isinstance(causal_condition, dict):
            causal_level = causal_condition.get("action_level")
    if isinstance(causal_level, str):
        try:
            return ActionLevel(causal_level)
        except ValueError:
            pass
    if execution is not None and execution.motor_intent is not None:
        return execution.motor_intent.action_level
    reported = status.get("episode_action_level")
    if isinstance(reported, str):
        try:
            return ActionLevel(reported)
        except ValueError:
            pass
    return ActionLevel.RAW


def _condition_target_track_id(
    intent: MotorIntent | None,
    blackboard: PerceptionBlackboard,
) -> str | None:
    if intent is None:
        return None
    latest = blackboard.latest()
    if latest is None:
        return None
    candidates = tuple(
        track
        for track in latest.tracks
        if intent.target_label is None or track.label.casefold() == intent.target_label.casefold()
    )
    if not candidates:
        return None
    return max(candidates, key=lambda track: track.confidence).track_id


def _semantic_refresh_allowed(
    *,
    cognition_requested: bool,
    cognition_pending: bool,
    operator_message_pending: bool,
    worker_available: bool,
) -> bool:
    """Keep optional semantic refreshes behind strategic and operator work."""
    return not (
        cognition_requested or cognition_pending or operator_message_pending or not worker_available
    )


def _sqlite_writer_contention(exc: sqlite3.OperationalError) -> bool:
    detail = str(exc).casefold()
    return "locked" in detail or "busy" in detail


def _terminal_run_event(
    run: SkillRun,
    *,
    observed_ns: int,
    trajectory_id: str | None,
) -> RuntimeEvent:
    """Build the append-only fact for one terminal skill execution."""

    event_kinds = {
        SkillOutcome.SUCCEEDED: RuntimeEventKind.SKILL_SUCCEEDED,
        SkillOutcome.FAILED: RuntimeEventKind.SKILL_FAILED,
        SkillOutcome.TIMED_OUT: RuntimeEventKind.SKILL_TIMED_OUT,
        SkillOutcome.CANCELLED: RuntimeEventKind.SKILL_CANCELLED,
    }
    try:
        kind = event_kinds[run.outcome]
    except KeyError as exc:
        raise ValueError("cannot create an event for a running skill") from exc
    payload: dict[str, str | int | float | bool] = {
        "run_id": run.run_id,
        "skill_id": run.skill_id,
        "context_key": run.context_key,
        "outcome": run.outcome.value,
        "started_monotonic_ns": run.started_ns,
        "parameters_json": json.dumps(
            run.parameters,
            sort_keys=True,
            separators=(",", ":"),
        ),
    }
    if run.ended_ns is not None:
        payload["ended_monotonic_ns"] = run.ended_ns
        payload["duration_ms"] = (run.ended_ns - run.started_ns) / 1_000_000
    if run.failure_reason is not None:
        payload["reported_reason"] = run.failure_reason
    if run.failure_code is not None:
        payload["failure_code"] = run.failure_code.value
    return RuntimeEvent(
        event_id=f"skill-run:{run.run_id}:terminal",
        kind=kind,
        observed_ns=observed_ns,
        trajectory_id=trajectory_id,
        payload=payload,
    )


def _verified_outcome_event(
    run: SkillRun,
    verification: OutcomeVerification,
    *,
    observed_ns: int,
    trajectory_id: str | None,
) -> RuntimeEvent | None:
    if (
        run.outcome != SkillOutcome.SUCCEEDED
        or verification.run_id != run.run_id
        or verification.status != OutcomeStatus.SUCCEEDED
        or verification.signal not in {
            OutcomeSignal.BLOCK_BROKEN, OutcomeSignal.RESOURCE_ACQUIRED
        }
    ):
        return None
    payload: dict[str, str | int | float | bool] = {
        "run_id": run.run_id,
        "skill_id": run.skill_id,
        "signal": verification.signal.value,
        "confidence": verification.confidence,
        "verified_monotonic_ns": verification.observed_ns,
        "reason": verification.reason,
        "evidence_keys_json": json.dumps(verification.evidence_keys),
    }
    if verification.target_kind is not None:
        payload["target_kind"] = verification.target_kind
    return RuntimeEvent(
        event_id=f"skill-run:{run.run_id}:{verification.signal.value.replace('_', '-')}",
        kind=(
            RuntimeEventKind.BLOCK_BROKEN
            if verification.signal == OutcomeSignal.BLOCK_BROKEN
            else RuntimeEventKind.RESOURCE_ACQUIRED
        ),
        observed_ns=observed_ns,
        trajectory_id=trajectory_id,
        payload=payload,
    )


def _terminal_run_memory(
    run: SkillRun,
    stats: SkillStats,
    *,
    observed_ns: int,
    existing: dict[str, MemoryRecord],
    outcome_verification: OutcomeVerification | None = None,
) -> MemoryRecord | None:
    """Create a stable, factual memory from a verified terminal outcome.

    Success/failure detection belongs to the skill contract. This function does
    not infer a cause or remedy from pixels; it only accumulates what the
    executor actually verified.
    """

    if run.outcome == SkillOutcome.CANCELLED:
        return None
    if _expected_keepalive_expiry(run):
        # Keepalives are deliberately bounded controller chunks. Expiry is the
        # normal scheduling boundary, not evidence that the skill is bad.
        return None
    if run.outcome == SkillOutcome.SUCCEEDED:
        kind = MemoryKind.PROCEDURAL
        identity = f"success:{run.skill_id}:{run.context_key}"
        prefix = "skill-procedural"
    elif run.outcome in {SkillOutcome.FAILED, SkillOutcome.TIMED_OUT}:
        kind = MemoryKind.FAILURE
        reason = run.failure_reason if run.failure_reason is not None else "<none-reported>"
        identity = f"{run.outcome.value}:{run.skill_id}:{run.context_key}:{reason}"
        prefix = "skill-failure"
    else:
        raise ValueError("cannot create memory for a running skill")

    memory_id = f"{prefix}:{uuid.uuid5(uuid.NAMESPACE_URL, f'minecraft-ai:{identity}').hex}"
    previous = existing.get(memory_id)
    previous_occurrences = 0
    if previous is not None:
        value = previous.metadata.get("occurrences", 0)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            previous_occurrences = value
    occurrences = previous_occurrences + 1
    updated_ns = max(observed_ns, 0 if previous is None else previous.updated_ns)
    created_ns = updated_ns if previous is None else previous.created_ns
    suffix = "occurrence" if occurrences == 1 else "occurrences"

    metadata: dict[str, str | int | float | bool] = {
        "occurrences": occurrences,
        "latest_run_id": run.run_id,
        "skill_id": run.skill_id,
        "context_key": run.context_key,
        "outcome": run.outcome.value,
        "context_successes": stats.successes,
        "context_failures": stats.failures,
        "context_timeouts": stats.timeouts,
        "context_consecutive_failures": stats.consecutive_failures,
    }
    if run.ended_ns is not None:
        metadata["latest_duration_ms"] = (run.ended_ns - run.started_ns) / 1_000_000
    if outcome_verification is not None:
        metadata.update(
            {
                "verified_outcome": outcome_verification.signal.value,
                "verified_outcome_confidence": outcome_verification.confidence,
                "verified_outcome_monotonic_ns": outcome_verification.observed_ns,
                "verified_outcome_evidence_json": json.dumps(
                    outcome_verification.evidence_keys
                ),
            }
        )
        if outcome_verification.target_kind is not None:
            metadata["verified_target_kind"] = outcome_verification.target_kind
    if run.outcome == SkillOutcome.SUCCEEDED:
        text = (
            f"Verified success for skill '{run.skill_id}' in context "
            f"'{run.context_key}' ({occurrences} {suffix})."
        )
        importance = 0.65
    else:
        reason = run.failure_reason if run.failure_reason is not None else "none reported"
        metadata["reported_reason"] = reason
        if run.failure_code is not None:
            metadata["failure_code"] = run.failure_code.value
        text = (
            f"Observed {run.outcome.value} for skill '{run.skill_id}' in context "
            f"'{run.context_key}'; reported reason: '{reason}' "
            f"({occurrences} {suffix})."
        )
        importance = 0.75
    return MemoryRecord(
        memory_id=memory_id,
        kind=kind,
        text=text,
        created_ns=created_ns,
        updated_ns=updated_ns,
        confidence=1.0,
        importance=importance,
        entity_tags=(run.skill_id,),
        source="runtime:verified-skill-outcome",
        metadata=metadata,
    )


def _skill_stats_totals(stats: Iterable[SkillStats]) -> dict[str, int]:
    totals = {
        "succeeded": 0,
        "failed": 0,
        "timed_out": 0,
        "cancelled": 0,
        "attempts": 0,
    }
    for item in stats:
        totals["succeeded"] += item.successes
        totals["failed"] += item.failures
        totals["timed_out"] += item.timeouts
        totals["cancelled"] += item.cancellations
        totals["attempts"] += item.attempts
    return totals


def _operator_target_facts(
    target: Track,
    current_hash: PerceptionFact | None,
    *,
    now_ns: int | None = None,
) -> tuple[PerceptionFact, ...]:
    """Convert a still-matching explicit region into geometric target facts.

    This does not infer object identity beyond the operator's label, mineability,
    range, or task success. The reference frame hash prevents a stale rectangle
    from becoming semantic ground truth after the view materially changes.
    """
    if target.attributes.get("source") != "operator":
        return ()
    observed_ns = time.monotonic_ns() if now_ns is None else now_ns
    facts: list[PerceptionFact] = []
    reference_path = target.attributes.get("reference_image_path")
    reference_sha256 = target.attributes.get("reference_image_sha256")
    if (
        isinstance(reference_path, str)
        and isinstance(reference_sha256, str)
        and len(reference_sha256) == 64
        and Path(reference_path).is_file()
    ):
        facts.append(
            PerceptionFact(
                key="target.reference_available",
                value=True,
                confidence=1.0,
                observed_ns=observed_ns,
                source=f"operator:cross-view-reference:{target.track_id}",
                expires_after_ms=250,
            )
        )
    reference = target.attributes.get("reference_dhash")
    observed = None if current_hash is None else current_hash.value
    if not isinstance(reference, str) or not isinstance(observed, str):
        return tuple(facts)
    try:
        if perceptual_hash_distance(reference, observed) > 6:
            return tuple(facts)
    except ValueError:
        return tuple(facts)
    center_x = target.region.x + target.region.width / 2.0
    center_y = target.region.y + target.region.height / 2.0
    source = f"operator:explicit-grounding:{target.track_id}"
    values: tuple[tuple[str, str | float | bool], ...] = (
        ("target.visible", True),
        ("target.kind", target.label),
        ("target.dx", max(-1.0, min(1.0, 2.0 * center_x - 1.0))),
        ("target.dy", max(-1.0, min(1.0, 2.0 * center_y - 1.0))),
    )
    facts.extend(
        PerceptionFact(
            key=key,
            value=value,
            confidence=1.0,
            observed_ns=observed_ns,
            source=source,
            expires_after_ms=250,
        )
        for key, value in values
    )
    return tuple(facts)


def _first_feasible_recovery(
    skills: SkillLibrary,
    recovery_ids: tuple[str, ...],
    blackboard: PerceptionBlackboard,
) -> SkillSpec | None:
    """Select recovery by current observed preconditions, preserving declared order."""
    for recovery_id in recovery_ids:
        if recovery_id not in skills.specs:
            continue
        candidate = skills.get(recovery_id)
        if initiation_satisfied(candidate, blackboard):
            return candidate
    return None


def _compatible_recovery_parameters(
    failed_run: SkillRun,
    recovery: SkillSpec,
) -> dict[str, str | int | float | bool]:
    """Carry only parameters declared by both the failed and recovery skills."""

    return {
        name: failed_run.parameters[name]
        for name in recovery.parameters
        if name in failed_run.parameters
    }


def _observed_scene_recovery(
    skills: SkillLibrary,
    blackboard: PerceptionBlackboard,
) -> SkillSpec | None:
    """Route verified blocking UI events to learned closed-loop options.

    This tactical event router selects an option contract only. It deliberately
    contains no GUI coordinates or actuator sequence; the configured learned
    policy must still perceive, act, and satisfy the option's visual outcome.
    """
    death = blackboard.fact("scene.death", min_confidence=0.9)
    if death is not None and bool(death.value):
        skill_id = "respawn_after_death"
    else:
        inventory_overlay = blackboard.fact(
            "scene.inventory_overlay",
            min_confidence=0.9,
        )
        playable = blackboard.fact("scene.playable", min_confidence=0.9)
        fast_inventory_interlock = bool(
            inventory_overlay is not None
            and inventory_overlay.value is True
            and playable is not None
            and playable.value is False
            and inventory_overlay.observed_ns == playable.observed_ns
            and inventory_overlay.source.startswith("safety:")
            and playable.source.startswith("safety:")
        )
        if fast_inventory_interlock:
            skill_id = "close_open_inventory"
        else:
            mode = blackboard.fact("scene.mode", min_confidence=0.9)
            if mode is None or mode.value != "inventory":
                return None
            if not _scene_claim_is_fresh(blackboard):
                # The mode belief may be a stale VLM hint. Without a matching
                # current frame hash we must not preempt world play over it;
                # an inventory recovery would otherwise freeze the agent in
                # close/open loops while the world sits fully playable.
                return None
            skill_id = "close_open_inventory"
    if skill_id not in skills.specs:
        return None
    candidate = skills.get(skill_id)
    return candidate if initiation_satisfied(candidate, blackboard) else None


def _scene_claim_is_fresh(blackboard: PerceptionBlackboard) -> bool:
    """True when the current mode claim was observed on the live frame hash.

    The VLM mode hint may be stale (frames change every 50ms). If the claim's
    observation dhash does not match the current frame dhash, it must not gate
    motor recovery; otherwise the agent freezes closing a phantom inventory.
    """
    observed = blackboard.fact("scene.observation_dhash", min_confidence=1.0)
    current = blackboard.fact("frame.dhash", min_confidence=1.0)
    if observed is None or current is None:
        return False
    if not isinstance(observed.value, str) or not isinstance(current.value, str):
        return False
    try:
        return perceptual_hash_distance(observed.value, current.value) <= 6
    except ValueError:
        return False


def _standing_goal_skill(goal: Goal, blackboard: PerceptionBlackboard) -> str | None:
    """Map a standing-goal description to the bootstrap skill that realizes it.

    Deterministic routing keeps the industrial loop persistent even when the
    high-level cognition is cold, using live inventory evidence so the loop
    actually progresses: gather while reserves are low, deposit when surplus
    exists, build once construction blocks are ready, explore otherwise.
    """
    text = goal.description.casefold()
    logs = _int_fact(blackboard, "inventory.logs", "inventory.oak_log", "inventory.wood")
    planks = _int_fact(blackboard, "inventory.planks", "inventory.oak_planks")
    chest = _int_fact(blackboard, "inventory.chest")
    if "gather" in text or "material" in text:
        if logs >= 16 and planks >= 8:
            return "deposit_in_storage"
        return "gather_nearby_wood"
    if "store" in text or "storage" in text:
        if chest >= 1:
            return "deposit_in_storage"
        if planks >= 8:
            return "craft_storage_units"
        if logs >= 3:
            return "gather_nearby_wood"
        return "craft_storage_units"
    if "build" in text or "workshop" in text or "expand" in text:
        if logs >= 8 and planks >= 8:
            return "build_workshop_shell"
        return "gather_nearby_wood"
    if "explore" in text:
        return "explore_forward"
    return None


def _int_fact(blackboard: PerceptionBlackboard, *keys: str) -> int:
    for key in keys:
        fact = blackboard.fact(key, min_confidence=0.3)
        if fact is not None:
            try:
                return int(fact.value)
            except (TypeError, ValueError):
                continue
    return 0


def _active_operator_messages(
    messages: tuple[OperatorMessage, ...],
) -> tuple[OperatorMessage, ...]:
    """Resolve operator-message authority without replaying stale commands.

    Fresh queued/delivered directives are the complete active command set for
    the next decision. Once those are handled, the newest acknowledged
    instruction remains the current directive. A correction authorizes one
    accepted bounded attempt, so an acknowledged correction acts as a
    tombstone: it is no longer active and an older instruction must not
    silently regain control underneath it. Persistent multi-project
    commitments belong in the goal portfolio rather than an ever-growing
    motor prompt.
    """
    pending = tuple(
        message
        for message in messages
        if message.status
        in {
            OperatorMessageStatus.QUEUED,
            OperatorMessageStatus.DELIVERED,
        }
    )
    if pending:
        return tuple(
            sorted(
                pending,
                key=lambda message: (
                    message.priority,
                    message.kind == OperatorMessageKind.CORRECTION,
                    message.created_ns,
                ),
                reverse=True,
            )
        )
    acknowledged = tuple(
        message
        for message in messages
        if message.status == OperatorMessageStatus.ACKNOWLEDGED
        and message.kind in {OperatorMessageKind.INSTRUCTION, OperatorMessageKind.CORRECTION}
    )
    if not acknowledged:
        return ()
    newest = max(acknowledged, key=lambda message: message.created_ns)
    return (newest,) if newest.kind == OperatorMessageKind.INSTRUCTION else ()


def _selected_operator_message_id(
    decision: CognitionDecision,
    pending_message_ids: tuple[str, ...],
) -> str | None:
    prefix = "operator:"
    if not decision.chosen_goal_id or not decision.chosen_goal_id.startswith(prefix):
        return None
    selected = decision.chosen_goal_id.removeprefix(prefix)
    return selected if selected in pending_message_ids else None


def _authorized_game_chat(
    decision: CognitionDecision,
    blackboard: PerceptionBlackboard,
    *,
    already_replied_ns: int | None = None,
    fact_source: str = "",
) -> str | None:
    """Return game chat only when perception carries explicit channel authority.

    Typing chat changes Bedrock focus and can interrupt/drown the embodied agent,
    so an LLM field alone is intentionally insufficient authority. A fresh
    grounded player-chat line (or an explicit operator authorization) grants the
    authority. ``already_replied_ns`` prevents re-answering the same line.
    """
    if decision.game_chat is None:
        return None
    for key in ("social.player_message", "operator.game_chat_authorized"):
        fact = blackboard.fact(key, min_confidence=0.7)
        if fact is None or not bool(fact.value):
            continue
        if not fact.fresh():
            continue
        if (
            already_replied_ns is not None
            and fact.observed_ns <= already_replied_ns
        ):
            continue
        return decision.game_chat
    return None


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


@dataclass
class AgentRuntime:
    perception: RealtimePerceptionService
    blackboard: PerceptionBlackboard
    executor: SkillExecutor
    skills: SkillLibrary
    role: RoleProfile
    lease_id: str
    high_level: HighLevelController | None = None
    memories: MemoryStore = field(default_factory=MemoryStore)
    social: SocialState = field(default_factory=SocialState)
    custom_goals: list[Goal] = field(default_factory=list)
    state_db: StateDatabase | None = None
    motor_hz: float = 20.0
    cognition_hz: float = 0.5
    semantic_hz: float = 2.0
    lease_renew_ms: int = 500
    stale_frame_consecutive_limit: int = 3
    metrics: RuntimeMetrics = field(default_factory=RuntimeMetrics)
    telemetry: TelemetryPublisher = field(default_factory=TelemetryPublisher)
    trajectory: TrajectoryRecorder | None = None
    trajectory_disabled_reason: str | None = None
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _sequence: int = field(default=0, init=False)
    _last_renew_ns: int = field(default=0, init=False)
    _last_cognition_ns: int = field(default=0, init=False)
    _last_player_chat_replied_ns: int | None = field(default=None, init=False)
    _last_player_chat_signature: str | None = field(default=None, init=False)
    _last_semantic_ns: int = field(default=0, init=False)
    _lease_thread: threading.Thread | None = field(default=None, init=False)
    _lease_fault: str | None = field(default=None, init=False)
    _pending_decision: concurrent.futures.Future[CognitionDecision] | None = field(
        default=None,
        init=False,
    )
    _pool: SingleWorkerDaemonExecutor = field(init=False)
    _last_decision: CognitionDecision | None = field(default=None, init=False)
    _pending_operator_message_ids: tuple[str, ...] = field(default=(), init=False)
    _pending_operator_message_kinds: dict[str, OperatorMessageKind] = field(
        default_factory=dict,
        init=False,
    )
    _pending_operator_status_updates: dict[
        str,
        tuple[OperatorMessageStatus, int, str | None],
    ] = field(default_factory=dict, init=False)
    _recent_skill_runs: deque[SkillRun] = field(
        default_factory=lambda: deque(maxlen=8),
        init=False,
    )
    _execution_revision: int = field(default=0, init=False)
    _pending_execution_revision: int = field(default=0, init=False)
    _plan_steps: tuple[str, ...] = field(default=(), init=False)
    _plan_goal_id: str | None = field(default=None, init=False)
    _plan_index: int = field(default=0, init=False)
    _plan_started_ns: int = field(default=0, init=False)
    _plan_step_completed_ns: int = field(default=0, init=False)
    _last_operator_target_id: str | None = field(default=None, init=False)
    _policy_warmup_error: str | None = field(default=None, init=False)
    _gui_fast_path_deferred: bool = field(default=False, init=False)
    _craft_semantic_probe: _CraftSemanticProbe | None = field(default=None, init=False)
    _cognition_perception_probe: _CognitionPerceptionProbe | None = field(
        default=None,
        init=False,
    )
    _idle_stall_probe_used_for_run_id: str | None = field(default=None, init=False)
    _cognition_requested: bool = field(default=True, init=False)
    _cognition_retry_count: int = field(default=0, init=False)
    _cognition_retry_not_before_ns: int = field(default=0, init=False)
    _pending_skill_stats: dict[tuple[str, str], SkillStats] = field(
        default_factory=dict,
        init=False,
    )
    _pending_runtime_events: dict[str, RuntimeEvent] = field(default_factory=dict, init=False)
    _pending_memories: dict[str, MemoryRecord] = field(default_factory=dict, init=False)
    _recorded_run_ids: set[str] = field(default_factory=set, init=False)
    _recorded_run_order: deque[str] = field(
        default_factory=lambda: deque(maxlen=_RECORDED_RUN_ID_LIMIT),
        init=False,
    )
    _last_storage_retry_ns: int = field(default=0, init=False)
    _last_operator_storage_retry_ns: int = field(default=0, init=False)
    _traversal_escalation_pending: bool = field(default=False, init=False)
    _headroom_recovery: _HeadroomRecovery | None = field(default=None, init=False)
    _gather_acquisition_continuation: _GatherAcquisitionContinuation | None = field(
        default=None,
        init=False,
    )
    _plan_neutral_recovery_runs: set[str] = field(default_factory=set, init=False)
    _planks_no_logs_failure_ns: int = field(default=0, init=False)
    _planks_failure_memory: MemoryRecord | None = field(default=None, init=False)
    _planks_failure_memory_initialized: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.motor_hz <= 0 or self.cognition_hz <= 0 or self.semantic_hz < 0:
            raise ValueError(
                "motor/cognition frequencies must be positive and semantic nonnegative"
            )
        if self.stale_frame_consecutive_limit < 1:
            raise ValueError("stale_frame_consecutive_limit must be positive")
        self._pool = SingleWorkerDaemonExecutor(
            thread_name="minecraft-ai-cognition",
        )

    def stop(self) -> None:
        self._stop.set()

    def run_forever(self) -> None:
        period = 1.0 / self.motor_hz
        self._lease_thread = threading.Thread(
            target=self._lease_heartbeat,
            name="minecraft-ai-lease-heartbeat",
            daemon=True,
        )
        # Secure the lease on the main thread before anything slow happens. The
        # CLI arms with only a 3s TTL; interpreter import + policy warmup can
        # easily exceed that before the heartbeat thread first renews, which
        # would make the supervisor watchdog expire the lease during a cold
        # start. An immediate, generous renewal hands ownership to the agent.
        try:
            send_command("renew", lease_id=self.lease_id, ttl_ms=5_000)
            self._last_renew_ns = time.monotonic_ns()
        except Exception as exc:
            self._lease_fault = f"{type(exc).__name__}: {exc}"
        self._lease_thread.start()
        if self.perception.active_vlm is not None:
            self.perception.active_vlm.start()
        try:
            self.telemetry.publish(self._telemetry_payload(state="warming"), force=True)
            # Strategic inference and policy checkpoint loading are independent.
            # Start the first typed decision from a real captured frame before
            # warming the learned policies so CPU model startup latency is not
            # paid serially while the avatar stands idle.
            if self.perception.last_capture is None:
                self.perception.capture_once()
            # Operator grounding is deterministic and may make a pending
            # directive executable before the first strategic snapshot. Merge
            # it before launching slow cognition, including after a process
            # restart where the message was already marked delivered.
            self._merge_operator_target()
            self._start_cognition_if_due()
            self._warmup_policy()
            while not self._stop.is_set():
                tick_started = time.perf_counter()
                self.tick()
                elapsed = time.perf_counter() - tick_started
                remaining = period - elapsed
                if remaining > 0:
                    self._stop.wait(remaining)
        except BaseException as exc:
            self._failsafe(f"agent-runtime:{type(exc).__name__}:{exc}")
            raise
        finally:
            self._stop.set()
            if self._lease_thread is not None:
                self._lease_thread.join(timeout=2.0)
            try:
                current = self.executor.run
                if current is not None and current.outcome == SkillOutcome.RUNNING:
                    cancelled = self.executor.cancel()
                    try:
                        if cancelled.action is not None:
                            self._send_motor(cancelled.action, execution=cancelled)
                    finally:
                        self._record_terminal_run(cancelled.run)
            except Exception:
                pass
            try:
                send_command("disarm")
            except Exception:
                pass
            self.perception.close()
            self.executor.close()
            if self.trajectory is not None:
                try:
                    self.trajectory.close()
                except Exception as exc:
                    self._failsafe(f"trajectory-flush:{type(exc).__name__}:{exc}")
            try:
                self._flush_pending_skill_stats(force=True)
                self._flush_pending_learning_records(force=True)
            except Exception as exc:
                self._failsafe(f"learning-flush:{type(exc).__name__}:{exc}")
            self.telemetry.publish(self._telemetry_payload(state="stopped"), force=True)
            self._pool.shutdown(wait=False, cancel_futures=True)

    def _warmup_policy(self) -> None:
        warmup = getattr(self.executor.policy, "warmup", None)
        if not callable(warmup):
            return
        if self.perception.last_capture is None:
            self.perception.capture_once()
        try:
            warmup()
            self._policy_warmup_error = None
        except Exception as exc:
            # Keep the agent available on its fallback route while surfacing the
            # exact checkpoint startup failure in operator telemetry.
            self._policy_warmup_error = f"{type(exc).__name__}: {exc}"

    def tick(self) -> None:
        # Capture is synchronous and precedes action selection. A later tick's
        # deterministic hotbar evidence can only arrive after _send_motor below
        # returns; a rejected send raises, while a suppressed send stops the run.
        capture_started = time.perf_counter()
        frame = self.perception.capture_once()
        self.metrics.frames += 1
        self.metrics.last_capture_ms = (time.perf_counter() - capture_started) * 1000.0
        self._merge_operator_target()
        self._merge_policy_perception()
        if self.perception.stale():
            self.metrics.stale_frame_skips += 1
            self.metrics.consecutive_stale_frames += 1
            # A late frame must never extend a previously accepted key/button
            # state. Preserve the authenticated lease so a transient CPU stall
            # can recover on the next fresh capture. Releasing input is
            # best-effort safety: a stalled supervisor or a missed reply must
            # never take down the whole agent while it is already degraded on a
            # stale capture, so a command failure here is tolerated — the lease
            # revocation path and release_all remain the authoritative release.
            try:
                send_command("release-inputs", lease_id=self.lease_id)
            except Exception:
                pass
            self.telemetry.publish(self._telemetry_payload(state="capture-stalled"))
            if self.metrics.consecutive_stale_frames >= self.stale_frame_consecutive_limit:
                raise RuntimeError(
                    "capture stream is stale for "
                    f"{self.metrics.consecutive_stale_frames} consecutive frames"
                )
            return
        self.metrics.consecutive_stale_frames = 0
        self._flush_pending_skill_stats()
        self._flush_pending_learning_records()
        self._flush_pending_operator_status_updates()
        self.telemetry.publish(self._telemetry_payload(state="running"))
        self._publish_player_chat_facts()
        self._planks_retry_requires_wood()
        self._consume_cognition()
        self._reconcile_cognition_perception_probe()
        self._start_cognition_if_due()
        self._request_semantics_if_due(frame.frame_id)
        self._route_observed_scene_recovery()
        self._advance_headroom_recovery()

        active = self.executor.run
        if active is None or active.outcome != SkillOutcome.RUNNING:
            # Reorientation and the one semantic query require stable pixels.
            # Keep the motor idle while this transaction owns the scene; its
            # mining/retry children appear as normal active runs below.
            if getattr(self, "_headroom_recovery", None) is not None:
                self._flush_pending_skill_stats()
                return
            if getattr(self, "_cognition_perception_probe", None) is not None:
                # A perception-only decision deliberately bought one stable
                # visual snapshot. Do not make that evidence stale by starting
                # the disposable exploration keepalive underneath it.
                self._flush_pending_skill_stats()
                return
            # Never idle the player while cognition is in flight: keep a
            # precondition-free exploration option running so motor keeps
            # emitting movement. Cognition switches skills when it returns.
            rescue = self._explore_keep_alive()
            if rescue is not None:
                self.executor.start(
                    rescue,
                    run_id=uuid.uuid4().hex,
                    context_key=_EXPLORE_KEEPALIVE_CONTEXT,
                )
                active = self.executor.run
            if active is None or active.outcome != SkillOutcome.RUNNING:
                self._flush_pending_skill_stats()
                return
        motor_started = time.perf_counter()
        result = self.executor.tick(
            self.blackboard,
            sequence=self._sequence,
            now_ns=time.monotonic_ns(),
        )
        self._merge_policy_perception()
        result, headroom_deadline_expired = self._expire_late_headroom_child(result)
        continuation = getattr(self, "_gather_acquisition_continuation", None)
        terminal = result.run.outcome != SkillOutcome.RUNNING
        verification = result.outcome_verification
        continuation_owned = bool(
            terminal
            and continuation is not None
            and continuation.active_run_id == result.run.run_id
            and continuation.context_key == result.run.context_key
        )
        frozen_gather_baseline = self.executor.mining_hotbar_log_baseline
        frozen_gather_count = _exact_frozen_log_count(
            frozen_gather_baseline,
            result.run,
        )
        verified_gather_break = bool(
            terminal
            and result.run.skill_id == "gather_nearby_wood"
            and result.run.outcome == SkillOutcome.SUCCEEDED
            and verification is not None
            and verification.run_id == result.run.run_id
            and _verified_oak_log_break(verification)
        )
        gather_break_claim = bool(
            terminal
            and result.run.skill_id == "gather_nearby_wood"
            and verification is not None
            and verification.signal == OutcomeSignal.BLOCK_BROKEN
        )
        gather_handoff = bool(
            terminal
            and verified_gather_break
            and "collect_recent_drop" in self.skills.specs
            and frozen_gather_count is not None
            and (
                continuation is None
                or (
                    continuation_owned
                    and frozen_gather_count == continuation.last_exact_count
                )
            )
        )
        verified_collection_count = getattr(
            self.executor,
            "verified_collection_hotbar_log_count",
            None,
        )
        verified_gather_collection = bool(
            terminal
            and continuation_owned
            and continuation is not None
            and _verified_gather_acquisition(
                result,
                continuation,
                exact_count=verified_collection_count,
            )
        )
        gather_collection_complete = bool(
            verified_gather_collection
            and continuation is not None
            and continuation.resource_acquired_events == _GATHER_ACQUISITIONS_REQUIRED - 1
        )
        gather_transaction_terminal = bool(
            terminal
            and (
                result.run.skill_id == "gather_nearby_wood"
                or continuation is not None
            )
        )
        inherited_plan_neutral = bool(
            terminal
            and result.run.run_id
            in getattr(self, "_plan_neutral_recovery_runs", set())
        )
        collect_recent_drop = bool(
            result.run.outcome == SkillOutcome.SUCCEEDED
            and result.run.skill_id == "mine_visible_block"
            and _verified_log_break(result.outcome_verification)
            and "collect_recent_drop" in self.skills.specs
        )
        headroom_child = bool(
            headroom_deadline_expired or self._is_headroom_child_result(result)
        )
        headroom = getattr(self, "_headroom_recovery", None)
        headroom_retry_advances_plan = bool(
            headroom_child
            and _headroom_retry_advances_plan(
                result,
                headroom,
                plan_steps=self._plan_steps,
                plan_index=self._plan_index,
                plan_goal_id=self._plan_goal_id,
            )
        )
        advance_plan = bool(
            not collect_recent_drop
            and (not headroom_child or headroom_retry_advances_plan)
        )
        if gather_transaction_terminal:
            # A gather break and the first two exact pickups are intermediate.
            # Only the third transaction-owned RESOURCE_ACQUIRED event consumes
            # the plan node.
            advance_plan = gather_collection_complete
        recorded_verification = verification
        if (gather_break_claim and not verified_gather_break) or (
            continuation is not None
            and terminal
            and result.run.skill_id == "collect_recent_drop"
            and not verified_gather_collection
        ):
            # Never persist a duplicate, unowned, or count-inexact acquisition
            # as one of this transaction's three facts.
            recorded_verification = None
        if continuation is not None and terminal and not (
            gather_handoff or (verified_gather_collection and not gather_collection_complete)
        ):
            self._gather_acquisition_continuation = None
        try:
            if result.action is not None:
                self._send_motor(result.action, execution=result)
        finally:
            if result.run.outcome != SkillOutcome.RUNNING:
                if recorded_verification is None:
                    self._record_terminal_run(
                        result.run,
                        advance_plan=advance_plan,
                    )
                else:
                    self._record_terminal_run(
                        result.run,
                        outcome_verification=recorded_verification,
                        advance_plan=advance_plan,
                    )
        self.metrics.last_motor_ms = (time.perf_counter() - motor_started) * 1000.0
        if terminal:
            stop_event = getattr(self, "_stop", None)
            if (
                (stop_event is not None and stop_event.is_set())
                or operator_pause_latched()
            ):
                # A pause/stop observed while releasing this terminal action
                # owns the executor. Never resurrect a successor transaction.
                self._gather_acquisition_continuation = None
                if stop_event is not None:
                    stop_event.set()
                return
            if gather_handoff:
                assert frozen_gather_baseline is not None
                assert frozen_gather_count is not None
                if continuation is None:
                    continuation = _GatherAcquisitionContinuation(
                        context_key=result.run.context_key,
                        parameters=dict(result.run.parameters),
                        instruction=self.executor.instruction,
                        active_run_id=result.run.run_id,
                        last_exact_count=frozen_gather_count,
                    )
                collection_run = self._start_drop_collection(
                    result.run,
                    frozen_gather_baseline,
                )
                continuation.active_run_id = collection_run.run_id
                self._gather_acquisition_continuation = continuation
                return
            if verified_gather_collection:
                assert continuation is not None
                assert verified_collection_count is not None
                continuation.last_exact_count = verified_collection_count
                continuation.resource_acquired_events += 1
                if gather_collection_complete:
                    self._note_terminal_for_cognition(
                        result.run,
                        recovery_started=False,
                    )
                    return
                gather_run_id = uuid.uuid4().hex
                self.executor.start(
                    self.skills.get("gather_nearby_wood"),
                    run_id=gather_run_id,
                    context_key=continuation.context_key,
                    parameters=continuation.parameters,
                    instruction=continuation.instruction,
                    gather_acquisitions_remaining=(
                        _GATHER_ACQUISITIONS_REQUIRED
                        - continuation.resource_acquired_events
                    ),
                )
                continuation.active_run_id = gather_run_id
                return
            if collect_recent_drop:
                self._start_drop_collection(
                    result.run,
                    self.executor.mining_hotbar_log_baseline,
                )
                return
            if self._route_headroom_terminal(result):
                return
            recovery = _first_feasible_recovery(
                self.skills,
                result.recovery_skills,
                self.blackboard,
            )
            self._note_terminal_for_cognition(
                result.run,
                recovery_started=recovery is not None,
            )
            if recovery is not None:
                self._start_recovery_skill(
                    recovery,
                    result.run,
                    plan_neutral=(
                        inherited_plan_neutral
                        or (
                            gather_transaction_terminal
                            and not gather_collection_complete
                        )
                    ),
                )

    def _start_drop_collection(
        self,
        broken_run: SkillRun,
        baseline: PerceptionFact | None,
    ) -> SkillRun:
        self.blackboard.merge_semantics(
            instance_id=self.perception.instance_id,
            facts=(PerceptionFact(
                key="collection.recent_log_break",
                value=True,
                confidence=0.995,
                observed_ns=time.monotonic_ns(),
                source=f"verified:{broken_run.run_id}:block-broken",
                expires_after_ms=6_000,
            ),),
        )
        return self.executor.start(
            self.skills.get("collect_recent_drop"),
            run_id=uuid.uuid4().hex,
            context_key=broken_run.context_key,
            collection_hotbar_log_baseline=baseline,
        )

    def _clear_drop_collection_authorization(self, run: SkillRun) -> None:
        """Revoke the short-lived pickup fact on every collector terminal path."""

        blackboard = getattr(self, "blackboard", None)
        perception = getattr(self, "perception", None)
        instance_id = getattr(perception, "instance_id", None)
        if blackboard is None or not isinstance(instance_id, str) or not instance_id:
            return
        blackboard.merge_semantics(
            instance_id=instance_id,
            facts=(
                PerceptionFact(
                    key="collection.recent_log_break",
                    value=False,
                    confidence=0.995,
                    observed_ns=time.monotonic_ns(),
                    source=f"runtime:{run.run_id}:collection-terminal",
                    expires_after_ms=250,
                ),
            ),
        )

    def _is_headroom_child_result(self, result: ExecutionTick) -> bool:
        recovery = getattr(self, "_headroom_recovery", None)
        if recovery is None:
            return False
        return bool(
            (recovery.phase == "mining" and result.run.run_id == recovery.mining_run_id)
            or (recovery.phase == "retry" and result.run.run_id == recovery.retry_run_id)
        )

    def _expire_late_headroom_child(
        self,
        result: ExecutionTick,
    ) -> tuple[ExecutionTick, bool]:
        """Fail closed when a blocking child tick returns after transaction expiry."""

        recovery = getattr(self, "_headroom_recovery", None)
        if (
            recovery is None
            or not self._is_headroom_child_result(result)
            or time.monotonic_ns() < recovery.deadline_ns
        ):
            return result, False

        now_ns = time.monotonic_ns()
        if result.run.outcome == SkillOutcome.RUNNING:
            expired = self.executor.cancel(now_ns=now_ns)
        else:
            expired = replace(
                result,
                run=result.run.model_copy(
                    update={
                        "ended_ns": now_ns,
                        "outcome": SkillOutcome.CANCELLED,
                        "failure_reason": "headroom-transaction-expired",
                        "failure_code": None,
                    }
                ),
                recovery_skills=(),
                outcome_verification=None,
            )
        self._clear_headroom_recovery(recovery)
        self._traversal_escalation_pending = True
        self._cognition_requested = True
        return expired, True

    def _route_headroom_terminal(self, result: ExecutionTick) -> bool:
        """Advance or end one clear-and-retry transaction without recursive recovery."""

        recovery = getattr(self, "_headroom_recovery", None)
        if recovery is not None and self._is_headroom_child_result(result):
            if recovery.phase == "mining":
                self._remove_headroom_target(recovery)
                if _verified_block_break(result):
                    retry_id = uuid.uuid4().hex
                    recovery.phase = "retry"
                    recovery.retry_run_id = retry_id
                    self.executor.start(
                        self.skills.get("traverse_visible_obstacle"),
                        run_id=retry_id,
                        context_key=recovery.context_key,
                        parameters=recovery.traversal_parameters,
                        complete_on_locomotion_progress=True,
                        locomotion_progress_events_required=3,
                        locomotion_progress_min_ms=750,
                    )
                else:
                    self._clear_headroom_recovery(recovery)
                    self._note_terminal_for_cognition(
                        result.run,
                        recovery_started=False,
                    )
                return True

            retry_succeeded = _verified_headroom_retry(result, recovery)
            self._clear_headroom_recovery(recovery)
            if retry_succeeded:
                self._traversal_escalation_pending = False
            else:
                self._traversal_escalation_pending = True
            self._note_terminal_for_cognition(
                result.run,
                recovery_started=False,
            )
            return True

        if not _verified_obstacle_stall(result):
            return False
        if (
            not self._headroom_scene_is_safe()
            or self.perception.active_vlm is None
            or "mine_visible_block" not in self.skills.specs
            or "traverse_visible_obstacle" not in self.skills.specs
        ):
            # Retain the ordinary declared recovery route (especially immediate
            # danger retreat) when the optional visual transaction cannot start.
            return False
        if not self._quiesce_headroom_inputs():
            # A classifier request is only meaningful when the avatar and
            # camera are actually still. Never infer from a scene whose held
            # input state could not be authoritatively released.
            return False
        self._note_terminal_for_cognition(result.run, recovery_started=True)
        if getattr(self, "_headroom_recovery", None) is None:
            now_ns = time.monotonic_ns()
            active_vlm = self.perception.active_vlm
            assert active_vlm is not None
            traversal = self.skills.get("traverse_visible_obstacle")
            self._headroom_recovery = _HeadroomRecovery(
                context_key=result.run.context_key,
                traversal_parameters=_compatible_recovery_parameters(
                    result.run,
                    traversal,
                ),
                deadline_ns=_headroom_deadline_ns(active_vlm, now_ns=now_ns),
                origin_skill_id=result.run.skill_id,
            )
        return True

    def _quiesce_headroom_inputs(self) -> bool:
        """Release every held input while preserving this runtime's live lease."""

        try:
            result = send_command("release-inputs", lease_id=self.lease_id)
        except Exception:
            return False
        return result.get("released") is True and result.get("lease_active") is True

    def _authoritative_world_camera_pitch_units(self) -> int | None:
        """Read the calibrated physical pitch accumulator from the supervisor."""

        try:
            status = send_command("status")
        except Exception:
            return None
        world_camera = status.get("world_camera")
        if not isinstance(world_camera, dict):
            return None
        pitch = world_camera.get("estimated_pitch_units")
        if (
            world_camera.get("origin_calibrated") is not True
            or not isinstance(pitch, int)
            or isinstance(pitch, bool)
        ):
            return None
        return pitch

    def _advance_headroom_recovery(self) -> None:
        """Request one exact grounding, then run one guarded clear and traversal retry."""

        recovery = getattr(self, "_headroom_recovery", None)
        if recovery is None:
            return
        now_ns = time.monotonic_ns()
        if now_ns >= recovery.deadline_ns:
            running = self.executor.run
            child_run_id = (
                recovery.mining_run_id
                if recovery.phase == "mining"
                else recovery.retry_run_id
            )
            if (
                running is not None
                and running.outcome == SkillOutcome.RUNNING
                and running.run_id == child_run_id
            ):
                cancelled = self.executor.cancel()
                try:
                    if cancelled.action is not None:
                        self._send_motor(cancelled.action, execution=cancelled)
                finally:
                    self._record_terminal_run(cancelled.run, advance_plan=False)
            self._clear_headroom_recovery(recovery)
            self._traversal_escalation_pending = True
            self._cognition_requested = True
            return
        if not self._headroom_scene_is_safe():
            running = self.executor.run
            child_run_ids = {recovery.mining_run_id, recovery.retry_run_id}
            if (
                running is not None
                and running.outcome == SkillOutcome.RUNNING
                and running.run_id in child_run_ids
            ):
                cancelled = self.executor.cancel()
                try:
                    if cancelled.action is not None:
                        self._send_motor(cancelled.action, execution=cancelled)
                finally:
                    self._record_terminal_run(cancelled.run, advance_plan=False)
            self._clear_headroom_recovery(recovery)
            self._traversal_escalation_pending = True
            self._cognition_requested = True
            return
        if recovery.phase in {"mining", "retry"}:
            return

        running = self.executor.run
        if running is not None and running.outcome == SkillOutcome.RUNNING:
            self._clear_headroom_recovery(recovery)
            return

        if recovery.phase == "reorient":
            latest = self.blackboard.raw_latest()
            captured = self.perception.last_capture
            if (
                latest is None
                or captured is None
                or captured.captured_ns != latest.captured_ns
            ):
                self._clear_headroom_recovery(recovery)
                return
            if recovery.pre_reorient_dhash is None:
                recovery.pre_reorient_dhash = frame_dhash(captured)
            current_pitch = self._authoritative_world_camera_pitch_units()
            if current_pitch is None:
                # A transient status timeout must not discard a verified stall
                # and immediately hand a stale camera estimate back to the
                # learned route. Keep this bounded transaction armed and retry
                # until its existing deadline or a safety preemption.
                return
            reorient_mouse_dy = _headroom_reorient_mouse_dy(current_pitch)
            if reorient_mouse_dy:
                recovery.reorientation_moved = True
                self._send_motor(
                    MotorAction(
                        sequence=self._sequence,
                        mouse_dy=reorient_mouse_dy,
                        camera_semantics="world",
                    )
                )
                if self._stop.is_set():
                    self._clear_headroom_recovery(recovery)
                    return
                _restore_policy_world_camera(
                    self.executor.policy,
                    pitch_units=current_pitch + reorient_mouse_dy,
                )
                return
            _restore_policy_world_camera(
                self.executor.policy,
                pitch_units=current_pitch,
            )
            recovery.phase = "settle"
            recovery.reoriented_frame_id = latest.frame_id
            recovery.settle_deadline_ns = (
                time.monotonic_ns() + _HEADROOM_SETTLE_TIMEOUT_NS
            )
            return

        if recovery.phase == "settle":
            if (
                recovery.settle_deadline_ns is None
                or now_ns >= recovery.settle_deadline_ns
            ):
                self._clear_headroom_recovery(recovery)
                self._traversal_escalation_pending = True
                self._cognition_requested = True
                return
            latest = self.blackboard.raw_latest()
            captured = self.perception.last_capture
            if (
                latest is None
                or captured is None
                or recovery.reoriented_frame_id is None
                or recovery.pre_reorient_dhash is None
                or latest.frame_id <= recovery.reoriented_frame_id
                or captured.captured_ns != latest.captured_ns
            ):
                return
            try:
                visibly_reoriented = (
                    perceptual_hash_distance(
                        recovery.pre_reorient_dhash,
                        frame_dhash(captured),
                    )
                    > 0
                )
            except ValueError:
                self._clear_headroom_recovery(recovery)
                return
            if recovery.reorientation_moved and not visibly_reoriented:
                recovery.settle_frame_id = None
                recovery.settle_crosshair_dhash = None
                recovery.settle_rgb_grid = None
                recovery.settle_stable_successors = 0
                return
            current_crosshair_dhash = crosshair_block_dhash(captured)
            current_rgb_grid = crosshair_block_rgb_grid(captured)
            if (
                recovery.settle_frame_id is None
                or recovery.settle_crosshair_dhash is None
                or recovery.settle_rgb_grid is None
            ):
                # The first visibly changed frame is the settle baseline, not
                # proof that sprint FOV, head bob, falling, or mouse easing has
                # stopped.
                recovery.settle_frame_id = latest.frame_id
                recovery.settle_crosshair_dhash = current_crosshair_dhash
                recovery.settle_rgb_grid = current_rgb_grid
                recovery.settle_stable_successors = 0
                return
            if latest.frame_id <= recovery.settle_frame_id:
                return
            try:
                crosshair_stable = perceptual_hash_distance(
                    recovery.settle_crosshair_dhash,
                    current_crosshair_dhash,
                ) <= 2
            except ValueError:
                crosshair_stable = False
            rgb_stable = (
                crosshair_block_rgb_grid_distance(
                    recovery.settle_rgb_grid,
                    current_rgb_grid,
                )
                <= 1.0
            )
            recovery.settle_frame_id = latest.frame_id
            recovery.settle_crosshair_dhash = current_crosshair_dhash
            recovery.settle_rgb_grid = current_rgb_grid
            if not crosshair_stable or not rgb_stable:
                recovery.settle_stable_successors = 0
                return
            recovery.settle_stable_successors += 1
            if recovery.settle_stable_successors < _HEADROOM_STABLE_SUCCESSOR_FRAMES:
                return
            recovery.phase = "request"

        if recovery.phase == "request":
            if self.perception.active_vlm is None:
                self._clear_headroom_recovery(recovery)
                return
            if (
                not self.perception.semantic_available()
                or not local_model_inference_available()
            ):
                return
            captured = self.perception.last_capture
            latest = self.blackboard.raw_latest()
            if (
                captured is None
                or latest is None
                or captured.captured_ns != latest.captured_ns
            ):
                self._clear_headroom_recovery(recovery)
                return
            query_id = uuid.uuid4().hex
            query_started_ns = time.monotonic_ns()
            query_frame_dhash = frame_dhash(captured)
            query_crosshair_dhash = crosshair_block_dhash(captured)
            query_pixel_sha256 = crosshair_block_pixel_sha256(captured)
            query_rgb_grid = crosshair_block_rgb_grid(captured)
            model_id = self.perception.active_vlm.model.model_id
            query = ActivePerceptionQuery(
                query_id=query_id,
                mode=PerceptionQueryMode.CROSSHAIR_BLOCK,
                question="Classify only the block exactly under the world crosshair.",
                skill_id="mine_visible_block",
                frame_id=latest.frame_id,
                deadline_ms=10_000,
            )
            if not self.perception.request_semantics(query, frame=captured):
                self._clear_headroom_recovery(recovery)
                return
            recovery.phase = "grounding"
            recovery.query_id = query_id
            recovery.query_started_ns = query_started_ns
            recovery.query_frame_dhash = query_frame_dhash
            recovery.query_crosshair_dhash = query_crosshair_dhash
            recovery.query_frame_id = latest.frame_id
            recovery.query_captured_ns = captured.captured_ns
            recovery.query_frame_width = captured.width
            recovery.query_frame_height = captured.height
            recovery.query_pixel_sha256 = query_pixel_sha256
            recovery.query_rgb_grid = query_rgb_grid
            recovery.query_source = f"vlm:{model_id}:{query_id}"
            self.metrics.semantic_requests += 1
            return

        target = _headroom_clear_target(
            self.blackboard,
            recovery,
            now_ns=time.monotonic_ns(),
            current_frame=self.perception.last_capture,
        )
        if target is not None:
            run_id = uuid.uuid4().hex
            track_id = self._materialize_headroom_target(recovery, target)
            if track_id is None:
                self._clear_headroom_recovery(recovery)
                self._traversal_escalation_pending = True
                self._cognition_requested = True
                return
            recovery.phase = "mining"
            recovery.mining_run_id = run_id
            self.executor.start(
                self.skills.get("mine_visible_block"),
                run_id=run_id,
                context_key=recovery.context_key,
                parameters={
                    "target": target.kind,
                    "target_track_id": track_id,
                },
                instruction=(
                    "Mine only the grounded soft block under the crosshair until it breaks."
                ),
            )
            return

        # Worker availability returns only after its publication or terminal
        # failure. An available worker with no exact accepted answer means this
        # single query abstained; never ask again or improvise another target.
        if self.perception.semantic_available():
            self._clear_headroom_recovery(recovery)
            # The completed recovery found no authorized block to clear. Let
            # cognition consume this failure before another disposable walk:
            # a new obstacle stall otherwise invalidates every slow decision
            # before it can return. Safety and operator work still preempt.
            self._traversal_escalation_pending = True
            self._cognition_requested = True

    def _headroom_scene_is_safe(self) -> bool:
        now_ns = time.monotonic_ns()
        latest = self.blackboard.raw_latest()
        if latest is None:
            return False
        unsafe_truths = ("danger.immediate", "scene.death", "scene.ui_overlay")
        if any(
            (fact := self.blackboard.fact(key, min_confidence=0.65, now_ns=now_ns))
            is not None
            and fact.value is True
            for key in unsafe_truths
        ):
            return False
        playable = self.blackboard.fact("scene.playable", min_confidence=0.65, now_ns=now_ns)
        mode = self.blackboard.fact("scene.mode", min_confidence=0.65, now_ns=now_ns)
        return bool(
            playable is not None
            and mode is not None
            and playable.value is True
            and mode.value == "world"
            and playable.source == mode.source
            and playable.source == BEDROCK_HUD_SAFETY_SOURCE
            and latest.captured_ns <= playable.observed_ns <= now_ns
            and latest.captured_ns <= mode.observed_ns <= now_ns
        )

    def _materialize_headroom_target(
        self,
        recovery: _HeadroomRecovery,
        target: _HeadroomTarget,
    ) -> str | None:
        """Bind an accepted classifier sample for the existing mining guard."""

        latest = self.blackboard.raw_latest()
        if latest is None or recovery.query_id is None:
            return None
        track_id = f"crosshair-probe:{recovery.query_id}"
        aperture_width = 1.0 / latest.width
        aperture_height = 1.0 / latest.height
        track = Track(
            track_id=track_id,
            label=target.kind,
            confidence=target.confidence,
            region=ScreenRegion(
                x=0.5 - aperture_width / 2,
                y=0.5 - aperture_height / 2,
                width=aperture_width,
                height=aperture_height,
            ),
            first_seen_ns=target.observed_ns,
            last_seen_ns=target.observed_ns,
            attributes={
                "source": "crosshair-block-probe",
                "tracking_source": target.source,
                "sampling_aperture": True,
                "crosshair_rgb_grid": recovery.query_rgb_grid or "",
            },
            evidence_refs=(target.evidence_id,),
        )
        facts = tuple(
            PerceptionFact(
                key=key,
                value=value,
                confidence=target.confidence,
                observed_ns=target.observed_ns,
                source=target.source,
                expires_after_ms=15_000,
                evidence_refs=(target.evidence_id,),
            )
            for key, value in (
                ("target.visible", True),
                ("target.kind", target.kind),
                ("target.reference_available", True),
            )
        )
        if not self.blackboard.merge_semantics(
            instance_id=self.perception.instance_id,
            facts=facts,
        ):
            return None
        if not self.blackboard.upsert_semantic_track(
            instance_id=self.perception.instance_id,
            track=track,
        ):
            self.blackboard.remove_semantic_facts(
                ("target.visible", "target.kind", "target.reference_available"),
                expected_source=target.source,
            )
            return None
        recovery.target_track_id = track_id
        current = self.blackboard.latest()
        if current is None or not any(item.track_id == track_id for item in current.tracks):
            self._remove_headroom_target(recovery)
            self.blackboard.remove_semantic_facts(
                ("target.visible", "target.kind", "target.reference_available"),
                expected_source=target.source,
            )
            return None
        return track_id

    def _remove_headroom_target(self, recovery: _HeadroomRecovery) -> None:
        if recovery.target_track_id is not None:
            self.blackboard.remove_semantic_track(recovery.target_track_id)
            recovery.target_track_id = None
        if recovery.query_source is not None:
            self.blackboard.remove_semantic_facts(
                ("target.visible", "target.kind", "target.reference_available"),
                expected_source=recovery.query_source,
            )

    def _clear_headroom_recovery(self, recovery: _HeadroomRecovery) -> None:
        """End one transaction and remove any temporary classifier binding."""

        self._remove_headroom_target(recovery)
        if recovery.query_source is not None:
            self.blackboard.remove_semantic_facts(
                (
                    "recovery.crosshair.block",
                    "recovery.crosshair.frame_dhash",
                    "recovery.crosshair.observation_dhash",
                    "target.visible",
                    "target.kind",
                    "target.reference_available",
                ),
                expected_source=recovery.query_source,
            )
        if getattr(self, "_headroom_recovery", None) is recovery:
            self._headroom_recovery = None

    def _explore_keep_alive(self) -> SkillSpec | None:
        """Pick a precondition-free option to keep motor busy while cognition decides.

        This runs only when no skill is currently running (the idle gap after a
        terminal run). The MOTION-level traversal option routes to the fast
        learned motion expert (VPT), which continuously emits locomotion even
        without fresh semantic/grounding data -- precisely what prevents the
        idle freeze that the latent STEVE body produces while cognition is in
        flight.
        """
        if getattr(self, "_traversal_escalation_pending", False):
            return None
        candidates: list[tuple[int, SkillSpec, SkillStats | None]] = []
        for order, skill_id in enumerate(("traverse_level_ground", "explore_forward")):
            skill = self.skills.specs.get(skill_id)
            if skill is None:
                continue
            stats = self.skills.stats.get((skill_id, _EXPLORE_KEEPALIVE_CONTEXT))
            candidates.append((order, skill, stats))
        if not candidates:
            return None
        healthy = [
            candidate
            for candidate in candidates
            if candidate[2] is None or candidate[2].consecutive_failures < 2
        ]
        pool = healthy or candidates
        return min(
            pool,
            key=lambda candidate: (
                0 if candidate[2] is None else candidate[2].consecutive_failures,
                0 if candidate[2] is None else candidate[2].attempts,
                candidate[0],
            ),
        )[1]

    def _note_terminal_for_cognition(
        self,
        run: SkillRun,
        *,
        recovery_started: bool,
    ) -> None:
        """Invalidate cognition only for execution changes it must observe.

        An exploration keepalive is explicitly disposable continuity work while
        cognition is pending. Its timeout must not discard that already-running
        strategic/operator decision. A failure that routes into a recovery, or
        any non-keepalive terminal result, still invalidates the old snapshot.
        """
        obstacle_recovery_exhausted = bool(
            run.skill_id == "traverse_visible_obstacle"
            and run.outcome not in {SkillOutcome.SUCCEEDED, SkillOutcome.CANCELLED}
        )
        if obstacle_recovery_exhausted:
            self._traversal_escalation_pending = True
        invalidates = (
            run.context_key != _EXPLORE_KEEPALIVE_CONTEXT
            or recovery_started
            or obstacle_recovery_exhausted
        )
        if invalidates:
            self._execution_revision += 1
            self._cognition_requested = True
        elif self._pending_decision is None:
            self._cognition_requested = True

    def _route_observed_scene_recovery(self) -> None:
        """Preempt stale world work when a verified blocking scene event arrives."""
        running = self.executor.run
        recovery = _observed_scene_recovery(self.skills, self.blackboard)
        if recovery is None:
            return
        continuation = getattr(self, "_gather_acquisition_continuation", None)
        headroom = getattr(self, "_headroom_recovery", None)
        incomplete_gather = bool(
            continuation is not None
            or (
                running is not None
                and running.outcome == SkillOutcome.RUNNING
                and (
                    running.skill_id == "gather_nearby_wood"
                    or running.run_id
                    in getattr(self, "_plan_neutral_recovery_runs", set())
                )
            )
            or (headroom is not None and headroom.origin_skill_id == "gather_nearby_wood")
        )
        # Death and modal UI recovery outrank the optional terrain-clear
        # transaction at every phase. The normal cancellation path below owns
        # releasing any active mining/traversal inputs.
        if headroom is not None:
            self._clear_headroom_recovery(headroom)
        if running is not None and running.outcome == SkillOutcome.RUNNING:
            active_spec = self.skills.get(running.skill_id)
            if (
                recovery.skill_id == "close_open_inventory"
                and active_spec.action_level == ActionLevel.GUI
                and running.skill_id != "close_open_inventory"
            ):
                # An inventory/crafting option deliberately owns the GUI until
                # its bounded verifier succeeds or fails. Treating that same
                # observed inventory as an obstruction would immediately close
                # the screen underneath it. Other safety events (notably death)
                # still preempt the GUI owner normally.
                return
        if (
            running is not None
            and running.outcome == SkillOutcome.RUNNING
            and running.skill_id == recovery.skill_id
        ):
            return
        self._gather_acquisition_continuation = None
        context_key = "scene-recovery"
        if running is not None and running.outcome == SkillOutcome.RUNNING:
            context_key = running.context_key
            cancelled = self.executor.cancel()
            try:
                if cancelled.action is not None:
                    self._send_motor(cancelled.action, execution=cancelled)
            finally:
                self._record_terminal_run(cancelled.run)
            self._execution_revision += 1
        recovery_run = self.executor.start(
            recovery,
            run_id=uuid.uuid4().hex,
            context_key=context_key,
        )
        if incomplete_gather:
            self._plan_neutral_recovery_runs = {
                *getattr(self, "_plan_neutral_recovery_runs", ()),
                recovery_run.run_id,
            }
        self._cognition_requested = True

    def _lease_heartbeat(self) -> None:
        """Keep the motor lease alive independently of inference/cognition latency."""
        interval_s = self.lease_renew_ms / 1000.0
        ttl_ms = min(5_000, max(3_000, self.lease_renew_ms * 12))
        missing = 0
        while not self._stop.is_set():
            try:
                send_command("renew", lease_id=self.lease_id, ttl_ms=ttl_ms)
                self._last_renew_ns = time.monotonic_ns()
                self._lease_fault = None
                missing = 0
            except Exception as exc:
                # A transiently busy supervisor must not silently terminate the
                # agent on one missed reply. The supervisor's own lease watchdog
                # revokes the motor if the lease truly lapses; tolerate a bounded
                # run of heartbeat failures before giving up and stopping.
                missing += 1
                self._lease_fault = f"{type(exc).__name__}: {exc}"
                if missing >= 2:
                    self._stop.set()
                    return
            self._stop.wait(interval_s)

    def _send_motor(
        self,
        action: MotorAction,
        *,
        execution: ExecutionTick | None = None,
    ) -> None:
        if self._stop.is_set() or operator_pause_latched():
            self._gather_acquisition_continuation = None
            self._stop.set()
            return
        # The supervisor lease has one global replay counter, while learned
        # policy bodies and synthetic controllers maintain independent local
        # counters. Runtime rebases a lagging route onto the wire counter, but
        # preserves an action that is already ahead: mining's post-release
        # verifier can consume local policy/reset sequences while deliberately
        # emitting no wire action. Collapsing that gap makes the next policy
        # call replay its own last sequence and crashes the agent even though
        # the supervisor would safely accept the monotonic jump.
        if action.sequence < self._sequence:
            action = action.model_copy(update={"sequence": self._sequence})
        provenance = _accepted_action_provenance(
            execution,
            self.blackboard,
            fallback_policy_id=self.executor.policy.policy_id,
        )
        try:
            accepted = send_command(
                "motor-action",
                lease_id=self.lease_id,
                action=action.model_dump(mode="json"),
            )
        except Exception:
            # Pause/stop can land after the preflight check while an already-running
            # tick is crossing the supervisor boundary. That revocation is an
            # expected shutdown, not an agent fault. Recheck after rejection to
            # close the TOCTOU window without hiding unrelated transport failures.
            if self._stop.is_set() or operator_pause_latched():
                self._stop.set()
                return
            raise
        if (
            execution is not None
            and execution.action_origin in {ActionOrigin.SYNTHETIC, ActionOrigin.RESET}
            and action.camera_semantics == "world"
        ):
            # Learned policy clients integrate their requested camera delta
            # before the mining/GUI guards can suppress or replace it. Rebind
            # every guarded or reset world-camera result to the physical
            # supervisor after acceptance so later routes never inherit an
            # unsent pitch.
            accepted_camera = accepted.get("world_camera")
            physical_pitch = (
                accepted_camera.get("estimated_pitch_units")
                if (
                    isinstance(accepted_camera, dict)
                    and accepted_camera.get("origin_calibrated") is True
                    and isinstance(
                        accepted_camera.get("estimated_pitch_units"),
                        int,
                    )
                    and not isinstance(
                        accepted_camera.get("estimated_pitch_units"),
                        bool,
                    )
                )
                else self._authoritative_world_camera_pitch_units()
            )
            if physical_pitch is not None:
                _restore_policy_world_camera(
                    self.executor.policy,
                    pitch_units=physical_pitch,
                )
        if self.trajectory is not None:
            frame = self.perception.last_capture
            blackboard = self.blackboard.latest()
            if frame is not None and blackboard is not None:
                running = self.executor.run if execution is None else execution.run
                reward_signals, event_ids = _trajectory_outcome_annotations(execution)
                try:
                    self.trajectory.record_accepted(
                        action=action,
                        provenance=provenance,
                        supervisor_response=accepted,
                        frame=frame,
                        blackboard=blackboard,
                        skill_run_id=None if running is None else running.run_id,
                        skill_id=None if running is None else running.skill_id,
                        goal_id=None
                        if self._last_decision is None
                        else self._last_decision.chosen_goal_id,
                        reward_signals=reward_signals,
                        event_ids=event_ids,
                    )
                except Exception as exc:
                    reason = f"{type(exc).__name__}: {exc}"
                    self.trajectory.disable(reason)
                    self.trajectory_disabled_reason = reason
        self._sequence = action.sequence + 1
        self.metrics.motor_actions += 1

    def _request_semantics_if_due(self, frame_id: int) -> None:
        # semantic_hz=0 is event-only active perception. Explicit questions from
        # cognition and bounded GUI transactions are still permitted events.
        if self.perception.active_vlm is None:
            return
        if getattr(self, "_cognition_perception_probe", None) is not None:
            # A decision-owned query must have the next available semantic
            # slot. Periodic work here would occupy the shared model lane,
            # outlive the short settle window, and recreate the starvation
            # this transaction exists to prevent.
            return
        if getattr(self, "_headroom_recovery", None) is not None:
            # The recovery owns exactly one narrowly scoped query. A periodic
            # request must neither race it nor replace its target facts.
            return
        if self.semantic_hz <= 0:
            executor = getattr(self, "executor", None)
            active = None if executor is None else executor.run
            if active is None or active.skill_id != "craft_wood_planks":
                return
        else:
            active = self.executor.run
        skill_id = active.skill_id if active is not None else None
        crafting_event = skill_id == "craft_wood_planks"
        now = time.monotonic_ns()
        if crafting_event and active is not None:
            self._reconcile_craft_semantic_probe(active)
            if not self.executor.plank_crafting_semantics_ready(
                self.blackboard,
                now_ns=now,
            ):
                # Runtime scheduling precedes motor execution in each loop. Do
                # not bind a 30s+ request to either the pre-toggle world or the
                # first partially-rendered inventory frame.
                return
        if not _semantic_refresh_allowed(
            cognition_requested=self._cognition_requested,
            cognition_pending=self._pending_decision is not None,
            operator_message_pending=bool(self._pending_operator_message_ids),
            worker_available=self.perception.semantic_available(),
        ):
            return
        effective_hz = self.semantic_hz if self.semantic_hz > 0 else 0.5
        interval = int(1e9 / effective_hz)
        if now - self._last_semantic_ns < interval:
            return
        terminal_count_before = self._active_vlm_terminal_count()
        if crafting_event and not self._craft_semantic_budget_available(now_ns=now):
            phase = self.executor.plank_crafting_phase
            if phase is not None:
                self.executor.note_plank_crafting_semantic_completion(phase)
            return
        question = self._semantic_question(skill_id)
        output_keys = list(
            (
                "scene.mode",
                "scene.playable",
                "gui.mode",
                "inventory.logs",
                "inventory.planks",
            )
            if crafting_event
            else (
                "scene.mode",
                "scene.playable",
                "danger.immediate",
                "obstacle.ahead",
                "target.visible",
                "target.dx",
                "target.dy",
            )
        )
        if skill_id is not None:
            spec = self.skills.get(skill_id)
            for condition in (
                *spec.preconditions,
                *(item for group in spec.initiation_alternatives for item in group),
                *spec.success_conditions,
                *spec.failure_conditions,
            ):
                if resolve_grounded_output_keys((), condition.key):
                    output_keys.append(condition.key)
        query = ActivePerceptionQuery(
            query_id=uuid.uuid4().hex,
            question=question,
            skill_id=skill_id,
            frame_id=frame_id,
            deadline_ms=_semantic_deadline_ms(effective_hz),
            output_keys=tuple(dict.fromkeys(output_keys)),
        )
        if self.perception.request_semantics(query):
            self.metrics.semantic_requests += 1
            self._last_semantic_ns = now
            phase = self.executor.plank_crafting_phase
            if (
                crafting_event
                and active is not None
                and phase is not None
                and terminal_count_before is not None
            ):
                self._craft_semantic_probe = _CraftSemanticProbe(
                    run_id=active.run_id,
                    phase=phase,
                    terminal_count_before=terminal_count_before,
                )

    def _active_vlm_status(self) -> dict[str, object]:
        active_vlm = self.perception.active_vlm
        status = getattr(active_vlm, "status", None)
        if not callable(status):
            return {}
        try:
            result = status()
        except Exception:
            return {}
        return result if isinstance(result, dict) else {}

    def _active_vlm_terminal_count(self) -> int | None:
        status = self._active_vlm_status()
        completed = status.get("completed")
        failures = status.get("failures")
        if (
            not isinstance(completed, int)
            or isinstance(completed, bool)
            or not isinstance(failures, int)
            or isinstance(failures, bool)
        ):
            return None
        return completed + failures

    def _reconcile_cognition_perception_probe(self) -> None:
        """Hold one matched observation through one bounded follow-up decision."""

        probe = getattr(self, "_cognition_perception_probe", None)
        if probe is None:
            return
        now_ns = time.monotonic_ns()
        stop_event = getattr(self, "_stop", None)
        if (stop_event is not None and stop_event.is_set()) or operator_pause_latched():
            self._clear_cognition_perception_probe(probe)
            if stop_event is not None:
                stop_event.set()
            return
        running = self.executor.run
        if (
            self._execution_revision != probe.execution_revision
            or (running is not None and running.outcome == SkillOutcome.RUNNING)
        ):
            # Safety/operator work may take ownership while this optional wait
            # is settling. Its visual publication remains independently
            # scene-matched, but it no longer owns runtime scheduling.
            self._clear_cognition_perception_probe(probe)
            self._cognition_requested = True
            return
        if _observed_scene_recovery(self.skills, self.blackboard) is not None:
            # The normal scene router below owns death and modal UI recovery.
            # Release this optional wait before considering any world motion.
            self._clear_cognition_perception_probe(probe)
            self._cognition_requested = True
            return
        safety = _first_feasible_recovery(
            self.skills,
            tuple(
                skill_id
                for key, skill_id in (
                    ("environment.underwater", "escape_submersion"),
                    ("danger.immediate", "retreat_from_danger"),
                )
                if (hazard := self.blackboard.fact(
                    key, min_confidence=0.7, now_ns=now_ns,
                )) is not None
                and hazard.value is True
            ),
            self.blackboard,
        )
        if safety is not None:
            # New hazard evidence must reach the existing learned escape route
            # immediately, even while the slow observation worker is occupied.
            self._clear_cognition_perception_probe(probe)
            self._cognition_requested = True
            stop_event = getattr(self, "_stop", None)
            if (stop_event is not None and stop_event.is_set()) or operator_pause_latched():
                if stop_event is not None:
                    stop_event.set()
                return
            safety_run = self.executor.start(
                safety,
                run_id=uuid.uuid4().hex,
                context_key="perception-safety-recovery",
            )
            self._plan_neutral_recovery_runs = {
                *getattr(self, "_plan_neutral_recovery_runs", ()),
                safety_run.run_id,
            }
            self._execution_revision += 1
            return
        if probe.handoff_deadline_ns is not None:
            if (
                now_ns >= probe.handoff_deadline_ns
                or not self._headroom_scene_is_safe()
                or not self._cognition_probe_scene_matches(probe, now_ns=now_ns)
                or any(
                    (current := self.blackboard.fact(fact.key, now_ns=now_ns)) is None
                    or current.source != fact.source
                    or current.observed_ns != fact.observed_ns
                    or current.value != fact.value
                    for fact in probe.retained_facts
                )
            ):
                self._clear_cognition_perception_probe(probe)
                self._schedule_cognition_retry(now_ns=now_ns)
            return
        if probe.query_id is None:
            if now_ns >= probe.settle_deadline_ns:
                self._clear_cognition_perception_probe(probe)
                self._schedule_cognition_retry(now_ns=now_ns)
                return
            latest = self.blackboard.raw_latest()
            captured = self.perception.last_capture
            current_hash_fact = self.blackboard.fact(
                "frame.dhash",
                min_confidence=1.0,
                now_ns=now_ns,
            )
            current_hash = (
                current_hash_fact.value
                if current_hash_fact is not None
                and isinstance(current_hash_fact.value, str)
                else None
            )
            if (
                latest is None
                or captured is None
                or latest.frame_id <= probe.frame_id
                or captured.captured_ns != latest.captured_ns
                or current_hash is None
            ):
                return
            if probe.settle_dhash is None:
                self._cognition_perception_probe = replace(
                    probe,
                    frame_id=latest.frame_id,
                    settle_dhash=current_hash,
                )
                return
            try:
                settled = perceptual_hash_distance(probe.settle_dhash, current_hash) <= 2
            except ValueError:
                settled = False
            if not settled:
                self._cognition_perception_probe = replace(
                    probe,
                    frame_id=latest.frame_id,
                    settle_dhash=current_hash,
                )
                return
            if (
                not self.perception.semantic_available()
                or not local_model_inference_available()
            ):
                return
            terminal_count = self._active_vlm_terminal_count()
            if terminal_count is None:
                return
            query_id = uuid.uuid4().hex
            query = ActivePerceptionQuery(
                query_id=query_id,
                question="Inspect only the requested canonical facts.",
                skill_id=None,
                frame_id=latest.frame_id,
                output_keys=probe.requested_keys,
            )
            if not self.perception.request_semantics(query, frame=captured):
                return
            self.metrics.semantic_requests += 1
            self._cognition_perception_probe = replace(
                probe,
                query_id=query_id,
                frame_id=latest.frame_id,
                settle_dhash=current_hash,
                terminal_count_before=terminal_count,
                grounding_deadline_ns=(
                    now_ns + _COGNITION_PERCEPTION_GROUNDING_TIMEOUT_NS
                ),
            )
            return
        if (
            probe.grounding_deadline_ns is not None
            and now_ns >= probe.grounding_deadline_ns
        ) or self._active_vlm_status().get("thread_alive") is False:
            # Bound ownership of the scene even if a worker stalls or stops.
            # The in-flight job is left alone; its eventual publication still
            # passes the independent visual freshness checks.
            self._clear_cognition_perception_probe(probe)
            self._schedule_cognition_retry(now_ns=now_ns)
            return
        if not self.perception.semantic_available():
            return
        terminal_count = self._active_vlm_terminal_count()
        if (
            terminal_count is None
            or probe.terminal_count_before is None
            or terminal_count <= probe.terminal_count_before
        ):
            return
        observation = self.blackboard.fact("scene.observation_dhash", min_confidence=1.0)
        source = None if observation is None else observation.source
        retained = tuple(
            fact
            for key in probe.requested_keys
            if key in {"target.visible", "obstacle.ahead"}
            and (fact := self.blackboard.fact(key, min_confidence=0.7, now_ns=now_ns)) is not None
            and fact.source == source
            and fact.value is True
            and observation is not None
            and fact.observed_ns == observation.observed_ns
        )
        if (
            source is None
            or not source.startswith("vlm:")
            or not source.endswith(f":{probe.query_id}")
            or observation is None
            or observation.value != probe.settle_dhash
            or not retained
            or not self._headroom_scene_is_safe()
            or not self._cognition_probe_scene_matches(probe, now_ns=now_ns)
        ):
            # Abstention, rejected publication, or scene drift must leave the
            # motor free and retry boundedly rather than buy an empty hold.
            self._clear_cognition_perception_probe(probe)
            self._schedule_cognition_retry(now_ns=now_ns)
            return
        deadline_ns = now_ns + _COGNITION_PERCEPTION_HANDOFF_TIMEOUT_NS
        retained = tuple(
            fact.model_copy(update={
                "expires_after_ms": max(1, (deadline_ns - fact.observed_ns) // 1_000_000),
            })
            for fact in retained
        )
        latest = self.blackboard.raw_latest()
        assert latest is not None
        self.blackboard.merge_semantics(instance_id=latest.instance_id, facts=retained)
        self._cognition_perception_probe = replace(
            probe,
            handoff_deadline_ns=deadline_ns,
            query_source=source,
            retained_facts=retained,
        )
        self._clear_cognition_retry()
        self._cognition_requested = True

    def _cognition_probe_scene_matches(
        self,
        probe: _CognitionPerceptionProbe,
        *,
        now_ns: int,
    ) -> bool:
        current = self.blackboard.fact("frame.dhash", min_confidence=1.0, now_ns=now_ns)
        if current is None or not isinstance(current.value, str) or probe.settle_dhash is None:
            return False
        try:
            return perceptual_hash_distance(probe.settle_dhash, current.value) <= 2
        except ValueError:
            return False

    def _clear_cognition_perception_probe(
        self,
        probe: _CognitionPerceptionProbe,
        *,
        action_grace: bool = False,
    ) -> None:
        """Revoke only this query's lease and decision, preserving newer producers."""

        future = probe.cognition_future
        if future is not None and self._pending_decision is future:
            future.cancel()
            self._pending_decision = None
            self._pending_operator_message_ids = ()
            self._pending_operator_message_kinds = {}
        if probe.query_source is not None:
            latest = self.blackboard.raw_latest()
            if action_grace and latest is not None:
                # Preserve provenance and the original observation timestamp.
                # Only a decision accepted on the matched view gets enough
                # remaining lifetime to cross the next motor boundary once.
                expires_ns = time.monotonic_ns() + _COGNITION_PERCEPTION_ACTION_GRACE_NS
                facts = tuple(
                    fact.model_copy(update={
                        "expires_after_ms": min(
                            fact.expires_after_ms,
                            max(1, (expires_ns - fact.observed_ns) // 1_000_000),
                        ),
                    })
                    for fact in probe.retained_facts
                    if (current := self.blackboard.fact(fact.key)) is not None
                    and current.source == probe.query_source
                    and current.observed_ns == fact.observed_ns
                )
                self.blackboard.merge_semantics(instance_id=latest.instance_id, facts=facts)
            else:
                self.blackboard.remove_semantic_facts(
                    tuple(fact.key for fact in probe.retained_facts),
                    expected_source=probe.query_source,
                )
        if getattr(self, "_cognition_perception_probe", None) is probe:
            self._cognition_perception_probe = None

    def _reconcile_craft_semantic_probe(self, active: SkillRun) -> None:
        probe = getattr(self, "_craft_semantic_probe", None)
        if probe is None:
            return
        phase = self.executor.plank_crafting_phase
        if active.run_id != probe.run_id or phase != probe.phase:
            self._craft_semantic_probe = None
            return
        # ActiveVLMWorker only becomes available after metrics and blackboard
        # publication are complete, avoiding a completion/publication race.
        if not self.perception.semantic_available():
            return
        terminal_count = self._active_vlm_terminal_count()
        if terminal_count is None or terminal_count <= probe.terminal_count_before:
            return
        self.executor.note_plank_crafting_semantic_completion(probe.phase)
        self._craft_semantic_probe = None

    def _craft_semantic_budget_available(self, *, now_ns: int) -> bool:
        remaining_ms = self.executor.plank_crafting_semantic_time_remaining_ms(
            now_ns=now_ns
        )
        if remaining_ms is None:
            return False
        latency = self._active_vlm_status().get("last_latency_ms")
        if not isinstance(latency, (int, float)) or isinstance(latency, bool) or latency <= 0:
            return remaining_ms >= 5_000
        required_ms = min(
            _CRAFT_SEMANTIC_MAX_REQUIRED_BUDGET_MS,
            max(5_000, int(float(latency) * _CRAFT_SEMANTIC_LATENCY_MARGIN)),
        )
        return remaining_ms >= required_ms

    def _merge_operator_target(self) -> None:
        """Publish a newly selected operator region as ROCKET's reference target."""
        if self.state_db is None:
            return
        target = self.state_db.load_operator_target()
        if target is None:
            if self._last_operator_target_id is not None:
                self.blackboard.remove_semantic_track(self._last_operator_target_id)
                self._last_operator_target_id = None
            return
        current_hash = self.blackboard.fact("frame.dhash", min_confidence=1.0)
        target_facts = _operator_target_facts(target, current_hash)
        if target_facts:
            self.blackboard.merge_semantics(
                instance_id=self.perception.instance_id,
                facts=target_facts,
            )
        if target.track_id == self._last_operator_target_id:
            return
        latest = self.blackboard.raw_latest()
        if latest is None:
            return
        if self._last_operator_target_id is not None:
            self.blackboard.remove_semantic_track(self._last_operator_target_id)
        if self.blackboard.upsert_semantic_track(
            instance_id=latest.instance_id,
            track=target,
        ):
            self._last_operator_target_id = target.track_id

    def _merge_policy_perception(self) -> None:
        """Merge optional learned motor-side perception into the blackboard."""
        merge = getattr(self.executor.policy, "merge_perception", None)
        if callable(merge):
            merge(self.blackboard)

    def _semantic_question(self, skill_id: str | None) -> str:
        if skill_id is None:
            return (
                "Describe only actionable visible state: immediate hazards, nearby resources, "
                "walkable direction, HUD danger, open GUI, and new chat. Emit target.visible, "
                "danger.immediate and normalized target.dx/target.dy when applicable."
            )
        if skill_id == "craft_wood_planks":
            return (
                "Inspect only the current Bedrock inventory GUI. Report gui.mode, "
                "inventory.logs, and inventory.planks from visible pixels. If a wood-planks "
                "recipe is visibly craftable, localize its clickable tile as a GUI track and "
                "label it exactly craftable_planks_recipe; otherwise do not emit that track."
            )
        spec = self.skills.get(skill_id)
        return (
            f"For skill {spec.name!r}, determine its preconditions, success/failure signals, "
            "target visibility and normalized target.dx/target.dy. Include immediate danger "
            "and chat."
        )

    def _cognition_due(self, *, operator_waiting: bool) -> bool:
        """Decide whether high-level cognition is worth invoking right now.

        Decouples planning from the motor loop's cadence: while a skill is
        actively executing under a fresh, non-exhausted plan there is nothing
        new to decide and re-invoking the (slow, local) VLM every cycle would
        churn planning effort for no benefit (motor never waits on cognition, so
        this is purely a planning-cadence decision). Fall through to True only
        when an event genuinely needs a decision: a skill finished, the plan is
        exhausted, an operator asked, or an explicit replan is requested.
        """
        if operator_waiting or self._cognition_requested:
            return True
        active = self.executor.run
        plan_active = bool(self._plan_steps) and 0 <= self._plan_index < len(self._plan_steps)
        return not (
            active is not None
            and active.outcome == SkillOutcome.RUNNING
            and plan_active
        )

    def _start_cognition_if_due(self) -> None:
        perception_probe = getattr(self, "_cognition_perception_probe", None)
        if perception_probe is not None:
            if self._new_queued_operator_message_waiting():
                # New operator authority revokes both the scene hold and its
                # one decision, including the query's extended planning cues.
                self._clear_cognition_perception_probe(perception_probe)
            elif (
                perception_probe.handoff_deadline_ns is None
                or perception_probe.cognition_future is not None
            ):
                # The VLM worker and planner share one serialized local-model
                # lane. Permit exactly one follow-up after publication; no
                # replanning can extend this query's bounded scene ownership.
                return
        headroom = getattr(self, "_headroom_recovery", None)
        if headroom is not None:
            if not self._queued_operator_message_waiting():
                return
            # Fresh operator authority cancels this optional autonomous
            # transaction. If a child is running, release it before the normal
            # operator fast path or model decision takes ownership.
            running = self.executor.run
            child_run_ids = {headroom.mining_run_id, headroom.retry_run_id}
            self._clear_headroom_recovery(headroom)
            if (
                running is not None
                and running.outcome == SkillOutcome.RUNNING
                and running.run_id in child_run_ids
            ):
                cancelled = self.executor.cancel()
                try:
                    if cancelled.action is not None:
                        self._send_motor(cancelled.action, execution=cancelled)
                finally:
                    self._record_terminal_run(cancelled.run, advance_plan=False)
                self._execution_revision += 1
        continuation = getattr(self, "_gather_acquisition_continuation", None)
        if continuation is not None:
            if not self._queued_operator_message_waiting():
                return
            running = self.executor.run
            self._gather_acquisition_continuation = None
            if (
                running is not None
                and running.outcome == SkillOutcome.RUNNING
                and running.run_id == continuation.active_run_id
            ):
                cancelled = self.executor.cancel()
                try:
                    if cancelled.action is not None:
                        self._send_motor(cancelled.action, execution=cancelled)
                finally:
                    self._record_terminal_run(cancelled.run, advance_plan=False)
                self._execution_revision += 1
        executor = getattr(self, "executor", None)
        active = None if executor is None else executor.run
        if (
            active is not None
            and active.outcome == SkillOutcome.RUNNING
            and active.skill_id in _ATOMIC_SKILL_IDS
        ):
            # Short closed-loop transactions retain ownership until their
            # bounded verifier or timeout finishes. Safety scene recovery and
            # the supervisor's pause/emergency paths remain independent.
            return
        if self._pending_decision is not None:
            if self._preempt_pending_cognition_for_operator():
                # The replacement is a completed deterministic decision. Apply
                # it on this motor-loop turn so operator authority can take
                # ownership from a disposable keepalive without waiting for
                # the stale model request to finish.
                self._consume_cognition()
            return
        now = time.monotonic_ns()
        new_operator_message = self._new_queued_operator_message_waiting()
        if (
            now < getattr(self, "_cognition_retry_not_before_ns", 0)
            and not new_operator_message
        ):
            return
        if new_operator_message:
            # New operator authority is a new decision problem, not another
            # attempt at the failed snapshot. It may bypass the old backoff
            # once; marking it DELIVERED below makes later retries wait.
            self._clear_cognition_retry()
        interval = int(1e9 / self.cognition_hz)
        operator_waiting = self._queued_operator_message_waiting()
        if (
            not self._cognition_requested
            and not operator_waiting
            and now - self._last_cognition_ns < interval
        ):
            return
        if not self._cognition_due(operator_waiting=operator_waiting):
            return
        context = self._cognition_context()
        if self._stage_operator_fast_path(context):
            # Literal, feasible operator authority does not need the model.
            # Apply the completed decision on this motor-loop turn even when a
            # previously detached model request still occupies the sole worker.
            self.metrics.cognition_calls += 1
            if getattr(self, "_cognition_perception_probe", None) is perception_probe:
                if perception_probe is not None:
                    self._cognition_perception_probe = replace(
                        perception_probe, cognition_future=self._pending_decision,
                    )
            self._consume_cognition()
            return
        if getattr(self, "_gui_fast_path_deferred", False):
            # Keep the world visible while the sole local-model lane drains.
            # Starting slow cognition here would only queue behind that same
            # lane and postpone the pixel-grounded GUI transaction again.
            return
        self._pending_operator_message_ids = tuple(
            message.message_id
            for message in context.operator_messages
            if message.status in {OperatorMessageStatus.QUEUED, OperatorMessageStatus.DELIVERED}
        )
        self._pending_operator_message_kinds = {
            message.message_id: message.kind
            for message in context.operator_messages
            if message.message_id in self._pending_operator_message_ids
        }
        if self.state_db is not None:
            for message in context.operator_messages:
                if message.status == OperatorMessageStatus.QUEUED:
                    self._persist_operator_message_status(
                        message.message_id,
                        OperatorMessageStatus.DELIVERED,
                        timestamp_ns=time.time_ns(),
                    )
        if self.high_level is None:
            engine = BootstrapCognitionPolicy(self.skills)
            self._pending_decision = self._pool.submit(
                engine.decide,
                self.blackboard,
                context,
            )
        else:
            self._pending_decision = self._pool.submit(
                self.high_level.decide,
                self.blackboard,
                context,
            )
        self._last_cognition_ns = now
        self._cognition_requested = False
        self._pending_execution_revision = self._execution_revision
        self.metrics.cognition_calls += 1
        if getattr(self, "_cognition_perception_probe", None) is perception_probe:
            if perception_probe is not None:
                self._cognition_perception_probe = replace(
                    perception_probe, cognition_future=self._pending_decision,
                )

    def _preempt_pending_cognition_for_operator(self) -> bool:
        """Replace a stale model future with one safe deterministic operator decision."""
        stale_future = self._pending_decision
        if stale_future is None or self.high_level is None or self.state_db is None:
            return False
        if not self._queued_operator_message_waiting():
            return False

        context = self._cognition_context()
        return self._stage_operator_fast_path(context, stale_future=stale_future)

    def _stage_operator_fast_path(
        self,
        context: CognitionContext,
        *,
        stale_future: concurrent.futures.Future[CognitionDecision] | None = None,
    ) -> bool:
        """Stage one executable operator decision without occupying the model worker."""
        self._gui_fast_path_deferred = False
        if self.high_level is None or self.state_db is None:
            return False
        fast_path = getattr(self.high_level, "_operator_fast_path_decision", None)
        if not callable(fast_path):
            return False
        decision = fast_path(self.blackboard, context)
        if decision is None:
            return False
        pending_ids = tuple(
            message.message_id
            for message in context.operator_messages
            if message.status in {OperatorMessageStatus.QUEUED, OperatorMessageStatus.DELIVERED}
        )
        selected_id = _selected_operator_message_id(decision, pending_ids)
        if selected_id is None:
            return False
        if decision.skill_id is not None:
            requested = self.skills.get(decision.skill_id)
            stale_request_running = stale_future is not None and not stale_future.done()
            if requested.action_level == ActionLevel.GUI and (
                stale_request_running or not local_model_inference_available()
            ):
                # A running HTTP request cannot be interrupted by Future.cancel.
                # Do not open a modal game GUI while its required semantic scan
                # is known to be queued behind that request.
                self._gui_fast_path_deferred = True
                return False

        running = self.executor.run
        if (
            running is not None
            and running.outcome == SkillOutcome.RUNNING
            and running.context_key == _EXPLORE_KEEPALIVE_CONTEXT
        ):
            cancelled = self.executor.cancel()
            try:
                if cancelled.action is not None:
                    self._send_motor(cancelled.action, execution=cancelled)
            finally:
                self._record_terminal_run(cancelled.run)
            self._execution_revision += 1

        for message in context.operator_messages:
            if message.status == OperatorMessageStatus.QUEUED:
                self._persist_operator_message_status(
                    message.message_id,
                    OperatorMessageStatus.DELIVERED,
                    timestamp_ns=time.time_ns(),
                )

        # A running thread-pool future cannot be interrupted safely. Cancel it
        # when it has not started, otherwise detach it; either way its stale
        # result can no longer alter runtime state. New fast-path decisions also
        # use this completed-future route so they never queue behind a detached
        # worker.
        if stale_future is not None:
            stale_future.cancel()
        replacement: concurrent.futures.Future[CognitionDecision] = (
            concurrent.futures.Future()
        )
        replacement.set_result(decision)
        self._pending_decision = replacement
        self._pending_operator_message_ids = pending_ids
        self._pending_operator_message_kinds = {
            message.message_id: message.kind
            for message in context.operator_messages
            if message.message_id in pending_ids
        }
        self._pending_execution_revision = self._execution_revision
        self._cognition_requested = False
        return True

    def _consume_cognition(self) -> None:
        probe = getattr(self, "_cognition_perception_probe", None)
        if probe is None or probe.handoff_deadline_ns is None:
            self._consume_cognition_decision()
            return
        # Tick consumes completed futures before its ordinary reconciliation.
        # Validate the current captured view here too, before any action starts.
        self._reconcile_cognition_perception_probe()
        if self._cognition_perception_probe is not probe:
            return
        assert probe is not None
        future = self._pending_decision
        if future is None or future is not probe.cognition_future or not future.done():
            return
        previous_run = self.executor.run
        try:
            self._consume_cognition_decision()
        finally:
            running = self.executor.run
            accepted_action = bool(
                running is not None
                and running is not previous_run
                and running.outcome == SkillOutcome.RUNNING
            )
            self._clear_cognition_perception_probe(probe, action_grace=accepted_action)

    def _idle_stall_probe_run_id(self, decision: CognitionDecision) -> str | None:
        """Buy one observation, never an action, after two autonomous stalls."""
        if (
            decision.skill_id is not None
            or decision.ask_perception
            or decision.request_replan
            or (decision.research_query or "").strip()
            or (decision.say or "").strip()
            or (decision.game_chat or "").strip()
            or not self._traversal_escalation_pending
        ):
            return None
        if any(
            getattr(self, owner, None) is not None
            for owner in (
                "_headroom_recovery", "_gather_acquisition_continuation",
                "_craft_semantic_probe", "_cognition_perception_probe",
            )
        ):
            return None
        executor = getattr(self, "executor", None)
        if executor is None or (
            executor.run is not None and executor.run.outcome == SkillOutcome.RUNNING
        ):
            return None
        stop = getattr(self, "_stop", None)
        if (
            (stop is not None and stop.is_set())
            or operator_pause_latched()
            or emergency_stop_latched()
            or not self._headroom_scene_is_safe()
        ):
            return None
        underwater = self.blackboard.fact("environment.underwater", min_confidence=0.7)
        if underwater is not None and underwater.value is True:
            return None
        if self._pending_operator_message_ids or self._pending_operator_status_updates:
            return None
        if self.state_db is not None:
            # Durable acknowledged instructions still own intent even when no
            # new message is queued. Corrections retain their existing tombstone.
            messages = self.state_db.load_operator_messages(
                statuses={OperatorMessageStatus.QUEUED, OperatorMessageStatus.DELIVERED},
                limit=20,
            )
            if messages or _active_operator_messages(self.state_db.load_operator_messages(
                statuses={OperatorMessageStatus.ACKNOWLEDGED}, limit=20,
            )):
                return None
        goal = self._plan_goal_id
        if (
            not goal
            or goal.startswith("operator:")
            or decision.chosen_goal_id != goal
            or not self._plan_steps[self._plan_index:]
            or self._plan_started_ns <= 0
        ):
            return None
        # Inspect attempts BEFORE filtering outcomes. A newer success, timeout,
        # cancellation or unrelated task must not expose older failures.
        attempts = tuple(
            run for run in self._recent_skill_runs
            if run.skill_id in _BOUNDED_KEEPALIVE_SKILL_IDS
            or run.skill_id == "gather_nearby_wood"
        )[:2]
        if len(attempts) != 2:
            return None
        newest, older = attempts
        if (
            newest.run_id == older.run_id
            or newest.run_id == self._idle_stall_probe_used_for_run_id
            or newest.context_key != older.context_key
            or newest.context_key not in {goal, _EXPLORE_KEEPALIVE_CONTEXT}
            or any(
                run.outcome != SkillOutcome.FAILED
                or run.failure_code != SkillFailureCode.LOCOMOTION_STALLED
                or run.ended_ns is None
                or run.started_ns < self._plan_started_ns
                for run in attempts
            )
        ):
            return None
        obstacle = self.blackboard.fact("obstacle.ahead", min_confidence=0.7)
        observation = self.blackboard.fact("scene.observation_dhash", min_confidence=1.0)
        current = self.blackboard.fact("frame.dhash", min_confidence=1.0)
        latest = self.blackboard.raw_latest()
        if (
            obstacle is not None and isinstance(obstacle.value, bool)
            and observation is not None and current is not None and latest is not None
            and obstacle.source == observation.source
            and obstacle.observed_ns == observation.observed_ns
            and current.observed_ns >= latest.captured_ns
            and isinstance(observation.value, str) and isinstance(current.value, str)
        ):
            try:
                if perceptual_hash_distance(observation.value, current.value) <= 2:
                    return None  # A trustworthy False answers the question too.
            except ValueError:
                pass
        return newest.run_id

    def _consume_cognition_decision(self) -> None:
        if getattr(self, "_headroom_recovery", None) is not None:
            # A decision completed against pre-recovery pixels cannot take the
            # executor while the bounded recovery owns a stable scene. In the
            # normal tick order, _start_cognition_if_due runs next and gives a
            # queued operator message authority to clear/release this owner
            # before any completed decision can be consumed.
            return
        executor = getattr(self, "executor", None)
        active = None if executor is None else executor.run
        continuation = getattr(self, "_gather_acquisition_continuation", None)
        if continuation is not None:
            if (
                active is not None
                and active.outcome == SkillOutcome.RUNNING
                and active.run_id == continuation.active_run_id
            ):
                return
            self._gather_acquisition_continuation = None
        if (
            active is not None
            and active.outcome == SkillOutcome.RUNNING
            and active.skill_id in _ATOMIC_SKILL_IDS
        ):
            return
        future = self._pending_decision
        if future is None or not future.done():
            return
        self._pending_decision = None
        try:
            decision = future.result()
        except Exception:
            now = time.monotonic_ns()
            self._last_cognition_ns = now
            self._pending_operator_message_ids = ()
            self._schedule_cognition_retry(now_ns=now)
            return
        now = time.monotonic_ns()
        self._last_cognition_ns = now
        if self._pending_execution_revision != self._execution_revision:
            # The decision was sampled before the option produced terminal
            # evidence. Re-evaluate with that failure/success in context rather
            # than immediately replaying the stale option choice.
            self._pending_operator_message_ids = ()
            self._cognition_requested = True
            return
        if self._operator_message_arrived_after_snapshot():
            # This decision was produced from an older context snapshot. A
            # fresh operator message has higher authority and must be included
            # before any skill switch or acknowledgement is applied.
            self._pending_operator_message_ids = ()
            self._cognition_requested = True
            return
        if (
            decision.skill_id in _WOOD_INVENTORY_AUDIT_SKILLS
            and self._planks_retry_requires_wood()
            and planks_retry_requires_wood(self._cognition_context())
        ):
            self._pending_operator_message_ids = ()
            self._cognition_requested = True
            return
        if self._close_crafting_gui_before_world_decision(decision):
            return
        selected_message_id = _selected_operator_message_id(
            decision,
            self._pending_operator_message_ids,
        )
        if (
            selected_message_id is not None
            and getattr(self, "_pending_operator_message_kinds", {}).get(
                selected_message_id
            )
            == OperatorMessageKind.CORRECTION
        ):
            # Corrections are bounded overrides, not durable multi-step goals.
            # The accepted skill still executes under its operator context, but
            # model-generated plan text cannot keep replaying it afterward.
            decision = decision.model_copy(update={"plan_steps": ()})
        idle_stall_run_id = self._idle_stall_probe_run_id(decision)
        idle_stall_decision = decision if idle_stall_run_id is not None else None
        if idle_stall_run_id is not None:
            decision = decision.model_copy(update={
                "ask_perception": ("obstacle.ahead",),
                "request_replan": True,
            })
        perception_output_keys = tuple(
            dict.fromkeys(
                key
                for question in decision.ask_perception
                for key in resolve_grounded_output_keys((), question)
            )
        )
        if decision.ask_perception:
            # A question is an explicit admission that the sampled decision is
            # missing current visual evidence. Never execute a simultaneously
            # proposed skill against that older snapshot; observe once, then
            # let fresh cognition select the action.
            decision = decision.model_copy(
                update={
                    "skill_id": None,
                    "request_replan": True,
                }
            )
        self._last_decision = decision
        if idle_stall_run_id is None:
            self._adopt_plan_if_revised(decision)
        operator_acknowledged = False
        if self.state_db is not None and self._pending_operator_message_ids:
            if selected_message_id is not None and not decision.request_replan:
                response = decision.say or decision.reasoning_summary
                operator_acknowledged = self._persist_operator_message_status(
                    selected_message_id,
                    OperatorMessageStatus.ACKNOWLEDGED,
                    timestamp_ns=time.time_ns(),
                    response_text=response,
                )
            self._pending_operator_message_ids = ()
            self._pending_operator_message_kinds = {}
        perception_probe_started = False
        if decision.ask_perception:
            # ``q`` means the planner cannot justify an action from its current
            # facts. Resolve every literal key into one request: the VLM queue
            # admits exactly one job, so submitting one job per key silently
            # discarded the second question. Invalid/free-form questions stay
            # fail-closed instead of expanding into an expensive open query.
            latest = self.blackboard.raw_latest()
            terminal_count_before = self._active_vlm_terminal_count()
            running = self.executor.run
            if (
                perception_output_keys
                and latest is not None
                and terminal_count_before is not None
                and self.perception.semantic_available()
                and local_model_inference_available()
            ):
                if running is not None and running.outcome == SkillOutcome.RUNNING:
                    cancelled = self.executor.cancel()
                    try:
                        if cancelled.action is not None:
                            self._send_motor(cancelled.action, execution=cancelled)
                    finally:
                        self._record_terminal_run(cancelled.run, advance_plan=False)
                    self._execution_revision += 1
                self._cognition_perception_probe = _CognitionPerceptionProbe(
                    query_id=None,
                    requested_keys=perception_output_keys,
                    frame_id=latest.frame_id,
                    execution_revision=self._execution_revision,
                    terminal_count_before=None,
                    settle_dhash=None,
                    settle_deadline_ns=(
                        time.monotonic_ns() + _COGNITION_PERCEPTION_SETTLE_TIMEOUT_NS
                    ),
                    trigger_run_id=idle_stall_run_id,
                    trigger_decision=idle_stall_decision,
                )
                if idle_stall_run_id is not None:
                    # Consume only on transaction creation, and never refund
                    # for unknown/stale answers, timeout or a new camera frame.
                    self._idle_stall_probe_used_for_run_id = idle_stall_run_id
                perception_probe_started = True
        game_chat = _authorized_game_chat(
            decision,
            self.blackboard,
            already_replied_ns=self._last_player_chat_replied_ns,
        )
        if game_chat:
            try:
                send_command("chat", lease_id=self.lease_id, text=game_chat)
                self.metrics.game_chat_messages += 1
                # Answer a player message once. The social fact stays merged
                # (expires in 30s) but the signature gate blocks re-replies.
                self._last_player_chat_replied_ns = time.monotonic_ns()
            except Exception:
                pass
        if decision.skill_id is not None:
            running = self.executor.run
            if running is not None and running.outcome == SkillOutcome.RUNNING:
                if (
                    running.skill_id != decision.skill_id
                    or self.executor.parameters != decision.skill_parameters
                    or running.context_key == _EXPLORE_KEEPALIVE_CONTEXT
                ):
                    cancelled = self.executor.cancel()
                    try:
                        if cancelled.action is not None:
                            self._send_motor(cancelled.action, execution=cancelled)
                    finally:
                        self._record_terminal_run(cancelled.run)
                    self._execution_revision += 1
                    spec = self.skills.get(decision.skill_id)
                    self.executor.start(
                        spec,
                        run_id=uuid.uuid4().hex,
                        context_key=decision.chosen_goal_id or "default",
                        parameters=decision.skill_parameters,
                        instruction=decision.instruction,
                    )
            else:
                spec = self.skills.get(decision.skill_id)
                self.executor.start(
                    spec,
                    run_id=uuid.uuid4().hex,
                    context_key=decision.chosen_goal_id or "default",
                    parameters=decision.skill_parameters,
                    instruction=decision.instruction,
                )
            if not decision.request_replan:
                self._traversal_escalation_pending = False
        operator_waiting = self._queued_operator_message_waiting()
        if perception_probe_started:
            self._clear_cognition_retry()
            self._cognition_requested = False
        elif decision.request_replan:
            self._schedule_cognition_retry(now_ns=now)
        elif operator_waiting and operator_acknowledged:
            self._schedule_operator_followup(now_ns=now)
        elif operator_waiting:
            # The model returned a valid shape but ignored pending operator
            # authority (or its acknowledgement could not be stored). Treat it
            # as an unfinished decision and retain bounded retry pressure.
            self._schedule_cognition_retry(now_ns=now)
        else:
            self._clear_cognition_retry()

    def _close_crafting_gui_before_world_decision(
        self,
        decision: CognitionDecision,
    ) -> bool:
        """Finish a verified inventory close before adopting world control.

        A completed decision was sampled while crafting owned the inventory.
        Reusing that decision after the visual scene changes would be stale, so
        discard it, close the GUI, and let the close terminal event request a
        fresh decision from the restored world frame.
        """
        if decision.skill_id is None or decision.skill_id == "craft_wood_planks":
            return False
        running = self.executor.run
        if (
            running is None
            or running.outcome != SkillOutcome.RUNNING
            or running.skill_id != "craft_wood_planks"
        ):
            return False
        requested = self.skills.get(decision.skill_id)
        if requested.action_level == ActionLevel.GUI:
            return False
        cancelled = self.executor.cancel()
        try:
            if cancelled.action is not None:
                self._send_motor(cancelled.action, execution=cancelled)
        finally:
            self._record_terminal_run(cancelled.run)
        self._execution_revision += 1
        recovery = _first_feasible_recovery(
            self.skills,
            cancelled.recovery_skills,
            self.blackboard,
        )
        if recovery is not None:
            self._start_recovery_skill(recovery, cancelled.run)
            # The recovery's terminal result is what requests fresh cognition.
            self._cognition_requested = False
        else:
            # Fail closed if the configured recovery was removed or became
            # infeasible; the next live scene-recovery pass can still route the
            # specific fast inventory interlock.
            self._cognition_requested = True
        self._pending_operator_message_ids = ()
        return True

    def _schedule_cognition_retry(self, *, now_ns: int) -> None:
        """Retry a failed/unfinished strategic decision without a hot loop."""
        retry_count = min(5, getattr(self, "_cognition_retry_count", 0) + 1)
        delay_ns = min(
            _COGNITION_RETRY_MAX_NS,
            _COGNITION_RETRY_BASE_NS * (2 ** (retry_count - 1)),
        )
        self._cognition_retry_count = retry_count
        self._cognition_retry_not_before_ns = now_ns + delay_ns
        self._cognition_requested = True

    def _clear_cognition_retry(self) -> None:
        self._cognition_retry_count = 0
        self._cognition_retry_not_before_ns = 0

    def _schedule_operator_followup(self, *, now_ns: int) -> None:
        """Drain another valid operator message promptly without failure backoff."""
        self._clear_cognition_retry()
        self._cognition_retry_not_before_ns = now_ns + _OPERATOR_FOLLOWUP_DELAY_NS
        self._cognition_requested = True

    def _advance_plan_on_step_complete(self, run: SkillRun) -> None:
        """Mark one persistent plan step done after a skill succeeds.

        Most steps are short free-form plans, so a completed world skill consumes
        the current position permissively. Modal inventory transitions are a
        narrow exception: opening or closing a GUI is often prerequisite cleanup,
        and advances only an explicit matching GUI step. The last decision's goal
        remains a sanity gate so an off-plan operator task cannot eat plan progress.
        """
        if not self._plan_steps:
            return
        if self._plan_index >= len(self._plan_steps):
            return
        if not _plan_step_requests_inventory_transition(
            run.skill_id, self._plan_steps[self._plan_index]
        ):
            return
        if self._last_decision is not None:
            decision_goal = self._last_decision.chosen_goal_id
            if (
                self._plan_goal_id is not None
                and decision_goal is not None
                and decision_goal != self._plan_goal_id
            ):
                return
        self._plan_index += 1
        self._plan_step_completed_ns = time.monotonic_ns()

    def _adopt_plan_if_revised(self, decision: CognitionDecision) -> None:
        """Persist a long-horizon plan across motor ticks.

        The high-level VLM re-decides on its own cadence; the motor loop acts on
        the current skill+instruction every 20 Hz. This plan state lets planning
        span many motor ticks instead of being discarded each decision.

        Replacement rules avoid thrashing a running plan:
          * New goal, or the current plan is exhausted -> adopt fresh (index 0).
          * Same goal and plan still running: only refresh the stored steps
            without moving our position when the new plan still opens with the
            current remaining steps (a genuine extension/refinement). An exact
            echo of the remaining steps changes nothing.
        """
        steps = decision.plan_steps
        if not steps:
            # A concrete operator command must not inherit a different
            # operator command's unfinished plan. Questions and status replies
            # have no execution instruction and deliberately preserve it.
            if (
                decision.skill_id is not None
                and decision.instruction
                and decision.chosen_goal_id is not None
                and decision.chosen_goal_id.startswith("operator:")
                and self._plan_goal_id is not None
                and self._plan_goal_id.startswith("operator:")
                and decision.chosen_goal_id != self._plan_goal_id
            ):
                self._plan_steps = ()
                self._plan_goal_id = decision.chosen_goal_id
                self._plan_index = 0
                self._plan_started_ns = time.monotonic_ns()
            return
        goal_changed = (
            decision.chosen_goal_id is not None
            and decision.chosen_goal_id != self._plan_goal_id
        )
        remaining = self._plan_steps[self._plan_index:]
        if not goal_changed and remaining:
            if self._is_prefix(remaining, steps):
                if steps != remaining:
                    self._plan_steps = steps
                return
            return
        if goal_changed or not remaining:
            self._plan_steps = steps
            self._plan_goal_id = decision.chosen_goal_id
            self._plan_index = 0
            self._plan_started_ns = time.monotonic_ns()

    @staticmethod
    def _is_prefix(prefix: tuple[str, ...], steps: tuple[str, ...]) -> bool:
        return len(prefix) <= len(steps) and steps[: len(prefix)] == prefix

    def _publish_player_chat_facts(self) -> None:
        """Turn freshly observed player chat lines into an authorizing fact.

        The grounded VLM extracts world-chat lines into the blackboard. Those
        lines are what give the high-level cognition authority to reply through
        Bedrock world chat (`game_chat`); otherwise the agent can never answer
        a player's question in game. One new line (different speaker/text from
        the last replied line) publishes a fresh `social.player_message`.
        """
        latest = self.blackboard.latest()
        if latest is None:
            return
        now_ns = time.monotonic_ns()
        latest_line = None
        for line in latest.chat:
            if line.speaker is None or line.speaker.casefold() in {
                self.role.role_id.casefold(),
                "eidos",
                "you",
                "console",
            }:
                continue
            if now_ns - line.observed_ns > 60_000_000_000:
                continue
            if latest_line is None or line.observed_ns > latest_line.observed_ns:
                latest_line = line
        if latest_line is None:
            return
        signature = f"{latest_line.speaker}:{latest_line.text}"
        if self._last_player_chat_signature == signature:
            return
        if (
            self._last_player_chat_replied_ns is not None
            and latest_line.observed_ns <= self._last_player_chat_replied_ns
        ):
            return
        self._last_player_chat_signature = signature
        self.blackboard.merge_semantics(
            instance_id=self.perception.instance_id,
            facts=(
                PerceptionFact(
                    key="social.player_message",
                    value=signature,
                    confidence=0.95,
                    observed_ns=now_ns,
                    source="grounded:player-chat",
                    expires_after_ms=30_000,
                ),
            ),
        )

    def _record_terminal_run(
        self,
        run: SkillRun,
        *,
        outcome_verification: OutcomeVerification | None = None,
        advance_plan: bool = True,
    ) -> None:
        """Record one terminal option exactly once across every exit path."""

        if run.outcome == SkillOutcome.RUNNING:
            raise ValueError("cannot record a running skill")
        if run.run_id in self._recorded_run_ids:
            return
        if run.skill_id == "collect_recent_drop":
            self._clear_drop_collection_authorization(run)
        neutral: set[str] = getattr(self, "_plan_neutral_recovery_runs", set())
        if run.run_id in neutral:
            advance_plan = False
            neutral.discard(run.run_id)
        if run.skill_id == "craft_wood_planks" and run.failure_reason == _PLANKS_NO_LOGS_REASON:
            self._planks_no_logs_failure_ns = run.ended_ns or time.monotonic_ns()
        stats = self.skills.record(run)
        if len(self._recorded_run_order) == _RECORDED_RUN_ID_LIMIT:
            self._recorded_run_ids.discard(self._recorded_run_order.popleft())
        self._recorded_run_order.append(run.run_id)
        self._recorded_run_ids.add(run.run_id)
        if not _expected_keepalive_expiry(run):
            self._recent_skill_runs.appendleft(run)

        if run.outcome == SkillOutcome.SUCCEEDED:
            self.metrics.skill_successes += 1
            if advance_plan:
                self._advance_plan_on_step_complete(run)
        elif run.outcome == SkillOutcome.FAILED:
            self.metrics.skill_failures += 1
            self.metrics.skill_failed_outcomes += 1
        elif run.outcome == SkillOutcome.TIMED_OUT:
            self.metrics.skill_failures += 1
            self.metrics.skill_timeouts += 1
        elif run.outcome == SkillOutcome.CANCELLED:
            self.metrics.skill_cancellations += 1

        observed_ns = time.time_ns()
        event = _terminal_run_event(
            run,
            observed_ns=observed_ns,
            trajectory_id=(
                None if self.trajectory is None else self.trajectory.manifest.trajectory_id
            ),
        )
        outcome_event = (
            None
            if outcome_verification is None
            else _verified_outcome_event(
                run,
                outcome_verification,
                observed_ns=observed_ns,
                trajectory_id=(
                    None
                    if self.trajectory is None
                    else self.trajectory.manifest.trajectory_id
                ),
            )
        )
        memory = _terminal_run_memory(
            run,
            stats,
            observed_ns=observed_ns,
            existing=self.memories.records,
            outcome_verification=outcome_verification,
        )
        if memory is not None:
            self.memories.upsert(memory)
            if run.skill_id == "craft_wood_planks" and run.failure_reason == _PLANKS_NO_LOGS_REASON:
                self._planks_failure_memory = memory
                self._planks_failure_memory_initialized = True

        if self.state_db is None:
            return
        self._pending_skill_stats[(run.skill_id, run.context_key)] = stats
        self._pending_runtime_events[event.event_id] = event
        if outcome_event is not None:
            self._pending_runtime_events[outcome_event.event_id] = outcome_event
        if memory is not None:
            self._pending_memories[memory.memory_id] = memory
        self._flush_pending_skill_stats(force=True)
        self._flush_pending_learning_records(force=True)

    def _flush_pending_skill_stats(self, *, force: bool = False) -> None:
        if self.state_db is None or not self._pending_skill_stats:
            return
        now = time.monotonic_ns()
        if not force and now - self._last_storage_retry_ns < 1_000_000_000:
            return
        self._last_storage_retry_ns = now
        for key, stats in tuple(self._pending_skill_stats.items()):
            try:
                self.state_db.save_skill_stats(key[0], key[1], stats)
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).casefold() and "busy" not in str(exc).casefold():
                    raise
                self.metrics.storage_contentions += 1
                self.metrics.last_storage_error = f"{type(exc).__name__}: {exc}"
                return
            else:
                self._pending_skill_stats.pop(key, None)
                self._clear_storage_error_if_drained()

    def _flush_pending_learning_records(self, *, force: bool = False) -> None:
        if self.state_db is None or not (
            self._pending_runtime_events or self._pending_memories
        ):
            return
        now = time.monotonic_ns()
        if not force and now - self._last_storage_retry_ns < 1_000_000_000:
            return
        self._last_storage_retry_ns = now
        for event_id, event in tuple(self._pending_runtime_events.items()):
            try:
                self.state_db.save_runtime_event(event)
            except sqlite3.OperationalError as exc:
                if not _sqlite_writer_contention(exc):
                    raise
                self.metrics.storage_contentions += 1
                self.metrics.last_storage_error = f"{type(exc).__name__}: {exc}"
                return
            else:
                self._pending_runtime_events.pop(event_id, None)
        for memory_id, memory in tuple(self._pending_memories.items()):
            try:
                self.state_db.save_memory(memory)
            except sqlite3.OperationalError as exc:
                if not _sqlite_writer_contention(exc):
                    raise
                self.metrics.storage_contentions += 1
                self.metrics.last_storage_error = f"{type(exc).__name__}: {exc}"
                return
            else:
                self._pending_memories.pop(memory_id, None)
        self._clear_storage_error_if_drained()

    def _persist_operator_message_status(
        self,
        message_id: str,
        status: OperatorMessageStatus,
        *,
        timestamp_ns: int,
        response_text: str | None = None,
    ) -> bool:
        """Commit an operator transition or retain it for bounded retry.

        Operator conversation is durable control-plane state, but it must not
        be able to terminate the 20 Hz motor process when a trajectory shard is
        publishing. The newest transition for each message supersedes an older
        pending transition and remains visible as storage backlog telemetry.
        """
        if self.state_db is None:
            return False
        update = (status, timestamp_ns, response_text)
        try:
            self.state_db.update_operator_message_status(
                message_id,
                status,
                timestamp_ns=timestamp_ns,
                response_text=response_text,
            )
        except KeyError:
            self._pending_operator_status_updates.pop(message_id, None)
            return False
        except sqlite3.OperationalError as exc:
            if not _sqlite_writer_contention(exc):
                raise
            self._pending_operator_status_updates[message_id] = update
            self._last_operator_storage_retry_ns = time.monotonic_ns()
            self.metrics.storage_contentions += 1
            self.metrics.last_storage_error = f"{type(exc).__name__}: {exc}"
            return False
        self._pending_operator_status_updates.pop(message_id, None)
        if status == OperatorMessageStatus.ACKNOWLEDGED:
            self.metrics.operator_responses += 1
        self._clear_storage_error_if_drained()
        return True

    def _flush_pending_operator_status_updates(self, *, force: bool = False) -> None:
        if self.state_db is None or not self._pending_operator_status_updates:
            return
        now = time.monotonic_ns()
        if not force and now - self._last_operator_storage_retry_ns < 1_000_000_000:
            return
        self._last_operator_storage_retry_ns = now
        for message_id, update in tuple(self._pending_operator_status_updates.items()):
            status, timestamp_ns, response_text = update
            if not self._persist_operator_message_status(
                message_id,
                status,
                timestamp_ns=timestamp_ns,
                response_text=response_text,
            ):
                return

    def _clear_storage_error_if_drained(self) -> None:
        if not (
            self._pending_skill_stats
            or self._pending_runtime_events
            or self._pending_memories
            or self._pending_operator_status_updates
        ):
            self.metrics.last_storage_error = None

    def _queued_operator_message_waiting(self) -> bool:
        """Return whether any operator message still awaits acknowledgement."""
        if self.state_db is None:
            return False
        return bool(
            self.state_db.load_operator_messages(
                statuses={
                    OperatorMessageStatus.QUEUED,
                    OperatorMessageStatus.DELIVERED,
                },
                limit=1,
            )
        )

    def _operator_message_arrived_after_snapshot(self) -> bool:
        """Detect pending authority that was absent from the in-flight decision."""
        if self.state_db is None:
            return False
        sampled = set(self._pending_operator_message_ids)
        pending = self.state_db.load_operator_messages(
            statuses={
                OperatorMessageStatus.QUEUED,
                OperatorMessageStatus.DELIVERED,
            },
            limit=20,
        )
        return any(message.message_id not in sampled for message in pending)

    def _new_queued_operator_message_waiting(self) -> bool:
        """Return whether fresh operator authority may bypass an old retry delay."""
        if self.state_db is None:
            return False
        pending_delivery = {
            message_id
            for message_id, update in self._pending_operator_status_updates.items()
            if update[0]
            in {
                OperatorMessageStatus.DELIVERED,
                OperatorMessageStatus.ACKNOWLEDGED,
            }
        }
        queued = self.state_db.load_operator_messages(
            statuses={OperatorMessageStatus.QUEUED},
            limit=20,
        )
        return any(message.message_id not in pending_delivery for message in queued)

    def _cognition_context(self) -> CognitionContext:
        goals = tuple((*role_standing_goals(self.role), *self.custom_goals))
        memories = tuple(self.memories.retrieve(limit=20))
        operator_messages: tuple[OperatorMessage, ...] = ()
        if self.state_db is not None:
            messages = self.state_db.load_operator_messages(
                statuses={
                    OperatorMessageStatus.QUEUED,
                    OperatorMessageStatus.DELIVERED,
                },
                limit=20,
            )
            if not messages:
                messages = self.state_db.load_operator_messages(
                    statuses={OperatorMessageStatus.ACKNOWLEDGED},
                    limit=20,
                )
            operator_messages = _active_operator_messages(messages)
        return CognitionContext(
            role=self.role,
            goals=goals,
            memories=memories,
            promises=self.social.active_promises(),
            wiki=(),
            operator_messages=operator_messages,
            recent_skill_runs=tuple(self._recent_skill_runs),
            current_plan=self._plan_steps,
            plan_goal_id=self._plan_goal_id,
            plan_index=self._plan_index,
            plan_started_ns=self._plan_started_ns,
            planks_retry_requires_wood=self._planks_retry_requires_wood(),
        )

    def _start_recovery_skill(
        self,
        recovery: SkillSpec,
        parent: SkillRun,
        *,
        plan_neutral: bool = False,
    ) -> SkillRun:
        run = self.executor.start(
            recovery,
            run_id=uuid.uuid4().hex,
            context_key=parent.context_key,
            parameters=_compatible_recovery_parameters(parent, recovery),
        )
        if plan_neutral or (
            parent.skill_id == "craft_wood_planks"
            and parent.outcome != SkillOutcome.SUCCEEDED
            and recovery.skill_id == "close_open_inventory"
        ):
            self._plan_neutral_recovery_runs = {
                *getattr(self, "_plan_neutral_recovery_runs", ()), run.run_id
            }
        return run

    def _planks_retry_requires_wood(self) -> bool:
        """Persist one prerequisite repair, not a stale claim of inventory absence."""
        memories = getattr(self, "memories", None)
        if memories is None:
            return False
        if not getattr(self, "_planks_failure_memory_initialized", False):
            self._planks_failure_memory = max(
                (
                    memory for memory in memories.records.values()
                    if memory.source == "runtime:verified-skill-outcome"
                    and memory.metadata.get("skill_id") == "craft_wood_planks"
                    and memory.metadata.get("reported_reason") == _PLANKS_NO_LOGS_REASON
                ),
                key=lambda memory: memory.updated_ns,
                default=None,
            )
            self._planks_failure_memory_initialized = True
        failure = self._planks_failure_memory
        if failure is None:
            return False  # Unknown initial inventory still receives one bounded audit.
        cleared = memories.records.get(_PLANKS_RETRY_CLEAR_MEMORY)
        if (
            cleared is not None
            and cleared.source == "runtime:craft-prerequisite-repair"
            and cleared.metadata.get("failure_revision_ns") == failure.updated_ns
        ):
            return False
        board = getattr(self, "blackboard", None)
        if board is None:
            return True
        now = time.monotonic_ns()
        for key in ("inventory.hotbar.logs", "inventory.logs"):
            fact = board.fact(key, min_confidence=0.9, now_ns=now)
            if (
                fact is None
                or not isinstance(fact.value, int)
                or isinstance(fact.value, bool)
                or fact.value < 1
                or fact.observed_ns <= getattr(self, "_planks_no_logs_failure_ns", 0)
                or not 0 <= now - fact.observed_ns <= fact.expires_after_ms * 1_000_000
            ):
                continue
            if key == "inventory.hotbar.logs":
                if fact.source != BEDROCK_HOTBAR_LOG_COUNT_SOURCE or fact.confidence < 0.99:
                    continue
            else:
                latest = board.latest()
                if not fact.source.startswith("vlm:") or latest is None or not any(
                    evidence.region_kind == EvidenceRegion.GUI
                    and evidence.evidence_id in fact.evidence_refs
                    # VLM observed_ns is completion time, not capture time.
                    # A delayed pre-failure image must not repair the prerequisite.
                    and getattr(self, "_planks_no_logs_failure_ns", 0)
                    < evidence.captured_ns <= fact.observed_ns
                    for evidence in latest.evidence
                ):
                    continue
            observed_ns = time.time_ns()
            marker = MemoryRecord(
                memory_id=_PLANKS_RETRY_CLEAR_MEMORY,
                kind=MemoryKind.WORKING,
                text="New positive log evidence permits a bounded planks inventory audit.",
                created_ns=observed_ns if cleared is None else cleared.created_ns,
                updated_ns=observed_ns,
                confidence=fact.confidence,
                importance=0.3,
                source="runtime:craft-prerequisite-repair",
                metadata={
                    "failure_revision_ns": failure.updated_ns,
                    "evidence_key": fact.key,
                    "evidence_source": fact.source,
                },
            )
            memories.upsert(marker)
            if getattr(self, "state_db", None) is not None:
                self._pending_memories[marker.memory_id] = marker
                self._flush_pending_learning_records(force=True)
            return False
        return True

    def _telemetry_payload(self, *, state: str) -> dict[str, object]:
        running = self.executor.run
        if running is not None and running.outcome != SkillOutcome.RUNNING:
            running = None
        decision = self._last_decision
        policy_status: dict[str, object] = {"policy_id": self.executor.policy.policy_id}
        status = getattr(self.executor.policy, "status", None)
        if callable(status):
            reported = status()
            if isinstance(reported, dict):
                policy_status = reported
        perception_status: dict[str, object] = {
            "fast_model_id": None
            if self.perception.fast_perception is None
            else self.perception.fast_perception.model_id,
            "fast_training_label_eligible": False
            if self.perception.fast_perception is None
            else self.perception.fast_perception.training_label_eligible,
            "active_vlm": None,
        }
        if self.perception.active_vlm is not None:
            perception_status["active_vlm"] = self.perception.active_vlm.status()
        cognition_probe = getattr(self, "_cognition_perception_probe", None)
        perception_status["cognition_probe"] = (
            None
            if cognition_probe is None
            else {
                "phase": (
                    "handoff"
                    if cognition_probe.handoff_deadline_ns is not None
                    else "settling" if cognition_probe.query_id is None else "grounding"
                ),
                "query_id": cognition_probe.query_id,
                "requested_keys": list(cognition_probe.requested_keys),
                "frame_id": cognition_probe.frame_id,
                "execution_revision": cognition_probe.execution_revision,
                "origin": (
                    "runtime:failure-triggered-observation"
                    if cognition_probe.trigger_run_id is not None else "model:requested"
                ),
                "trigger_run_id": cognition_probe.trigger_run_id,
                "trigger_decision": (
                    None if cognition_probe.trigger_decision is None
                    else cognition_probe.trigger_decision.model_dump(mode="json")
                ),
            }
        )
        perception_status["idle_stall_probe_used_for_run_id"] = (
            self._idle_stall_probe_used_for_run_id
        )
        cognition_status: dict[str, object] | None = None
        if self.high_level is not None:
            cognition_status = self.high_level.status()
        fresh_facts = self.blackboard.fresh_facts(min_confidence=0.35)
        perception_status["fresh_facts"] = {
            key: {
                "value": fact.value,
                "confidence": round(fact.confidence, 3),
                "source": fact.source,
            }
            for key, fact in sorted(fresh_facts.items())
        }
        latest = self.blackboard.latest()
        perception_status["tracks"] = (
            [] if latest is None else [track.model_dump(mode="json") for track in latest.tracks]
        )
        session_skill_totals = {
            "succeeded": self.metrics.skill_successes,
            "failed": self.metrics.skill_failed_outcomes,
            "timed_out": self.metrics.skill_timeouts,
            "cancelled": self.metrics.skill_cancellations,
            "attempts": (
                self.metrics.skill_successes
                + self.metrics.skill_failed_outcomes
                + self.metrics.skill_timeouts
                + self.metrics.skill_cancellations
            ),
        }
        trajectory_status = (
            self.trajectory.status()
            if self.trajectory is not None
            else {
                "enabled": False,
                "disabled_reason": self.trajectory_disabled_reason
                or "disabled-by-configuration",
                "written_steps": 0,
                "dropped_steps": 0,
                "queued_samples": 0,
                "queue_capacity": 0,
            }
        )
        return {
            "schema_version": 1,
            "state": state,
            "role": self.role.role_id,
            "lease_id": self.lease_id,
            "frames": self.metrics.frames,
            "motor_actions": self.metrics.motor_actions,
            "cognition_calls": self.metrics.cognition_calls,
            "semantic_requests": self.metrics.semantic_requests,
            "operator_responses": self.metrics.operator_responses,
            "game_chat_messages": self.metrics.game_chat_messages,
            # Compatibility alias for existing telemetry consumers. This now
            # means actual Bedrock chat transmissions, never console replies.
            "chat_messages": self.metrics.game_chat_messages,
            "skill_successes": self.metrics.skill_successes,
            "skill_failures": self.metrics.skill_failures,
            "skill_totals": {
                "session": session_skill_totals,
                "lifetime": _skill_stats_totals(self.skills.stats.values()),
            },
            "last_capture_ms": round(self.metrics.last_capture_ms, 3),
            "last_motor_ms": round(self.metrics.last_motor_ms, 3),
            "stale_frame_skips": self.metrics.stale_frame_skips,
            "consecutive_stale_frames": self.metrics.consecutive_stale_frames,
            "storage_contentions": self.metrics.storage_contentions,
            "storage_backlog": (
                len(self._pending_skill_stats)
                + len(self._pending_runtime_events)
                + len(self._pending_memories)
                + len(self._pending_operator_status_updates)
            ),
            "last_storage_error": self.metrics.last_storage_error,
            "trajectory_recording": trajectory_status,
            "active_skill": None if running is None else running.skill_id,
            "active_skill_parameters": ({} if running is None else self.executor.policy_parameters),
            "active_instruction": None if running is None else self.executor.instruction,
            "plan_steps": [] if decision is None else list(decision.plan_steps),
            "persistent_plan": {
                "goal": self._plan_goal_id,
                "steps": list(self._plan_steps),
                "next": self._plan_index,
                "started_ago_ms": (
                    0
                    if self._plan_started_ns == 0
                    else int((time.monotonic_ns() - self._plan_started_ns) // 1_000_000)
                ),
            },
            "skill_outcome": None if running is None else running.outcome.value,
            "recent_skill_runs": [run.model_dump(mode="json") for run in self._recent_skill_runs],
            "chosen_goal_id": None if decision is None else decision.chosen_goal_id,
            "reasoning_summary": None if decision is None else decision.reasoning_summary,
            "perception_questions": [] if decision is None else list(decision.ask_perception),
            "request_replan": False if decision is None else decision.request_replan,
            "operator_response": None if decision is None else decision.say,
            "pending_game_chat": None if decision is None else decision.game_chat,
            "cognition": cognition_status,
            "policy": policy_status,
            "policy_warmup_error": self._policy_warmup_error,
            "perception": perception_status,
            "lease_heartbeat_error": self._lease_fault,
            "updated_monotonic_ns": time.monotonic_ns(),
        }

    def _bootstrap_if_idle(self) -> None:
        running = self.executor.run
        if running is not None and running.outcome == SkillOutcome.RUNNING:
            return
        goals = role_standing_goals(self.role)
        scheduler = CurriculumScheduler(self.role)
        chosen = scheduler.choose(
            [CurriculumCandidate(goal=goal, progression_novelty=0.4) for goal in goals]
        )
        skill_id = None
        if chosen is not None:
            skill_id = _standing_goal_skill(chosen.goal, self.blackboard)
        if skill_id is None or skill_id not in self.skills.specs:
            skill_id = "explore_forward"
        self.executor.start(
            self.skills.get(skill_id),
            run_id=uuid.uuid4().hex,
            context_key=f"role:{self.role.role_id}",
        )

    def _failsafe(self, reason: str) -> None:
        try:
            send_command("fault", reason=reason)
        except Exception:
            pass
