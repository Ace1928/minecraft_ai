from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Iterable
from pathlib import Path

from minecraft_ai.cognition import (
    CognitionDecision,
)
from minecraft_ai.action_levels import ActionLevel
from minecraft_ai.episodes import RuntimeEvent, RuntimeEventKind
from minecraft_ai.execution import ExecutionTick, initiation_satisfied
from minecraft_ai.grounded_perception import (
    CROSSHAIR_BLOCK_FAST_SOURCE,
    crosshair_block_crop_dimensions,
    crosshair_block_region,
    crosshair_block_visually_equivalent,
)
from minecraft_ai.memory import MemoryKind, MemoryRecord
from minecraft_ai.mining_control import (
    is_hand_safe_soft_block,
    normalize_block_kind,
)
from minecraft_ai.motor import MotorIntent
from minecraft_ai.outcome_verifier import (
    OutcomeKind,
    OutcomeSignal,
    OutcomeStatus,
    OutcomeVerification,
)
from minecraft_ai.perception import (
    EvidenceRegion,
    PerceptionBlackboard,
    PerceptionFact,
    Track,
)
from minecraft_ai.perception_service import (
    BEDROCK_HOTBAR_LOG_COUNT_SOURCE,
    perceptual_hash_distance,
)
from minecraft_ai.planning import Goal
from minecraft_ai.platforms.bedrock_x11 import CapturedFrame
from minecraft_ai.skills import (
    SkillLibrary,
    SkillFailureCode,
    SkillOutcome,
    SkillRun,
    SkillSpec,
    SkillStats,
)
from minecraft_ai.social import (
    OperatorMessage,
    OperatorMessageKind,
    OperatorMessageStatus,
)
from minecraft_ai.trajectory import ActionOrigin, ActionProvenance, motor_condition_id

from minecraft_ai.runtime_support.types import (
    _GatherAcquisitionContinuation,
    _HeadroomRecovery,
    _HeadroomTarget,
)

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
        "dismiss_away_overlay",
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

_GUI_PLAN_SKILLS = frozenset(
    {
        "activate_visible_gui_control",
        "close_open_inventory",
        "open_inventory",
    }
)


def _normalized_plan_text(value: str) -> str:
    return " ".join(value.casefold().replace("_", " ").replace("-", " ").split())


def _plan_step_matches_skill(skill_id: str, step: str) -> bool:
    """True only when this skill is the current plan node, not any world option."""

    if skill_id in _GUI_PLAN_SKILLS:
        return _plan_step_requests_inventory_transition(skill_id, step)
    return _normalized_plan_text(skill_id) == _normalized_plan_text(step)


def _plan_step_requests_inventory_transition(skill_id: str, step: str) -> bool:
    """Require an explicit GUI plan node before its transition can consume progress."""

    if skill_id == "close_open_inventory" and step.strip().casefold() == skill_id:
        return True
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
    origin = ActionOrigin.POLICY if execution is None else execution.action_origin
    source_frame_id = None
    source_captured_ns = None
    source_frame_value = causal_fields.get("source_frame_id")
    source_time_value = causal_fields.get("source_captured_ns")
    # Source identity belongs to the consumed prediction, never the latest
    # blackboard or a reset/manual/release action that shares a status snapshot.
    if (
        origin == ActionOrigin.POLICY
        and policy_action_kind in {"prediction", "prediction_hold"}
        and policy_request_id is not None
        and type(source_frame_value) is int
        and source_frame_value >= 0
        and type(source_time_value) is int
        and source_time_value >= 0
    ):
        source_frame_id = source_frame_value
        source_captured_ns = source_time_value
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
        source_frame_id=source_frame_id,
        source_captured_ns=source_captured_ns,
        action_level=action_level,
        origin=origin,
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
        away = blackboard.fact("scene.away", min_confidence=0.9)
        if away is not None and bool(away.value):
            skill_id = "dismiss_away_overlay"
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

