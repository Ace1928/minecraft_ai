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
from .execution import ExecutionTick, SkillExecutor, initiation_satisfied
from .grounded_perception import resolve_grounded_output_keys
from .memory import MemoryKind, MemoryRecord, MemoryStore
from .mining_control import (
    is_hand_safe_soft_block,
    normalize_block_kind,
    track_contains_crosshair,
)
from .models import local_model_inference_available
from .motor import MotorIntent
from .outcome_verifier import OutcomeKind, OutcomeSignal, OutcomeStatus, OutcomeVerification
from .perception import (
    ActivePerceptionQuery,
    EvidenceRegion,
    PerceptionBlackboard,
    PerceptionFact,
    Track,
)
from .perception_service import (
    BEDROCK_HOTBAR_LOG_COUNT_SOURCE,
    RealtimePerceptionService,
    frame_dhash,
    perceptual_hash_distance,
)
from .planning import Goal
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
    {"open_inventory", "close_open_inventory", "collect_recent_drop"}
)
_RECORDED_RUN_ID_LIMIT = 4_096
_COGNITION_RETRY_BASE_NS = 2_000_000_000
_COGNITION_RETRY_MAX_NS = 30_000_000_000
_OPERATOR_FOLLOWUP_DELAY_NS = 250_000_000
_CRAFT_SEMANTIC_LATENCY_MARGIN = 1.25
_CRAFT_SEMANTIC_MAX_REQUIRED_BUDGET_MS = 60_000
_PLANKS_NO_LOGS_REASON = "crafting-no-logs-observed-in-inventory"
_PLANKS_RETRY_CLEAR_MEMORY = "working:planks-retry-positive-log-evidence"
_HEADROOM_QUERY_OUTPUT_KEYS = (
    "scene.mode",
    "scene.playable",
    "danger.immediate",
    "target.visible",
    "target.kind",
    "target.mineable",
    "target.dx",
    "target.dy",
)
_HEADROOM_CENTER_TOLERANCE = 0.15
# Positive Bedrock pitch moves the open-sky crosshair toward the near terrain
# lip that commonly causes this verified wedge. This is one bounded look-down
# adjustment, not a scan; the next capture must confirm that the view changed.
_HEADROOM_REORIENT_MOUSE_DY = 96
_HEADROOM_MIN_TIMEOUT_S = 60.0
_HEADROOM_TIMEOUT_MULTIPLIER = 5.0
_HEADROOM_TIMEOUT_MARGIN_S = 5.0
_HEADROOM_TRANSACTION_MAX_S = 180.0
_HEADROOM_SETTLE_TIMEOUT_NS = 2_000_000_000


@dataclass(frozen=True)
class _CraftSemanticProbe:
    run_id: str
    phase: PlankCraftPhase
    terminal_count_before: int


@dataclass
class _HeadroomRecovery:
    """One fail-closed clear-and-retry transaction after a verified traversal stall."""

    context_key: str
    traversal_parameters: dict[str, str | int | float | bool]
    deadline_ns: int
    phase: str = "reorient"
    reoriented_frame_id: int | None = None
    pre_reorient_dhash: str | None = None
    settle_deadline_ns: int | None = None
    query_id: str | None = None
    query_started_ns: int = 0
    query_frame_dhash: str | None = None
    mining_run_id: str | None = None
    retry_run_id: str | None = None


@dataclass(frozen=True)
class _HeadroomTarget:
    kind: str
    track_id: str


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


def _expected_keepalive_expiry(run: SkillRun) -> bool:
    return (
        run.outcome == SkillOutcome.TIMED_OUT
        and run.context_key == _EXPLORE_KEEPALIVE_CONTEXT
        and run.skill_id in _BOUNDED_KEEPALIVE_SKILL_IDS
    )


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


def _verified_obstacle_stall(result: ExecutionTick) -> bool:
    """Accept only the traversal verifier's exact obstacle-stall terminal signal."""

    verification = result.outcome_verification
    return bool(
        result.run.skill_id == "traverse_visible_obstacle"
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
) -> _HeadroomTarget | None:
    """Resolve one exact, current, soft block from the recovery's own VLM query."""

    query_id = recovery.query_id
    requested_hash = recovery.query_frame_dhash
    if query_id is None or requested_hash is None:
        return None

    required_keys = (
        "scene.mode",
        "scene.playable",
        "danger.immediate",
        "target.visible",
        "target.kind",
        "target.mineable",
        "target.dx",
        "target.dy",
        "scene.observation_dhash",
    )
    facts = {
        key: blackboard.fact(key, min_confidence=0.70, now_ns=now_ns)
        for key in required_keys
    }
    if any(fact is None for fact in facts.values()):
        return None
    observed = tuple(fact for fact in facts.values() if fact is not None)
    source = observed[0].source
    observed_ns = observed[0].observed_ns
    if (
        not source.startswith("vlm:")
        or not source.endswith(f":{query_id}")
        or any(fact.source != source for fact in observed)
        or any(fact.observed_ns != observed_ns for fact in observed)
        or observed_ns <= recovery.query_started_ns
    ):
        return None

    mode = facts["scene.mode"]
    playable = facts["scene.playable"]
    danger = facts["danger.immediate"]
    visible = facts["target.visible"]
    kind = facts["target.kind"]
    mineable = facts["target.mineable"]
    dx = facts["target.dx"]
    dy = facts["target.dy"]
    scene_hash = facts["scene.observation_dhash"]
    assert mode is not None
    assert playable is not None
    assert danger is not None
    assert visible is not None
    assert kind is not None
    assert mineable is not None
    assert dx is not None
    assert dy is not None
    assert scene_hash is not None
    if (
        mode.value != "world"
        or playable.value is not True
        or danger.value is not False
        or visible.value is not True
        or mineable.value is not True
        or not isinstance(kind.value, str)
        or not is_hand_safe_soft_block(kind.value)
        or not isinstance(dx.value, (int, float))
        or isinstance(dx.value, bool)
        or not isinstance(dy.value, (int, float))
        or isinstance(dy.value, bool)
        or abs(float(dx.value)) > _HEADROOM_CENTER_TOLERANCE
        or abs(float(dy.value)) > _HEADROOM_CENTER_TOLERANCE
        or scene_hash.value != requested_hash
    ):
        return None

    current_hash = blackboard.fact("frame.dhash", min_confidence=1.0, now_ns=now_ns)
    if current_hash is None or not isinstance(current_hash.value, str):
        return None
    try:
        if perceptual_hash_distance(requested_hash, current_hash.value) > 6:
            return None
    except ValueError:
        return None

    latest = blackboard.latest()
    if latest is None:
        return None
    normalized_kind = normalize_block_kind(kind.value)
    candidates = tuple(
        track
        for track in latest.tracks
        if track.track_id.startswith(f"vlm:{query_id}:")
        and track.confidence >= 0.70
        and track.last_seen_ns == observed_ns
        and normalize_block_kind(track.label) == normalized_kind
        and track_contains_crosshair(track)
    )
    if len(candidates) != 1:
        return None
    return _HeadroomTarget(kind=normalized_kind, track_id=candidates[0].track_id)


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
        self._start_cognition_if_due()
        self._request_semantics_if_due(frame.frame_id)
        self._route_observed_scene_recovery()
        self._advance_headroom_recovery()

        active = self.executor.run
        if active is None or active.outcome != SkillOutcome.RUNNING:
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
        if (
            result.run.skill_id == "collect_recent_drop"
            and result.run.outcome != SkillOutcome.RUNNING
        ):
            self.blackboard.merge_semantics(
                instance_id=self.perception.instance_id,
                facts=(
                    PerceptionFact(
                        key="collection.recent_log_break",
                        value=False,
                        confidence=0.995,
                        observed_ns=time.monotonic_ns(),
                        source=f"runtime:{result.run.run_id}:collection-terminal",
                        expires_after_ms=250,
                    ),
                ),
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
        try:
            if result.action is not None:
                self._send_motor(result.action, execution=result)
        finally:
            if result.run.outcome != SkillOutcome.RUNNING:
                if result.outcome_verification is None:
                    self._record_terminal_run(
                        result.run,
                        advance_plan=advance_plan,
                    )
                else:
                    self._record_terminal_run(
                        result.run,
                        outcome_verification=result.outcome_verification,
                        advance_plan=advance_plan,
                    )
        self.metrics.last_motor_ms = (time.perf_counter() - motor_started) * 1000.0
        if result.run.outcome != SkillOutcome.RUNNING:
            if collect_recent_drop:
                self.blackboard.merge_semantics(
                    instance_id=self.perception.instance_id,
                    facts=(
                        PerceptionFact(
                            key="collection.recent_log_break",
                            value=True,
                            confidence=0.995,
                            observed_ns=time.monotonic_ns(),
                            source=f"verified:{result.run.run_id}:block-broken",
                            expires_after_ms=6_000,
                        ),
                    ),
                )
                self.executor.start(
                    self.skills.get("collect_recent_drop"),
                    run_id=uuid.uuid4().hex,
                    context_key=result.run.context_key,
                    collection_hotbar_log_baseline=(
                        self.executor.mining_hotbar_log_baseline
                    ),
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
                self._start_recovery_skill(recovery, result.run)

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
        self._headroom_recovery = None
        self._traversal_escalation_pending = True
        self._cognition_requested = True
        return expired, True

    def _route_headroom_terminal(self, result: ExecutionTick) -> bool:
        """Advance or end one clear-and-retry transaction without recursive recovery."""

        recovery = getattr(self, "_headroom_recovery", None)
        if recovery is not None and self._is_headroom_child_result(result):
            if recovery.phase == "mining":
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
                    )
                else:
                    self._headroom_recovery = None
                    self._note_terminal_for_cognition(
                        result.run,
                        recovery_started=False,
                    )
                return True

            retry_succeeded = _verified_headroom_retry(result, recovery)
            self._headroom_recovery = None
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
        self._note_terminal_for_cognition(result.run, recovery_started=True)
        if getattr(self, "_headroom_recovery", None) is None:
            now_ns = time.monotonic_ns()
            active_vlm = self.perception.active_vlm
            assert active_vlm is not None
            self._headroom_recovery = _HeadroomRecovery(
                context_key=result.run.context_key,
                traversal_parameters=dict(result.run.parameters),
                deadline_ns=_headroom_deadline_ns(active_vlm, now_ns=now_ns),
            )
        return True

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
            self._headroom_recovery = None
            self._traversal_escalation_pending = True
            self._cognition_requested = True
            return
        if recovery.phase in {"mining", "retry"}:
            return
        if not self._headroom_scene_is_safe():
            self._headroom_recovery = None
            return

        running = self.executor.run
        if running is not None and running.outcome == SkillOutcome.RUNNING:
            self._headroom_recovery = None
            return

        if recovery.phase == "reorient":
            latest = self.blackboard.raw_latest()
            captured = self.perception.last_capture
            if (
                latest is None
                or captured is None
                or captured.captured_ns != latest.captured_ns
            ):
                self._headroom_recovery = None
                return
            recovery.pre_reorient_dhash = frame_dhash(captured)
            self._send_motor(
                MotorAction(
                    sequence=self._sequence,
                    mouse_dy=_HEADROOM_REORIENT_MOUSE_DY,
                    camera_semantics="world",
                )
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
                self._headroom_recovery = None
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
                self._headroom_recovery = None
                return
            if not visibly_reoriented:
                return
            recovery.phase = "request"

        if recovery.phase == "request":
            if self.perception.active_vlm is None:
                self._headroom_recovery = None
                return
            if not self.perception.semantic_available():
                return
            captured = self.perception.last_capture
            latest = self.blackboard.raw_latest()
            if captured is None or latest is None:
                self._headroom_recovery = None
                return
            query_id = uuid.uuid4().hex
            query_started_ns = time.monotonic_ns()
            query_frame_dhash = frame_dhash(captured)
            query = ActivePerceptionQuery(
                query_id=query_id,
                question=(
                    "Inspect only the single block exactly under the world crosshair that may "
                    "be preventing forward or upward movement. Ground that exact block with one "
                    "bounding box and report its canonical Minecraft block identifier, whether "
                    "it is visibly mineable, world/playable scene state, immediate danger, and "
                    "target dx/dy. Do not select an adjacent block and do not infer hidden state."
                ),
                skill_id="mine_visible_block",
                frame_id=latest.frame_id,
                deadline_ms=10_000,
                output_keys=_HEADROOM_QUERY_OUTPUT_KEYS,
            )
            if not self.perception.request_semantics(query, frame=captured):
                self._headroom_recovery = None
                return
            recovery.phase = "grounding"
            recovery.query_id = query_id
            recovery.query_started_ns = query_started_ns
            recovery.query_frame_dhash = query_frame_dhash
            self.metrics.semantic_requests += 1
            return

        target = _headroom_clear_target(
            self.blackboard,
            recovery,
            now_ns=time.monotonic_ns(),
        )
        if target is not None:
            run_id = uuid.uuid4().hex
            recovery.phase = "mining"
            recovery.mining_run_id = run_id
            self.executor.start(
                self.skills.get("mine_visible_block"),
                run_id=run_id,
                context_key=recovery.context_key,
                parameters={
                    "target": target.kind,
                    "target_track_id": target.track_id,
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
            self._headroom_recovery = None

    def _headroom_scene_is_safe(self) -> bool:
        now_ns = time.monotonic_ns()
        unsafe_truths = ("danger.immediate", "scene.death", "scene.ui_overlay")
        if any(
            (fact := self.blackboard.fact(key, min_confidence=0.65, now_ns=now_ns))
            is not None
            and fact.value is True
            for key in unsafe_truths
        ):
            return False
        playable = self.blackboard.fact("scene.playable", min_confidence=0.65, now_ns=now_ns)
        return playable is None or playable.value is not False

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
        # Death and modal UI recovery outrank the optional terrain-clear
        # transaction at every phase. The normal cancellation path below owns
        # releasing any active mining/traversal inputs.
        self._headroom_recovery = None
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
        self.executor.start(
            recovery,
            run_id=uuid.uuid4().hex,
            context_key=context_key,
        )
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
        headroom = getattr(self, "_headroom_recovery", None)
        if headroom is not None:
            if not self._queued_operator_message_waiting():
                return
            # Fresh operator authority cancels this optional autonomous
            # transaction. If a child is running, release it before the normal
            # operator fast path or model decision takes ownership.
            running = self.executor.run
            child_run_ids = {headroom.mining_run_id, headroom.retry_run_id}
            self._headroom_recovery = None
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
        executor = getattr(self, "executor", None)
        active = None if executor is None else executor.run
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
            decision.skill_id == "craft_wood_planks"
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
        self._last_decision = decision
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
        for question in decision.ask_perception:
            latest = self.blackboard.raw_latest()
            if latest is None:
                break
            query = ActivePerceptionQuery(
                query_id=uuid.uuid4().hex,
                question=question,
                skill_id=decision.skill_id,
                frame_id=latest.frame_id,
                output_keys=resolve_grounded_output_keys((), question),
            )
            self.perception.request_semantics(query)
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
        if decision.request_replan:
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

        Steps are short free-form plans from the high-level VLM, so we cannot
        reliably attribute a success to a specific step string. Instead we treat
        a completed skill as consuming the current step position and advance the
        plan index, preserving ordering while remaining permissive. The goal of
        the last accepted decision is used only as a sanity gate so an off-plan
        success (e.g. an operator-message task that overrides the standing plan)
        does not silently eat plan progress.
        """
        if not self._plan_steps:
            return
        if self._plan_index >= len(self._plan_steps):
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

    def _start_recovery_skill(self, recovery: SkillSpec, parent: SkillRun) -> SkillRun:
        run = self.executor.start(
            recovery,
            run_id=uuid.uuid4().hex,
            context_key=parent.context_key,
            parameters=_compatible_recovery_parameters(parent, recovery),
        )
        if (
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
