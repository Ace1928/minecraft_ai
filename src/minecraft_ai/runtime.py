from __future__ import annotations

import concurrent.futures
import copy
import logging
import math
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, cast

from .cognition import (
    BootstrapCognitionPolicy,
    CognitionContext,
    CognitionDecision,
    HighLevelController,
    cognition_decision_sha256,
    planks_retry_requires_wood,
)
from .action_levels import ActionLevel
from .curriculum import CurriculumCandidate, CurriculumScheduler, role_standing_goals
from .daemon_executor import SingleWorkerDaemonExecutor
from .episodes import RuntimeEvent
from .emergency import emergency_stop_latched
from .execution import ExecutionTick, SkillExecutor, initiation_satisfied
from .grounded_perception import (
    crosshair_block_pixel_sha256,
    crosshair_block_rgb_grid,
    crosshair_block_rgb_grid_distance,
    resolve_grounded_output_keys,
)
from .memory import MemoryKind, MemoryRecord, MemoryStore
from .models import BoundCognitionModel, local_model_inference_available
from .model_requests import ModelRequestLifecycle, RequestBinding
from .outcome_verifier import OutcomeSignal, OutcomeVerification
from .perception import (
    ActivePerceptionQuery,
    EvidenceRegion,
    PerceptionBlackboard,
    CognitionBlackboardSnapshot,
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
from .trajectory import ActionOrigin, TrajectoryRecorder
from .storage import OperatorContextSnapshot, StateDatabase
from .supervisor import operator_intent_lock, operator_pause_latched, send_command


from minecraft_ai.runtime_support.helpers import (
    _ATOMIC_SKILL_IDS,
    _BOUNDED_KEEPALIVE_SKILL_IDS,
    _COGNITION_PERCEPTION_ACTION_GRACE_NS,
    _COGNITION_PERCEPTION_GROUNDING_TIMEOUT_NS,
    _COGNITION_PERCEPTION_HANDOFF_TIMEOUT_NS,
    _COGNITION_PERCEPTION_SETTLE_TIMEOUT_NS,
    _COGNITION_RETRY_BASE_NS,
    _COGNITION_RETRY_MAX_NS,
    _CRAFT_SEMANTIC_LATENCY_MARGIN,
    _CRAFT_SEMANTIC_MAX_REQUIRED_BUDGET_MS,
    _EXPLORE_KEEPALIVE_CONTEXT,
    _GATHER_ACQUISITIONS_REQUIRED,
    _HEADROOM_MIN_TIMEOUT_S,
    _HEADROOM_REORIENT_MAX_ABS_DY,
    _HEADROOM_REORIENT_TARGET_PITCH_UNITS,
    _HEADROOM_REORIENT_TARGET_TOLERANCE_UNITS,
    _HEADROOM_SETTLE_TIMEOUT_NS,
    _HEADROOM_STABLE_SUCCESSOR_FRAMES,
    _HEADROOM_TIMEOUT_MARGIN_S,
    _HEADROOM_TIMEOUT_MULTIPLIER,
    _HEADROOM_TRANSACTION_MAX_S,
    _OPERATOR_FOLLOWUP_DELAY_NS,
    _PLANKS_NO_LOGS_REASON,
    _PLANKS_RETRY_CLEAR_MEMORY,
    _RECORDED_RUN_ID_LIMIT,
    _WOOD_INVENTORY_AUDIT_SKILLS,
    _accepted_action_provenance,
    _active_operator_messages,
    _authorized_game_chat,
    _compatible_recovery_parameters,
    _condition_target_track_id,
    _exact_frozen_log_count,
    _expected_keepalive_expiry,
    _first_feasible_recovery,
    _headroom_clear_target,
    _headroom_deadline_ns,
    _headroom_reorient_mouse_dy,
    _headroom_retry_advances_plan,
    _int_fact,
    _observed_scene_recovery,
    _operator_target_facts,
    _plan_step_requests_inventory_transition,
    _reported_action_level,
    _restore_policy_world_camera,
    _scene_claim_is_fresh,
    _selected_operator_message_id,
    _semantic_deadline_ms,
    _semantic_refresh_allowed,
    _skill_stats_totals,
    _sqlite_writer_contention,
    _standing_goal_skill,
    _terminal_run_event,
    _terminal_run_memory,
    _trajectory_outcome_annotations,
    _verified_block_break,
    _verified_gather_acquisition,
    _verified_headroom_retry,
    _verified_log_break,
    _verified_oak_log_break,
    _verified_obstacle_stall,
    _verified_outcome_event,
    _verified_traversal_progress,
)
from minecraft_ai.runtime_support.types import (
    RuntimeMetrics,
    SkillDecisionOrigin,
    SkillStartSource,
    _CognitionPerceptionProbe,
    _CraftSemanticProbe,
    _GatherAcquisitionContinuation,
    _HeadroomRecovery,
    _HeadroomTarget,
)

__all__ = [
    'AgentRuntime',
    'RuntimeMetrics',
    'SkillDecisionOrigin',
    'SkillStartSource',
    '_ATOMIC_SKILL_IDS',
    '_BOUNDED_KEEPALIVE_SKILL_IDS',
    '_COGNITION_PERCEPTION_ACTION_GRACE_NS',
    '_COGNITION_PERCEPTION_GROUNDING_TIMEOUT_NS',
    '_COGNITION_PERCEPTION_HANDOFF_TIMEOUT_NS',
    '_COGNITION_PERCEPTION_SETTLE_TIMEOUT_NS',
    '_COGNITION_RETRY_BASE_NS',
    '_COGNITION_RETRY_MAX_NS',
    '_CRAFT_SEMANTIC_LATENCY_MARGIN',
    '_CRAFT_SEMANTIC_MAX_REQUIRED_BUDGET_MS',
    '_CognitionPerceptionProbe',
    '_CraftSemanticProbe',
    '_EXPLORE_KEEPALIVE_CONTEXT',
    '_GATHER_ACQUISITIONS_REQUIRED',
    '_GatherAcquisitionContinuation',
    '_HEADROOM_MIN_TIMEOUT_S',
    '_HEADROOM_REORIENT_MAX_ABS_DY',
    '_HEADROOM_REORIENT_TARGET_PITCH_UNITS',
    '_HEADROOM_REORIENT_TARGET_TOLERANCE_UNITS',
    '_HEADROOM_SETTLE_TIMEOUT_NS',
    '_HEADROOM_STABLE_SUCCESSOR_FRAMES',
    '_HEADROOM_TIMEOUT_MARGIN_S',
    '_HEADROOM_TIMEOUT_MULTIPLIER',
    '_HEADROOM_TRANSACTION_MAX_S',
    '_HeadroomRecovery',
    '_HeadroomTarget',
    '_OPERATOR_FOLLOWUP_DELAY_NS',
    '_PLANKS_NO_LOGS_REASON',
    '_PLANKS_RETRY_CLEAR_MEMORY',
    '_RECORDED_RUN_ID_LIMIT',
    '_WOOD_INVENTORY_AUDIT_SKILLS',
    '_accepted_action_provenance',
    '_active_operator_messages',
    '_authorized_game_chat',
    '_compatible_recovery_parameters',
    '_condition_target_track_id',
    '_exact_frozen_log_count',
    '_expected_keepalive_expiry',
    '_first_feasible_recovery',
    '_headroom_clear_target',
    '_headroom_deadline_ns',
    '_headroom_reorient_mouse_dy',
    '_headroom_retry_advances_plan',
    '_int_fact',
    '_observed_scene_recovery',
    '_operator_target_facts',
    '_plan_step_requests_inventory_transition',
    '_reported_action_level',
    '_restore_policy_world_camera',
    '_scene_claim_is_fresh',
    '_selected_operator_message_id',
    '_semantic_deadline_ms',
    '_semantic_refresh_allowed',
    '_skill_stats_totals',
    '_sqlite_writer_contention',
    '_standing_goal_skill',
    '_terminal_run_event',
    '_terminal_run_memory',
    '_trajectory_outcome_annotations',
    '_verified_block_break',
    '_verified_gather_acquisition',
    '_verified_headroom_retry',
    '_verified_log_break',
    '_verified_oak_log_break',
    '_verified_obstacle_stall',
    '_verified_outcome_event',
    '_verified_traversal_progress',
]


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
    cognition_request_timeout_ms: int = 60_000
    semantic_hz: float = 2.0
    lease_renew_ms: int = 500
    stale_frame_consecutive_limit: int = 3
    metrics: RuntimeMetrics = field(default_factory=RuntimeMetrics)
    telemetry: TelemetryPublisher = field(default_factory=TelemetryPublisher)
    trajectory: TrajectoryRecorder | None = None
    trajectory_disabled_reason: str | None = None
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _run_started: bool = field(default=False, init=False)
    _pre_run_closed_resources: set[str] = field(default_factory=set, init=False, repr=False)
    _sequence: int = field(default=0, init=False)
    _last_renew_ns: int = field(default=0, init=False)
    _last_cognition_ns: int = field(default=0, init=False)
    _last_player_chat_replied_ns: int | None = field(default=None, init=False)
    _last_player_chat_signature: str | None = field(default=None, init=False)
    _last_semantic_ns: int = field(default=0, init=False)
    _lease_thread: threading.Thread | None = field(default=None, init=False)
    _lease_fault: str | None = field(default=None, init=False)
    _input_release_pending_ns: int | None = field(default=None, init=False)
    _pending_decision: concurrent.futures.Future[CognitionDecision] | None = field(
        default=None,
        init=False,
    )
    _bound_cognition_requests: dict[
        concurrent.futures.Future[CognitionDecision], tuple[ModelRequestLifecycle, object],
    ] = field(default_factory=dict, init=False, repr=False)
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
        if (type(self.cognition_request_timeout_ms) is not int
                or not 1 <= self.cognition_request_timeout_ms <= 300_000):
            raise ValueError("bound cognition timeout must be between 1 and 300000 ms")
        if self.stale_frame_consecutive_limit < 1:
            raise ValueError("stale_frame_consecutive_limit must be positive")
        self._pool = SingleWorkerDaemonExecutor(
            thread_name="minecraft-ai-cognition",
        )

    def stop(self) -> None:
        self._stop.set()

    def close_before_run(self, *, timeout_s: float = 2.0) -> bool:
        """Drain an unstarted runtime without claiming supervisor authority.

        The factory must not start inference/gameplay or replace owned resources.
        Its private lifecycle must be drained first. Explicit waits share one
        deadline; perception/executor close callbacks must cooperate because this
        method cannot hard-preempt arbitrary callbacks. False retains ownership
        for retry, including the caller-owned database, which is never closed here.
        """
        self._stop.set()
        if self._run_started:
            return False
        if not math.isfinite(timeout_s) or timeout_s < 0:
            raise ValueError("pre-run cleanup timeout must be finite and nonnegative")
        deadline = time.monotonic() + timeout_s
        closed = self._pre_run_closed_resources
        try:
            if "pool" not in closed:
                self._pool.shutdown(wait=False, cancel_futures=True)
                if not self._pool.wait_closed(max(0.0, deadline - time.monotonic())):
                    return False
                closed.add("pool")
            for name in ("perception", "executor", "trajectory"):
                if name in closed:
                    continue
                resource = getattr(self, name)
                if resource is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    if name == "trajectory":
                        resource.close(timeout_s=remaining)
                    else:
                        resource.close()
                closed.add(name)
        except Exception:
            return False
        return True

    def run_forever(self) -> None:
        self._run_started = True
        period = 1.0 / self.motor_hz
        self._lease_thread = None
        try:
            if self._stop.is_set():
                return
            # Confirm ownership before starting workers. A rejected renewal
            # goes straight through cleanup; an expired lease cannot be revived
            # by background retries. Earlier process assembly is a separate
            # startup phase and is not covered by this runtime heartbeat.
            try:
                send_command("renew", lease_id=self.lease_id, ttl_ms=5_000)
                self._last_renew_ns = time.monotonic_ns()
            except Exception as exc:
                self._lease_fault = f"{type(exc).__name__}: {exc}"
                if self._stop.is_set() or operator_pause_latched():
                    return
                raise
            if self._stop.is_set():
                return
            lease_thread = threading.Thread(
                target=self._lease_heartbeat,
                name="minecraft-ai-lease-heartbeat",
                daemon=True,
            )
            lease_thread.start()
            self._lease_thread = lease_thread
            if self._stop.is_set():
                return
            if self.perception.active_vlm is not None:
                self.perception.active_vlm.start()
            if self._stop.is_set():
                return
            self.telemetry.publish(self._telemetry_payload(state="warming"), force=True)
            if self._stop.is_set():
                return
            # Strategic inference and policy checkpoint loading are independent.
            # Start the first typed decision from a real captured frame before
            # warming the learned policies so CPU model startup latency is not
            # paid serially while the avatar stands idle.
            if self.perception.last_capture is None:
                self.perception.capture_once()
            if self._stop.is_set():
                return
            # Operator grounding is deterministic and may make a pending
            # directive executable before the first strategic snapshot. Merge
            # it before launching slow cognition, including after a process
            # restart where the message was already marked delivered.
            self._merge_operator_target()
            if self._stop.is_set():
                return
            self._start_cognition_if_due()
            if self._stop.is_set():
                return
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
            # Retire requests before fallible device/telemetry cleanup. Running
            # workers retain their accounting and discard only after completion.
            # Partial/legacy runtime assembly may never initialize this optional
            # registry. An existing invalid registry still fails visibly.
            for future in tuple(getattr(self, "_bound_cognition_requests", {})):
                self._reject_bound_cognition(future, "runtime_shutdown")
            self._pool.shutdown(wait=False, cancel_futures=True)
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
            self._release_and_reconcile_inputs()
            self.telemetry.publish(self._telemetry_payload(state="capture-stalled"))
            if self.metrics.consecutive_stale_frames >= self.stale_frame_consecutive_limit:
                raise RuntimeError(
                    "capture stream is stale for "
                    f"{self.metrics.consecutive_stale_frames} consecutive frames"
                )
            return
        self.metrics.consecutive_stale_frames = 0
        if self._input_release_pending_ns is not None:
            self._release_and_reconcile_inputs()
            self.telemetry.publish(self._telemetry_payload(state="input-release-pending"))
            # Even an acknowledged retry happened after this tick's capture.
            # The next tick must observe the released body before selecting
            # another action; never replay the pre-release image or command.
            return
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
                self._start_skill(
                    rescue,
                    source=SkillStartSource.KEEPALIVE,
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
            capture=getattr(self.perception, "last_capture", None),
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
                self._start_skill(
                    self.skills.get("gather_nearby_wood"),
                    source=SkillStartSource.CONTINUATION,
                    parent_run_id=result.run.run_id,
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
        return self._start_skill(
            self.skills.get("collect_recent_drop"),
            source=SkillStartSource.CONTINUATION,
            parent_run_id=broken_run.run_id,
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
                    self._start_skill(
                        self.skills.get("traverse_visible_obstacle"),
                        source=SkillStartSource.RECOVERY,
                        parent_run_id=result.run.run_id,
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
                origin_run_id=result.run.run_id,
            )
        return True

    def _quiesce_headroom_inputs(self) -> bool:
        """Release every held input while preserving this runtime's live lease."""
        return self._release_and_reconcile_inputs()

    def _release_and_reconcile_inputs(self) -> bool:
        """Require a physical acknowledgement before clearing controller holds."""
        if self._input_release_pending_ns is None:
            self._input_release_pending_ns = time.monotonic_ns()
        try:
            result = send_command("release-inputs", lease_id=self.lease_id)
        except Exception:
            return False
        if (not isinstance(result, dict) or result.get("released") is not True
                or result.get("lease_active") is not True):
            return False
        # The protocol has no exact physical release timestamp. The first
        # request is a conservative action-duration cutoff, not sensor time.
        # Keep pending on notification failure; no positive action may escape.
        self.executor.notify_inputs_released(now_ns=self._input_release_pending_ns)
        self._input_release_pending_ns = None
        return True

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
            self._start_skill(
                self.skills.get("mine_visible_block"),
                source=SkillStartSource.RECOVERY,
                parent_run_id=recovery.origin_run_id,
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
        parent_run_id = None
        if running is not None and running.outcome == SkillOutcome.RUNNING:
            context_key = running.context_key
            parent_run_id = running.run_id
            cancelled = self.executor.cancel()
            try:
                if cancelled.action is not None:
                    self._send_motor(cancelled.action, execution=cancelled)
            finally:
                self._record_terminal_run(cancelled.run)
            self._execution_revision += 1
        recovery_run = self._start_skill(
            recovery,
            source=SkillStartSource.RECOVERY,
            parent_run_id=parent_run_id,
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
        if self._input_release_pending_ns is not None:
            raise RuntimeError("input release acknowledgement is pending")
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
            safety_run = self._start_skill(
                safety,
                source=SkillStartSource.RECOVERY,
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
            self._reject_bound_cognition(future, "perception_probe_revoked")
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
            observed = self.blackboard.latest()
            if observed is not None and any(track == target for track in observed.tracks):
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
        bound_inputs = None
        if self._uses_bound_cognition():
            try:
                bound_inputs = self._capture_bound_cognition_inputs()
                context = bound_inputs[0]
            except (ValueError, RuntimeError, sqlite3.Error):
                self._schedule_cognition_retry(now_ns=now)
                return
        else:
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
        elif bound_inputs is not None:
            _, snapshot, operator_revision = bound_inputs
            self._pending_decision = self._submit_bound_cognition(
                context, snapshot, operator_revision=operator_revision,
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

    def _uses_bound_cognition(self) -> bool:
        controller = self.high_level
        model = getattr(controller, "model", None)
        return model is not None and all(
            callable(getattr(model, name, None))
            for name in (
                "complete_bound_constrained", "admit_bound_decision", "discard_bound_request",
            )
        )

    def _capture_bound_cognition_inputs(
        self,
    ) -> tuple[CognitionContext, CognitionBlackboardSnapshot, int]:
        """Freeze semantic observations and intent before queueing model work."""
        if self.state_db is None or self.high_level is None:
            raise RuntimeError("bound cognition requires durable operator authority")
        model = self.high_level.model
        if not all(callable(getattr(model, name, None)) for name in (
            "complete_bound_constrained", "admit_bound_decision", "discard_bound_request",
        )):
            raise RuntimeError("bound cognition adapter lacks its complete lifecycle")
        with operator_intent_lock(timeout_s=0.05):
            if self._stop.is_set() or operator_pause_latched() or emergency_stop_latched():
                raise RuntimeError("operator suspension prevents model submission")
            operator = self.state_db.load_operator_context(
                statuses={OperatorMessageStatus.QUEUED, OperatorMessageStatus.DELIVERED},
                limit=20,
            )
            with self.state_db.admit_operator_revision(operator.revision) as current:
                if not current:
                    raise RuntimeError("operator changed during cognition snapshot")
                if not operator.messages:
                    operator = self.state_db.load_operator_context(
                        statuses={OperatorMessageStatus.ACKNOWLEDGED}, limit=20,
                    )
                # Keep the target and semantic snapshot on the captured revision.
                self._merge_operator_target()
                snapshot = self.blackboard.cognition_snapshot()
        # Reconciliation can flush durable learning records. Only do it after
        # successful snapshot construction and outside the rollback boundary,
        # using exactly the observation the queued planner will receive.
        requires_wood = self._planks_retry_requires_wood(snapshot=snapshot)
        context = copy.deepcopy(self._cognition_context(
            operator, requires_wood=requires_wood,
        ))
        return context, snapshot, operator.revision

    @staticmethod
    def _notify_bound_discard(request: ModelRequestLifecycle, model: object) -> None:
        reason = request.take_discard_notice()
        if reason is None:
            return
        try:
            discard = cast(BoundCognitionModel, model).discard_bound_request
            discard(request=request.binding, reason=reason)
        except Exception as error:
            # A broken adapter notification cannot reopen a rejected request.
            # Detailed model work remains adapter-owned; delivery is not claimed.
            logging.getLogger(__name__).error(
                "Bound cognition discard notification failed: %s", type(error).__name__,
            )

    def _submit_bound_cognition(
        self, context: CognitionContext, snapshot: CognitionBlackboardSnapshot,
        *, operator_revision: int,
    ) -> concurrent.futures.Future[CognitionDecision]:
        controller = self.high_level
        assert controller is not None
        model = controller.model
        now = time.monotonic_ns()
        request = ModelRequestLifecycle(RequestBinding.from_snapshot(
            snapshot, request_id=uuid.uuid4().hex, operator_revision=operator_revision,
            execution_revision=self._execution_revision, submitted_ns=now,
            deadline_ns=now + self.cognition_request_timeout_ms * 1_000_000,
        ))

        def compute() -> CognitionDecision:
            try:
                if time.monotonic_ns() >= request.binding.deadline_ns:
                    request.reject("deadline_before_start")
                    raise RuntimeError("bound cognition expired while queued")
                # The call scope is installed inside this worker by decide().
                # No runtime, blackboard or database lock crosses this call.
                return controller.decide(snapshot, context, request=request)
            finally:
                request.mark_computation_complete()

        def completed(future: concurrent.futures.Future[CognitionDecision]) -> None:
            # Capture this exact request; a newer pending decision is unrelated.
            if future.cancelled():
                request.reject("cancelled_before_start")
                request.mark_computation_complete()
            elif future.exception() is not None:
                request.reject("computation_failed")
            self._notify_bound_discard(request, model)

        try:
            future = self._pool.submit(compute)
        except Exception:
            request.reject("submission_failed")
            request.mark_computation_complete()
            self._notify_bound_discard(request, model)
            raise
        self._bound_cognition_requests[future] = request, model
        future.add_done_callback(completed)
        return future

    def _reject_bound_cognition(
        self, future: concurrent.futures.Future[CognitionDecision], reason: str,
    ) -> None:
        record = getattr(self, "_bound_cognition_requests", {}).pop(future, None)
        if record is not None:
            request, model = record
            request.reject(reason)  # Must precede cancel(), which may call back immediately.
            self._notify_bound_discard(request, model)

    def _preflight_bound_cognition(
        self, record: tuple[ModelRequestLifecycle, object],
    ) -> bool:
        """Discard already stale requests early; this check never authorizes inputs.

        This short metadata check grants no publication and retains no lock over
        recovery or inputs. Final transactional admission must still repeat the
        authority checks after any decision rewriting.
        """
        request, model = record
        binding = request.binding
        snapshot = request.snapshot()
        if (self.state_db is None or snapshot.disposition != "pending"
                or snapshot.computation_completed_ns is None):
            return False
        try:
            with operator_intent_lock(timeout_s=0.05):
                with self.state_db.admit_operator_revision(binding.operator_revision) as current:
                    latest = self.blackboard.raw_latest()
                    return not (
                        not current or self._stop.is_set() or operator_pause_latched()
                        or emergency_stop_latched()
                        or self._input_release_pending_ns is not None
                        or self._execution_revision != binding.execution_revision
                        or self.high_level is None
                        or getattr(self.high_level, "model", None) is not model
                        or latest is None or latest.instance_id != binding.instance_id
                        or time.monotonic_ns() >= binding.deadline_ns
                    )
        except Exception as error:
            logging.getLogger(__name__).error(
                "Bound cognition preflight rejected: %s", type(error).__name__,
            )
            return False

    def _admit_bound_cognition(
        self, record: tuple[ModelRequestLifecycle, object], decision: CognitionDecision,
        *, adopt_plan: bool,
    ) -> bool:
        """Compare authority and publish once; never perform inference or inputs here.

        New ordinary frames do not invalidate an older semantic snapshot. Current
        instance, intent, execution, selected-skill preconditions and existing
        perception-probe rules remain the publication authority.
        """
        request, model = record
        binding = request.binding
        origin = decision.model_origin
        previous = (
            self._last_decision, self._plan_steps, self._plan_goal_id,
            self._plan_index, self._plan_started_ns,
        )

        def restore_metadata() -> None:
            (
                self._last_decision, self._plan_steps, self._plan_goal_id,
                self._plan_index, self._plan_started_ns,
            ) = previous

        def publish() -> None:
            try:
                if self.state_db is None:
                    raise RuntimeError("operator_authority_unavailable")
                with operator_intent_lock(timeout_s=0.05):
                    with self.state_db.admit_operator_revision(
                        binding.operator_revision,
                    ) as current:
                        latest = self.blackboard.raw_latest()
                        if (not current or self._stop.is_set() or operator_pause_latched()
                                or emergency_stop_latched()
                                or self._input_release_pending_ns is not None
                                or self._execution_revision != binding.execution_revision
                                or self.high_level is None or self.high_level.model is not model
                                or latest is None or latest.instance_id != binding.instance_id
                                or time.monotonic_ns() >= binding.deadline_ns):
                            raise RuntimeError("publication_authority_changed")
                        if decision.skill_id is not None:
                            if (decision.skill_id not in self.skills.specs
                                    or not initiation_satisfied(
                                        self.skills.get(decision.skill_id), self.blackboard,
                                    )):
                                raise RuntimeError("selected_skill_no_longer_feasible")
                        if origin is not None:
                            admit: Callable[..., object] = (
                                cast(BoundCognitionModel, model).admit_bound_decision
                            )
                            result = admit(
                                request=binding, attempt_id=origin.attempt_id,
                                source_decision_sha256=origin.source_decision_sha256,
                                final_decision=decision.model_dump(mode="json"),
                                final_decision_sha256=final_sha,
                                rewritten=final_sha != origin.source_decision_sha256,
                            )
                            if result is not None:
                                raise TypeError("bound admission must return None or raise")
                        self._last_decision = decision
                        if adopt_plan:
                            self._adopt_plan_if_revised(decision)
            except BaseException:
                restore_metadata()
                raise

        try:
            if origin is None:
                # A fallback may be adopted, but earlier model attempts never
                # become its publication. Claim the discard only after the
                # complete authority transaction succeeds.
                if request.snapshot().disposition != "pending":
                    return False
                try:
                    publish()
                    if request.reject("non_model_decision"):
                        return True
                except BaseException:
                    restore_metadata()
                    raise
                restore_metadata()
                return False
            if origin.request_id != binding.request_id:
                raise RuntimeError("model_attempt_origin_changed")
            final_sha = cognition_decision_sha256(decision)
            # Context-manager exits belong inside accept's callback: a failed
            # database commit must trigger the existing publication compensation.
            return request.accept(origin.attempt_id, final_sha, publish)
        except BaseException as error:
            request.reject(type(error).__name__)
            if not isinstance(error, Exception):
                raise
            logging.getLogger(__name__).error(
                "Bound cognition publication rejected: %s", type(error).__name__,
            )
            return False
        finally:
            self._notify_bound_discard(request, model)

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
            self._reject_bound_cognition(stale_future, "operator_preempted")
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
            if messages:
                return None
            acknowledged = self.state_db.load_operator_messages(
                statuses={OperatorMessageStatus.ACKNOWLEDGED}, limit=20,
            )
            if _active_operator_messages(acknowledged) or (
                len(acknowledged) == 20 and not any(
                    message.kind in {
                        OperatorMessageKind.INSTRUCTION, OperatorMessageKind.CORRECTION,
                    }
                    for message in acknowledged
                )
            ):
                # A full page of replies must not hide an older durable wait
                # instruction. Unknown authority is not autonomous permission.
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
            self._reject_bound_cognition(future, "result_failed")
            now = time.monotonic_ns()
            self._last_cognition_ns = now
            self._pending_operator_message_ids = ()
            self._schedule_cognition_retry(now_ns=now)
            return
        now = time.monotonic_ns()
        self._last_cognition_ns = now
        record = getattr(self, "_bound_cognition_requests", {}).get(future)
        if record is not None and not self._preflight_bound_cognition(record):
            self._reject_bound_cognition(future, "consumption_authority_changed")
            self._pending_operator_message_ids = ()
            self._pending_operator_message_kinds = {}
            self._schedule_cognition_retry(now_ns=now)
            return
        if self._pending_execution_revision != self._execution_revision:
            self._reject_bound_cognition(future, "execution_changed")
            # The decision was sampled before the option produced terminal
            # evidence. Re-evaluate with that failure/success in context rather
            # than immediately replaying the stale option choice.
            self._pending_operator_message_ids = ()
            self._cognition_requested = True
            return
        if self._operator_message_arrived_after_snapshot():
            self._reject_bound_cognition(future, "operator_changed")
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
            self._reject_bound_cognition(future, "wood_prerequisite_changed")
            self._pending_operator_message_ids = ()
            self._cognition_requested = True
            return
        if record is not None:
            if self._crafting_gui_close_required(decision):
                # This shortcut performs inputs without final publication.
                # A preflight cannot authorize them after its locks expire.
                # Keep crafting under its executor/operator recovery authority
                # and let a later observation produce another bound decision.
                self._reject_bound_cognition(future, "crafting_gui_close_deferred")
                self._pending_operator_message_ids = ()
                self._pending_operator_message_kinds = {}
                self._schedule_cognition_retry(now_ns=now)
                return
        elif self._close_crafting_gui_before_world_decision(decision):
            self._reject_bound_cognition(future, "crafting_gui_close_required")
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
        record = getattr(self, "_bound_cognition_requests", {}).pop(future, None)
        if record is not None:
            if not self._admit_bound_cognition(
                record, decision, adopt_plan=idle_stall_run_id is None,
            ):
                self._pending_operator_message_ids = ()
                self._schedule_cognition_retry(now_ns=now)
                return
        else:
            self._last_decision = decision
            if idle_stall_run_id is None:
                self._adopt_plan_if_revised(decision)
        skill_origin = self._skill_decision_origin(record, decision)
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
                    self._start_skill(
                        spec,
                        source=SkillStartSource.COGNITION,
                        origin=skill_origin,
                        run_id=uuid.uuid4().hex,
                        context_key=decision.chosen_goal_id or "default",
                        parameters=decision.skill_parameters,
                        instruction=decision.instruction,
                    )
            else:
                spec = self.skills.get(decision.skill_id)
                self._start_skill(
                    spec,
                    source=SkillStartSource.COGNITION,
                    origin=skill_origin,
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

    def _crafting_gui_close_required(self, decision: CognitionDecision) -> bool:
        """Inspect the GUI-close shortcut without cancelling, recovering or sending inputs."""
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
        return requested.action_level != ActionLevel.GUI

    def _close_crafting_gui_before_world_decision(
        self,
        decision: CognitionDecision,
    ) -> bool:
        """Finish a verified inventory close before adopting legacy world control.

        A completed decision was sampled while crafting owned the inventory.
        Reusing that decision after the visual scene changes would be stale, so
        discard it, close the GUI, and let the close terminal event request a
        fresh decision from the restored world frame. Bound decisions defer
        this shortcut because its inputs have no final admission boundary.
        """
        if not self._crafting_gui_close_required(decision):
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

    @staticmethod
    def _skill_decision_origin(
        record: tuple[ModelRequestLifecycle, object] | None,
        decision: CognitionDecision,
    ) -> SkillDecisionOrigin | None:
        """Bind only this accepted decision, never a latest-admission lookup.

        A deterministic fallback can pass runtime admission while its failed
        model request is rejected. It deliberately has no model origin.
        """
        origin = decision.model_origin
        if record is None or origin is None:
            return None
        snapshot = record[0].snapshot()
        if (
            snapshot.disposition != "accepted"
            or snapshot.binding.request_id != origin.request_id
            or snapshot.selected_attempt_id != origin.attempt_id
            or snapshot.final_decision_sha256 != cognition_decision_sha256(decision)
        ):
            return None
        assert snapshot.final_decision_sha256 is not None
        return SkillDecisionOrigin(
            request=snapshot.binding,
            attempt_id=origin.attempt_id,
            source_decision_sha256=origin.source_decision_sha256,
            final_decision_sha256=snapshot.final_decision_sha256,
        )

    def on_skill_run_started(
        self, *, run: SkillRun, source: SkillStartSource,
        origin: SkillDecisionOrigin | None, parent_run_id: str | None,
    ) -> None:
        """Observe an actual new run; default implementation does nothing.

        Called synchronously on the runtime thread, after executor.start returns.
        Overrides must be short, nonblocking metadata operations: no input,
        inference or IO. The run is detached from executor-owned parameters.
        Non-model/recovery/continuation runs have no model origin; a parent ID
        is a factual relationship, not inherited model credit. Admission of a
        same-action decision does not create a run or call this hook again.
        """

    def on_skill_run_terminal(
        self, *, run: SkillRun, outcome_verification: OutcomeVerification | None,
    ) -> None:
        """Observe a terminal run and its matching, runtime-filtered evidence.

        Same nonblocking contract as on_skill_run_started. Called before storage,
        even without a database or a matching observed start. Deduplication uses
        the existing 4096-ID window, not a durable exactly-once guarantee. Consumers
        own run-ID joins and replay handling; never use the latest model decision.
        Terminal recording can follow failed input delivery: this hook does not
        prove supervisor-accepted actuation or authorize learning from it.
        """

    def _start_skill(
        self, spec: SkillSpec, *, source: SkillStartSource,
        origin: SkillDecisionOrigin | None = None, parent_run_id: str | None = None,
        **kwargs: Any,
    ) -> SkillRun:
        run = self.executor.start(spec, **kwargs)
        try:
            self.on_skill_run_started(
                run=run.model_copy(deep=True), source=source, origin=origin,
                parent_run_id=parent_run_id,
            )
        except Exception as error:
            logging.getLogger(__name__).error(
                "Skill-start observer failed: %s", type(error).__name__,
            )
        return run

    def _record_terminal_run(
        self,
        run: SkillRun,
        *,
        outcome_verification: OutcomeVerification | None = None,
        advance_plan: bool = True,
    ) -> None:
        """Record one terminal option, deduplicated within the recent ID window."""

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
        matching_verification = outcome_verification
        if matching_verification is not None and matching_verification.run_id != run.run_id:
            matching_verification = None
            logging.getLogger(__name__).error("Skill-terminal evidence rejected: ValueError")
        try:
            self.on_skill_run_terminal(
                run=run.model_copy(deep=True), outcome_verification=matching_verification,
            )
        except Exception as error:
            logging.getLogger(__name__).error(
                "Skill-terminal observer failed: %s", type(error).__name__,
            )
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

    def _cognition_context(
        self, operator_context: OperatorContextSnapshot | None = None,
        *, requires_wood: bool | None = None,
    ) -> CognitionContext:
        goals = tuple((*role_standing_goals(self.role), *self.custom_goals))
        memories = tuple(self.memories.retrieve(limit=20))
        operator_messages: tuple[OperatorMessage, ...] = ()
        if operator_context is not None:
            messages = tuple(
                message for message in operator_context.messages
                if message.status in {
                    OperatorMessageStatus.QUEUED, OperatorMessageStatus.DELIVERED,
                }
            )[:20]
            if not messages:
                messages = tuple(message for message in operator_context.messages
                                 if message.status == OperatorMessageStatus.ACKNOWLEDGED)[:20]
            operator_messages = _active_operator_messages(messages)
        elif self.state_db is not None:
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
            planks_retry_requires_wood=(
                self._planks_retry_requires_wood() if requires_wood is None else requires_wood
            ),
        )

    def _start_recovery_skill(
        self,
        recovery: SkillSpec,
        parent: SkillRun,
        *,
        plan_neutral: bool = False,
    ) -> SkillRun:
        run = self._start_skill(
            recovery,
            source=SkillStartSource.RECOVERY,
            parent_run_id=parent.run_id,
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

    def _planks_retry_requires_wood(
        self, *, snapshot: CognitionBlackboardSnapshot | None = None,
    ) -> bool:
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
        board = snapshot if snapshot is not None else getattr(self, "blackboard", None)
        if board is None:
            return True
        now = time.monotonic_ns() if snapshot is None else snapshot.snapshot_ns
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
            "input_release_pending": self._input_release_pending_ns is not None,
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
        self._start_skill(
            self.skills.get(skill_id),
            source=SkillStartSource.BOOTSTRAP,
            run_id=uuid.uuid4().hex,
            context_key=f"role:{self.role.role_id}",
        )

    def _failsafe(self, reason: str) -> None:
        try:
            send_command("fault", reason=reason)
        except Exception:
            pass

