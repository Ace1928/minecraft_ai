from __future__ import annotations

import concurrent.futures
import sqlite3
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from .cognition import (
    BootstrapCognitionPolicy,
    CognitionContext,
    CognitionDecision,
    HighLevelController,
)
from .action_levels import ActionLevel
from .curriculum import CurriculumCandidate, CurriculumScheduler, role_standing_goals
from .execution import ExecutionTick, SkillExecutor, initiation_satisfied
from .grounded_perception import resolve_grounded_output_keys
from .memory import MemoryStore
from .motor import MotorIntent
from .perception import ActivePerceptionQuery, PerceptionBlackboard, PerceptionFact, Track
from .perception_service import RealtimePerceptionService, perceptual_hash_distance
from .planning import Goal
from .roles import RoleProfile
from .safety import MotorAction
from .skills import SkillLibrary, SkillOutcome, SkillRun, SkillSpec, SkillStats
from .social import (
    OperatorMessage,
    OperatorMessageKind,
    OperatorMessageStatus,
    SocialState,
)
from .telemetry import TelemetryPublisher
from .trajectory import ActionOrigin, ActionProvenance, TrajectoryRecorder, motor_condition_id
from .storage import StateDatabase
from .supervisor import send_command


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
    instruction or correction remains the current directive. Older messages
    stay visible in conversation history but cannot silently regain control.
    Persistent multi-project commitments belong in the goal portfolio rather
    than an ever-growing motor prompt.
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
    return tuple(
        sorted(
            acknowledged,
            key=lambda message: message.created_ns,
            reverse=True,
        )[:1]
    )


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
    _pool: concurrent.futures.ThreadPoolExecutor = field(init=False)
    _last_decision: CognitionDecision | None = field(default=None, init=False)
    _pending_operator_message_ids: tuple[str, ...] = field(default=(), init=False)
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
    _cognition_requested: bool = field(default=True, init=False)
    _pending_skill_stats: dict[tuple[str, str], SkillStats] = field(
        default_factory=dict,
        init=False,
    )
    _last_storage_retry_ns: int = field(default=0, init=False)
    _last_operator_storage_retry_ns: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.motor_hz <= 0 or self.cognition_hz <= 0 or self.semantic_hz < 0:
            raise ValueError(
                "motor/cognition frequencies must be positive and semantic nonnegative"
            )
        if self.stale_frame_consecutive_limit < 1:
            raise ValueError("stale_frame_consecutive_limit must be positive")
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="minecraft-ai-cognition",
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
            send_command("renew", lease_id=self.lease_id, ttl_ms=10_000)
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
                    if cancelled.action is not None:
                        self._send_motor(cancelled.action, execution=cancelled)
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
        self._flush_pending_operator_status_updates()
        self.telemetry.publish(self._telemetry_payload(state="running"))
        self._publish_player_chat_facts()
        self._consume_cognition()
        self._start_cognition_if_due()
        self._request_semantics_if_due(frame.frame_id)
        self._route_observed_scene_recovery()

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
                    context_key="explore-keepalive",
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
        if result.action is not None:
            self._send_motor(result.action, execution=result)
        self.metrics.last_motor_ms = (time.perf_counter() - motor_started) * 1000.0
        if result.run.outcome != SkillOutcome.RUNNING:
            self._recent_skill_runs.appendleft(result.run)
            self._execution_revision += 1
            self._cognition_requested = True
            stats = self.skills.record(result.run)
            if self.state_db is not None:
                self._pending_skill_stats[(result.run.skill_id, result.run.context_key)] = stats
                self._flush_pending_skill_stats(force=True)
            if result.run.outcome == SkillOutcome.SUCCEEDED:
                self.metrics.skill_successes += 1
                self._advance_plan_on_step_complete(result.run)
            elif result.run.outcome in {SkillOutcome.FAILED, SkillOutcome.TIMED_OUT}:
                self.metrics.skill_failures += 1

            recovery = _first_feasible_recovery(
                self.skills,
                result.recovery_skills,
                self.blackboard,
            )
            if recovery is not None:
                self.executor.start(
                    recovery,
                    run_id=uuid.uuid4().hex,
                    context_key=result.run.context_key,
                )

    def _explore_keep_alive(self) -> SkillSpec | None:
        """Pick a precondition-free option to keep motor busy while cognition decides.

        This runs only when no skill is currently running (the idle gap after a
        terminal run). The MOTION-level traversal option routes to the fast
        learned motion expert (VPT), which continuously emits locomotion even
        without fresh semantic/grounding data -- precisely what prevents the
        idle freeze that the latent STEVE body produces while cognition is in
        flight.
        """
        for skill_id in ("traverse_level_ground", "explore_forward"):
            return self.skills.specs.get(skill_id)

    def _route_observed_scene_recovery(self) -> None:
        """Preempt stale world work when a verified blocking scene event arrives."""
        recovery = _observed_scene_recovery(self.skills, self.blackboard)
        if recovery is None:
            return
        running = self.executor.run
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
            if cancelled.action is not None:
                self._send_motor(cancelled.action, execution=cancelled)
            self._recent_skill_runs.appendleft(cancelled.run)
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
        ttl_ms = min(10_000, max(5_000, self.lease_renew_ms * 12))
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
        provenance = _accepted_action_provenance(
            execution,
            self.blackboard,
            fallback_policy_id=self.executor.policy.policy_id,
        )
        accepted = send_command(
            "motor-action",
            lease_id=self.lease_id,
            action=action.model_dump(mode="json"),
        )
        if self.trajectory is not None:
            frame = self.perception.last_capture
            blackboard = self.blackboard.latest()
            if frame is not None and blackboard is not None:
                running = self.executor.run if execution is None else execution.run
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
                )
        self._sequence = action.sequence + 1
        self.metrics.motor_actions += 1

    def _request_semantics_if_due(self, frame_id: int) -> None:
        # semantic_hz=0 is event-only active perception. Explicit questions from
        # cognition still flow through _consume_cognition below.
        if self.semantic_hz <= 0 or self.perception.active_vlm is None:
            return
        if not _semantic_refresh_allowed(
            cognition_requested=self._cognition_requested,
            cognition_pending=self._pending_decision is not None,
            operator_message_pending=bool(self._pending_operator_message_ids),
            worker_available=self.perception.semantic_available(),
        ):
            return
        now = time.monotonic_ns()
        interval = int(1e9 / self.semantic_hz)
        if now - self._last_semantic_ns < interval:
            return
        active = self.executor.run
        skill_id = active.skill_id if active is not None else None
        question = self._semantic_question(skill_id)
        output_keys = list(
            (
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
            deadline_ms=_semantic_deadline_ms(self.semantic_hz),
            output_keys=tuple(dict.fromkeys(output_keys)),
        )
        if self.perception.request_semantics(query):
            self.metrics.semantic_requests += 1
            self._last_semantic_ns = now

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
        if self._pending_decision is not None:
            return
        now = time.monotonic_ns()
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
        self._pending_operator_message_ids = tuple(
            message.message_id
            for message in context.operator_messages
            if message.status in {OperatorMessageStatus.QUEUED, OperatorMessageStatus.DELIVERED}
        )
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

    def _consume_cognition(self) -> None:
        future = self._pending_decision
        if future is None or not future.done():
            return
        self._pending_decision = None
        try:
            decision = future.result()
        except Exception:
            self._last_cognition_ns = time.monotonic_ns()
            return
        self._last_cognition_ns = time.monotonic_ns()
        if self._pending_execution_revision != self._execution_revision:
            # The decision was sampled before the option produced terminal
            # evidence. Re-evaluate with that failure/success in context rather
            # than immediately replaying the stale option choice.
            self._pending_operator_message_ids = ()
            self._cognition_requested = True
            return
        if self._queued_operator_message_waiting():
            # This decision was produced from an older context snapshot. A
            # fresh operator message has higher authority and must be included
            # before any skill switch or acknowledgement is applied.
            self._pending_operator_message_ids = ()
            self._cognition_requested = True
            return
        self._last_decision = decision
        self._adopt_plan_if_revised(decision)
        if self.state_db is not None and self._pending_operator_message_ids:
            selected_message_id = _selected_operator_message_id(
                decision,
                self._pending_operator_message_ids,
            )
            if selected_message_id is not None:
                response = decision.say or decision.reasoning_summary
                self._persist_operator_message_status(
                    selected_message_id,
                    OperatorMessageStatus.ACKNOWLEDGED,
                    timestamp_ns=time.time_ns(),
                    response_text=response,
                )
            self._pending_operator_message_ids = ()
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
                ):
                    cancelled = self.executor.cancel()
                    if cancelled.action is not None:
                        self._send_motor(cancelled.action, execution=cancelled)
                    self._recent_skill_runs.appendleft(cancelled.run)
                    self._execution_revision += 1
                    spec = self.skills.get(decision.skill_id)
                    self.executor.start(
                        spec,
                        run_id=uuid.uuid4().hex,
                        parameters=decision.skill_parameters,
                        instruction=decision.instruction,
                    )
            else:
                spec = self.skills.get(decision.skill_id)
                self.executor.start(
                    spec,
                    run_id=uuid.uuid4().hex,
                    parameters=decision.skill_parameters,
                    instruction=decision.instruction,
                )

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
                self.cfg_role().casefold(), "eidos", "you", "console",
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
        if not self._pending_skill_stats and not self._pending_operator_status_updates:
            self.metrics.last_storage_error = None

    def _queued_operator_message_waiting(self) -> bool:
        if self.state_db is None:
            return False
        return bool(
            self.state_db.load_operator_messages(
                statuses={OperatorMessageStatus.QUEUED},
                limit=1,
            )
        )

    def _cognition_context(self) -> CognitionContext:
        goals = tuple((*role_standing_goals(self.role), *self.custom_goals))
        memories = tuple(self.memories.retrieve(limit=20))
        operator_messages: tuple[OperatorMessage, ...] = ()
        if self.state_db is not None:
            operator_messages = _active_operator_messages(
                self.state_db.load_operator_messages(
                    statuses={
                        OperatorMessageStatus.QUEUED,
                        OperatorMessageStatus.DELIVERED,
                        OperatorMessageStatus.ACKNOWLEDGED,
                    },
                    limit=20,
                )
            )
        return CognitionContext(
            role=self.role,
            goals=goals,
            memories=memories,
            promises=self.social.active_promises(),
            wiki=(),
            operator_messages=operator_messages,
            recent_skill_runs=tuple(self._recent_skill_runs),
            current_plan=self._plan_steps[self._plan_index:],
            plan_goal_id=self._plan_goal_id,
            plan_index=self._plan_index,
            plan_started_ns=self._plan_started_ns,
        )

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
            "last_capture_ms": round(self.metrics.last_capture_ms, 3),
            "last_motor_ms": round(self.metrics.last_motor_ms, 3),
            "stale_frame_skips": self.metrics.stale_frame_skips,
            "consecutive_stale_frames": self.metrics.consecutive_stale_frames,
            "storage_contentions": self.metrics.storage_contentions,
            "storage_backlog": (
                len(self._pending_skill_stats) + len(self._pending_operator_status_updates)
            ),
            "last_storage_error": self.metrics.last_storage_error,
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
