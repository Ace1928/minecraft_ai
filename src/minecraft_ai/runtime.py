from __future__ import annotations

import concurrent.futures
import threading
import time
import uuid
from dataclasses import dataclass, field

from .cognition import (
    BootstrapCognitionPolicy,
    CognitionContext,
    CognitionDecision,
    HighLevelController,
)
from .datasets import ActionLevel
from .curriculum import CurriculumCandidate, CurriculumScheduler, role_standing_goals
from .execution import SkillExecutor
from .memory import MemoryStore
from .perception import ActivePerceptionQuery, PerceptionBlackboard
from .perception_service import RealtimePerceptionService
from .planning import Goal
from .roles import RoleProfile
from .safety import MotorAction
from .skills import SkillLibrary, SkillOutcome
from .social import (
    OperatorMessage,
    OperatorMessageKind,
    OperatorMessageStatus,
    SocialState,
)
from .telemetry import TelemetryPublisher
from .trajectory import TrajectoryRecorder
from .storage import StateDatabase
from .supervisor import send_command


def _semantic_deadline_ms(semantic_hz: float) -> int:
    """Bound request lifetime independently from a slower query cadence."""
    return min(10_000, max(250, int(1000 / semantic_hz)))


def _active_operator_messages(
    messages: tuple[OperatorMessage, ...],
) -> tuple[OperatorMessage, ...]:
    """Retain acknowledged directives as commitments until explicitly archived."""
    return tuple(
        message
        for message in messages
        if message.status in {
            OperatorMessageStatus.QUEUED,
            OperatorMessageStatus.DELIVERED,
        }
        or (
            message.status == OperatorMessageStatus.ACKNOWLEDGED
            and message.kind
            in {OperatorMessageKind.INSTRUCTION, OperatorMessageKind.CORRECTION}
        )
    )


@dataclass
class RuntimeMetrics:
    frames: int = 0
    motor_actions: int = 0
    cognition_calls: int = 0
    semantic_requests: int = 0
    chat_messages: int = 0
    skill_successes: int = 0
    skill_failures: int = 0
    started_ns: int = field(default_factory=time.monotonic_ns)
    last_capture_ms: float = 0.0
    last_motor_ms: float = 0.0
    stale_frame_skips: int = 0
    consecutive_stale_frames: int = 0


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
    _last_operator_target_id: str | None = field(default=None, init=False)
    _policy_warmup_error: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.motor_hz <= 0 or self.cognition_hz <= 0 or self.semantic_hz <= 0:
            raise ValueError("runtime frequencies must be positive")
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
        self._lease_thread.start()
        if self.perception.active_vlm is not None:
            self.perception.active_vlm.start()
        try:
            self.telemetry.publish(self._telemetry_payload(state="warming"), force=True)
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
                    release = self.executor.cancel().action
                    if release is not None:
                        self._send_motor(release)
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
        if self.perception.stale():
            self.metrics.stale_frame_skips += 1
            self.metrics.consecutive_stale_frames += 1
            # A late frame must never extend a previously accepted key/button
            # state. Preserve the authenticated lease so a transient CPU stall
            # can recover on the next fresh capture.
            send_command("release-inputs", lease_id=self.lease_id)
            self.telemetry.publish(self._telemetry_payload(state="capture-stalled"))
            if (
                self.metrics.consecutive_stale_frames
                >= self.stale_frame_consecutive_limit
            ):
                raise RuntimeError(
                    "capture stream is stale for "
                    f"{self.metrics.consecutive_stale_frames} consecutive frames"
                )
            return
        self.metrics.consecutive_stale_frames = 0
        self.telemetry.publish(self._telemetry_payload(state="running"))
        self._consume_cognition()
        self._request_semantics_if_due(frame.frame_id)
        self._start_cognition_if_due()

        active = self.executor.run
        if active is None or active.outcome != SkillOutcome.RUNNING:
            return
        motor_started = time.perf_counter()
        result = self.executor.tick(
            self.blackboard,
            sequence=self._sequence,
            now_ns=time.monotonic_ns(),
        )
        if result.action is not None:
            self._send_motor(result.action)
        self.metrics.last_motor_ms = (time.perf_counter() - motor_started) * 1000.0
        if result.run.outcome != SkillOutcome.RUNNING:
            stats = self.skills.record(result.run)
            if self.state_db is not None:
                self.state_db.save_skill_stats(
                    result.run.skill_id,
                    result.run.context_key,
                    stats,
                )
            if result.run.outcome == SkillOutcome.SUCCEEDED:
                self.metrics.skill_successes += 1
            elif result.run.outcome in {SkillOutcome.FAILED, SkillOutcome.TIMED_OUT}:
                self.metrics.skill_failures += 1

            for recovery_id in result.recovery_skills:
                if recovery_id in self.skills.specs:
                    self.executor.start(
                        self.skills.get(recovery_id),
                        run_id=uuid.uuid4().hex,
                        context_key=result.run.context_key,
                    )
                    break

    def _lease_heartbeat(self) -> None:
        """Keep the motor lease alive independently of inference/cognition latency."""
        interval_s = self.lease_renew_ms / 1000.0
        ttl_ms = min(5000, max(3000, self.lease_renew_ms * 8))
        while not self._stop.is_set():
            try:
                send_command("renew", lease_id=self.lease_id, ttl_ms=ttl_ms)
                self._last_renew_ns = time.monotonic_ns()
                self._lease_fault = None
            except Exception as exc:
                self._lease_fault = f"{type(exc).__name__}: {exc}"
                self._stop.set()
                return
            self._stop.wait(interval_s)

    def _send_motor(self, action: MotorAction) -> None:
        accepted = send_command(
            "motor-action",
            lease_id=self.lease_id,
            action=action.model_dump(mode="json"),
        )
        if self.trajectory is not None:
            frame = self.perception.last_capture
            blackboard = self.blackboard.latest()
            if frame is not None and blackboard is not None:
                running = self.executor.run
                self.trajectory.record_accepted(
                    action=action,
                    supervisor_response=accepted,
                    frame=frame,
                    blackboard=blackboard,
                    action_level=ActionLevel.RAW,
                    skill_run_id=None if running is None else running.run_id,
                    skill_id=None if running is None else running.skill_id,
                    goal_id=None
                    if self._last_decision is None
                    else self._last_decision.chosen_goal_id,
                )
        self._sequence = action.sequence + 1
        self.metrics.motor_actions += 1

    def _request_semantics_if_due(self, frame_id: int) -> None:
        if self.perception.active_vlm is None:
            return
        now = time.monotonic_ns()
        interval = int(1e9 / self.semantic_hz)
        if now - self._last_semantic_ns < interval:
            return
        active = self.executor.run
        skill_id = active.skill_id if active is not None else None
        question = self._semantic_question(skill_id)
        query = ActivePerceptionQuery(
            query_id=uuid.uuid4().hex,
            question=question,
            skill_id=skill_id,
            frame_id=frame_id,
            deadline_ms=_semantic_deadline_ms(self.semantic_hz),
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

    def _start_cognition_if_due(self) -> None:
        if self._pending_decision is not None:
            return
        now = time.monotonic_ns()
        interval = int(1e9 / self.cognition_hz)
        if now - self._last_cognition_ns < interval:
            return
        context = self._cognition_context()
        self._pending_operator_message_ids = tuple(
            message.message_id
            for message in context.operator_messages
            if message.status
            in {OperatorMessageStatus.QUEUED, OperatorMessageStatus.DELIVERED}
        )
        if self.state_db is not None:
            for message in context.operator_messages:
                if message.status == OperatorMessageStatus.QUEUED:
                    self.state_db.update_operator_message_status(
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
        self.metrics.cognition_calls += 1

    def _consume_cognition(self) -> None:
        future = self._pending_decision
        if future is None or not future.done():
            return
        self._pending_decision = None
        try:
            decision = future.result()
        except Exception:
            return
        self._last_decision = decision
        if self.state_db is not None and self._pending_operator_message_ids:
            response = decision.say or decision.reasoning_summary
            for message_id in self._pending_operator_message_ids:
                try:
                    self.state_db.update_operator_message_status(
                        message_id,
                        OperatorMessageStatus.ACKNOWLEDGED,
                        timestamp_ns=time.time_ns(),
                        response_text=response,
                    )
                except KeyError:
                    continue
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
            )
            self.perception.request_semantics(query)
        if decision.say:
            try:
                send_command("chat", lease_id=self.lease_id, text=decision.say)
                self.metrics.chat_messages += 1
            except Exception:
                pass
        if decision.skill_id is not None:
            running = self.executor.run
            if running is not None and running.outcome == SkillOutcome.RUNNING:
                if running.skill_id != decision.skill_id:
                    cancelled = self.executor.cancel()
                    if cancelled.action is not None:
                        self._send_motor(cancelled.action)
                    spec = self.skills.get(decision.skill_id)
                    self.executor.start(
                        spec,
                        run_id=uuid.uuid4().hex,
                        parameters=decision.skill_parameters,
                    )
            else:
                spec = self.skills.get(decision.skill_id)
                self.executor.start(
                    spec,
                    run_id=uuid.uuid4().hex,
                    parameters=decision.skill_parameters,
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
        )

    def _telemetry_payload(self, *, state: str) -> dict[str, object]:
        running = self.executor.run
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
        perception_status["tracks"] = [] if latest is None else [
            track.model_dump(mode="json") for track in latest.tracks
        ]
        return {
            "schema_version": 1,
            "state": state,
            "role": self.role.role_id,
            "lease_id": self.lease_id,
            "frames": self.metrics.frames,
            "motor_actions": self.metrics.motor_actions,
            "cognition_calls": self.metrics.cognition_calls,
            "semantic_requests": self.metrics.semantic_requests,
            "chat_messages": self.metrics.chat_messages,
            "skill_successes": self.metrics.skill_successes,
            "skill_failures": self.metrics.skill_failures,
            "last_capture_ms": round(self.metrics.last_capture_ms, 3),
            "last_motor_ms": round(self.metrics.last_motor_ms, 3),
            "stale_frame_skips": self.metrics.stale_frame_skips,
            "consecutive_stale_frames": self.metrics.consecutive_stale_frames,
            "active_skill": None if running is None else running.skill_id,
            "active_instruction": self.executor.instruction,
            "skill_outcome": None if running is None else running.outcome.value,
            "chosen_goal_id": None if decision is None else decision.chosen_goal_id,
            "reasoning_summary": None if decision is None else decision.reasoning_summary,
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
        scheduler.choose(
            [CurriculumCandidate(goal=goal, progression_novelty=0.4) for goal in goals]
        )
        if "explore_forward" in self.skills.specs:
            self.executor.start(
                self.skills.get("explore_forward"),
                run_id=uuid.uuid4().hex,
                context_key=f"role:{self.role.role_id}",
            )

    def _failsafe(self, reason: str) -> None:
        try:
            send_command("fault", reason=reason)
        except Exception:
            pass
