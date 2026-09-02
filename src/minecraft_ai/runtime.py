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
from .curriculum import CurriculumCandidate, CurriculumScheduler, role_standing_goals
from .execution import SkillExecutor
from .memory import MemoryStore
from .perception import ActivePerceptionQuery, PerceptionBlackboard
from .perception_service import RealtimePerceptionService
from .planning import Goal
from .roles import RoleProfile
from .safety import MotorAction
from .skills import SkillLibrary, SkillOutcome
from .social import OperatorMessage, OperatorMessageStatus, SocialState
from .telemetry import TelemetryPublisher
from .storage import StateDatabase
from .supervisor import send_command


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
    metrics: RuntimeMetrics = field(default_factory=RuntimeMetrics)
    telemetry: TelemetryPublisher = field(default_factory=TelemetryPublisher)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _sequence: int = field(default=0, init=False)
    _last_renew_ns: int = field(default=0, init=False)
    _last_cognition_ns: int = field(default=0, init=False)
    _last_semantic_ns: int = field(default=0, init=False)
    _pending_decision: concurrent.futures.Future[CognitionDecision] | None = field(
        default=None,
        init=False,
    )
    _pool: concurrent.futures.ThreadPoolExecutor = field(init=False)
    _last_decision: CognitionDecision | None = field(default=None, init=False)
    _pending_operator_message_ids: tuple[str, ...] = field(default=(), init=False)

    def __post_init__(self) -> None:
        if self.motor_hz <= 0 or self.cognition_hz <= 0 or self.semantic_hz <= 0:
            raise ValueError("runtime frequencies must be positive")
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="minecraft-ai-cognition",
        )

    def stop(self) -> None:
        self._stop.set()

    def run_forever(self) -> None:
        period = 1.0 / self.motor_hz
        if self.perception.active_vlm is not None:
            self.perception.active_vlm.start()
        try:
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
            self.telemetry.publish(self._telemetry_payload(state="stopped"), force=True)
            self._pool.shutdown(wait=False, cancel_futures=True)

    def tick(self) -> None:
        capture_started = time.perf_counter()
        frame = self.perception.capture_once()
        self.metrics.frames += 1
        self.metrics.last_capture_ms = (time.perf_counter() - capture_started) * 1000.0
        self.telemetry.publish(self._telemetry_payload(state="running"))
        if self.perception.stale():
            raise RuntimeError("capture stream is stale")
        self._renew_lease_if_due()
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

    def _renew_lease_if_due(self) -> None:
        now = time.monotonic_ns()
        if now - self._last_renew_ns < self.lease_renew_ms * 1_000_000:
            return
        send_command(
            "renew",
            lease_id=self.lease_id,
            ttl_ms=max(2000, self.lease_renew_ms * 4),
        )
        self._last_renew_ns = now

    def _send_motor(self, action: MotorAction) -> None:
        send_command(
            "motor-action",
            lease_id=self.lease_id,
            action=action.model_dump(mode="json"),
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
            deadline_ms=max(250, int(1000 / self.semantic_hz)),
        )
        if self.perception.request_semantics(query):
            self.metrics.semantic_requests += 1
            self._last_semantic_ns = now

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
            message.message_id for message in context.operator_messages
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
                    self.executor.cancel()
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
            operator_messages = self.state_db.load_operator_messages(
                statuses={OperatorMessageStatus.QUEUED, OperatorMessageStatus.DELIVERED},
                limit=20,
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
            "active_skill": None if running is None else running.skill_id,
            "skill_outcome": None if running is None else running.outcome.value,
            "chosen_goal_id": None if decision is None else decision.chosen_goal_id,
            "reasoning_summary": None if decision is None else decision.reasoning_summary,
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
