from __future__ import annotations

import platform
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..builtin_skills import build_bootstrap_skill_library
from ..datasets import DatasetSource, DatasetSourceType, TrajectoryManifest
from ..execution import ExecutionTick, SkillExecutor
from ..perception import PerceptionBlackboard, Track
from ..perception_service import RealtimePerceptionService
from ..runtime import _accepted_action_provenance, _operator_target_facts
from ..skills import SkillOutcome, SkillRun
from ..trajectory import TrajectoryRecorder, new_trajectory_id
from .bedrock_worlds import world_contract
from .evaluator import BenchmarkReport, BenchmarkRunner, EvaluationEvidence
from .tasks import BenchmarkTask, bedrock_baseline_suite


ARENA_TASK_IDS: tuple[str, ...] = (
    "a_move_forward",
    "a_turn_to_target",
    "a_jump_obstacle",
    "b_ground_log",
    "b_mine_log",
    "b_open_inventory",
)


@dataclass(frozen=True)
class ArenaSkillPhase:
    skill_id: str
    parameters: dict[str, str | int | float | bool] = field(default_factory=dict)


@dataclass(frozen=True)
class ArenaTaskBinding:
    task_id: str
    fixture_id: str
    phases: tuple[ArenaSkillPhase, ...]


ARENA_TASK_BINDINGS: dict[str, ArenaTaskBinding] = {
    "a_move_forward": ArenaTaskBinding(
        task_id="a_move_forward",
        fixture_id="movement-range",
        phases=(ArenaSkillPhase("traverse_level_ground"),),
    ),
    "a_turn_to_target": ArenaTaskBinding(
        task_id="a_turn_to_target",
        fixture_id="target-range",
        phases=(ArenaSkillPhase("reacquire_target", {"target": "marked target"}),),
    ),
    "a_jump_obstacle": ArenaTaskBinding(
        task_id="a_jump_obstacle",
        fixture_id="movement-range",
        phases=(
            ArenaSkillPhase(
                "traverse_visible_obstacle",
                {"allow_attack": False, "allow_use": False, "allow_jump": True},
            ),
        ),
    ),
    "b_ground_log": ArenaTaskBinding(
        task_id="b_ground_log",
        fixture_id="resource-range",
        phases=(ArenaSkillPhase("reacquire_target", {"target": "log"}),),
    ),
    # The generic atomic mining option correctly refuses an unverified
    # ``target.mineable`` claim. The learned wood-acquisition option can act on
    # an explicit visual log reference while the evaluator alone determines
    # whether the block actually broke.
    "b_mine_log": ArenaTaskBinding(
        task_id="b_mine_log",
        fixture_id="resource-range",
        phases=(
            ArenaSkillPhase("gather_nearby_wood"),
        ),
    ),
    "b_open_inventory": ArenaTaskBinding(
        task_id="b_open_inventory",
        fixture_id="gui-range",
        phases=(
            ArenaSkillPhase("open_inventory"),
            ArenaSkillPhase("close_open_inventory"),
        ),
    ),
}


PrepareAttempt = Callable[[BenchmarkTask, int], None]
CollectEvidence = Callable[[BenchmarkTask, int, Path], EvaluationEvidence | None]
SupervisorCommand = Callable[..., dict[str, object]]
TargetLoader = Callable[[], Track | None]


@dataclass(frozen=True)
class ArenaAttempt:
    task_id: str
    repetition: int
    trajectory_id: str
    trajectory_path: Path
    skill_runs: tuple[SkillRun, ...]
    evidence: EvaluationEvidence | None


@dataclass(frozen=True)
class ArenaRun:
    task_id: str
    attempts: tuple[ArenaAttempt, ...]
    report: BenchmarkReport


@dataclass
class BedrockArenaRunner:
    """Execute only the first six frozen Bedrock tasks against one live client.

    Fixture reset and privileged outcome collection stay outside the agent
    observation path. The runner receives those two narrow callbacks, plays a
    typed learned skill episode through the existing supervisor, seals the
    accepted-action trajectory, and only then asks the evaluator for evidence.
    """

    perception: RealtimePerceptionService
    blackboard: PerceptionBlackboard
    executor: SkillExecutor
    lease_id: str
    trajectory_root: Path
    state_db_path: Path
    game_version: str
    instance_id: str
    prepare_attempt: PrepareAttempt
    collect_evidence: CollectEvidence
    supervisor_command: SupervisorCommand
    target_loader: TargetLoader | None = None
    role: str = "arena"
    launcher_profile: str = "bedrock-on-linux/winegdk:arena"
    motor_hz: float = 20.0
    lease_renew_ms: int = 500
    shard_steps: int = 256
    queue_size: int = 512
    clock_ns: Callable[[], int] = time.monotonic_ns
    sleep: Callable[[float], None] = time.sleep
    _sequence: int = field(default=0, init=False)
    _last_target_id: str | None = field(default=None, init=False)
    _lease_stop: threading.Event = field(default_factory=threading.Event, init=False)
    _lease_thread: threading.Thread | None = field(default=None, init=False)
    _lease_fault: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.motor_hz <= 0:
            raise ValueError("arena motor_hz must be positive")
        if self.lease_renew_ms < 100 or self.lease_renew_ms > 2000:
            raise ValueError("arena lease_renew_ms must be in [100, 2000]")
        if not self.lease_id:
            raise ValueError("arena requires an authenticated supervisor lease")

    def run(self, task_id: str, *, repetitions: int = 1) -> ArenaRun:
        if repetitions < 1:
            raise ValueError("arena repetitions must be positive")
        task, binding = arena_task(task_id)
        attempts: list[ArenaAttempt] = []
        evidence_by_trajectory: dict[str, EvaluationEvidence] = {}
        self._start_lease_heartbeat()
        try:
            for repetition in range(repetitions):
                self._require_healthy_lease()
                attempt = self._run_attempt(task, binding, repetition)
                attempts.append(attempt)
                if attempt.evidence is not None:
                    evidence_by_trajectory[attempt.trajectory_id] = attempt.evidence
            self._require_healthy_lease()
            report = BenchmarkRunner(bedrock_baseline_suite()).evaluate_many(
                tuple(attempt.trajectory_path for attempt in attempts),
                evidence_by_trajectory=evidence_by_trajectory,
            )
        finally:
            self._stop_lease_heartbeat()
        return ArenaRun(task_id=task_id, attempts=tuple(attempts), report=report)

    def _run_attempt(
        self,
        task: BenchmarkTask,
        binding: ArenaTaskBinding,
        repetition: int,
    ) -> ArenaAttempt:
        # Revoke any retained inputs before the privileged fixture reset. The
        # independent lease heartbeat keeps the watchdog live even if a real
        # reset/evaluation callback takes longer than one inference interval.
        self.supervisor_command("release-inputs", lease_id=self.lease_id)
        self.prepare_attempt(task, repetition)
        self._require_healthy_lease()
        first_state = self.perception.capture_once()
        first_frame = self.perception.last_capture
        if first_frame is None:
            raise RuntimeError("arena capture did not retain its first frame")
        trajectory_id = new_trajectory_id(f"arena-{task.task_id}-r{repetition:03d}")
        recorder = TrajectoryRecorder(
            manifest=TrajectoryManifest(
                trajectory_id=trajectory_id,
                source=DatasetSource(
                    source_id=f"minecraft-ai:{trajectory_id}",
                    source_type=DatasetSourceType.BEDROCK_AGENT,
                    license="operator-owned-gameplay",
                    redistribution_allowed=False,
                    training_allowed=True,
                    edition="bedrock",
                    game_versions=(self.game_version,),
                ),
                role=self.role,
                label=f"bedrock-arena:{task.task_id}",
                task_id=task.task_id,
                game_version=self.game_version,
                platform=platform.platform(),
                launcher_profile=self.launcher_profile,
                resolution=(first_state.width, first_state.height),
                started_ns=time.time_ns(),
            ),
            artifact_root=self.trajectory_root,
            state_db_path=self.state_db_path,
            shard_steps=self.shard_steps,
            queue_size=self.queue_size,
        )
        runs: list[SkillRun] = []
        phase_index = 0
        deadline_ns = self.clock_ns() + int(task.timeout_s * 1e9)
        self._start_phase(binding, phase_index, repetition)
        try:
            while self.clock_ns() < deadline_ns:
                tick_started_ns = self.clock_ns()
                self._require_healthy_lease()
                self.perception.capture_once()
                self._merge_agent_visible_target()
                self._merge_policy_perception()
                execution = self.executor.tick(
                    self.blackboard,
                    sequence=self._sequence,
                    now_ns=self.clock_ns(),
                )
                self._merge_policy_perception()
                if execution.action is not None:
                    self._send_and_record(execution, recorder)
                if execution.run.outcome != SkillOutcome.RUNNING:
                    runs.append(execution.run)
                    if execution.run.outcome == SkillOutcome.SUCCEEDED and phase_index + 1 < len(
                        binding.phases
                    ):
                        phase_index += 1
                        self._start_phase(binding, phase_index, repetition)
                    else:
                        break
                elapsed_s = (self.clock_ns() - tick_started_ns) / 1e9
                self.sleep(max(0.0, 1.0 / self.motor_hz - elapsed_s))
        finally:
            active = self.executor.run
            if active is not None and active.outcome == SkillOutcome.RUNNING:
                cancelled = self.executor.cancel(now_ns=self.clock_ns())
                runs.append(cancelled.run)
                if cancelled.action is not None:
                    self._send_and_record(cancelled, recorder)
            self.supervisor_command("release-inputs", lease_id=self.lease_id)
            recorder.close()

        trajectory_path = self.trajectory_root / trajectory_id
        # Privileged labels are collected only after the agent-visible episode
        # and trajectory have ended, so they cannot leak into motor decisions.
        evidence = self.collect_evidence(task, repetition, trajectory_path)
        self._require_healthy_lease()
        return ArenaAttempt(
            task_id=task.task_id,
            repetition=repetition,
            trajectory_id=trajectory_id,
            trajectory_path=trajectory_path,
            skill_runs=tuple(runs),
            evidence=evidence,
        )

    def _start_lease_heartbeat(self) -> None:
        if self._lease_thread is not None:
            raise RuntimeError("arena lease heartbeat is already running")
        self._lease_stop.clear()
        self._lease_fault = None
        ttl_ms = min(5000, max(3000, self.lease_renew_ms * 8))
        # Renew synchronously before any potentially slow fixture callback.
        self.supervisor_command("renew", lease_id=self.lease_id, ttl_ms=ttl_ms)

        def renew_until_stopped() -> None:
            interval_s = self.lease_renew_ms / 1000.0
            while not self._lease_stop.wait(interval_s):
                try:
                    self.supervisor_command(
                        "renew",
                        lease_id=self.lease_id,
                        ttl_ms=ttl_ms,
                    )
                except Exception as exc:
                    self._lease_fault = f"{type(exc).__name__}: {exc}"
                    self._lease_stop.set()
                    return

        thread = threading.Thread(
            target=renew_until_stopped,
            name="bedrock-arena-lease",
            daemon=True,
        )
        self._lease_thread = thread
        thread.start()

    def _stop_lease_heartbeat(self) -> None:
        thread = self._lease_thread
        self._lease_thread = None
        self._lease_stop.set()
        if thread is not None:
            thread.join(timeout=max(1.0, self.lease_renew_ms / 500.0))

    def _require_healthy_lease(self) -> None:
        fault = self._lease_fault
        if fault is not None:
            raise RuntimeError(f"arena lease heartbeat failed: {fault}")

    def _start_phase(
        self,
        binding: ArenaTaskBinding,
        phase_index: int,
        repetition: int,
    ) -> None:
        phase = binding.phases[phase_index]
        spec = build_bootstrap_skill_library().get(phase.skill_id)
        self.executor.start(
            spec,
            run_id=(
                f"arena:{binding.task_id}:r{repetition:03d}:p{phase_index:02d}:"
                f"{uuid.uuid4().hex[:12]}"
            ),
            context_key=f"arena:{binding.fixture_id}",
            parameters=phase.parameters,
            now_ns=self.clock_ns(),
        )

    def _merge_policy_perception(self) -> None:
        merge = getattr(self.executor.policy, "merge_perception", None)
        if callable(merge):
            merge(self.blackboard)

    def _merge_agent_visible_target(self) -> None:
        if self.target_loader is None:
            return
        target = self.target_loader()
        if target is None:
            return
        current_hash = self.blackboard.fact("frame.dhash", min_confidence=1.0)
        facts = _operator_target_facts(target, current_hash, now_ns=self.clock_ns())
        if facts:
            self.blackboard.merge_semantics(instance_id=self.instance_id, facts=facts)
        latest = self.blackboard.raw_latest()
        if latest is None or target.track_id == self._last_target_id:
            return
        if self._last_target_id is not None:
            self.blackboard.remove_semantic_track(self._last_target_id)
        if self.blackboard.upsert_semantic_track(instance_id=self.instance_id, track=target):
            self._last_target_id = target.track_id

    def _send_and_record(
        self,
        execution: ExecutionTick,
        recorder: TrajectoryRecorder,
    ) -> None:
        action = execution.action
        if action is None:
            return
        response = self.supervisor_command(
            "motor-action",
            lease_id=self.lease_id,
            action=action.model_dump(mode="json"),
        )
        frame = self.perception.last_capture
        state = self.blackboard.latest()
        if frame is None or state is None:
            raise RuntimeError("arena lost frame/action alignment before recording")
        recorded = recorder.record_accepted(
            action=action,
            provenance=_accepted_action_provenance(
                execution,
                self.blackboard,
                fallback_policy_id=self.executor.policy.policy_id,
            ),
            supervisor_response=response,
            frame=frame,
            blackboard=state,
            skill_run_id=execution.run.run_id,
            skill_id=execution.run.skill_id,
        )
        if not recorded:
            raise RuntimeError("arena could not retain a supervisor-accepted action")
        self._sequence = action.sequence + 1


def arena_task(task_id: str) -> tuple[BenchmarkTask, ArenaTaskBinding]:
    try:
        binding = ARENA_TASK_BINDINGS[task_id]
    except KeyError as exc:
        raise ValueError(
            f"arena task must be one of {', '.join(ARENA_TASK_IDS)}; got {task_id!r}"
        ) from exc
    task = bedrock_baseline_suite().task(task_id)
    contract = world_contract(binding.fixture_id)
    if task.world_fixture_id != contract.world_fixture_id:
        raise RuntimeError("arena task binding does not match frozen world contract")
    return task, binding
