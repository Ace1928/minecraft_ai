from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import pytest

import minecraft_ai.eval.arena as arena_module
from minecraft_ai.eval.arena import (
    ARENA_TASK_IDS,
    ARENA_TASK_BINDINGS,
    BedrockArenaRunner,
    arena_task,
)
from minecraft_ai.eval.evaluator import EvaluationEvidence, EvaluationStatus
from minecraft_ai.eval.tasks import BenchmarkTask
from minecraft_ai.execution import SkillExecutor
from minecraft_ai.motor import MotorIntent
from minecraft_ai.perception import PerceptionBlackboard
from minecraft_ai.perception_service import RealtimePerceptionService
from minecraft_ai.platforms.bedrock_x11 import CapturedFrame
from minecraft_ai.safety import MotorAction
from minecraft_ai.skills import SkillOutcome
from minecraft_ai.trajectory import TrajectoryReader, TrajectoryRecorder


@dataclass
class _Clock:
    value: int = field(default_factory=time.monotonic_ns)

    def now(self) -> int:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += max(1, int(seconds * 1e9))


@dataclass
class _Capture:
    clock: _Clock
    events: list[str]
    frame_id: int = 0

    def capture(self) -> CapturedFrame:
        self.events.append("capture")
        self.clock.advance(0.001)
        self.frame_id += 1
        return CapturedFrame(
            frame_id=self.frame_id,
            captured_ns=self.clock.now(),
            width=4,
            height=3,
            bgra=bytes((25, 50, 75, 255)) * 12,
        )

    def close(self) -> None:
        return


@dataclass
class _ForwardPolicy:
    policy_id: str = "test:vpt-fast-body"
    last_sequence: int = -1
    emitted_forward: bool = False

    def act(
        self,
        blackboard: PerceptionBlackboard,
        intent: MotorIntent,
        *,
        sequence: int,
    ) -> MotorAction:
        del blackboard, intent
        self.last_sequence = sequence
        if not self.emitted_forward:
            self.emitted_forward = True
            return MotorAction(sequence=sequence, keys_down=("w",))
        return MotorAction(sequence=sequence)

    def reset(self) -> MotorAction:
        self.last_sequence += 1
        return MotorAction(sequence=self.last_sequence, keys_up=("w",))


def _runner(tmp_path: Path) -> tuple[BedrockArenaRunner, list[str], PerceptionBlackboard]:
    clock = _Clock()
    events: list[str] = []
    blackboard = PerceptionBlackboard()
    perception = RealtimePerceptionService(
        capture_source=_Capture(clock, events),
        blackboard=blackboard,
        instance_id="bedrock:test-arena",
        fast_perception=None,
    )
    policy = _ForwardPolicy()
    executor = SkillExecutor(policy)

    def supervisor(command: str, **payload: object) -> dict[str, object]:
        events.append(f"supervisor:{command}")
        if command == "motor-action":
            raw = payload["action"]
            assert isinstance(raw, dict)
            action = MotorAction.model_validate(raw)
            return {
                "accepted_sequence": action.sequence,
                "accepted_monotonic_ns": clock.now() + 1,
            }
        return {"ok": True}

    def prepare(task: BenchmarkTask, repetition: int) -> None:
        assert task.task_id == "a_move_forward"
        assert repetition == 0
        events.append("prepare")

    def evidence(
        task: BenchmarkTask,
        repetition: int,
        trajectory: Path,
    ) -> EvaluationEvidence:
        assert task.task_id == "a_move_forward"
        assert repetition == 0
        assert trajectory.is_dir()
        events.append("evaluate")
        return EvaluationEvidence(
            source="controlled-world:test-movement-range",
            metrics={"event.destination_reached": 1},
            artifact_refs=("fixture://movement-range/r000",),
        )

    return (
        BedrockArenaRunner(
            perception=perception,
            blackboard=blackboard,
            executor=executor,
            lease_id="arena-lease",
            trajectory_root=tmp_path / "trajectories",
            state_db_path=tmp_path / "state.sqlite3",
            game_version="1.test",
            instance_id="bedrock:test-arena",
            prepare_attempt=prepare,
            collect_evidence=evidence,
            supervisor_command=supervisor,
            motor_hz=1.0,
            lease_renew_ms=500,
            shard_steps=16,
            queue_size=32,
            clock_ns=clock.now,
            sleep=clock.advance,
        ),
        events,
        blackboard,
    )


def test_first_arena_task_plays_records_then_scores_without_label_leakage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        arena_module,
        "TrajectoryRecorder",
        partial(TrajectoryRecorder, min_free_disk_bytes=0),
    )
    runner, events, blackboard = _runner(tmp_path)

    result = runner.run("a_move_forward")

    assert result.task_id == "a_move_forward"
    assert result.report.results[0].status == EvaluationStatus.PASSED
    assert result.report.results[0].metrics["action.forward_presses"] == 1
    assert result.attempts[0].skill_runs[-1].outcome == SkillOutcome.TIMED_OUT
    assert events.index("prepare") < events.index("capture") < events.index("evaluate")
    assert blackboard.fact("event.destination_reached") is None

    reader = TrajectoryReader(result.attempts[0].trajectory_path)
    assert reader.manifest.task_id == "a_move_forward"
    validation = reader.validate()
    assert validation.valid
    samples = tuple(reader.iter_samples())
    assert samples
    assert samples[0].step.policy_id == "test:vpt-fast-body"
    assert samples[0].step.skill_id == "traverse_level_ground"
    assert samples[0].step.action.keys_down == ("w",)


def test_arena_scope_is_exactly_the_six_declared_tasks() -> None:
    assert tuple(ARENA_TASK_BINDINGS) == ARENA_TASK_IDS
    for task_id in ARENA_TASK_IDS:
        task, binding = arena_task(task_id)
        assert task.task_id == binding.task_id
        assert task.world_fixture_id == binding.fixture_id
        assert binding.phases

    with pytest.raises(ValueError, match="arena task must be one of"):
        arena_task("c_build_wall")


def test_inventory_task_is_an_explicit_open_then_close_sequence() -> None:
    binding = ARENA_TASK_BINDINGS["b_open_inventory"]

    assert tuple(phase.skill_id for phase in binding.phases) == (
        "open_inventory",
        "close_open_inventory",
    )
