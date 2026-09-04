from __future__ import annotations

from collections import deque
from concurrent.futures import Future
from pathlib import Path
import sqlite3
import threading
import time
from types import SimpleNamespace

import pytest

from minecraft_ai.action_levels import ActionLevel
from minecraft_ai.builtin_skills import build_bootstrap_skill_library
from minecraft_ai.execution import ExecutionTick, SkillExecutor
from minecraft_ai.episodes import RuntimeEventKind
from minecraft_ai.knowledge import Edition, GameVersion, KnowledgeGraph
from minecraft_ai.memory import MemoryKind, MemoryRecord, MemoryStore
from minecraft_ai.models import ModelMessage, ModelResponse
from minecraft_ai.perception import (
    FrameState,
    PerceptionBlackboard,
    PerceptionFact,
    ScreenRegion,
    Track,
)
from minecraft_ai.planning import Goal, GoalScorer
from minecraft_ai.motor import BootstrapMotorPolicy, MotorIntent
from minecraft_ai.platforms.bedrock_x11 import CapturedFrame
from minecraft_ai.roles import BUILTIN_ROLES, get_role
from minecraft_ai.cognition import CognitionDecision, HighLevelController
from minecraft_ai.runtime import (
    AgentRuntime,
    RuntimeMetrics,
    _accepted_action_provenance,
    _authorized_game_chat,
    _active_operator_messages,
    _first_feasible_recovery,
    _observed_scene_recovery,
    _operator_target_facts,
    _selected_operator_message_id,
    _semantic_deadline_ms,
    _semantic_refresh_allowed,
    _skill_stats_totals,
)
from minecraft_ai.safety import MotorAction
from minecraft_ai.social import (
    OperatorMessage,
    OperatorMessageKind,
    OperatorMessageStatus,
    SocialState,
)
from minecraft_ai.skills import (
    SkillLibrary,
    SkillOutcome,
    SkillRun,
    SkillSpec,
    SkillStage,
    SkillStats,
)
from minecraft_ai.storage import StateDatabase
from minecraft_ai.trajectory import ActionOrigin, motor_condition_id


def test_accepted_action_provenance_resolves_bound_model_route_and_condition() -> None:
    board = PerceptionBlackboard()
    board.publish(
        FrameState(
            frame_id=7,
            captured_ns=time.monotonic_ns(),
            instance_id="bedrock:provenance",
            width=1280,
            height=720,
            tracks=(
                Track(
                    track_id="track:weak-log",
                    label="oak_log",
                    confidence=0.7,
                    region=ScreenRegion(x=0.1, y=0.1, width=0.1, height=0.2),
                    first_seen_ns=1,
                    last_seen_ns=1,
                ),
                Track(
                    track_id="track:strong-log",
                    label="oak_log",
                    confidence=0.95,
                    region=ScreenRegion(x=0.5, y=0.2, width=0.2, height=0.4),
                    first_seen_ns=1,
                    last_seen_ns=1,
                ),
            ),
        )
    )
    intent = MotorIntent(
        skill_id="mine_log",
        mode="mine",
        episode_id="run-7",
        action_level=ActionLevel.SKILL,
        instruction="mine log",
        target_label="oak_log",
    )
    causal_condition = intent.model_dump(mode="json")
    causal_condition["action_level"] = ActionLevel.GROUNDED.value
    causal_condition["target_track"] = {
        "track_id": "track:weak-log",
        "label": "oak_log",
        "confidence": 0.7,
    }
    causal_condition["interaction_id"] = 2
    run = SkillRun(run_id="run-7", skill_id="mine_log", started_ns=1)
    execution = ExecutionTick(
        run=run,
        action=MotorAction(sequence=17, keys_down=("w",)),
        motor_intent=intent,
        policy_status={
            "policy_id": "router:test",
            "active_route": "semantic",
            "episode_action_level": "grounded",
            "primary": {
                "policy_id": "learned:minestudio-steve1:steve1-1x",
                "model_version": "steve1-1x",
                "last_action_provenance": {
                    "action_kind": "prediction",
                    "request_id": "request-from-earlier-frame",
                    "prediction_id": "prediction-from-earlier-frame",
                    "action_level": "grounded",
                    "condition": causal_condition,
                    "target_track_id": "track:weak-log",
                    "model_version": "steve1-1x-output",
                    "behavior_token": 91,
                    "latent_id": "z_091",
                },
                "last_prediction": {
                    "behavior_token": 41,
                    "latent_id": "z_041",
                },
            },
        },
        action_origin=ActionOrigin.POLICY,
    )

    provenance = _accepted_action_provenance(
        execution,
        board,
        fallback_policy_id="fallback",
    )

    assert provenance.policy_id == "learned:minestudio-steve1:steve1-1x"
    assert provenance.model_version == "steve1-1x-output"
    assert provenance.route_id == "semantic"
    assert provenance.policy_action_kind == "prediction"
    assert provenance.policy_request_id == "request-from-earlier-frame"
    assert provenance.prediction_id == "prediction-from-earlier-frame"
    assert provenance.action_level == ActionLevel.GROUNDED
    assert provenance.behavior_token == 91
    assert provenance.latent_id == "z_091"
    # The current frame prefers the strong track, but the emitted action came
    # from an asynchronous request conditioned on the earlier weak track.
    assert provenance.target_track_id == "track:weak-log"
    assert provenance.condition == causal_condition
    assert provenance.condition_id == motor_condition_id(
        causal_condition,
        route_id="semantic",
        target_track_id="track:weak-log",
    )


def test_reset_action_does_not_inherit_previous_prediction_condition() -> None:
    board = PerceptionBlackboard()
    intent = MotorIntent(
        skill_id="mine_log",
        mode="mine",
        episode_id="run-7",
        action_level=ActionLevel.GROUNDED,
    )
    execution = ExecutionTick(
        run=SkillRun(run_id="run-7", skill_id="mine_log", started_ns=1),
        action=MotorAction(sequence=18, keys_up=("w",)),
        motor_intent=intent,
        policy_status={
            "policy_id": "router:test",
            "active_route": "semantic",
            "last_action_provenance": {
                "action_kind": "prediction_hold",
                "condition": intent.model_dump(mode="json"),
                "policy_id": "learned:stale",
            },
        },
        action_origin=ActionOrigin.RESET,
    )

    provenance = _accepted_action_provenance(
        execution,
        board,
        fallback_policy_id="fallback",
    )

    assert provenance.origin == ActionOrigin.RESET
    assert provenance.policy_id == "router:test"
    assert provenance.route_id == "reset"
    assert provenance.policy_action_kind == "reset"
    assert provenance.action_level == ActionLevel.GROUNDED
    assert provenance.condition is None
    assert provenance.condition_id is None
    assert provenance.policy_request_id is None
    assert provenance.prediction_id is None
    assert provenance.target_track_id is None


def test_blackboard_rejects_instance_switch_and_stale_fact() -> None:
    board = PerceptionBlackboard(frame_capacity=2)
    now = time.monotonic_ns()
    board.publish(
        FrameState(
            frame_id=1,
            captured_ns=now,
            instance_id="bedrock:1",
            width=1280,
            height=720,
            facts=(
                PerceptionFact(
                    key="crosshair.target",
                    value="oak_log",
                    confidence=0.9,
                    observed_ns=now,
                    source="detector",
                    expires_after_ms=1000,
                ),
            ),
        )
    )
    assert board.fact("crosshair.target", min_confidence=0.8) is not None

    try:
        board.publish(
            FrameState(
                frame_id=2,
                captured_ns=now + 1,
                instance_id="bedrock:2",
                width=1280,
                height=720,
            )
        )
    except ValueError:
        pass
    else:
        raise AssertionError("instance identity changes must be rejected")


def test_game_chat_requires_explicit_perception_authority() -> None:
    decision = CognitionDecision(game_chat="I found the base.")
    board = PerceptionBlackboard()

    assert _authorized_game_chat(decision, board) is None

    now = time.monotonic_ns()
    board.publish(
        FrameState(
            frame_id=1,
            captured_ns=now,
            instance_id="bedrock:chat",
            width=1280,
            height=720,
            facts=(
                PerceptionFact(
                    key="social.player_message",
                    value=True,
                    confidence=0.9,
                    observed_ns=now,
                    source="verified-chat-perception",
                    expires_after_ms=1000,
                ),
            ),
        )
    )

    assert _authorized_game_chat(decision, board) == "I found the base."


def test_operator_console_response_never_authorizes_game_chat() -> None:
    board = PerceptionBlackboard()
    decision = CognitionDecision(say="Working on it.")

    assert _authorized_game_chat(decision, board) is None


def test_verified_death_scene_routes_to_learned_respawn_option() -> None:
    now = time.monotonic_ns()
    board = PerceptionBlackboard()
    board.publish(
        FrameState(
            frame_id=1,
            captured_ns=now,
            instance_id="bedrock:death",
            width=1280,
            height=720,
            facts=(
                PerceptionFact(
                    key="scene.death",
                    value=True,
                    confidence=0.995,
                    observed_ns=now,
                    source="safety:bedrock-hud-v1:not-training-label",
                ),
            ),
        )
    )

    recovery = _observed_scene_recovery(build_bootstrap_skill_library(), board)

    assert recovery is not None
    assert recovery.skill_id == "respawn_after_death"
    assert recovery.policy_ref == "death_gui"
    assert recovery.policy_instruction == "respawn"


def test_learned_inventory_scene_routes_to_learned_inventory_toggle() -> None:
    now = time.monotonic_ns()
    board = PerceptionBlackboard()
    board.publish(
        FrameState(
            frame_id=1,
            captured_ns=now,
            instance_id="bedrock:inventory",
            width=1280,
            height=720,
            facts=(
                PerceptionFact(
                    key="scene.mode",
                    value="inventory",
                    confidence=0.9,
                    observed_ns=now,
                    source="learned:steve1:mineclip-scene:not-training-label",
                ),
                # The claim must be currently observed on the live frame hash.
                PerceptionFact(
                    key="scene.observation_dhash",
                    value="0000000000000000",
                    confidence=1.0,
                    observed_ns=now,
                    source="bootstrap:safety",
                ),
                PerceptionFact(
                    key="frame.dhash",
                    value="0000000000000000",
                    confidence=1.0,
                    observed_ns=now,
                    source="bootstrap:safety",
                ),
            ),
        )
    )

    recovery = _observed_scene_recovery(build_bootstrap_skill_library(), board)

    assert recovery is not None
    assert recovery.skill_id == "close_open_inventory"
    assert recovery.policy_ref == "close_inventory"


def test_stale_inventory_scene_does_not_preempt_world_play() -> None:
    now = time.monotonic_ns()
    board = PerceptionBlackboard()
    board.publish(
        FrameState(
            frame_id=1,
            captured_ns=now,
            instance_id="bedrock:inventory",
            width=1280,
            height=720,
            facts=(
                PerceptionFact(
                    key="scene.mode",
                    value="inventory",
                    confidence=0.9,
                    observed_ns=now,
                    source="learned:steve1:mineclip-scene:not-training-label",
                ),
                # Stale: mode was observed on an old frame, current frame differs.
                PerceptionFact(
                    key="scene.observation_dhash",
                    value="1111111111111111",
                    confidence=1.0,
                    observed_ns=now,
                    source="bootstrap:safety",
                ),
                PerceptionFact(
                    key="frame.dhash",
                    value="0000000000000000",
                    confidence=1.0,
                    observed_ns=now,
                    source="bootstrap:safety",
                ),
            ),
        )
    )

    recovery = _observed_scene_recovery(build_bootstrap_skill_library(), board)

    assert recovery is None


def test_skill_library_uses_smoothed_context_score_and_lifecycle() -> None:
    library = SkillLibrary()
    library.register(
        SkillSpec(
            skill_id="approach",
            name="Approach target",
            expected_effects=("near_target",),
        )
    )
    library.record(
        SkillRun(
            run_id="1",
            skill_id="approach",
            started_ns=1,
            ended_ns=2,
            outcome=SkillOutcome.SUCCEEDED,
        )
    )
    assert 0.5 < library.contextual_score("approach") < 1.0
    promoted = library.promote("approach", SkillStage.EXPERIMENTAL)
    assert promoted.stage == SkillStage.EXPERIMENTAL
    assert promoted.version == 2


def test_builtin_roles_change_goal_utility() -> None:
    assert "farmer" in BUILTIN_ROLES
    farming_goal = Goal(
        goal_id="food",
        description="Build wheat farm",
        source="role",
        priority=0.8,
        domain="farming",
    )
    farmer_score = GoalScorer(get_role("farmer")).score(farming_goal)
    fighter_score = GoalScorer(get_role("fighter")).score(farming_goal)
    assert farmer_score > fighter_score


def test_memory_retrieval_prioritizes_relevant_location_and_goal() -> None:
    store = MemoryStore()
    now = time.monotonic_ns()
    store.upsert(
        MemoryRecord(
            memory_id="iron",
            kind=MemoryKind.SPATIAL,
            text="Iron exposed beside river",
            created_ns=now,
            updated_ns=now,
            goal_tags=("iron",),
            location_key="river",
        )
    )
    store.upsert(
        MemoryRecord(
            memory_id="house",
            kind=MemoryKind.EPISODIC,
            text="Finished roof",
            created_ns=now,
            updated_ns=now,
        )
    )
    result = store.retrieve(goal_tags={"iron"}, location_key="river", limit=1, now_ns=now)
    assert result[0].memory_id == "iron"


def test_associative_recall_ignores_legacy_keepalive_expiry_memory() -> None:
    now = time.time_ns()
    store = MemoryStore()
    store.upsert(
        MemoryRecord(
            memory_id="legacy-expected-expiry",
            kind=MemoryKind.FAILURE,
            text="Observed timed_out for an ordinary keepalive chunk.",
            created_ns=now,
            updated_ns=now,
            source="runtime:verified-skill-outcome",
            metadata={
                "context_key": "explore-keepalive",
                "outcome": "timed_out",
                "skill_id": "traverse_level_ground",
            },
        )
    )
    store.upsert(
        MemoryRecord(
            memory_id="real-failure",
            kind=MemoryKind.FAILURE,
            text="A verified non-keepalive failure.",
            created_ns=now,
            updated_ns=now,
            source="runtime:verified-skill-outcome",
            metadata={"context_key": "operator:test", "outcome": "failed"},
        )
    )

    assert [memory.memory_id for memory in store.retrieve(now_ns=now)] == [
        "real-failure"
    ]


def test_empty_dependency_graph_is_valid() -> None:
    graph = KnowledgeGraph(GameVersion(edition=Edition.BEDROCK, version_id="1.0"))
    assert graph.validate() == []


def test_semantic_request_deadline_is_bounded_below_query_cadence() -> None:
    assert _semantic_deadline_ms(2.0) == 500
    assert _semantic_deadline_ms(0.03) == 10_000
    with pytest.raises(ValueError, match="must be positive"):
        _semantic_deadline_ms(0.0)


def test_zero_semantic_frequency_disables_only_periodic_refresh() -> None:
    runtime = object.__new__(AgentRuntime)
    runtime.semantic_hz = 0.0

    class _Perception:
        active_vlm = object()

        def semantic_available(self) -> bool:
            raise AssertionError("event-only mode must not schedule periodic semantics")

    runtime.perception = _Perception()  # type: ignore[assignment]

    runtime._request_semantics_if_due(frame_id=1)


def test_trajectory_failure_degrades_recording_without_stopping_motor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = time.monotonic_ns()
    board = PerceptionBlackboard()
    board.publish(
        FrameState(
            frame_id=1,
            captured_ns=now,
            instance_id="bedrock:test",
            width=1,
            height=1,
        )
    )

    class _BrokenTrajectory:
        disabled_reason: str | None = None

        def record_accepted(self, **_kwargs: object) -> bool:
            raise OSError("simulated recorder fault")

        def disable(self, reason: str) -> None:
            self.disabled_reason = reason

    trajectory = _BrokenTrajectory()
    runtime = object.__new__(AgentRuntime)
    runtime._stop = threading.Event()
    runtime.blackboard = board
    runtime.perception = SimpleNamespace(
        last_capture=CapturedFrame(1, now, 1, 1, b"\x00\x00\x00\xff")
    )
    runtime.executor = SimpleNamespace(
        policy=SimpleNamespace(policy_id="learned:test"),
        run=None,
    )
    runtime.trajectory = trajectory
    runtime.trajectory_disabled_reason = None
    runtime.lease_id = "lease-test"
    runtime._last_decision = None
    runtime._last_keepalive_skill_id = None
    runtime._sequence = 0
    runtime.metrics = RuntimeMetrics()
    monkeypatch.setattr("minecraft_ai.runtime.operator_pause_latched", lambda: False)
    monkeypatch.setattr(
        "minecraft_ai.runtime.send_command",
        lambda *_args, **_kwargs: {
            "accepted_sequence": 0,
            "accepted_monotonic_ns": now + 1,
        },
    )

    runtime._send_motor(MotorAction(sequence=0))

    assert runtime.metrics.motor_actions == 1
    assert runtime._sequence == 1
    assert trajectory.disabled_reason == "OSError: simulated recorder fault"
    assert runtime.trajectory_disabled_reason == "OSError: simulated recorder fault"


class _AcceptedTrajectorySpy:
    def __init__(self) -> None:
        self.accepted_calls = 0

    def record_accepted(self, **_kwargs: object) -> bool:
        self.accepted_calls += 1
        return True


def _motor_shutdown_runtime() -> tuple[AgentRuntime, _AcceptedTrajectorySpy]:
    now = time.monotonic_ns()
    board = PerceptionBlackboard()
    board.publish(
        FrameState(
            frame_id=1,
            captured_ns=now,
            instance_id="bedrock:shutdown-test",
            width=1,
            height=1,
        )
    )
    trajectory = _AcceptedTrajectorySpy()
    runtime = object.__new__(AgentRuntime)
    runtime._stop = threading.Event()
    runtime.blackboard = board
    runtime.perception = SimpleNamespace(
        last_capture=CapturedFrame(1, now, 1, 1, b"\x00\x00\x00\xff")
    )
    runtime.executor = SimpleNamespace(
        policy=SimpleNamespace(policy_id="learned:test"),
        run=None,
    )
    runtime.trajectory = trajectory  # type: ignore[assignment]
    runtime.trajectory_disabled_reason = None
    runtime.lease_id = "lease-test"
    runtime._last_decision = None
    runtime._sequence = 7
    runtime.metrics = RuntimeMetrics(motor_actions=3)
    return runtime, trajectory


def test_motor_send_noops_when_runtime_is_already_stopping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, trajectory = _motor_shutdown_runtime()
    runtime._stop.set()
    calls: list[str] = []
    monkeypatch.setattr("minecraft_ai.runtime.operator_pause_latched", lambda: False)
    monkeypatch.setattr(
        "minecraft_ai.runtime.send_command",
        lambda command, **_kwargs: calls.append(command),
    )

    runtime._send_motor(MotorAction(sequence=7, keys_down=("w",)))

    assert calls == []
    assert runtime._sequence == 7
    assert runtime.metrics.motor_actions == 3
    assert trajectory.accepted_calls == 0


def test_motor_send_noops_when_operator_pause_is_already_latched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, trajectory = _motor_shutdown_runtime()
    calls: list[str] = []
    monkeypatch.setattr("minecraft_ai.runtime.operator_pause_latched", lambda: True)
    monkeypatch.setattr(
        "minecraft_ai.runtime.send_command",
        lambda command, **_kwargs: calls.append(command),
    )

    runtime._send_motor(MotorAction(sequence=7, buttons_down=("left",)))

    assert calls == []
    assert runtime._stop.is_set()
    assert runtime._sequence == 7
    assert runtime.metrics.motor_actions == 3
    assert trajectory.accepted_calls == 0


def test_motor_send_treats_pause_latched_during_ipc_as_expected_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, trajectory = _motor_shutdown_runtime()
    paused = False

    def pause_during_send(_command: str, **_kwargs: object) -> dict[str, object]:
        nonlocal paused
        paused = True
        raise RuntimeError("RuntimeError: operator pause is latched")

    monkeypatch.setattr("minecraft_ai.runtime.operator_pause_latched", lambda: paused)
    monkeypatch.setattr("minecraft_ai.runtime.send_command", pause_during_send)

    runtime._send_motor(MotorAction(sequence=7, buttons_down=("left",)))

    assert runtime._stop.is_set()
    assert runtime._sequence == 7
    assert runtime.metrics.motor_actions == 3
    assert trajectory.accepted_calls == 0


def test_motor_send_propagates_unrelated_ipc_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, trajectory = _motor_shutdown_runtime()
    monkeypatch.setattr("minecraft_ai.runtime.operator_pause_latched", lambda: False)

    def fail_send(_command: str, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError("transport failure")

    monkeypatch.setattr("minecraft_ai.runtime.send_command", fail_send)

    with pytest.raises(RuntimeError, match="transport failure"):
        runtime._send_motor(MotorAction(sequence=7))

    assert not runtime._stop.is_set()
    assert runtime._sequence == 7
    assert runtime.metrics.motor_actions == 3
    assert trajectory.accepted_calls == 0


def test_motor_send_treats_transient_stop_during_ipc_as_expected_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, trajectory = _motor_shutdown_runtime()
    monkeypatch.setattr("minecraft_ai.runtime.operator_pause_latched", lambda: False)

    def stop_during_send(_command: str, **_kwargs: object) -> dict[str, object]:
        runtime.stop()
        raise FileNotFoundError("control.json")

    monkeypatch.setattr("minecraft_ai.runtime.send_command", stop_during_send)

    runtime._send_motor(MotorAction(sequence=7, keys_up=("w",)))

    assert runtime._stop.is_set()
    assert runtime._sequence == 7
    assert runtime.metrics.motor_actions == 3
    assert trajectory.accepted_calls == 0


def test_run_forever_does_not_failsafe_when_pause_wins_motor_send_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _trajectory = _motor_shutdown_runtime()
    runtime.motor_hz = 20.0
    runtime._lease_thread = None
    runtime._last_renew_ns = 0
    runtime._lease_fault = None
    runtime.perception = SimpleNamespace(
        active_vlm=None,
        last_capture=object(),
        close=lambda: None,
    )
    runtime.executor = SimpleNamespace(
        policy=SimpleNamespace(policy_id="learned:test"),
        run=None,
        close=lambda: None,
    )
    runtime.trajectory = None
    runtime.telemetry = SimpleNamespace(publish=lambda *_args, **_kwargs: None)
    runtime._pool = SimpleNamespace(shutdown=lambda **_kwargs: None)
    runtime._lease_heartbeat = lambda: None  # type: ignore[method-assign]
    runtime._merge_operator_target = lambda: None  # type: ignore[method-assign]
    runtime._start_cognition_if_due = lambda: None  # type: ignore[method-assign]
    runtime._warmup_policy = lambda: None  # type: ignore[method-assign]
    runtime._flush_pending_skill_stats = lambda **_kwargs: None  # type: ignore[method-assign]
    runtime._flush_pending_learning_records = (  # type: ignore[method-assign]
        lambda **_kwargs: None
    )
    runtime._telemetry_payload = lambda **_kwargs: {}  # type: ignore[method-assign]
    failsafe_calls: list[str] = []
    runtime._failsafe = failsafe_calls.append  # type: ignore[method-assign]
    paused = False

    def send(command: str, **_kwargs: object) -> dict[str, object]:
        nonlocal paused
        if command == "motor-action":
            paused = True
            raise RuntimeError("RuntimeError: operator pause is latched")
        return {}

    monkeypatch.setattr("minecraft_ai.runtime.operator_pause_latched", lambda: paused)
    monkeypatch.setattr("minecraft_ai.runtime.send_command", send)
    runtime.tick = lambda: runtime._send_motor(  # type: ignore[method-assign]
        MotorAction(sequence=7, keys_down=("w",))
    )

    runtime.run_forever()

    assert failsafe_calls == []
    assert runtime._stop.is_set()
    assert runtime._sequence == 7
    assert runtime.metrics.motor_actions == 3


def test_optional_semantics_yield_to_cognition_and_operator_work() -> None:
    assert _semantic_refresh_allowed(
        cognition_requested=False,
        cognition_pending=False,
        operator_message_pending=False,
        worker_available=True,
    )
    for blocked in (
        {"cognition_requested": True},
        {"cognition_pending": True},
        {"operator_message_pending": True},
        {"worker_available": False},
    ):
        inputs = {
            "cognition_requested": False,
            "cognition_pending": False,
            "operator_message_pending": False,
            "worker_available": True,
            **blocked,
        }
        assert not _semantic_refresh_allowed(**inputs)


def test_realtime_skill_stats_survive_transient_database_contention() -> None:
    class _FlakyDatabase:
        def __init__(self) -> None:
            self.calls = 0

        def save_skill_stats(
            self,
            skill_id: str,
            context_key: str,
            stats: SkillStats,
        ) -> None:
            del skill_id, context_key, stats
            self.calls += 1
            if self.calls == 1:
                raise sqlite3.OperationalError("database is locked")

    database = _FlakyDatabase()
    runtime = object.__new__(AgentRuntime)
    runtime.state_db = database
    runtime.metrics = RuntimeMetrics()
    runtime._pending_skill_stats = {("navigate", "default"): SkillStats(successes=1)}
    runtime._pending_runtime_events = {}
    runtime._pending_memories = {}
    runtime._pending_operator_status_updates = {}
    runtime._last_storage_retry_ns = 0

    runtime._flush_pending_skill_stats(force=True)

    assert runtime.metrics.storage_contentions == 1
    assert runtime.metrics.last_storage_error == "OperationalError: database is locked"
    assert runtime._pending_skill_stats

    runtime._flush_pending_skill_stats(force=True)

    assert not runtime._pending_skill_stats
    assert runtime.metrics.last_storage_error is None


def test_realtime_operator_ack_survives_transient_database_contention() -> None:
    class _FlakyDatabase:
        def __init__(self) -> None:
            self.calls = 0

        def update_operator_message_status(
            self,
            message_id: str,
            status: OperatorMessageStatus,
            *,
            timestamp_ns: int,
            response_text: str | None = None,
        ) -> None:
            del message_id, status, timestamp_ns, response_text
            self.calls += 1
            if self.calls == 1:
                raise sqlite3.OperationalError("database is locked")

    database = _FlakyDatabase()
    runtime = object.__new__(AgentRuntime)
    runtime.state_db = database
    runtime.metrics = RuntimeMetrics()
    runtime._pending_skill_stats = {}
    runtime._pending_runtime_events = {}
    runtime._pending_memories = {}
    runtime._pending_operator_status_updates = {}
    runtime._last_operator_storage_retry_ns = 0

    persisted = runtime._persist_operator_message_status(
        "operator-1",
        OperatorMessageStatus.ACKNOWLEDGED,
        timestamp_ns=123,
        response_text="I am continuing the current plan.",
    )

    assert not persisted
    assert runtime.metrics.storage_contentions == 1
    assert runtime.metrics.operator_responses == 0
    assert "operator-1" in runtime._pending_operator_status_updates

    runtime._flush_pending_operator_status_updates(force=True)

    assert not runtime._pending_operator_status_updates
    assert runtime.metrics.operator_responses == 1
    assert runtime.metrics.last_storage_error is None


def test_matching_operator_region_publishes_geometry_without_guessing_semantics() -> None:
    target = Track(
        track_id="operator:oak",
        label="oak_log",
        confidence=1.0,
        region=ScreenRegion(x=0.15, y=0.10, width=0.10, height=0.30),
        first_seen_ns=1,
        last_seen_ns=1,
        attributes={
            "source": "operator",
            "reference_dhash": "0123456789abcdef",
        },
    )
    current_hash = PerceptionFact(
        key="frame.dhash",
        value="0123456789abcdef",
        confidence=1.0,
        observed_ns=1,
        source="bootstrap:test",
    )

    facts = {fact.key: fact for fact in _operator_target_facts(target, current_hash, now_ns=2)}

    assert facts["target.visible"].value is True
    assert facts["target.kind"].value == "oak_log"
    assert facts["target.dx"].value == pytest.approx(-0.6)
    assert facts["target.dy"].value == pytest.approx(-0.5)
    assert "target.mineable" not in facts
    assert "target.near" not in facts


def test_changed_frame_invalidates_operator_region_facts() -> None:
    target = Track(
        track_id="operator:oak",
        label="oak_log",
        confidence=1.0,
        region=ScreenRegion(x=0.15, y=0.10, width=0.10, height=0.30),
        first_seen_ns=1,
        last_seen_ns=1,
        attributes={"source": "operator", "reference_dhash": "0000000000000000"},
    )
    changed = PerceptionFact(
        key="frame.dhash",
        value="ffffffffffffffff",
        confidence=1.0,
        observed_ns=1,
        source="bootstrap:test",
    )

    assert _operator_target_facts(target, changed, now_ns=2) == ()


def test_changed_frame_preserves_verified_cross_view_reference_fact(tmp_path) -> None:
    reference = tmp_path / "target.jpg"
    reference.write_bytes(b"reference")
    target = Track(
        track_id="operator:oak",
        label="oak_log",
        confidence=1.0,
        region=ScreenRegion(x=0.15, y=0.10, width=0.10, height=0.30),
        first_seen_ns=1,
        last_seen_ns=1,
        attributes={
            "source": "operator",
            "reference_dhash": "0000000000000000",
            "reference_image_path": str(reference),
            "reference_image_sha256": "a" * 64,
        },
    )
    changed = PerceptionFact(
        key="frame.dhash",
        value="ffffffffffffffff",
        confidence=1.0,
        observed_ns=1,
        source="bootstrap:test",
    )

    facts = _operator_target_facts(target, changed, now_ns=2)

    assert tuple(fact.key for fact in facts) == ("target.reference_available",)
    assert facts[0].value is True


def test_recovery_selection_requires_observed_option_preconditions() -> None:
    skills = build_bootstrap_skill_library()
    recoveries = ("escape_submersion", "retreat_from_danger")
    board = PerceptionBlackboard()
    now = time.monotonic_ns()
    board.publish(
        FrameState(
            frame_id=1,
            captured_ns=now,
            instance_id="bedrock:test",
            width=1280,
            height=635,
        )
    )

    assert _first_feasible_recovery(skills, recoveries, board) is None

    board.merge_semantics(
        instance_id="bedrock:test",
        facts=(
            PerceptionFact(
                key="environment.underwater",
                value=True,
                confidence=0.995,
                observed_ns=now,
                source="safety:test",
            ),
            PerceptionFact(
                key="danger.immediate",
                value=True,
                confidence=0.995,
                observed_ns=now,
                source="safety:test",
            ),
        ),
    )

    selected = _first_feasible_recovery(skills, recoveries, board)
    assert selected is not None
    assert selected.skill_id == "escape_submersion"


def test_newest_acknowledged_operator_directive_remains_active() -> None:
    messages = (
        OperatorMessage(
            message_id="instruction",
            created_ns=1,
            text="Collect three logs",
            status=OperatorMessageStatus.ACKNOWLEDGED,
        ),
        OperatorMessage(
            message_id="question",
            created_ns=2,
            text="What can you see?",
            kind=OperatorMessageKind.QUESTION,
            status=OperatorMessageStatus.ACKNOWLEDGED,
        ),
        OperatorMessage(
            message_id="archived",
            created_ns=3,
            text="Old instruction",
            status=OperatorMessageStatus.ARCHIVED,
        ),
    )

    active = _active_operator_messages(messages)

    assert tuple(message.message_id for message in active) == ("instruction",)


def test_fresh_correction_supersedes_older_acknowledged_instruction() -> None:
    messages = (
        OperatorMessage(
            message_id="old",
            created_ns=1,
            text="Keep collecting logs",
            priority=0.8,
            status=OperatorMessageStatus.ACKNOWLEDGED,
        ),
        OperatorMessage(
            message_id="correction",
            created_ns=2,
            text="Stop and climb the hill",
            kind=OperatorMessageKind.CORRECTION,
            priority=1.0,
            status=OperatorMessageStatus.DELIVERED,
        ),
    )

    active = _active_operator_messages(messages)

    assert tuple(message.message_id for message in active) == ("correction",)


def test_fresh_directive_precedes_an_older_acknowledged_correction() -> None:
    messages = (
        OperatorMessage(
            message_id="old-correction",
            created_ns=1,
            text="Leave the canopy and explore",
            kind=OperatorMessageKind.CORRECTION,
            priority=1.0,
            status=OperatorMessageStatus.ACKNOWLEDGED,
        ),
        OperatorMessage(
            message_id="new-instruction",
            created_ns=2,
            text="Mine the selected tree",
            priority=1.0,
            status=OperatorMessageStatus.DELIVERED,
        ),
    )

    active = _active_operator_messages(messages)

    assert tuple(message.message_id for message in active) == ("new-instruction",)


def test_newer_acknowledged_instruction_supersedes_old_high_priority_correction() -> None:
    messages = (
        OperatorMessage(
            message_id="old-correction",
            created_ns=1,
            text="Keep gathering the selected log",
            kind=OperatorMessageKind.CORRECTION,
            priority=1.0,
            status=OperatorMessageStatus.ACKNOWLEDGED,
        ),
        OperatorMessage(
            message_id="new-instruction",
            created_ns=2,
            text="Leave the pit and traverse open ground",
            priority=0.8,
            status=OperatorMessageStatus.ACKNOWLEDGED,
        ),
    )

    active = _active_operator_messages(messages)

    assert tuple(message.message_id for message in active) == ("new-instruction",)


def test_only_selected_operator_goal_is_acknowledgeable() -> None:
    pending = ("new-correction", "old-instruction")

    assert (
        _selected_operator_message_id(
            CognitionDecision(chosen_goal_id="operator:new-correction"),
            pending,
        )
        == "new-correction"
    )
    assert (
        _selected_operator_message_id(
            CognitionDecision(chosen_goal_id="role:generalist:0:survive"),
            pending,
        )
        is None
    )


def test_player_chat_authorizes_world_chat_reply() -> None:
    from minecraft_ai.cognition import CognitionDecision
    from minecraft_ai.perception import ChatLine, FrameState, PerceptionBlackboard
    from minecraft_ai.runtime import _authorized_game_chat

    now = time.monotonic_ns()
    board = PerceptionBlackboard()
    board.publish(
        FrameState(
            frame_id=1,
            captured_ns=now,
            instance_id="bedrock:world",
            width=1280,
            height=720,
            chat=(
                ChatLine(
                    speaker="Steve",
                    text="how do I craft a chest?",
                    observed_ns=now,
                    confidence=0.95,
                ),
            ),
        )
    )
    board.merge_semantics(
        instance_id="bedrock:world",
        facts=(
            PerceptionFact(
                key="social.player_message",
                value="Steve:how do I craft a chest?",
                confidence=0.95,
                observed_ns=now,
                source="grounded:player-chat",
                expires_after_ms=30_000,
            ),
        ),
    )

    decision = CognitionDecision(skill_id="explore_forward", game_chat="Planks x8 gives a chest.")
    assert _authorized_game_chat(decision, board) == "Planks x8 gives a chest."
    # Re-answer is blocked after the reply timestamp advances.
    assert (
        _authorized_game_chat(decision, board, already_replied_ns=now + 1)
        is None
    )
    # A new player line (newer observation) authorizes again.
    board.merge_semantics(
        instance_id="bedrock:world",
        facts=(
            PerceptionFact(
                key="social.player_message",
                value="Steve:wiki cheats?",
                confidence=0.95,
                observed_ns=now + 100,
                source="grounded:player-chat",
                expires_after_ms=30_000,
            ),
        ),
    )
    assert (
        _authorized_game_chat(decision, board, already_replied_ns=now + 1)
        == "Planks x8 gives a chest."
    )


def test_no_game_chat_without_authority() -> None:
    from minecraft_ai.cognition import CognitionDecision
    from minecraft_ai.perception import FrameState, PerceptionBlackboard
    from minecraft_ai.runtime import _authorized_game_chat

    now = time.monotonic_ns()
    board = PerceptionBlackboard()
    board.publish(
        FrameState(frame_id=1, captured_ns=now, instance_id="bedrock:world", width=1280, height=720)
    )
    decision = CognitionDecision(skill_id="explore_forward", game_chat="answer")
    assert _authorized_game_chat(decision, board) is None


def test_persistent_plan_adoption_and_advancement() -> None:
    from minecraft_ai.runtime import AgentRuntime

    runtime = object.__new__(AgentRuntime)
    runtime._plan_steps = ()
    runtime._plan_goal_id = None
    runtime._plan_index = 0
    runtime._plan_started_ns = 0
    runtime._plan_step_completed_ns = 0
    runtime._last_decision = None

    # Adopt a fresh plan for a goal.
    decision = CognitionDecision(
        chosen_goal_id="progression:build",
        plan_steps=("gather wood", "craft planks", "place blocks"),
    )
    runtime._adopt_plan_if_revised(decision)
    assert runtime._plan_steps == ("gather wood", "craft planks", "place blocks")
    assert runtime._plan_goal_id == "progression:build"
    assert runtime._plan_index == 0
    assert runtime._plan_started_ns > 0

    # A skill success advances the plan index under the same goal.
    runtime._last_decision = decision
    runtime._advance_plan_on_step_complete(_run("gather_wood"))
    assert runtime._plan_index == 1

    # Re-declaring the same remaining plan (no goal change, not exhausted,
    # identical remaining steps) must not reset the index (no thrash).
    echo = CognitionDecision(
        chosen_goal_id="progression:build",
        plan_steps=("gather wood", "craft planks", "place blocks"),
    )
    runtime._adopt_plan_if_revised(echo)
    assert runtime._plan_index == 1

    # Exhausting the plan lets a new decision revive/restart it.
    runtime._plan_index = 3
    renewed = CognitionDecision(
        chosen_goal_id="progression:build",
        plan_steps=("expand house", "add roof"),
    )
    runtime._adopt_plan_if_revised(renewed)
    assert runtime._plan_steps == ("expand house", "add roof")
    assert runtime._plan_index == 0

    # Success under an operator override goal does not consume plan progress.
    runtime._plan_index = 0
    runtime._plan_goal_id = "progression:build"
    runtime._last_decision = CognitionDecision(chosen_goal_id="operator:abc")
    runtime._advance_plan_on_step_complete(_run("navigate"))
    assert runtime._plan_index == 0


def _run(skill_id: str):
    from minecraft_ai.skills import SkillOutcome, SkillRun

    return SkillRun(
        run_id="r",
        skill_id=skill_id,
        started_ns=1,
        ended_ns=2,
        outcome=SkillOutcome.SUCCEEDED,
    )


def test_cognition_skips_while_plan_executing() -> None:
    from minecraft_ai.runtime import AgentRuntime
    from minecraft_ai.skills import SkillRun

    runtime = object.__new__(AgentRuntime)
    runtime._cognition_requested = False
    runtime._plan_steps = ("a", "b")
    runtime._plan_index = 0

    class _Executor:
        run = SkillRun(run_id="r", skill_id="explore_forward", started_ns=1)

    runtime.executor = _Executor()  # type: ignore[assignment]
    runtime._pending_decision = None

    assert runtime._cognition_due(operator_waiting=False) is False


def test_cognition_due_when_plan_exhausted() -> None:
    from minecraft_ai.runtime import AgentRuntime
    from minecraft_ai.skills import SkillRun

    runtime = object.__new__(AgentRuntime)
    runtime._cognition_requested = False
    runtime._plan_steps = ("a",)
    runtime._plan_index = 1  # exhausted

    class _Executor:
        run = SkillRun(run_id="r", skill_id="explore_forward", started_ns=1)

    runtime.executor = _Executor()  # type: ignore[assignment]
    runtime._pending_decision = None

    assert runtime._cognition_due(operator_waiting=False) is True


def _runtime_for_learning(database: StateDatabase) -> AgentRuntime:
    runtime = object.__new__(AgentRuntime)
    runtime.skills = SkillLibrary()
    runtime.memories = MemoryStore()
    runtime.state_db = database
    runtime.trajectory = None
    runtime.metrics = RuntimeMetrics()
    runtime._recorded_run_ids = set()
    runtime._recorded_run_order = deque(maxlen=4_096)
    runtime._recent_skill_runs = deque(maxlen=8)
    runtime._last_keepalive_skill_id = None
    runtime._pending_skill_stats = {}
    runtime._pending_runtime_events = {}
    runtime._pending_memories = {}
    runtime._pending_operator_status_updates = {}
    runtime._last_storage_retry_ns = 0
    runtime._last_decision = None
    runtime._plan_steps = ()
    runtime._plan_goal_id = None
    runtime._plan_index = 0
    runtime._plan_step_completed_ns = 0
    return runtime


def test_terminal_runs_persist_stats_events_and_factual_memories(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite3"
    with StateDatabase(path) as database:
        runtime = _runtime_for_learning(database)
        runs = (
            SkillRun(
                run_id="timeout-1",
                skill_id="traverse_level_ground",
                context_key="test-terminal",
                started_ns=10,
                ended_ns=20,
                outcome=SkillOutcome.TIMED_OUT,
                failure_reason="skill-timeout",
            ),
            SkillRun(
                run_id="timeout-2",
                skill_id="traverse_level_ground",
                context_key="test-terminal",
                started_ns=20,
                ended_ns=30,
                outcome=SkillOutcome.TIMED_OUT,
                failure_reason="skill-timeout",
            ),
            SkillRun(
                run_id="failed-1",
                skill_id="traverse_level_ground",
                context_key="test-terminal",
                started_ns=30,
                ended_ns=40,
                outcome=SkillOutcome.FAILED,
                failure_reason="failure-condition:danger.immediate",
            ),
            SkillRun(
                run_id="success-1",
                skill_id="traverse_level_ground",
                context_key="test-terminal",
                started_ns=40,
                ended_ns=50,
                outcome=SkillOutcome.SUCCEEDED,
            ),
            SkillRun(
                run_id="cancelled-1",
                skill_id="traverse_level_ground",
                context_key="test-terminal",
                started_ns=50,
                ended_ns=60,
                outcome=SkillOutcome.CANCELLED,
                failure_reason="cancelled",
            ),
        )
        for run in runs:
            runtime._record_terminal_run(run)
        runtime._record_terminal_run(runs[2])

        assert runtime.metrics.skill_successes == 1
        assert runtime.metrics.skill_failed_outcomes == 1
        assert runtime.metrics.skill_timeouts == 2
        assert runtime.metrics.skill_cancellations == 1
        assert runtime.metrics.skill_failures == 3
        assert not runtime._pending_skill_stats
        assert not runtime._pending_runtime_events
        assert not runtime._pending_memories

    with StateDatabase(path) as database:
        stats = database.load_skills().stats[
            ("traverse_level_ground", "test-terminal")
        ]
        events = database.load_runtime_events(limit=10)
        memories = tuple(database.load_memories().records.values())

    assert (stats.successes, stats.failures, stats.timeouts, stats.cancellations) == (1, 1, 2, 1)
    assert {event.kind for event in events} == {
        RuntimeEventKind.SKILL_SUCCEEDED,
        RuntimeEventKind.SKILL_FAILED,
        RuntimeEventKind.SKILL_TIMED_OUT,
        RuntimeEventKind.SKILL_CANCELLED,
    }
    assert len(events) == 5
    assert all(event.observed_ns > 1_000_000_000_000_000_000 for event in events)
    failure_memories = [memory for memory in memories if memory.kind == MemoryKind.FAILURE]
    procedural_memories = [memory for memory in memories if memory.kind == MemoryKind.PROCEDURAL]
    assert len(failure_memories) == 2
    assert len(procedural_memories) == 1
    timeout_memory = next(
        memory for memory in failure_memories if memory.metadata["outcome"] == "timed_out"
    )
    assert timeout_memory.metadata["occurrences"] == 2
    assert timeout_memory.metadata["reported_reason"] == "skill-timeout"
    assert "cause" not in timeout_memory.text.casefold()
    assert procedural_memories[0].metadata["occurrences"] == 1
    assert procedural_memories[0].text.startswith("Verified success")
    assert all(memory.created_ns > 1_000_000_000_000_000_000 for memory in memories)


def test_keepalive_timeout_is_an_event_not_a_permanent_failure_memory(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.sqlite3"
    with StateDatabase(path) as database:
        runtime = _runtime_for_learning(database)
        run = SkillRun(
            run_id="expected-expiry",
            skill_id="traverse_level_ground",
            context_key="explore-keepalive",
            started_ns=10,
            ended_ns=20,
            outcome=SkillOutcome.TIMED_OUT,
            failure_reason="skill-timeout",
        )
        runtime._record_terminal_run(run)

    with StateDatabase(path) as database:
        stats = database.load_skills().stats[
            ("traverse_level_ground", "explore-keepalive")
        ]
        events = database.load_runtime_events(limit=10)
        memories = tuple(database.load_memories().records.values())

    assert stats.timeouts == 1
    assert [event.kind for event in events] == [RuntimeEventKind.SKILL_TIMED_OUT]
    assert not memories
    assert not runtime._recent_skill_runs


def test_real_scene_recovery_timeout_in_keepalive_context_remains_learning_evidence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.sqlite3"
    with StateDatabase(path) as database:
        runtime = _runtime_for_learning(database)
        run = SkillRun(
            run_id="submersion-timeout",
            skill_id="escape_submersion",
            context_key="explore-keepalive",
            started_ns=10,
            ended_ns=20,
            outcome=SkillOutcome.TIMED_OUT,
            failure_reason="skill-timeout",
        )
        runtime._record_terminal_run(run)

    assert runtime._recent_skill_runs[0] == run
    assert any(memory.kind == MemoryKind.FAILURE for memory in runtime.memories.records.values())


def test_terminal_run_deduplication_window_is_bounded(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite3"
    with StateDatabase(path) as database:
        runtime = _runtime_for_learning(database)
        for index in range(4_100):
            runtime._record_terminal_run(
                SkillRun(
                    run_id=f"cancelled-{index}",
                    skill_id="traverse_level_ground",
                    context_key="bounded-dedup",
                    started_ns=index * 2 + 1,
                    ended_ns=index * 2 + 2,
                    outcome=SkillOutcome.CANCELLED,
                    failure_reason="cancelled",
                )
            )

        assert len(runtime._recorded_run_ids) == 4_096
        assert len(runtime._recorded_run_order) == 4_096
        cancellations = runtime.metrics.skill_cancellations
        latest = SkillRun(
            run_id="cancelled-4099",
            skill_id="traverse_level_ground",
            context_key="bounded-dedup",
            started_ns=8_199,
            ended_ns=8_200,
            outcome=SkillOutcome.CANCELLED,
            failure_reason="cancelled",
        )
        runtime._record_terminal_run(latest)
        assert runtime.metrics.skill_cancellations == cancellations


def test_skill_totals_distinguish_exact_lifetime_outcomes() -> None:
    assert _skill_stats_totals(
        (
            SkillStats(successes=2, failures=3, timeouts=4, cancellations=5),
            SkillStats(successes=7, failures=11, timeouts=13, cancellations=17),
        )
    ) == {
        "succeeded": 9,
        "failed": 14,
        "timed_out": 17,
        "cancelled": 22,
        "attempts": 62,
    }


def test_keepalive_rotates_away_from_persisted_consecutive_failures(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.sqlite3"
    bootstrap = build_bootstrap_skill_library()
    with StateDatabase(path) as database:
        for skill_id in ("traverse_level_ground", "explore_forward"):
            database.save_skill(bootstrap.get(skill_id))
        database.save_skill_stats(
            "traverse_level_ground",
            "explore-keepalive",
            SkillStats(timeouts=7, consecutive_failures=7),
        )

    with StateDatabase(path) as database:
        persisted = database.load_skills()
    runtime = object.__new__(AgentRuntime)
    runtime.skills = persisted

    selected = runtime._explore_keep_alive()

    assert selected is not None
    assert selected.skill_id == "explore_forward"


def test_static_failed_keepalive_rotates_between_existing_vpt_recoveries(
    tmp_path: Path,
) -> None:
    database = StateDatabase(tmp_path / "state.sqlite3")
    runtime = _runtime_for_learning(database)
    runtime.skills = build_bootstrap_skill_library()
    runtime.skills.stats[("traverse_level_ground", "explore-keepalive")] = SkillStats(
        timeouts=466,
        consecutive_failures=466,
    )
    runtime.skills.stats[("explore_forward", "explore-keepalive")] = SkillStats(
        timeouts=54,
        consecutive_failures=54,
    )
    runtime._keepalive_stagnant_failures = 1
    runtime._record_terminal_run(
        SkillRun(
            run_id="semantic-timeout",
            skill_id="explore_forward",
            context_key="explore-keepalive",
            started_ns=1,
            ended_ns=2,
            outcome=SkillOutcome.TIMED_OUT,
        )
    )

    first = runtime._explore_keep_alive()
    assert first is not None
    assert first.skill_id == "traverse_visible_obstacle"

    runtime._record_terminal_run(
        SkillRun(
            run_id="obstacle-timeout",
            skill_id="traverse_visible_obstacle",
            context_key="explore-keepalive",
            started_ns=2,
            ended_ns=3,
            outcome=SkillOutcome.TIMED_OUT,
        )
    )
    second = runtime._explore_keep_alive()
    assert second is not None
    assert second.skill_id == "traverse_level_ground"
    database.close()


def test_keepalive_stagnation_uses_visual_hash_displacement() -> None:
    now = time.monotonic_ns()
    runtime = object.__new__(AgentRuntime)
    runtime.blackboard = PerceptionBlackboard()
    runtime.blackboard.publish(
        FrameState(
            frame_id=1,
            captured_ns=now,
            instance_id="bedrock:stagnation",
            width=1280,
            height=720,
            facts=(
                PerceptionFact(
                    key="frame.dhash",
                    value="ae579e554aa525a6",
                    confidence=1.0,
                    observed_ns=now,
                    source="bootstrap:bootstrap-rgb-v1:not-training-label",
                ),
            ),
        )
    )
    runtime._keepalive_start_dhash = "ae579e554aa525a6"
    runtime._keepalive_stagnant_failures = 0

    runtime._update_keepalive_stagnation(
        SkillRun(
            run_id="static-timeout",
            skill_id="explore_forward",
            context_key="explore-keepalive",
            started_ns=1,
            ended_ns=2,
            outcome=SkillOutcome.TIMED_OUT,
        )
    )

    assert runtime._keepalive_stagnant_failures == 1

    later = now + 1
    runtime.blackboard.publish(
        FrameState(
            frame_id=2,
            captured_ns=later,
            instance_id="bedrock:stagnation",
            width=1280,
            height=720,
            facts=(
                PerceptionFact(
                    key="frame.dhash",
                    value="ffffffffffffffff",
                    confidence=1.0,
                    observed_ns=later,
                    source="bootstrap:bootstrap-rgb-v1:not-training-label",
                ),
            ),
        )
    )
    runtime._keepalive_start_dhash = "0000000000000000"
    runtime._update_keepalive_stagnation(
        SkillRun(
            run_id="moving-timeout",
            skill_id="traverse_visible_obstacle",
            context_key="explore-keepalive",
            started_ns=2,
            ended_ns=3,
            outcome=SkillOutcome.TIMED_OUT,
        )
    )

    assert runtime._keepalive_stagnant_failures == 0


def test_keepalive_timeout_does_not_stale_pending_operator_cognition() -> None:
    runtime = object.__new__(AgentRuntime)
    runtime._execution_revision = 7
    runtime._cognition_requested = False
    runtime._pending_decision = object()  # type: ignore[assignment]
    keepalive = SkillRun(
        run_id="keepalive-timeout",
        skill_id="explore_forward",
        context_key="explore-keepalive",
        started_ns=1,
        ended_ns=2,
        outcome=SkillOutcome.TIMED_OUT,
    )

    runtime._note_terminal_for_cognition(keepalive, recovery_started=False)

    assert runtime._execution_revision == 7
    assert runtime._cognition_requested is False

    runtime._note_terminal_for_cognition(keepalive, recovery_started=True)
    assert runtime._execution_revision == 8
    assert runtime._cognition_requested is True


def test_real_skill_terminal_still_invalidates_pending_cognition() -> None:
    runtime = object.__new__(AgentRuntime)
    runtime._execution_revision = 4
    runtime._cognition_requested = False
    runtime._pending_decision = object()  # type: ignore[assignment]

    runtime._note_terminal_for_cognition(
        SkillRun(
            run_id="real-timeout",
            skill_id="gather_nearby_wood",
            context_key="operator:mine-tree",
            started_ns=1,
            ended_ns=2,
            outcome=SkillOutcome.TIMED_OUT,
        ),
        recovery_started=False,
    )

    assert runtime._execution_revision == 5
    assert runtime._cognition_requested is True


def test_same_skill_cognition_takes_ownership_from_disposable_keepalive() -> None:
    runtime = object.__new__(AgentRuntime)
    runtime.skills = build_bootstrap_skill_library()
    runtime.executor = SkillExecutor(BootstrapMotorPolicy())
    runtime.executor.start(
        runtime.skills.get("explore_forward"),
        run_id="keepalive-run",
        context_key="explore-keepalive",
    )
    decision = CognitionDecision(
        skill_id="explore_forward",
        instruction="Explore toward open ground.",
    )
    future: Future[CognitionDecision] = Future()
    future.set_result(decision)
    runtime._pending_decision = future
    runtime._pending_execution_revision = 3
    runtime._execution_revision = 3
    runtime._pending_operator_message_ids = ()
    runtime.state_db = None
    runtime.blackboard = PerceptionBlackboard()
    runtime.metrics = RuntimeMetrics()
    runtime._last_decision = None
    runtime._last_cognition_ns = 0
    runtime._queued_operator_message_waiting = lambda: False  # type: ignore[method-assign]
    runtime._adopt_plan_if_revised = lambda _decision: None  # type: ignore[method-assign]
    terminal: list[SkillRun] = []
    runtime._record_terminal_run = terminal.append  # type: ignore[method-assign]
    runtime._send_motor = lambda *_args, **_kwargs: None  # type: ignore[method-assign]

    runtime._consume_cognition()

    assert terminal[0].run_id == "keepalive-run"
    assert terminal[0].outcome == SkillOutcome.CANCELLED
    assert runtime.executor.run is not None
    assert runtime.executor.run.run_id != "keepalive-run"
    assert runtime.executor.run.context_key == "default"
    assert runtime.executor.instruction == "Explore toward open ground."


def _runtime_with_completed_decision(
    decision: CognitionDecision,
    *,
    database: StateDatabase | None = None,
    pending_message_ids: tuple[str, ...] = (),
) -> AgentRuntime:
    runtime = object.__new__(AgentRuntime)
    future: Future[CognitionDecision] = Future()
    future.set_result(decision)
    runtime._pending_decision = future
    runtime._pending_execution_revision = 0
    runtime._execution_revision = 0
    runtime._pending_operator_message_ids = pending_message_ids
    runtime._cognition_requested = False
    runtime._cognition_retry_count = 0
    runtime._cognition_retry_not_before_ns = 0
    runtime._last_cognition_ns = 0
    runtime._last_decision = None
    runtime._last_player_chat_replied_ns = None
    runtime.state_db = database
    runtime.blackboard = PerceptionBlackboard()
    runtime.metrics = RuntimeMetrics()
    runtime._pending_skill_stats = {}
    runtime._pending_runtime_events = {}
    runtime._pending_memories = {}
    runtime._pending_operator_status_updates = {}
    runtime._adopt_plan_if_revised = lambda _decision: None  # type: ignore[method-assign]
    return runtime


def test_request_replan_rearms_cognition_with_bounded_backoff() -> None:
    runtime = _runtime_with_completed_decision(
        CognitionDecision(
            reasoning_summary="Structured decision failed closed.",
            request_replan=True,
        )
    )

    runtime._consume_cognition()

    assert runtime._cognition_requested is True
    assert runtime._cognition_retry_count == 1
    assert runtime._cognition_retry_not_before_ns > runtime._last_cognition_ns
    retry_deadline = runtime._cognition_retry_not_before_ns

    runtime._pending_decision = None
    runtime._start_cognition_if_due()

    assert runtime._pending_decision is None
    assert runtime._cognition_retry_not_before_ns == retry_deadline


def test_cognition_retry_backoff_is_exponential_and_capped() -> None:
    runtime = object.__new__(AgentRuntime)
    runtime._cognition_retry_count = 0
    runtime._cognition_retry_not_before_ns = 0
    runtime._cognition_requested = False
    delays: list[int] = []

    for attempt in range(8):
        now = attempt * 100
        runtime._schedule_cognition_retry(now_ns=now)
        delays.append(runtime._cognition_retry_not_before_ns - now)

    assert delays[:5] == [
        2_000_000_000,
        4_000_000_000,
        8_000_000_000,
        16_000_000_000,
        30_000_000_000,
    ]
    assert delays[5:] == [30_000_000_000] * 3


def test_new_queued_operator_message_bypasses_old_backoff_once(tmp_path: Path) -> None:
    class _RecordingPool:
        def __init__(self) -> None:
            self.calls = 0

        def submit(self, _function, *_args):  # type: ignore[no-untyped-def]
            self.calls += 1
            return Future()

    database = StateDatabase(tmp_path / "state.sqlite3")
    try:
        message = OperatorMessage(
            message_id="new-instruction",
            created_ns=1,
            text="Climb toward the surface.",
        )
        database.save_operator_message(message)
        runtime = object.__new__(AgentRuntime)
        runtime._pending_decision = None
        runtime._cognition_requested = True
        runtime._cognition_retry_count = 5
        runtime._cognition_retry_not_before_ns = 2**63 - 1
        runtime._last_cognition_ns = 0
        runtime._execution_revision = 3
        runtime._pending_operator_message_ids = ()
        runtime._pending_operator_status_updates = {}
        runtime._pending_skill_stats = {}
        runtime._pending_runtime_events = {}
        runtime._pending_memories = {}
        runtime.state_db = database
        runtime.high_level = SimpleNamespace(decide=lambda *_args: None)
        runtime.blackboard = PerceptionBlackboard()
        runtime.cognition_hz = 0.03
        runtime.metrics = RuntimeMetrics()
        pool = _RecordingPool()
        runtime._pool = pool  # type: ignore[assignment]
        runtime._cognition_context = lambda: SimpleNamespace(  # type: ignore[method-assign]
            operator_messages=(message,)
        )

        runtime._start_cognition_if_due()

        assert pool.calls == 1
        assert runtime._pending_decision is not None
        assert runtime._cognition_retry_count == 0
        assert database.load_operator_messages(limit=1)[0].status == (
            OperatorMessageStatus.DELIVERED
        )

        # Once delivered, the same message no longer bypasses a failed
        # snapshot's retry deadline.
        runtime._pending_decision = None
        runtime._cognition_requested = True
        runtime._cognition_retry_not_before_ns = 2**63 - 1
        runtime._start_cognition_if_due()
        assert pool.calls == 1
    finally:
        database.close()


class _OperatorFastPathOnlyModel:
    model_id = "operator-fast-path-only"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages: tuple[ModelMessage, ...]) -> ModelResponse:
        del messages
        self.calls += 1
        raise AssertionError("operator preemption must not invoke the slow model")


def _runtime_with_inflight_operator_cognition(
    database: StateDatabase,
    *,
    message_text: str,
    status: OperatorMessageStatus = OperatorMessageStatus.QUEUED,
    target_reference: bool = True,
    danger: bool = False,
    sampled_message: bool = False,
) -> tuple[
    AgentRuntime,
    Future[CognitionDecision],
    _OperatorFastPathOnlyModel,
    list[SkillRun],
    list[MotorAction],
]:
    now = time.monotonic_ns()
    facts: list[PerceptionFact] = []
    if target_reference:
        facts.append(
            PerceptionFact(
                key="target.reference_available",
                value=True,
                confidence=1.0,
                observed_ns=now,
                source="operator",
                expires_after_ms=10_000,
            )
        )
    blackboard = PerceptionBlackboard()
    blackboard.publish(
        FrameState(
            frame_id=1,
            captured_ns=now,
            instance_id="bedrock:operator-preemption",
            width=1280,
            height=720,
            facts=tuple(facts),
            tracks=(
                Track(
                    track_id="operator:marked-dirt",
                    label="dirt",
                    confidence=1.0,
                    region=ScreenRegion(x=0.3, y=0.2, width=0.4, height=0.6),
                    first_seen_ns=now,
                    last_seen_ns=now,
                    attributes={"source": "operator"},
                ),
            ),
        )
    )
    message = OperatorMessage(
        message_id="new-operator-correction",
        created_ns=now,
        text=message_text,
        kind=OperatorMessageKind.CORRECTION,
        status=status,
    )
    database.save_operator_message(message)
    skills = build_bootstrap_skill_library()
    executor = SkillExecutor(BootstrapMotorPolicy())
    executor.start(
        skills.get("explore_forward"),
        run_id="disposable-keepalive",
        context_key="explore-keepalive",
    )
    keepalive_tick = executor.tick(blackboard, sequence=0, now_ns=now + 1)
    assert keepalive_tick.action is not None
    assert "w" in keepalive_tick.action.keys_down
    if danger:
        blackboard.merge_semantics(
            instance_id="bedrock:operator-preemption",
            facts=(
                PerceptionFact(
                    key="danger.immediate",
                    value=True,
                    confidence=1.0,
                    observed_ns=now + 2,
                    source="test",
                    expires_after_ms=10_000,
                ),
            ),
        )
    model = _OperatorFastPathOnlyModel()
    stale_future: Future[CognitionDecision] = Future()
    assert stale_future.set_running_or_notify_cancel()
    terminal: list[SkillRun] = []
    sent_actions: list[MotorAction] = []

    runtime = object.__new__(AgentRuntime)
    runtime.blackboard = blackboard
    runtime.executor = executor
    runtime.skills = skills
    runtime.role = get_role("generalist")
    runtime.memories = MemoryStore()
    runtime.social = SocialState()
    runtime.custom_goals = []
    runtime.state_db = database
    runtime.high_level = HighLevelController(model, skills)
    runtime._pending_decision = stale_future
    runtime._pending_execution_revision = 0
    runtime._execution_revision = 0
    runtime._pending_operator_message_ids = (
        (message.message_id,) if sampled_message else ()
    )
    runtime._pending_operator_status_updates = {}
    runtime._pending_skill_stats = {}
    runtime._pending_runtime_events = {}
    runtime._pending_memories = {}
    runtime._recent_skill_runs = deque(maxlen=8)
    runtime._cognition_requested = False
    runtime._cognition_retry_count = 0
    runtime._cognition_retry_not_before_ns = 0
    runtime._last_cognition_ns = 0
    runtime._last_player_chat_replied_ns = None
    runtime._last_decision = None
    runtime._plan_steps = ()
    runtime._plan_goal_id = None
    runtime._plan_index = 0
    runtime._plan_started_ns = 0
    runtime.metrics = RuntimeMetrics()
    runtime._record_terminal_run = terminal.append  # type: ignore[method-assign]
    runtime._send_motor = (  # type: ignore[method-assign]
        lambda action, **_kwargs: sent_actions.append(action)
    )
    return runtime, stale_future, model, terminal, sent_actions


@pytest.mark.parametrize(
    ("status", "sampled_message"),
    (
        (OperatorMessageStatus.QUEUED, False),
        (OperatorMessageStatus.DELIVERED, False),
        # A target can become available after this same delivered message was
        # already captured in a slow, pre-grounding cognition snapshot.
        (OperatorMessageStatus.DELIVERED, True),
    ),
)
def test_actionable_operator_command_preempts_inflight_cognition_and_keepalive(
    tmp_path: Path,
    status: OperatorMessageStatus,
    sampled_message: bool,
) -> None:
    with StateDatabase(tmp_path / "state.sqlite3") as database:
        runtime, stale_future, model, terminal, sent_actions = (
            _runtime_with_inflight_operator_cognition(
                database,
                message_text="Mine the marked dirt block.",
                status=status,
                sampled_message=sampled_message,
            )
        )

        runtime._start_cognition_if_due()

        # Python cannot interrupt a running worker thread, but its stale result
        # is detached and can no longer delay or mutate live runtime state.
        assert stale_future.running()
        assert runtime._pending_decision is None
        assert model.calls == 0
        assert terminal[0].run_id == "disposable-keepalive"
        assert terminal[0].outcome == SkillOutcome.CANCELLED
        assert "w" in sent_actions[0].keys_up
        assert runtime.executor.run is not None
        assert runtime.executor.run.skill_id == "mine_visible_block"
        assert runtime.executor.parameters == {"target": "dirt"}
        assert runtime.executor.instruction == "Mine the marked dirt block."
        assert runtime._last_decision is not None
        assert runtime._last_decision.chosen_goal_id == "operator:new-operator-correction"
        persisted = database.load_operator_messages(limit=1)[0]
        assert persisted.status == OperatorMessageStatus.ACKNOWLEDGED
        assert persisted.response_text == "Starting that now."


@pytest.mark.parametrize(
    ("message_text", "target_reference"),
    (
        ("Move forward through safe open terrain.", True),
        ("Mine the marked dirt block.", False),
    ),
)
def test_ambiguous_or_infeasible_operator_command_keeps_inflight_cognition(
    tmp_path: Path,
    message_text: str,
    target_reference: bool,
) -> None:
    with StateDatabase(tmp_path / "state.sqlite3") as database:
        runtime, stale_future, model, terminal, sent_actions = (
            _runtime_with_inflight_operator_cognition(
                database,
                message_text=message_text,
                target_reference=target_reference,
            )
        )

        runtime._start_cognition_if_due()

        assert runtime._pending_decision is stale_future
        assert stale_future.running()
        assert model.calls == 0
        assert terminal == []
        assert sent_actions == []
        assert runtime.executor.run is not None
        assert runtime.executor.run.run_id == "disposable-keepalive"


def test_urgent_safety_blocks_operator_fast_path_preemption(
    tmp_path: Path,
) -> None:
    with StateDatabase(tmp_path / "state.sqlite3") as database:
        runtime, stale_future, model, terminal, sent_actions = (
            _runtime_with_inflight_operator_cognition(
                database,
                message_text="Mine the marked dirt block.",
                danger=True,
            )
        )

        runtime._start_cognition_if_due()

        assert runtime._pending_decision is stale_future
        assert model.calls == 0
        assert runtime.executor.run is not None
        assert runtime.executor.run.run_id == "disposable-keepalive"
        assert terminal == []
        assert sent_actions == []
        assert database.load_operator_messages(limit=1)[0].status == (
            OperatorMessageStatus.QUEUED
        )


def test_delivered_operator_message_remains_waiting_until_acknowledged(tmp_path: Path) -> None:
    database = StateDatabase(tmp_path / "state.sqlite3")
    try:
        database.save_operator_message(
            OperatorMessage(
                message_id="climb-out",
                created_ns=1,
                text="Look up and build stairs to the surface.",
                status=OperatorMessageStatus.DELIVERED,
                delivered_ns=2,
            )
        )
        runtime = object.__new__(AgentRuntime)
        runtime.state_db = database
        runtime._pending_operator_message_ids = ("climb-out",)

        assert runtime._queued_operator_message_waiting() is True
        assert runtime._operator_message_arrived_after_snapshot() is False

        database.save_operator_message(
            OperatorMessage(
                message_id="new-correction",
                created_ns=3,
                text="Stop if you see lava.",
                kind=OperatorMessageKind.CORRECTION,
            )
        )

        assert runtime._operator_message_arrived_after_snapshot() is True
    finally:
        database.close()


def test_pending_operator_message_is_not_hidden_by_newer_acknowledged_history(
    tmp_path: Path,
) -> None:
    database = StateDatabase(tmp_path / "state.sqlite3")
    try:
        database.save_operator_message(
            OperatorMessage(
                message_id="old-pending",
                created_ns=1,
                text="Keep climbing toward the surface.",
                status=OperatorMessageStatus.DELIVERED,
                delivered_ns=2,
            )
        )
        for index in range(2, 22):
            database.save_operator_message(
                OperatorMessage(
                    message_id=f"acknowledged-{index}",
                    created_ns=index,
                    text=f"Previous instruction {index}.",
                    status=OperatorMessageStatus.ACKNOWLEDGED,
                    delivered_ns=index,
                    acknowledged_ns=index,
                )
            )

        runtime = object.__new__(AgentRuntime)
        runtime.state_db = database
        runtime.role = get_role("generalist")
        runtime.custom_goals = []
        runtime.memories = MemoryStore()
        runtime.social = SocialState()
        runtime._recent_skill_runs = deque(maxlen=8)
        runtime._plan_steps = ()
        runtime._plan_index = 0
        runtime._plan_goal_id = None
        runtime._plan_started_ns = 0

        context = runtime._cognition_context()

        assert tuple(message.message_id for message in context.operator_messages) == (
            "old-pending",
        )
    finally:
        database.close()


def test_replan_does_not_acknowledge_delivered_operator_message(tmp_path: Path) -> None:
    database = StateDatabase(tmp_path / "state.sqlite3")
    try:
        message = OperatorMessage(
            message_id="climb-out",
            created_ns=1,
            text="Look up and build stairs to the surface.",
            status=OperatorMessageStatus.DELIVERED,
            delivered_ns=2,
        )
        database.save_operator_message(message)
        runtime = _runtime_with_completed_decision(
            CognitionDecision(
                chosen_goal_id="operator:climb-out",
                reasoning_summary="Invalid structured response; retry.",
                request_replan=True,
            ),
            database=database,
            pending_message_ids=("climb-out",),
        )

        runtime._consume_cognition()

        persisted = database.load_operator_messages(limit=1)[0]
        assert persisted.status == OperatorMessageStatus.DELIVERED
        assert runtime._cognition_requested is True
        assert runtime._cognition_retry_count == 1
    finally:
        database.close()


def test_valid_decision_acknowledges_its_own_delivered_message(tmp_path: Path) -> None:
    database = StateDatabase(tmp_path / "state.sqlite3")
    try:
        message = OperatorMessage(
            message_id="status-question",
            created_ns=1,
            text="What are you doing?",
            kind=OperatorMessageKind.QUESTION,
            status=OperatorMessageStatus.DELIVERED,
            delivered_ns=2,
        )
        database.save_operator_message(message)
        runtime = _runtime_with_completed_decision(
            CognitionDecision(
                chosen_goal_id="operator:status-question",
                reasoning_summary="Continuing safe exploration.",
                say="I am exploring for a route out.",
            ),
            database=database,
            pending_message_ids=("status-question",),
        )

        runtime._consume_cognition()

        persisted = database.load_operator_messages(limit=1)[0]
        assert persisted.status == OperatorMessageStatus.ACKNOWLEDGED
        assert persisted.response_text == "I am exploring for a route out."
        assert runtime._cognition_requested is False
        assert runtime._cognition_retry_count == 0
    finally:
        database.close()


def test_valid_acknowledgement_drains_next_message_without_failure_backoff(
    tmp_path: Path,
) -> None:
    database = StateDatabase(tmp_path / "state.sqlite3")
    try:
        for message_id, created_ns in (("first", 2), ("second", 1)):
            database.save_operator_message(
                OperatorMessage(
                    message_id=message_id,
                    created_ns=created_ns,
                    text=f"Handle {message_id} request.",
                    status=OperatorMessageStatus.DELIVERED,
                    delivered_ns=created_ns,
                )
            )
        runtime = _runtime_with_completed_decision(
            CognitionDecision(
                chosen_goal_id="operator:first",
                reasoning_summary="Handled the first request.",
                say="First request handled.",
            ),
            database=database,
            pending_message_ids=("first", "second"),
        )

        runtime._consume_cognition()

        messages = {
            message.message_id: message for message in database.load_operator_messages(limit=10)
        }
        assert messages["first"].status == OperatorMessageStatus.ACKNOWLEDGED
        assert messages["second"].status == OperatorMessageStatus.DELIVERED
        assert runtime._cognition_requested is True
        assert runtime._cognition_retry_count == 0
        delay = runtime._cognition_retry_not_before_ns - runtime._last_cognition_ns
        assert 0 < delay <= 250_000_000
    finally:
        database.close()
