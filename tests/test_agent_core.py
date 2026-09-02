from __future__ import annotations

import sqlite3
import time

import pytest

from minecraft_ai.builtin_skills import build_bootstrap_skill_library
from minecraft_ai.knowledge import Edition, GameVersion, KnowledgeGraph
from minecraft_ai.memory import MemoryKind, MemoryRecord, MemoryStore
from minecraft_ai.perception import (
    FrameState,
    PerceptionBlackboard,
    PerceptionFact,
    ScreenRegion,
    Track,
)
from minecraft_ai.planning import Goal, GoalScorer
from minecraft_ai.roles import BUILTIN_ROLES, get_role
from minecraft_ai.cognition import CognitionDecision
from minecraft_ai.runtime import (
    AgentRuntime,
    RuntimeMetrics,
    _authorized_game_chat,
    _active_operator_messages,
    _first_feasible_recovery,
    _observed_scene_recovery,
    _operator_target_facts,
    _selected_operator_message_id,
    _semantic_deadline_ms,
    _semantic_refresh_allowed,
)
from minecraft_ai.social import OperatorMessage, OperatorMessageKind, OperatorMessageStatus
from minecraft_ai.skills import (
    SkillLibrary,
    SkillOutcome,
    SkillRun,
    SkillSpec,
    SkillStage,
    SkillStats,
)


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
            ),
        )
    )

    recovery = _observed_scene_recovery(build_bootstrap_skill_library(), board)

    assert recovery is not None
    assert recovery.skill_id == "close_open_inventory"
    assert recovery.policy_ref == "close_inventory"


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


def test_empty_dependency_graph_is_valid() -> None:
    graph = KnowledgeGraph(GameVersion(edition=Edition.BEDROCK, version_id="1.0"))
    assert graph.validate() == []


def test_semantic_request_deadline_is_bounded_below_query_cadence() -> None:
    assert _semantic_deadline_ms(2.0) == 500
    assert _semantic_deadline_ms(0.03) == 10_000


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
    runtime._last_storage_retry_ns = 0

    runtime._flush_pending_skill_stats(force=True)

    assert runtime.metrics.storage_contentions == 1
    assert runtime.metrics.last_storage_error == "OperationalError: database is locked"
    assert runtime._pending_skill_stats

    runtime._flush_pending_skill_stats(force=True)

    assert not runtime._pending_skill_stats
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
