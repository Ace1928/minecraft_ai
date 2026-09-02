from __future__ import annotations

import time

from minecraft_ai.knowledge import Edition, GameVersion, KnowledgeGraph
from minecraft_ai.memory import MemoryKind, MemoryRecord, MemoryStore
from minecraft_ai.perception import FrameState, PerceptionBlackboard, PerceptionFact
from minecraft_ai.planning import Goal, GoalScorer
from minecraft_ai.roles import BUILTIN_ROLES, get_role
from minecraft_ai.cognition import CognitionDecision
from minecraft_ai.runtime import (
    _active_operator_messages,
    _selected_operator_message_id,
    _semantic_deadline_ms,
)
from minecraft_ai.social import OperatorMessage, OperatorMessageKind, OperatorMessageStatus
from minecraft_ai.skills import SkillLibrary, SkillOutcome, SkillRun, SkillSpec, SkillStage


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


def test_acknowledged_operator_directive_remains_active_until_archived() -> None:
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


def test_urgent_correction_precedes_older_acknowledged_instruction() -> None:
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

    assert tuple(message.message_id for message in active) == ("correction", "old")


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

    assert tuple(message.message_id for message in active) == (
        "new-instruction",
        "old-correction",
    )


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
