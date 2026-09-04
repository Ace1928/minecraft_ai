from __future__ import annotations

import time

import pytest

from minecraft_ai.builtin_skills import build_bootstrap_skill_library
from minecraft_ai.crafting_control import BoundedPlankCraftController, PlankCraftPhase
from minecraft_ai.execution import SkillExecutor
from minecraft_ai.motor import MotorIntent
from minecraft_ai.outcome_verifier import OutcomeSignal
from minecraft_ai.perception import (
    EvidenceRegion,
    FrameState,
    PerceptionBlackboard,
    PerceptionEvidence,
    PerceptionFact,
    ScreenRegion,
    Track,
)
from minecraft_ai.safety import MotorAction
from minecraft_ai.skills import SkillOutcome
from minecraft_ai.runtime import AgentRuntime, RuntimeMetrics


_SECOND = 1_000_000_000


class _UnusedPolicy:
    policy_id = "unused:test"

    def __init__(self) -> None:
        self.last_sequence = -1

    def act(
        self,
        _blackboard: PerceptionBlackboard,
        _intent: MotorIntent,
        *,
        sequence: int,
    ) -> MotorAction:
        raise AssertionError("bounded crafting must not delegate GUI input to the policy")

    def reset(self) -> MotorAction:
        self.last_sequence += 1
        return MotorAction(sequence=max(0, self.last_sequence))


def _fact(
    key: str,
    value: str | int | float | bool,
    *,
    observed_ns: int,
    evidence_refs: tuple[str, ...] = (),
    source: str = "vlm:test",
) -> PerceptionFact:
    return PerceptionFact(
        key=key,
        value=value,
        confidence=0.95,
        observed_ns=observed_ns,
        source=source,
        expires_after_ms=60_000,
        evidence_refs=evidence_refs,
    )


def _board(now_ns: int) -> PerceptionBlackboard:
    board = PerceptionBlackboard()
    board.publish(
        FrameState(
            frame_id=0,
            captured_ns=now_ns,
            instance_id="bedrock:test",
            width=1280,
            height=720,
            facts=(
                _fact("scene.mode", "world", observed_ns=now_ns),
                _fact("scene.playable", True, observed_ns=now_ns),
            ),
        )
    )
    return board


def _gui_evidence(observed_ns: int, suffix: str) -> PerceptionEvidence:
    return PerceptionEvidence(
        evidence_id=f"frame-gui:{suffix}",
        frame_id=0,
        captured_ns=observed_ns,
        region_kind=EvidenceRegion.GUI,
        region=ScreenRegion(x=0.1, y=0.05, width=0.8, height=0.9),
        pixel_sha256="0" * 64,
        crop_width=1024,
        crop_height=648,
    )


def _merge_inventory(
    board: PerceptionBlackboard,
    *,
    observed_ns: int,
    logs: int,
    planks: int,
    recipe: bool,
    recipe_label: str = "craftable_planks_recipe",
    captured_ns: int | None = None,
) -> None:
    evidence = _gui_evidence(
        observed_ns if captured_ns is None else captured_ns,
        str(observed_ns),
    )
    references = (evidence.evidence_id,)
    tracks = (
        (
            Track(
                track_id=f"recipe:{observed_ns}",
                label=recipe_label,
                confidence=0.95,
                region=ScreenRegion(x=0.2, y=0.2, width=0.1, height=0.1),
                first_seen_ns=observed_ns,
                last_seen_ns=observed_ns,
                evidence_refs=references,
            ),
        )
        if recipe
        else ()
    )
    board.merge_semantics(
        instance_id="bedrock:test",
        facts=(
            _fact("scene.mode", "gui", observed_ns=observed_ns, evidence_refs=references),
            _fact("gui.mode", "inventory", observed_ns=observed_ns, evidence_refs=references),
            _fact("inventory.logs", logs, observed_ns=observed_ns, evidence_refs=references),
            _fact(
                "inventory.planks",
                planks,
                observed_ns=observed_ns,
                evidence_refs=references,
            ),
        ),
        tracks=tracks,
        evidence=(evidence,),
    )


def _merge_fast_overlay(board: PerceptionBlackboard, *, observed_ns: int) -> None:
    board.merge_semantics(
        instance_id="bedrock:test",
        facts=(
            _fact(
                "scene.ui_overlay",
                True,
                observed_ns=observed_ns,
                source="bootstrap:bootstrap-rgb-v1:not-training-label",
            ),
            _fact(
                "scene.playable",
                False,
                observed_ns=observed_ns,
                source="bootstrap:bootstrap-rgb-v1:not-training-label",
            ),
        ),
    )


def test_fresh_world_skill_opens_crafts_verifies_delta_and_closes() -> None:
    now = time.monotonic_ns()
    board = _board(now)
    executor = SkillExecutor(_UnusedPolicy())
    executor.start(
        build_bootstrap_skill_library().get("craft_wood_planks"),
        run_id="craft:one",
        now_ns=now - 1,
    )

    opened = executor.tick(board, sequence=1, now_ns=now)

    assert opened.run.outcome == SkillOutcome.RUNNING
    assert opened.action == MotorAction(
        sequence=1,
        keys_down=("e",),
        keys_up=("e",),
        duration_ms=50,
    )

    _merge_fast_overlay(board, observed_ns=now + 100_000_000)
    awaiting_semantics = executor.tick(
        board,
        sequence=2,
        now_ns=now + 100_000_000,
    )
    assert awaiting_semantics.action is None

    _merge_inventory(
        board,
        observed_ns=now + 70 * _SECOND,
        logs=2,
        planks=0,
        recipe=True,
    )
    clicked = executor.tick(board, sequence=3, now_ns=now + 70 * _SECOND)

    assert clicked.run.outcome == SkillOutcome.RUNNING
    assert clicked.action is not None
    assert clicked.action.buttons_down == clicked.action.buttons_up == ("right",)
    assert clicked.action.camera_semantics == "cursor"
    assert clicked.action.cursor_x == 0.25
    assert clicked.action.cursor_y == 0.25
    assert clicked.action.duration_ms == 50

    _merge_inventory(
        board,
        observed_ns=now + 140 * _SECOND,
        logs=1,
        planks=4,
        recipe=False,
    )
    closed = executor.tick(board, sequence=4, now_ns=now + 140 * _SECOND)

    assert closed.run.outcome == SkillOutcome.RUNNING
    assert closed.action == MotorAction(
        sequence=4,
        keys_down=("e",),
        keys_up=("e",),
        duration_ms=50,
    )

    board.merge_semantics(
        instance_id="bedrock:test",
        facts=(
            _fact("scene.mode", "world", observed_ns=now + 141 * _SECOND),
            _fact("scene.playable", True, observed_ns=now + 141 * _SECOND),
        ),
    )
    finished = executor.tick(board, sequence=5, now_ns=now + 141 * _SECOND)

    assert finished.run.outcome == SkillOutcome.SUCCEEDED
    assert finished.outcome_verification is not None
    assert finished.outcome_verification.signal == OutcomeSignal.PLANKS_CRAFTED
    assert finished.action is not None and finished.action.sequence == 5


def test_runtime_does_not_submit_crafting_semantics_before_its_gui_toggle() -> None:
    now = time.monotonic_ns()
    board = _board(now)
    skills = build_bootstrap_skill_library()
    executor = SkillExecutor(_UnusedPolicy())
    executor.start(
        skills.get("craft_wood_planks"),
        run_id="craft:query-order",
        now_ns=now - 1,
    )

    class _Perception:
        active_vlm = object()

        def __init__(self) -> None:
            self.requests: list[object] = []

        def semantic_available(self) -> bool:
            return True

        def request_semantics(self, query: object) -> bool:
            self.requests.append(query)
            return True

    perception = _Perception()
    runtime = object.__new__(AgentRuntime)
    runtime.blackboard = board
    runtime.skills = skills
    runtime.executor = executor
    runtime.perception = perception  # type: ignore[assignment]
    runtime.semantic_hz = 0.0
    runtime._cognition_requested = False
    runtime._pending_decision = None
    runtime._pending_operator_message_ids = ()
    runtime._last_semantic_ns = 0
    runtime.metrics = RuntimeMetrics()

    runtime._request_semantics_if_due(frame_id=1)
    assert perception.requests == []

    opened = executor.tick(board, sequence=1, now_ns=now)
    assert opened.action is not None and opened.action.keys_down == ("e",)
    runtime._request_semantics_if_due(frame_id=2)
    assert perception.requests == []

    _merge_fast_overlay(board, observed_ns=now + 100_000_000)
    runtime._request_semantics_if_due(frame_id=3)

    assert len(perception.requests) == 1


def test_runtime_reserves_first_post_click_semantic_request_for_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = time.monotonic_ns()
    clock = [now]
    monkeypatch.setattr("minecraft_ai.runtime.time.monotonic_ns", lambda: clock[0])
    board = _board(now)
    skills = build_bootstrap_skill_library()
    executor = SkillExecutor(_UnusedPolicy())
    executor.start(
        skills.get("craft_wood_planks"),
        run_id="craft:post-click-query",
        now_ns=now - 1,
    )

    class _Perception:
        active_vlm = object()

        def __init__(self) -> None:
            self.requests: list[object] = []

        def semantic_available(self) -> bool:
            return True

        def request_semantics(self, query: object) -> bool:
            self.requests.append(query)
            return True

    perception = _Perception()
    runtime = object.__new__(AgentRuntime)
    runtime.blackboard = board
    runtime.skills = skills
    runtime.executor = executor
    runtime.perception = perception  # type: ignore[assignment]
    runtime.semantic_hz = 0.0
    runtime._cognition_requested = False
    runtime._pending_decision = None
    runtime._pending_operator_message_ids = ()
    runtime._last_semantic_ns = 0
    runtime.metrics = RuntimeMetrics()

    opened = executor.tick(board, sequence=1, now_ns=now)
    assert opened.action is not None and opened.action.keys_down == ("e",)
    clock[0] = now + 100_000_000
    _merge_fast_overlay(board, observed_ns=clock[0])

    # This is the one request that discovers the log baseline and recipe.
    runtime._request_semantics_if_due(frame_id=2)
    assert len(perception.requests) == 1
    executor.tick(board, sequence=2, now_ns=clock[0])

    clock[0] = now + 3 * _SECOND
    _merge_inventory(
        board,
        observed_ns=clock[0],
        logs=2,
        planks=0,
        recipe=True,
    )

    # Scheduling precedes control on every runtime loop. The click-ready facts
    # must suppress a redundant request from this still-pre-click frame.
    runtime._request_semantics_if_due(frame_id=3)
    assert len(perception.requests) == 1
    clicked = executor.tick(board, sequence=3, now_ns=clock[0])
    assert clicked.action is not None and clicked.action.buttons_down == ("right",)

    # The next captured frame is the first request after interaction.
    clock[0] += 100_000_000
    runtime._request_semantics_if_due(frame_id=4)
    assert len(perception.requests) == 2
    assert perception.requests[-1].frame_id == 4


def test_near_miss_recipe_label_never_authorizes_click() -> None:
    now = time.monotonic_ns()
    board = _board(now)
    controller = BoundedPlankCraftController(retry_interval_ms=0)
    controller.step(board, run_id="craft:near-miss", sequence=1, now_ns=now)
    _merge_fast_overlay(board, observed_ns=now + 100_000_000)
    controller.step(
        board,
        run_id="craft:near-miss",
        sequence=2,
        now_ns=now + 100_000_000,
    )
    _merge_inventory(
        board,
        observed_ns=now + _SECOND,
        logs=2,
        planks=0,
        recipe=True,
        recipe_label="uncraftable_planks_recipe",
    )

    step = controller.step(
        board,
        run_id="craft:near-miss",
        sequence=3,
        now_ns=now + _SECOND,
    )

    assert step.action is None
    assert controller.phase == PlankCraftPhase.LOCATE_RECIPE


def test_existing_planks_without_log_consumption_are_not_success() -> None:
    now = time.monotonic_ns()
    board = _board(now)
    controller = BoundedPlankCraftController()
    controller.step(board, run_id="craft:one", sequence=1, now_ns=now)
    _merge_fast_overlay(board, observed_ns=now + 100_000_000)
    controller.step(
        board,
        run_id="craft:one",
        sequence=2,
        now_ns=now + 100_000_000,
    )
    _merge_inventory(
        board,
        observed_ns=now + 3 * _SECOND,
        logs=2,
        planks=8,
        recipe=True,
    )
    clicked = controller.step(
        board,
        run_id="craft:one",
        sequence=3,
        now_ns=now + 3 * _SECOND,
    )
    assert clicked.action is not None

    _merge_inventory(
        board,
        observed_ns=now + 4 * _SECOND,
        logs=2,
        planks=8,
        recipe=False,
    )
    pending = controller.step(
        board,
        run_id="craft:one",
        sequence=4,
        now_ns=now + 4 * _SECOND,
    )

    assert pending.verification is None
    assert pending.action is None
    assert controller.phase == PlankCraftPhase.VERIFY_OUTPUT


def test_pre_click_pixels_cannot_verify_a_post_click_inventory_delta() -> None:
    now = time.monotonic_ns()
    board = _board(now)
    controller = BoundedPlankCraftController()
    controller.step(board, run_id="craft:one", sequence=1, now_ns=now)
    _merge_fast_overlay(board, observed_ns=now + 100_000_000)
    controller.step(
        board,
        run_id="craft:one",
        sequence=2,
        now_ns=now + 100_000_000,
    )
    _merge_inventory(
        board,
        observed_ns=now + 3 * _SECOND,
        logs=2,
        planks=0,
        recipe=True,
    )
    clicked = controller.step(
        board,
        run_id="craft:one",
        sequence=3,
        now_ns=now + 3 * _SECOND,
    )
    assert clicked.action is not None

    # Simulate a slow job that was captured before the click but only published
    # afterward. Semantic completion time alone must not make those pixels new.
    _merge_inventory(
        board,
        observed_ns=now + 4 * _SECOND,
        captured_ns=now + 2 * _SECOND,
        logs=1,
        planks=4,
        recipe=False,
    )
    pending = controller.step(
        board,
        run_id="craft:one",
        sequence=4,
        now_ns=now + 4 * _SECOND,
    )

    assert pending.verification is None
    assert pending.action is None
    assert controller.phase == PlankCraftPhase.VERIFY_OUTPUT


def test_gui_grounded_zero_logs_fails_without_clicking() -> None:
    now = time.monotonic_ns()
    board = _board(now)
    controller = BoundedPlankCraftController()
    controller.step(board, run_id="craft:one", sequence=1, now_ns=now)
    _merge_fast_overlay(board, observed_ns=now + 100_000_000)
    controller.step(
        board,
        run_id="craft:one",
        sequence=2,
        now_ns=now + 100_000_000,
    )
    _merge_inventory(
        board,
        observed_ns=now + 3 * _SECOND,
        logs=0,
        planks=0,
        recipe=True,
    )

    failed = controller.step(
        board,
        run_id="craft:one",
        sequence=3,
        now_ns=now + 3 * _SECOND,
    )

    assert failed.failure_reason == "crafting-no-logs-observed-in-inventory"
    assert failed.action is None
    assert controller.phase == PlankCraftPhase.FAILED


def test_crafting_global_timeout_requests_inventory_close_recovery() -> None:
    now = time.monotonic_ns()
    executor = SkillExecutor(_UnusedPolicy())
    spec = build_bootstrap_skill_library().get("craft_wood_planks")
    executor.start(spec, run_id="craft:timeout", now_ns=now)

    result = executor.tick(
        _board(now),
        sequence=1,
        now_ns=now + spec.max_duration_ms * 1_000_000,
    )

    assert result.run.outcome == SkillOutcome.TIMED_OUT
    assert result.recovery_skills == ("close_open_inventory",)


def test_crafting_cancellation_requests_inventory_close_recovery() -> None:
    executor = SkillExecutor(_UnusedPolicy())
    executor.start(
        build_bootstrap_skill_library().get("craft_wood_planks"),
        run_id="craft:cancel",
    )

    result = executor.cancel()

    assert result.run.outcome == SkillOutcome.CANCELLED
    assert result.recovery_skills == ("close_open_inventory",)
