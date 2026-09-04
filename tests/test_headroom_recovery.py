from __future__ import annotations

import threading
import time
from collections import deque
from concurrent.futures import Future
from dataclasses import replace
from itertools import count

import pytest

from minecraft_ai.builtin_skills import build_bootstrap_skill_library
from minecraft_ai.execution import ExecutionTick, SkillExecutor
from minecraft_ai.grounded_perception import (
    CROSSHAIR_BLOCK_FAST_SOURCE,
    crosshair_block_crop_dimensions,
    crosshair_block_pixel_sha256,
    crosshair_block_region,
    crosshair_block_rgb_grid,
)
from minecraft_ai.memory import MemoryStore
from minecraft_ai.motor import BootstrapMotorPolicy, MotorIntent
from minecraft_ai.outcome_verifier import (
    OutcomeKind,
    OutcomeSignal,
    OutcomeStatus,
    OutcomeVerification,
)
from minecraft_ai.perception import (
    ActivePerceptionQuery,
    EvidenceRegion,
    FrameState,
    PerceptionBlackboard,
    PerceptionEvidence,
    PerceptionFact,
    PerceptionQueryMode,
    ScreenRegion,
    Track,
)
from minecraft_ai.perception_service import crosshair_block_dhash, frame_dhash
from minecraft_ai.platforms.bedrock_x11 import CapturedFrame
from minecraft_ai.runtime import (
    AgentRuntime,
    RuntimeMetrics,
    _HeadroomRecovery,
    _headroom_clear_target,
    _headroom_deadline_ns,
    _headroom_reorient_mouse_dy,
    _headroom_retry_advances_plan,
    _restore_policy_world_camera,
    _verified_headroom_retry,
    _verified_obstacle_stall,
)
from minecraft_ai.safety import MotorAction
from minecraft_ai.skills import SkillFailureCode, SkillOutcome, SkillRun


_HASH_A = "0000000000000000"
_HASH_FAR = "ffffffffffffffff"
_ROCKET_SOURCE = "learned:learned:minestudio-rocket2:test:aux-localization:not-training-label"


def _install_strict_monotonic_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamps = count(start=time.monotonic_ns(), step=1_000_000)
    monkeypatch.setattr(time, "monotonic_ns", lambda: next(timestamps))


def _stall_result(run_id: str = "obstacle-stall") -> ExecutionTick:
    run = SkillRun(
        run_id=run_id,
        skill_id="traverse_visible_obstacle",
        context_key="explore-keepalive",
        started_ns=1,
        ended_ns=2,
        outcome=SkillOutcome.FAILED,
        failure_reason=SkillFailureCode.LOCOMOTION_STALLED.value,
        failure_code=SkillFailureCode.LOCOMOTION_STALLED,
        parameters={"allow_jump": True},
    )
    return ExecutionTick(
        run=run,
        action=None,
        outcome_verification=OutcomeVerification(
            run_id=run_id,
            kind=OutcomeKind.TRAVERSAL,
            status=OutcomeStatus.STALLED,
            signal=OutcomeSignal.LOCOMOTION_STALLED,
            observed_ns=2,
            confidence=0.92,
            reason="verified no displacement",
        ),
    )


def _publish_headroom_answer(
    board: PerceptionBlackboard,
    recovery: _HeadroomRecovery,
    frame: CapturedFrame,
    *,
    kind: str = "dirt",
    confidence: float = 0.95,
    block_source: str | None = None,
    crop_hash: str | None = None,
    evidence_region: ScreenRegion | None = None,
    evidence_sha: str | None = None,
    observed_ns: int | None = None,
) -> None:
    assert recovery.query_id is not None and recovery.query_source is not None
    assert recovery.query_frame_id is not None
    now = observed_ns or max(time.monotonic_ns(), recovery.query_started_ns + 1)
    source = recovery.query_source
    evidence_id = f"frame-{recovery.query_frame_id}:crosshair-block"
    crop_width, crop_height = crosshair_block_crop_dimensions(frame.width, frame.height)
    facts = (
        PerceptionFact(
            key="recovery.crosshair.block",
            value=kind,
            confidence=confidence,
            observed_ns=now,
            source=block_source or source,
            expires_after_ms=60_000,
            evidence_refs=(evidence_id,),
        ),
        PerceptionFact(
            key="recovery.crosshair.frame_dhash",
            value=recovery.query_frame_dhash or _HASH_A,
            confidence=1.0,
            observed_ns=now,
            source=source,
            expires_after_ms=60_000,
        ),
        PerceptionFact(
            key="recovery.crosshair.observation_dhash",
            value=crop_hash or recovery.query_crosshair_dhash or _HASH_A,
            confidence=1.0,
            observed_ns=now,
            source=source,
            expires_after_ms=60_000,
        ),
    )
    board.merge_semantics(
        instance_id="bedrock:headroom",
        facts=facts,
        evidence=(
            PerceptionEvidence(
                evidence_id=evidence_id,
                frame_id=recovery.query_frame_id,
                captured_ns=recovery.query_captured_ns or frame.captured_ns,
                region_kind=EvidenceRegion.WORLD,
                region=evidence_region or crosshair_block_region(frame.width, frame.height),
                pixel_sha256=evidence_sha or recovery.query_pixel_sha256 or "0" * 64,
                crop_width=crop_width,
                crop_height=crop_height,
            ),
        ),
    )


def _pixels(width: int, height: int, *, changed: bool = False) -> bytes:
    return bytes(
        channel
        for y in range(height)
        for x in range(width)
        for channel in (
            ((x * 5 + y * 3 + (91 if changed else 0)) % 256),
            ((x * 2 + y * 7 + (37 if changed else 0)) % 256),
            ((x * 11 + y + (173 if changed else 0)) % 256),
            255,
        )
    )


def _capture(
    *,
    frame_id: int = 41,
    captured_ns: int | None = None,
    changed: bool = False,
) -> CapturedFrame:
    width = height = 64
    return CapturedFrame(
        frame_id=frame_id,
        captured_ns=time.monotonic_ns() if captured_ns is None else captured_ns,
        width=width,
        height=height,
        bgra=_pixels(width, height, changed=changed),
    )


def _publish_frame(board: PerceptionBlackboard, frame: CapturedFrame) -> None:
    observed_ns = max(time.monotonic_ns(), frame.captured_ns)
    board.publish(
        FrameState(
            frame_id=frame.frame_id,
            captured_ns=frame.captured_ns,
            instance_id="bedrock:headroom",
            width=frame.width,
            height=frame.height,
        )
    )
    board.merge_semantics(
        instance_id="bedrock:headroom",
        facts=(
            PerceptionFact(
                key="scene.playable", value=True, confidence=1.0,
                observed_ns=observed_ns,
                source="safety:bedrock-hud-v1:not-training-label",
                expires_after_ms=60_000,
            ),
            PerceptionFact(
                key="scene.mode", value="world", confidence=1.0,
                observed_ns=observed_ns,
                source="safety:bedrock-hud-v1:not-training-label",
                expires_after_ms=60_000,
            ),
            PerceptionFact(
                key="frame.dhash", value=frame_dhash(frame), confidence=1.0,
                observed_ns=observed_ns, source=CROSSHAIR_BLOCK_FAST_SOURCE,
                expires_after_ms=60_000,
            ),
            PerceptionFact(
                key="frame.crosshair_block_dhash", value=crosshair_block_dhash(frame),
                confidence=1.0, observed_ns=observed_ns,
                source=CROSSHAIR_BLOCK_FAST_SOURCE,
                expires_after_ms=60_000,
            ),
            PerceptionFact(
                key="frame.crosshair_block_rgb_grid", value=crosshair_block_rgb_grid(frame),
                confidence=1.0, observed_ns=observed_ns,
                source=CROSSHAIR_BLOCK_FAST_SOURCE,
                expires_after_ms=60_000,
            ),
            PerceptionFact(
                key="frame.crosshair_dhash", value=_HASH_A, confidence=1.0,
                observed_ns=observed_ns, source="bootstrap:frame",
                expires_after_ms=60_000,
            ),
        ),
    )


def _board(frame: CapturedFrame | None = None) -> PerceptionBlackboard:
    frame = frame or _capture()
    board = PerceptionBlackboard()
    _publish_frame(board, frame)
    return board


def _recovery_for_frame(frame: CapturedFrame, *, query_id: str = "query-one") -> _HeadroomRecovery:
    return _HeadroomRecovery(
        context_key="explore-keepalive",
        traversal_parameters={},
        deadline_ns=time.monotonic_ns() + 60_000_000_000,
        phase="grounding",
        query_id=query_id,
        query_started_ns=time.monotonic_ns() - 1_000,
        query_frame_dhash=frame_dhash(frame),
        query_crosshair_dhash=crosshair_block_dhash(frame),
        query_frame_id=frame.frame_id,
        query_captured_ns=frame.captured_ns,
        query_frame_width=frame.width,
        query_frame_height=frame.height,
        query_pixel_sha256=crosshair_block_pixel_sha256(frame),
        query_rgb_grid=crosshair_block_rgb_grid(frame),
        query_source=f"vlm:test:{query_id}",
    )


def test_headroom_trigger_requires_exact_verified_obstacle_stall() -> None:
    exact = _stall_result()
    assert _verified_obstacle_stall(exact) is True

    no_verification = replace(exact, outcome_verification=None)
    assert _verified_obstacle_stall(no_verification) is False

    level_ground = replace(
        exact,
        run=exact.run.model_copy(update={"skill_id": "traverse_level_ground"}),
    )
    assert _verified_obstacle_stall(level_ground) is False


def test_headroom_deadline_respects_serialized_vlm_timeout_budget() -> None:
    class _Model:
        timeout_s = 75.0

    class _Worker:
        model = _Model()

    now = 12_000
    assert _headroom_deadline_ns(_Worker(), now_ns=now) == now + 180_000_000_000


def test_headroom_reorientation_targets_absolute_calibrated_pitch() -> None:
    class _Policy:
        def __init__(self, pitch: int) -> None:
            self.pitch = pitch

        def status(self) -> dict[str, object]:
            return {"world_camera": {"estimated_pitch_units": self.pitch}}

        def restore_world_camera_state(self, *, estimated_pitch_units: int) -> None:
            self.pitch = estimated_pitch_units

    upward = _Policy(-251)
    chunks: list[int] = []
    while delta := _headroom_reorient_mouse_dy(upward.pitch):
        chunks.append(delta)
        _restore_policy_world_camera(upward, pitch_units=upward.pitch + delta)
    assert chunks == [96, 96, 96, 59]
    assert upward.pitch == 96

    assert _headroom_reorient_mouse_dy(-2_000) == 96
    assert _headroom_reorient_mouse_dy(2_000) == -96
    assert _headroom_reorient_mouse_dy(96) == 0
    assert _headroom_reorient_mouse_dy(65) == 0
    assert _headroom_reorient_mouse_dy(128) == 0


def test_headroom_authoritative_pitch_requires_calibrated_supervisor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = object.__new__(AgentRuntime)
    response: dict[str, object] = {
        "world_camera": {
            "estimated_pitch_units": -251,
            "origin_calibrated": True,
        }
    }
    monkeypatch.setattr(
        "minecraft_ai.runtime.send_command",
        lambda command: response if command == "status" else {},
    )
    assert runtime._authoritative_world_camera_pitch_units() == -251

    response["world_camera"] = {
        "estimated_pitch_units": -251,
        "origin_calibrated": False,
    }
    assert runtime._authoritative_world_camera_pitch_units() is None


@pytest.mark.parametrize("kind", ("stone", "bedrock", "oak_log", "unknown"))
def test_headroom_target_abstains_on_hard_or_unknown_exact_center_block(kind: str) -> None:
    frame = _capture()
    board = _board(frame)
    recovery = _recovery_for_frame(frame)
    _publish_headroom_answer(board, recovery, frame, kind=kind)

    assert (
        _headroom_clear_target(
            board, recovery, now_ns=time.monotonic_ns(), current_frame=frame
        )
        is None
    )


@pytest.mark.parametrize("defect", ("source", "crop_hash", "region", "pixel_sha", "future"))
def test_headroom_target_abstains_on_stale_or_incoherent_provenance(defect: str) -> None:
    frame = _capture()
    board = _board(frame)
    recovery = _recovery_for_frame(frame)
    kwargs: dict[str, object] = {}
    if defect == "source":
        kwargs["block_source"] = "vlm:test:another-query"
    elif defect == "crop_hash":
        kwargs["crop_hash"] = _HASH_FAR
    elif defect == "region":
        kwargs["evidence_region"] = ScreenRegion(x=0, y=0, width=0.5, height=0.5)
    elif defect == "pixel_sha":
        kwargs["evidence_sha"] = "f" * 64
    else:
        kwargs["observed_ns"] = time.monotonic_ns() + 60_000_000_000
    _publish_headroom_answer(board, recovery, frame, **kwargs)  # type: ignore[arg-type]

    assert (
        _headroom_clear_target(
            board, recovery, now_ns=time.monotonic_ns(), current_frame=frame
        )
        is None
    )


def test_headroom_target_requires_post_request_evidence() -> None:
    frame = _capture()
    board = _board(frame)
    recovery = _recovery_for_frame(frame)
    _publish_headroom_answer(board, recovery, frame)
    assert _headroom_clear_target(
        board, recovery, now_ns=time.monotonic_ns(), current_frame=frame
    ) is not None

    recovery.query_started_ns = time.monotonic_ns() + 1
    assert _headroom_clear_target(
        board, recovery, now_ns=time.monotonic_ns(), current_frame=frame
    ) is None


class _Perception:
    def __init__(self, captured: CapturedFrame) -> None:
        class _Model:
            model_id = "test"
            timeout_s = 30.0

        class _Worker:
            model = _Model()

        self.active_vlm = _Worker()
        self.instance_id = "bedrock:headroom"
        self.last_capture = captured
        self.available = True
        self.requests: list[tuple[ActivePerceptionQuery, CapturedFrame | None]] = []

    def semantic_available(self) -> bool:
        return self.available

    def request_semantics(
        self,
        query: ActivePerceptionQuery,
        frame: CapturedFrame | None = None,
    ) -> bool:
        self.requests.append((query, frame))
        self.available = False
        return True


class _QuietMotorPolicy:
    policy_id = "quiet-test-policy"

    def act(
        self,
        blackboard: PerceptionBlackboard,
        intent: MotorIntent,
        *,
        sequence: int,
    ) -> MotorAction:
        del blackboard, intent
        return MotorAction(sequence=sequence)

    def reset(self) -> MotorAction:
        return MotorAction(sequence=0)


def _runtime_for_probe(
    captured: CapturedFrame | None = None,
) -> tuple[AgentRuntime, _Perception, list[MotorAction]]:
    captured = captured or _capture()
    board = _board(captured)
    perception = _Perception(captured)
    runtime = object.__new__(AgentRuntime)
    runtime.blackboard = board
    runtime.perception = perception  # type: ignore[assignment]
    runtime.skills = build_bootstrap_skill_library()
    runtime.executor = SkillExecutor(BootstrapMotorPolicy())
    runtime.metrics = RuntimeMetrics()
    runtime.semantic_hz = 0.0
    runtime._headroom_recovery = None
    runtime._traversal_escalation_pending = False
    runtime._execution_revision = 0
    runtime._cognition_requested = False
    runtime._pending_decision = None
    runtime._sequence = 0
    runtime.lease_id = "test-lease"
    runtime._stop = threading.Event()
    pitch = {"value": 0}
    sent: list[MotorAction] = []

    def _send(action: MotorAction, **_kwargs: object) -> None:
        sent.append(action)
        pitch["value"] += action.mouse_dy
        runtime._sequence += 1

    runtime._send_motor = _send  # type: ignore[method-assign]
    runtime._quiesce_headroom_inputs = lambda: True  # type: ignore[method-assign]
    runtime._authoritative_world_camera_pitch_units = lambda: pitch["value"]  # type: ignore[method-assign]
    return runtime, perception, sent


def test_verified_stall_requests_one_event_query_at_zero_semantic_hz_then_mines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_strict_monotonic_clock(monkeypatch)
    runtime, perception, sent = _runtime_for_probe()

    assert runtime._route_headroom_terminal(_stall_result()) is True
    recovery = runtime._headroom_recovery
    assert recovery is not None
    assert recovery.phase == "reorient"
    assert perception.requests == []

    runtime._advance_headroom_recovery()
    assert recovery.phase == "reorient"
    assert len(sent) == 1
    runtime._advance_headroom_recovery()
    assert recovery.phase == "settle"
    assert len(sent) == 1
    reorientation = sent[0]
    assert isinstance(reorientation, MotorAction)
    assert reorientation.mouse_dy == 96
    assert not reorientation.keys_down
    assert not reorientation.buttons_down

    runtime._advance_headroom_recovery()
    assert perception.requests == []

    previous = runtime.blackboard.raw_latest()
    assert previous is not None
    next_ns = max(time.monotonic_ns(), previous.captured_ns + 1)
    next_capture = replace(perception.last_capture, frame_id=42, captured_ns=next_ns)
    perception.last_capture = next_capture
    _publish_frame(runtime.blackboard, next_capture)
    runtime._advance_headroom_recovery()
    assert recovery.phase == "settle"
    assert perception.requests == []

    changed_ns = max(time.monotonic_ns(), next_ns + 1)
    changed_capture = _capture(frame_id=43, captured_ns=changed_ns, changed=True)
    perception.last_capture = changed_capture
    _publish_frame(runtime.blackboard, changed_capture)
    runtime._advance_headroom_recovery()
    assert recovery.phase == "settle"
    assert perception.requests == []

    stable_one = replace(
        changed_capture,
        frame_id=44,
        captured_ns=max(time.monotonic_ns(), changed_capture.captured_ns + 1),
    )
    perception.last_capture = stable_one
    _publish_frame(runtime.blackboard, stable_one)
    runtime._advance_headroom_recovery()
    assert recovery.phase == "settle"
    assert perception.requests == []

    stable_two = replace(
        changed_capture,
        frame_id=45,
        captured_ns=max(time.monotonic_ns(), stable_one.captured_ns + 1),
    )
    perception.last_capture = stable_two
    _publish_frame(runtime.blackboard, stable_two)
    runtime._advance_headroom_recovery()
    assert recovery.phase == "grounding"
    assert len(perception.requests) == 1
    query = perception.requests[0][0]
    assert query.mode == PerceptionQueryMode.CROSSHAIR_BLOCK
    assert query.skill_id == "mine_visible_block"
    assert query.output_keys == ()

    runtime._advance_headroom_recovery()
    assert len(perception.requests) == 1

    _publish_headroom_answer(runtime.blackboard, recovery, stable_two)
    perception.available = True
    runtime._advance_headroom_recovery()

    assert len(perception.requests) == 1
    assert recovery.phase == "mining"
    assert runtime.executor.run is not None
    assert runtime.executor.run.skill_id == "mine_visible_block"
    assert runtime.executor.parameters == {
        "target": "dirt",
        "target_track_id": f"crosshair-probe:{recovery.query_id}",
    }

    first = runtime.executor.tick(
        runtime.blackboard,
        sequence=10,
        now_ns=time.monotonic_ns(),
    )
    assert first.action is not None
    assert "left" not in first.action.buttons_down
    assert "w" in first.action.keys_down

    post_motion = replace(
        stable_two,
        frame_id=46,
        captured_ns=time.monotonic_ns(),
    )
    perception.last_capture = post_motion
    _publish_frame(runtime.blackboard, post_motion)
    settling = runtime.executor.tick(
        runtime.blackboard,
        sequence=11,
        now_ns=time.monotonic_ns(),
    )
    assert settling.action is not None
    assert "left" not in settling.action.buttons_down
    assert "w" in settling.action.keys_up

    settled = replace(
        stable_two,
        frame_id=47,
        captured_ns=time.monotonic_ns(),
    )
    perception.last_capture = settled
    _publish_frame(runtime.blackboard, settled)
    started = runtime.executor.tick(
        runtime.blackboard,
        sequence=12,
        now_ns=time.monotonic_ns(),
    )
    assert started.action is not None
    assert started.action.buttons_down == ("left",)


def test_headroom_already_near_target_pitch_settles_without_camera_overshoot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_strict_monotonic_clock(monkeypatch)
    runtime, perception, sent = _runtime_for_probe()
    runtime._authoritative_world_camera_pitch_units = lambda: 96  # type: ignore[method-assign]
    restored: list[int] = []
    monkeypatch.setattr(
        "minecraft_ai.runtime._restore_policy_world_camera",
        lambda _policy, *, pitch_units: restored.append(pitch_units),
    )
    assert runtime._route_headroom_terminal(_stall_result()) is True
    recovery = runtime._headroom_recovery
    assert recovery is not None

    runtime._advance_headroom_recovery()
    assert recovery.phase == "settle"
    assert sent == []
    assert restored == [96]

    for frame_id in (42, 43, 44):
        settled = replace(
            perception.last_capture,
            frame_id=frame_id,
            captured_ns=time.monotonic_ns(),
        )
        perception.last_capture = settled
        _publish_frame(runtime.blackboard, settled)
        runtime._advance_headroom_recovery()

    assert recovery.phase == "grounding"
    assert len(perception.requests) == 1


def test_headroom_retries_transient_missing_physical_pitch() -> None:
    runtime, _, sent = _runtime_for_probe()
    assert runtime._route_headroom_terminal(_stall_result()) is True
    recovery = runtime._headroom_recovery
    assert recovery is not None
    runtime._authoritative_world_camera_pitch_units = lambda: None  # type: ignore[method-assign]

    runtime._advance_headroom_recovery()

    assert runtime._headroom_recovery is recovery
    assert recovery.phase == "reorient"
    assert sent == []

    runtime._authoritative_world_camera_pitch_units = lambda: 96  # type: ignore[method-assign]
    runtime._advance_headroom_recovery()
    assert recovery.phase == "settle"


def test_current_crosshair_probe_can_initiate_mining_without_learned_attack() -> None:
    frame = _capture()
    runtime, _, _ = _runtime_for_probe(frame)
    recovery = _recovery_for_frame(frame, query_id="quiet-policy-query")
    runtime._headroom_recovery = recovery
    _publish_headroom_answer(runtime.blackboard, recovery, frame)
    target = _headroom_clear_target(
        runtime.blackboard,
        recovery,
        now_ns=time.monotonic_ns(),
        current_frame=frame,
    )
    assert target is not None
    track_id = runtime._materialize_headroom_target(recovery, target)
    assert track_id is not None
    runtime.executor = SkillExecutor(_QuietMotorPolicy())
    runtime.executor.start(
        runtime.skills.get("mine_visible_block"),
        run_id="quiet-probe-mining",
        context_key="explore-keepalive",
        parameters={"target": "dirt", "target_track_id": track_id},
    )

    started = runtime.executor.tick(
        runtime.blackboard,
        sequence=1,
        now_ns=time.monotonic_ns(),
    )

    assert started.run.outcome == SkillOutcome.RUNNING
    assert started.action is not None
    assert started.action.buttons_down == ("left",)


def test_changed_crosshair_probe_never_initiates_quiet_policy_attack() -> None:
    frame = _capture()
    runtime, perception, _ = _runtime_for_probe(frame)
    recovery = _recovery_for_frame(frame, query_id="quiet-stale-query")
    runtime._headroom_recovery = recovery
    _publish_headroom_answer(runtime.blackboard, recovery, frame)
    target = _headroom_clear_target(
        runtime.blackboard,
        recovery,
        now_ns=time.monotonic_ns(),
        current_frame=frame,
    )
    assert target is not None
    track_id = runtime._materialize_headroom_target(recovery, target)
    assert track_id is not None
    runtime.executor = SkillExecutor(_QuietMotorPolicy())
    runtime.executor.start(
        runtime.skills.get("mine_visible_block"),
        run_id="quiet-stale-mining",
        context_key="explore-keepalive",
        parameters={"target": "dirt", "target_track_id": track_id},
    )
    changed = CapturedFrame(
        frame_id=frame.frame_id + 1,
        captured_ns=frame.captured_ns + 1,
        width=frame.width,
        height=frame.height,
        bgra=bytes((12, 34, 56, 255)) * frame.width * frame.height,
    )
    perception.last_capture = changed
    _publish_frame(runtime.blackboard, changed)

    waiting = runtime.executor.tick(
        runtime.blackboard,
        sequence=1,
        now_ns=time.monotonic_ns(),
    )

    assert waiting.run.outcome == SkillOutcome.RUNNING
    assert waiting.action is not None
    assert "left" not in waiting.action.buttons_down


def test_verified_stall_abstains_when_inputs_cannot_be_quiesced() -> None:
    runtime, perception, sent = _runtime_for_probe()
    runtime._quiesce_headroom_inputs = lambda: False  # type: ignore[method-assign]

    assert runtime._route_headroom_terminal(_stall_result()) is False
    assert runtime._headroom_recovery is None
    assert perception.requests == []
    assert sent == []


def test_headroom_waits_for_shared_model_lane_before_capturing_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_strict_monotonic_clock(monkeypatch)
    model_available = False
    monkeypatch.setattr(
        "minecraft_ai.runtime.local_model_inference_available",
        lambda: model_available,
    )
    runtime, perception, _ = _runtime_for_probe()
    assert runtime._route_headroom_terminal(_stall_result()) is True
    recovery = runtime._headroom_recovery
    assert recovery is not None
    runtime._advance_headroom_recovery()
    runtime._advance_headroom_recovery()

    for frame_id in (43, 44, 45):
        captured = _capture(
            frame_id=frame_id,
            captured_ns=time.monotonic_ns(),
            changed=True,
        )
        perception.last_capture = captured
        _publish_frame(runtime.blackboard, captured)
        runtime._advance_headroom_recovery()

    assert recovery.phase == "request"
    assert perception.requests == []

    model_available = True
    runtime._advance_headroom_recovery()
    assert recovery.phase == "grounding"
    assert len(perception.requests) == 1
    assert perception.requests[0][1] is perception.last_capture


def test_headroom_settle_drift_restarts_stable_successor_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_strict_monotonic_clock(monkeypatch)
    runtime, perception, _ = _runtime_for_probe()
    assert runtime._route_headroom_terminal(_stall_result()) is True
    recovery = runtime._headroom_recovery
    assert recovery is not None
    runtime._advance_headroom_recovery()
    runtime._advance_headroom_recovery()

    baseline = _capture(
        frame_id=43,
        captured_ns=time.monotonic_ns(),
        changed=True,
    )
    perception.last_capture = baseline
    _publish_frame(runtime.blackboard, baseline)
    runtime._advance_headroom_recovery()

    stable_one = replace(
        baseline,
        frame_id=44,
        captured_ns=time.monotonic_ns(),
    )
    perception.last_capture = stable_one
    _publish_frame(runtime.blackboard, stable_one)
    runtime._advance_headroom_recovery()
    assert recovery.settle_stable_successors == 1

    drifted = CapturedFrame(
        frame_id=45,
        captured_ns=time.monotonic_ns(),
        width=baseline.width,
        height=baseline.height,
        bgra=bytes((12, 34, 56, 255)) * baseline.width * baseline.height,
    )
    perception.last_capture = drifted
    _publish_frame(runtime.blackboard, drifted)
    runtime._advance_headroom_recovery()
    assert recovery.phase == "settle"
    assert recovery.settle_stable_successors == 0
    assert perception.requests == []

    for frame_id in (46, 47):
        stable = replace(
            drifted,
            frame_id=frame_id,
            captured_ns=time.monotonic_ns(),
        )
        perception.last_capture = stable
        _publish_frame(runtime.blackboard, stable)
        runtime._advance_headroom_recovery()

    assert recovery.phase == "grounding"
    assert len(perception.requests) == 1


def test_crosshair_color_swap_after_acquisition_motion_never_starts_attack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_strict_monotonic_clock(monkeypatch)
    width = height = 64
    first = CapturedFrame(
        frame_id=51,
        captured_ns=time.monotonic_ns(),
        width=width,
        height=height,
        bgra=bytes((100, 100, 100, 255)) * width * height,
    )
    runtime, perception, _ = _runtime_for_probe(first)
    recovery = _recovery_for_frame(first, query_id="swap-query")
    runtime._headroom_recovery = recovery
    _publish_headroom_answer(runtime.blackboard, recovery, first)
    target = _headroom_clear_target(
        runtime.blackboard,
        recovery,
        now_ns=time.monotonic_ns(),
        current_frame=first,
    )
    assert target is not None
    track_id = runtime._materialize_headroom_target(recovery, target)
    assert track_id is not None
    recovery.phase = "mining"
    recovery.mining_run_id = "swap-mining"
    runtime.executor.start(
        runtime.skills.get("mine_visible_block"),
        run_id="swap-mining",
        context_key=recovery.context_key,
        parameters={"target": "dirt", "target_track_id": track_id},
    )

    moving = runtime.executor.tick(
        runtime.blackboard,
        sequence=20,
        now_ns=time.monotonic_ns(),
    )
    assert moving.action is not None and "w" in moving.action.keys_down
    assert "left" not in moving.action.buttons_down

    swapped = CapturedFrame(
        frame_id=52,
        captured_ns=time.monotonic_ns(),
        width=width,
        height=height,
        # Equal-luma-ish colors retain the flat-image dHash while changing
        # the target's absolute color signature around the crosshair.
        bgra=bytes((200, 132, 0, 255)) * width * height,
    )
    assert crosshair_block_dhash(first) == crosshair_block_dhash(swapped)
    perception.last_capture = swapped
    _publish_frame(runtime.blackboard, swapped)
    after_swap = runtime.executor.tick(
        runtime.blackboard,
        sequence=21,
        now_ns=time.monotonic_ns(),
    )
    assert after_swap.action is not None
    assert "left" not in after_swap.action.buttons_down

    # Even if the normal grounded router later emits fresh positive ROCKET
    # observations, this synthetic aperture must never be promoted into the
    # generic ROCKET authorization route and bypass its RGB mismatch.
    for sequence in (22, 23, 24):
        observed_ns = time.monotonic_ns()
        newer_swap = replace(
            swapped,
            frame_id=swapped.frame_id + sequence,
            captured_ns=observed_ns,
        )
        perception.last_capture = newer_swap
        _publish_frame(runtime.blackboard, newer_swap)
        latest = runtime.blackboard.latest()
        assert latest is not None
        probe = next(track for track in latest.tracks if track.track_id == track_id)
        attributes = dict(probe.attributes)
        attributes["tracking_source"] = _ROCKET_SOURCE
        assert runtime.blackboard.upsert_semantic_track(
            instance_id="bedrock:headroom",
            track=probe.model_copy(
                update={
                    "last_seen_ns": observed_ns,
                    "attributes": attributes,
                }
            ),
        )
        runtime.blackboard.merge_semantics(
            instance_id="bedrock:headroom",
            facts=(
                PerceptionFact(
                    key="target.visible", value=True, confidence=0.99,
                    observed_ns=observed_ns, source=_ROCKET_SOURCE,
                    expires_after_ms=5_000,
                ),
                PerceptionFact(
                    key="target.kind", value="dirt", confidence=0.99,
                    observed_ns=observed_ns, source=_ROCKET_SOURCE,
                    expires_after_ms=5_000,
                ),
            ),
        )
        routed = runtime.executor.tick(
            runtime.blackboard,
            sequence=sequence,
            now_ns=time.monotonic_ns(),
        )
        assert routed.action is not None
        assert "left" not in routed.action.buttons_down


@pytest.mark.parametrize("kind", ("stone", "unknown"))
def test_hard_or_unknown_answer_abstains_after_exactly_one_query(kind: str) -> None:
    runtime, perception, sent = _runtime_for_probe()
    runtime._headroom_recovery = _HeadroomRecovery(
        context_key="explore-keepalive",
        traversal_parameters={},
        deadline_ns=time.monotonic_ns() + 60_000_000_000,
        phase="request",
    )
    runtime._advance_headroom_recovery()
    recovery = runtime._headroom_recovery
    assert recovery is not None and recovery.phase == "grounding"
    assert len(perception.requests) == 1
    assert perception.last_capture is not None
    _publish_headroom_answer(
        runtime.blackboard,
        recovery,
        perception.last_capture,
        kind=kind,
    )
    perception.available = True

    runtime._advance_headroom_recovery()
    runtime._advance_headroom_recovery()

    assert runtime._headroom_recovery is None
    assert len(perception.requests) == 1
    assert runtime.executor.run is None
    assert sent == []


@pytest.mark.parametrize(
    "spoofed_key",
    ("frame.crosshair_block_dhash", "frame.crosshair_block_rgb_grid"),
)
def test_headroom_target_rejects_spoofed_fast_source(spoofed_key: str) -> None:
    frame = _capture()
    board = _board(frame)
    recovery = _recovery_for_frame(frame)
    _publish_headroom_answer(board, recovery, frame)
    canonical = board.fact(spoofed_key, now_ns=time.monotonic_ns())
    assert canonical is not None
    board.merge_semantics(
        instance_id="bedrock:headroom",
        facts=(
            canonical.model_copy(
                update={
                    "observed_ns": time.monotonic_ns(),
                    "source": "bootstrap:lookalike:not-training-label",
                }
            ),
        ),
    )

    assert _headroom_clear_target(
        board,
        recovery,
        now_ns=time.monotonic_ns(),
        current_frame=frame,
    ) is None


@pytest.mark.parametrize("state", ("missing", "spoofed", "stale"))
def test_headroom_requires_current_canonical_full_frame_safety(state: str) -> None:
    runtime, _, _ = _runtime_for_probe()
    runtime.blackboard.remove_semantic_facts(
        ("scene.playable", "scene.mode"),
        expected_source="safety:bedrock-hud-v1:not-training-label",
    )
    if state == "spoofed":
        now = time.monotonic_ns()
        runtime.blackboard.merge_semantics(
            instance_id="bedrock:headroom",
            facts=(
                PerceptionFact(
                    key="scene.playable", value=True, confidence=1.0,
                    observed_ns=now, source="safety:bedrock-hud-v1:spoof",
                    expires_after_ms=60_000,
                ),
                PerceptionFact(
                    key="scene.mode", value="world", confidence=1.0,
                    observed_ns=now, source="safety:bedrock-hud-v1:spoof",
                    expires_after_ms=60_000,
                ),
            ),
        )
    elif state == "stale":
        old = time.monotonic_ns() - 1_000_000_000
        runtime.blackboard.merge_semantics(
            instance_id="bedrock:headroom",
            facts=(
                PerceptionFact(
                    key="scene.playable", value=True, confidence=1.0,
                    observed_ns=old,
                    source="safety:bedrock-hud-v1:not-training-label",
                    expires_after_ms=60_000,
                ),
                PerceptionFact(
                    key="scene.mode", value="world", confidence=1.0,
                    observed_ns=old,
                    source="safety:bedrock-hud-v1:not-training-label",
                    expires_after_ms=60_000,
                ),
            ),
        )
    assert runtime._headroom_scene_is_safe() is False


def test_headroom_target_upsert_preserves_peer_tracks_and_cleanup_is_transaction_local() -> None:
    runtime, perception, _ = _runtime_for_probe()
    assert perception.last_capture is not None
    frame = perception.last_capture
    recovery = _recovery_for_frame(frame, query_id="peer-query")
    runtime._headroom_recovery = recovery
    peer = Track(
        track_id="rocket:peer",
        label="oak_log",
        confidence=0.9,
        region=ScreenRegion(x=0.1, y=0.1, width=0.2, height=0.3),
        first_seen_ns=time.monotonic_ns(),
        last_seen_ns=time.monotonic_ns(),
        attributes={"source": "rocket"},
    )
    assert runtime.blackboard.upsert_semantic_track(
        instance_id="bedrock:headroom", track=peer
    )
    _publish_headroom_answer(runtime.blackboard, recovery, frame)
    target = _headroom_clear_target(
        runtime.blackboard,
        recovery,
        now_ns=time.monotonic_ns(),
        current_frame=frame,
    )
    assert target is not None
    track_id = runtime._materialize_headroom_target(recovery, target)
    assert track_id is not None
    assert {track.track_id for track in runtime.blackboard.latest().tracks} == {
        "rocket:peer",
        track_id,
    }

    runtime._clear_headroom_recovery(recovery)

    latest = runtime.blackboard.latest()
    assert latest is not None
    assert {track.track_id for track in latest.tracks} == {"rocket:peer"}
    for key in (
        "recovery.crosshair.block",
        "recovery.crosshair.frame_dhash",
        "recovery.crosshair.observation_dhash",
        "target.visible",
        "target.kind",
        "target.reference_available",
    ):
        assert runtime.blackboard.fact(key, now_ns=time.monotonic_ns()) is None


def test_unsafe_stall_preserves_ordinary_recovery_route() -> None:
    runtime, perception, _ = _runtime_for_probe()
    now = time.monotonic_ns()
    runtime.blackboard.merge_semantics(
        instance_id="bedrock:headroom",
        facts=(
            PerceptionFact(
                key="danger.immediate",
                value=True,
                confidence=1.0,
                observed_ns=now,
                source="safety:test",
                expires_after_ms=5_000,
            ),
        ),
    )

    assert runtime._route_headroom_terminal(_stall_result()) is False
    assert runtime._headroom_recovery is None
    assert perception.requests == []


def test_expired_headroom_transaction_abstains_without_input_or_query() -> None:
    runtime, perception, sent = _runtime_for_probe()
    runtime._headroom_recovery = _HeadroomRecovery(
        context_key="explore-keepalive",
        traversal_parameters={},
        deadline_ns=time.monotonic_ns() - 1,
    )

    runtime._advance_headroom_recovery()

    assert runtime._headroom_recovery is None
    assert perception.requests == []
    assert sent == []
    assert runtime._traversal_escalation_pending is True
    assert runtime._cognition_requested is True


def test_unchanged_reorientation_settle_abstains_after_two_seconds() -> None:
    runtime, perception, sent = _runtime_for_probe()
    assert perception.last_capture is not None
    latest = runtime.blackboard.raw_latest()
    assert latest is not None
    runtime._headroom_recovery = _HeadroomRecovery(
        context_key="explore-keepalive",
        traversal_parameters={},
        deadline_ns=time.monotonic_ns() + 60_000_000_000,
        phase="settle",
        reoriented_frame_id=latest.frame_id,
        pre_reorient_dhash=frame_dhash(perception.last_capture),
        settle_deadline_ns=time.monotonic_ns() - 1,
    )

    runtime._advance_headroom_recovery()

    assert runtime._headroom_recovery is None
    assert perception.requests == []
    assert sent == []
    assert runtime._traversal_escalation_pending is True
    assert runtime._cognition_requested is True


def test_expired_headroom_transaction_cancels_and_releases_running_child() -> None:
    runtime, _, sent = _runtime_for_probe()
    runtime.executor.start(
        runtime.skills.get("mine_visible_block"),
        run_id="clear-one",
        context_key="explore-keepalive",
        parameters={"target": "dirt"},
    )
    runtime._headroom_recovery = _HeadroomRecovery(
        context_key="explore-keepalive",
        traversal_parameters={},
        deadline_ns=time.monotonic_ns() - 1,
        phase="mining",
        mining_run_id="clear-one",
    )
    recorded: list[SkillRun] = []
    runtime._record_terminal_run = (  # type: ignore[method-assign]
        lambda run, **_kwargs: recorded.append(run)
    )

    runtime._advance_headroom_recovery()

    assert runtime._headroom_recovery is None
    assert recorded[0].outcome == SkillOutcome.CANCELLED
    assert sent
    assert runtime._traversal_escalation_pending is True
    assert runtime._cognition_requested is True


def _mining_success(run_id: str) -> ExecutionTick:
    run = SkillRun(
        run_id=run_id,
        skill_id="mine_visible_block",
        context_key="explore-keepalive",
        started_ns=2,
        ended_ns=3,
        outcome=SkillOutcome.SUCCEEDED,
    )
    return ExecutionTick(
        run=run,
        action=None,
        outcome_verification=OutcomeVerification(
            run_id=run_id,
            kind=OutcomeKind.MINING,
            status=OutcomeStatus.SUCCEEDED,
            signal=OutcomeSignal.BLOCK_BROKEN,
            observed_ns=3,
            confidence=0.9,
            reason="bound dirt block disappeared after damage",
            target_kind="dirt",
        ),
    )


def _retry_progress(
    run_id: str,
    *,
    context_key: str = "explore-keepalive",
) -> ExecutionTick:
    run = SkillRun(
        run_id=run_id,
        skill_id="traverse_visible_obstacle",
        context_key=context_key,
        started_ns=3,
        ended_ns=4,
        outcome=SkillOutcome.SUCCEEDED,
    )
    return ExecutionTick(
        run=run,
        action=None,
        outcome_verification=OutcomeVerification(
            run_id=run_id,
            kind=OutcomeKind.TRAVERSAL,
            status=OutcomeStatus.PROGRESS,
            signal=OutcomeSignal.LOCOMOTION_PROGRESS,
            observed_ns=4,
            confidence=0.91,
            reason="verified displacement after the bounded retry",
        ),
    )


def test_verified_clear_retries_obstacle_once_and_retry_stall_does_not_loop() -> None:
    runtime, _, _ = _runtime_for_probe()
    recovery = _HeadroomRecovery(
        context_key="explore-keepalive",
        traversal_parameters={"allow_jump": True},
        deadline_ns=time.monotonic_ns() + 60_000_000_000,
        phase="mining",
        mining_run_id="clear-one",
    )
    runtime._headroom_recovery = recovery
    runtime._traversal_escalation_pending = True

    assert runtime._route_headroom_terminal(_mining_success("clear-one")) is True
    assert recovery.phase == "retry"
    assert recovery.retry_run_id is not None
    assert runtime.executor.run is not None
    assert runtime.executor.run.skill_id == "traverse_visible_obstacle"
    assert runtime.executor.parameters == {"allow_jump": True}
    assert runtime._traversal_escalation_pending is True

    retry_stall = _stall_result(recovery.retry_run_id)
    assert runtime._route_headroom_terminal(retry_stall) is True
    assert runtime._headroom_recovery is None
    assert runtime._traversal_escalation_pending is True
    assert runtime.executor.run is not None
    assert runtime.executor.run.run_id == retry_stall.run.run_id


def test_verified_retry_progress_clears_escalation_and_advances_plan_once() -> None:
    runtime, _, _ = _runtime_for_probe()
    recovery = _HeadroomRecovery(
        context_key="explore-keepalive",
        traversal_parameters={"allow_jump": True},
        deadline_ns=time.monotonic_ns() + 60_000_000_000,
        phase="retry",
        retry_run_id="retry-one",
    )
    runtime._headroom_recovery = recovery
    runtime._traversal_escalation_pending = True
    result = _retry_progress("retry-one")

    assert _verified_headroom_retry(result, recovery) is True
    assert runtime._route_headroom_terminal(result) is True
    assert runtime._headroom_recovery is None
    assert runtime._traversal_escalation_pending is False

    assert (
        _headroom_retry_advances_plan(
            result,
            recovery,
            plan_steps=("unrelated strategic work",),
            plan_index=0,
            plan_goal_id="strategic-goal",
        )
        is False
    )

    planned_recovery = _HeadroomRecovery(
        context_key="strategic-goal",
        traversal_parameters={},
        deadline_ns=time.monotonic_ns() + 60_000_000_000,
        phase="retry",
        retry_run_id="planned-retry",
    )
    planned_result = _retry_progress(
        "planned-retry",
        context_key="strategic-goal",
    )
    advances_plan = _headroom_retry_advances_plan(
        planned_result,
        planned_recovery,
        plan_steps=("clear the obstacle", "continue exploring"),
        plan_index=0,
        plan_goal_id="strategic-goal",
    )
    assert advances_plan is True

    recorder = object.__new__(AgentRuntime)
    recorder.skills = build_bootstrap_skill_library()
    recorder.memories = MemoryStore()
    recorder.state_db = None
    recorder.trajectory = None
    recorder.metrics = RuntimeMetrics()
    recorder._recorded_run_ids = set()
    recorder._recorded_run_order = deque(maxlen=4_096)
    recorder._recent_skill_runs = deque(maxlen=8)
    recorder._plan_steps = ("clear the obstacle", "continue exploring")
    recorder._plan_index = 0
    recorder._plan_goal_id = "strategic-goal"
    recorder._last_decision = None
    recorder._plan_step_completed_ns = 0
    recorder._record_terminal_run(
        planned_result.run,
        outcome_verification=planned_result.outcome_verification,
        advance_plan=advances_plan,
    )
    recorder._record_terminal_run(
        planned_result.run,
        outcome_verification=planned_result.outcome_verification,
        advance_plan=advances_plan,
    )

    assert recorder._plan_index == 1
    assert recorder.metrics.skill_successes == 1


def test_unverified_retry_success_keeps_escalation_latched() -> None:
    runtime, _, _ = _runtime_for_probe()
    recovery = _HeadroomRecovery(
        context_key="explore-keepalive",
        traversal_parameters={},
        deadline_ns=time.monotonic_ns() + 60_000_000_000,
        phase="retry",
        retry_run_id="retry-one",
    )
    runtime._headroom_recovery = recovery
    runtime._traversal_escalation_pending = True
    unverified = replace(_retry_progress("retry-one"), outcome_verification=None)

    assert _verified_headroom_retry(unverified, recovery) is False
    assert runtime._route_headroom_terminal(unverified) is True
    assert runtime._headroom_recovery is None
    assert runtime._traversal_escalation_pending is True


def test_retry_result_returning_after_transaction_deadline_is_cancelled() -> None:
    runtime, _, _ = _runtime_for_probe()
    recovery = _HeadroomRecovery(
        context_key="strategic-goal",
        traversal_parameters={},
        deadline_ns=time.monotonic_ns() - 1,
        phase="retry",
        retry_run_id="late-retry",
    )
    runtime._headroom_recovery = recovery
    runtime._traversal_escalation_pending = True
    late = _retry_progress("late-retry", context_key="strategic-goal")
    runtime.executor._run = late.run  # mirror the terminal executor state after tick

    expired, was_expired = runtime._expire_late_headroom_child(late)

    assert was_expired is True
    assert expired.run.outcome == SkillOutcome.CANCELLED
    assert expired.run.failure_reason == "headroom-transaction-expired"
    assert expired.outcome_verification is None
    assert expired.action is None or not expired.action.keys_down
    assert runtime._headroom_recovery is None
    assert runtime._traversal_escalation_pending is True
    assert runtime._cognition_requested is True
    assert (
        _headroom_retry_advances_plan(
            expired,
            recovery,
            plan_steps=("clear the obstacle",),
            plan_index=0,
            plan_goal_id="strategic-goal",
        )
        is False
    )


def test_operator_authority_cancels_running_headroom_child() -> None:
    runtime, _, _ = _runtime_for_probe()
    runtime.executor.start(
        runtime.skills.get("mine_visible_block"),
        run_id="clear-one",
        context_key="explore-keepalive",
        parameters={"target": "dirt"},
    )
    runtime._headroom_recovery = _HeadroomRecovery(
        context_key="explore-keepalive",
        traversal_parameters={},
        deadline_ns=time.monotonic_ns() + 60_000_000_000,
        phase="mining",
        mining_run_id="clear-one",
    )
    runtime._pending_decision = Future()
    runtime._pending_decision.set_result(object())
    runtime._queued_operator_message_waiting = lambda: True  # type: ignore[method-assign]
    runtime._preempt_pending_cognition_for_operator = lambda: False  # type: ignore[method-assign]
    recorded: list[SkillRun] = []
    sent = []
    runtime._record_terminal_run = (  # type: ignore[method-assign]
        lambda run, **_kwargs: recorded.append(run)
    )
    runtime._send_motor = lambda action, **_kwargs: sent.append(action)  # type: ignore[method-assign]

    runtime._consume_cognition()
    assert runtime._headroom_recovery is not None
    assert runtime.executor.run is not None
    assert runtime.executor.run.outcome == SkillOutcome.RUNNING

    runtime._start_cognition_if_due()

    assert runtime._headroom_recovery is None
    assert recorded[0].outcome == SkillOutcome.CANCELLED
    assert sent


def test_death_scene_preempts_running_headroom_child() -> None:
    runtime, perception, _ = _runtime_for_probe()
    assert perception.last_capture is not None
    recovery = _recovery_for_frame(perception.last_capture, query_id="death-query")
    runtime._headroom_recovery = recovery
    _publish_headroom_answer(
        runtime.blackboard,
        recovery,
        perception.last_capture,
    )
    target = _headroom_clear_target(
        runtime.blackboard,
        recovery,
        now_ns=time.monotonic_ns(),
        current_frame=perception.last_capture,
    )
    assert target is not None
    track_id = runtime._materialize_headroom_target(recovery, target)
    assert track_id is not None
    runtime.executor.start(
        runtime.skills.get("mine_visible_block"),
        run_id="clear-one",
        context_key="explore-keepalive",
        parameters={"target": "dirt", "target_track_id": track_id},
    )
    recovery.phase = "mining"
    recovery.mining_run_id = "clear-one"
    now = time.monotonic_ns()
    runtime.blackboard.merge_semantics(
        instance_id="bedrock:headroom",
        facts=(
            PerceptionFact(
                key="scene.death",
                value=True,
                confidence=1.0,
                observed_ns=now,
                source="safety:test",
                expires_after_ms=5_000,
            ),
        ),
    )
    recorded: list[SkillRun] = []
    sent = []
    runtime._record_terminal_run = (  # type: ignore[method-assign]
        lambda run, **_kwargs: recorded.append(run)
    )
    runtime._send_motor = lambda action, **_kwargs: sent.append(action)  # type: ignore[method-assign]

    runtime._route_observed_scene_recovery()

    assert runtime._headroom_recovery is None
    assert recorded[0].outcome == SkillOutcome.CANCELLED
    assert sent
    assert runtime.executor.run is not None
    assert runtime.executor.run.skill_id == "respawn_after_death"
    assert all(track.track_id != track_id for track in runtime.blackboard.latest().tracks)
    for key in (
        "recovery.crosshair.block",
        "recovery.crosshair.frame_dhash",
        "recovery.crosshair.observation_dhash",
        "target.visible",
        "target.kind",
        "target.reference_available",
    ):
        assert runtime.blackboard.fact(key, now_ns=time.monotonic_ns()) is None
