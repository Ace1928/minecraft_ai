from pathlib import Path
import time

import pytest

from minecraft_ai.config import PolicyConfig
from minecraft_ai.motor import MotorIntent
from minecraft_ai.perception import (
    FrameState,
    PerceptionBlackboard,
    PerceptionFact,
    ScreenRegion,
    Track,
)
from minecraft_ai.platforms.bedrock_x11 import CapturedFrame
from minecraft_ai.policy_service import (
    LearnedPolicyOutput,
    GroundedPolicyRouter,
    TemporalPolicyClient,
    _decoded_policy_output,
    _intent_instruction,
    _learned_scene_blocked,
    _rocket_interaction_id,
    _track_mask,
    _validate_policy_config,
)
from minecraft_ai.safety import MotorAction


class _RoutingPolicy:
    def __init__(self, policy_id: str, *, key: str) -> None:
        self.policy_id = policy_id
        self.key = key
        self.calls = 0
        self.resets = 0
        self.warmups = 0

    def act(
        self,
        blackboard: PerceptionBlackboard,
        intent: MotorIntent,
        *,
        sequence: int,
    ) -> MotorAction:
        self.calls += 1
        return MotorAction(sequence=sequence, keys_down=(self.key,))

    def reset(self) -> MotorAction:
        self.resets += 1
        return MotorAction(sequence=self.calls, keys_up=(self.key,))

    def warmup(self) -> None:
        self.warmups += 1


def _tracked_board(*, age_ms: int = 0) -> PerceptionBlackboard:
    now = time.monotonic_ns()
    board = PerceptionBlackboard()
    board.publish(
        FrameState(
            frame_id=1,
            captured_ns=now,
            instance_id="bedrock:test",
            width=1280,
            height=720,
            tracks=(
                Track(
                    track_id="log-1",
                    label="oak log",
                    confidence=0.9,
                    region=ScreenRegion(x=0.4, y=0.25, width=0.2, height=0.5),
                    first_seen_ns=now - age_ms * 1_000_000,
                    last_seen_ns=now - age_ms * 1_000_000,
                ),
            ),
        )
    )
    return board


def _operator_tracked_board(*, age_ms: int) -> PerceptionBlackboard:
    board = _tracked_board(age_ms=age_ms)
    current = board.latest()
    assert current is not None
    track = current.tracks[0].model_copy(
        update={"attributes": {"source": "operator", "grounding": "explicit-region"}}
    )
    board.upsert_semantic_track(instance_id=current.instance_id, track=track)
    return board


def test_grounded_router_requires_fresh_track_and_supported_interaction() -> None:
    primary = _RoutingPolicy("steve", key="w")
    grounded = _RoutingPolicy("rocket", key="a")
    router = GroundedPolicyRouter(primary, grounded, max_track_age_ms=100)
    mine = MotorIntent(skill_id="mine", mode="mine")

    first = router.act(PerceptionBlackboard(), mine, sequence=1)
    assert first.keys_down == ("w",)
    assert grounded.calls == 0

    second = router.act(_tracked_board(), mine, sequence=2)
    assert second.keys_down == ("a",)
    assert second.keys_up == ("w",)
    assert router.status()["active_route"] == "grounded"

    third = router.act(_tracked_board(age_ms=500), mine, sequence=3)
    assert third.keys_down == ("w",)
    assert third.keys_up == ("a",)
    assert router.status()["switches"] == 2

    explore = MotorIntent(skill_id="explore", mode="explore")
    router.act(_tracked_board(), explore, sequence=4)
    assert grounded.calls == 1


def test_grounded_router_prewarms_both_learned_controllers() -> None:
    primary = _RoutingPolicy("steve", key="w")
    grounded = _RoutingPolicy("rocket", key="a")
    router = GroundedPolicyRouter(primary, grounded)

    router.warmup()

    assert primary.warmups == 1
    assert grounded.warmups == 1


def test_grounded_router_keeps_explicit_cross_view_goal_until_cleared() -> None:
    primary = _RoutingPolicy("steve", key="w")
    grounded = _RoutingPolicy("rocket", key="a")
    router = GroundedPolicyRouter(primary, grounded, max_track_age_ms=100)

    action = router.act(
        _operator_tracked_board(age_ms=10_000),
        MotorIntent(skill_id="mine", mode="mine"),
        sequence=1,
    )

    assert action.keys_down == ("a",)
    assert router.status()["active_route"] == "grounded"


def _policy_config(tmp_path: Path, *, deadline_ms: int = 48) -> PolicyConfig:
    executable = tmp_path / "python"
    source = tmp_path / "source"
    model = tmp_path / "model"
    weights = tmp_path / "weights"
    executable.touch()
    source.mkdir()
    model.touch()
    weights.touch()
    return PolicyConfig(
        enabled=True,
        python_path=str(executable),
        source_path=str(source),
        model_path=str(model),
        weights_path=str(weights),
        model_sha256="a" * 64,
        weights_sha256="b" * 64,
        model_version="official-v1",
        source_commit="c" * 40,
        license="MIT",
        deadline_ms=deadline_ms,
    )


def test_learned_policy_output_rejects_unknown_action_fields() -> None:
    with pytest.raises(ValueError):
        LearnedPolicyOutput.model_validate(
            {
                "keys": [],
                "buttons": [],
                "mouse_dx": 0,
                "mouse_dy": 0,
                "inference_ns": 1,
                "model_version": "v1",
                "scripted_recovery": True,
            }
        )


def test_policy_config_requires_hashes_provenance_and_paths(tmp_path: Path) -> None:
    config = _policy_config(tmp_path)
    _validate_policy_config(config)

    with pytest.raises(ValueError, match="license"):
        _validate_policy_config(config.model_copy(update={"license": "unknown"}))

    _validate_policy_config(
        config.model_copy(update={"license": "unverified", "research_only": True})
    )
    with pytest.raises(ValueError, match="camera_recovery_release"):
        _validate_policy_config(
            config.model_copy(update={"camera_pitch_limit": 20, "camera_recovery_release": 20})
        )


def test_temporal_policy_warmup_requires_and_uses_current_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = CapturedFrame(frame_id=1, captured_ns=1, width=2, height=2, bgra=b"\0" * 16)
    client = TemporalPolicyClient(
        config=_policy_config(tmp_path).model_copy(update={"startup_timeout_s": 75.0}),
        frame_provider=lambda: frame,
    )
    required_sizes: list[int] = []
    monkeypatch.setattr(client, "_ensure_started", required_sizes.append)

    client.warmup()

    assert required_sizes == [len(frame.bgra)]


def test_goal_instruction_prefers_semantic_skill_contract() -> None:
    assert (
        _intent_instruction(
            {
                "skill_id": "mine_visible_log",
                "instruction": "Approach and mine the visible oak log",
            }
        )
        == "Approach and mine the visible oak log"
    )
    assert _intent_instruction({"skill_id": "mine_visible_log"}) == "mine visible log"


def test_decoded_policy_camera_scale_is_bounded_adapter_calibration() -> None:
    decoded = {
        "camera": [[10.0, -10.0]],
        "attack": [1],
        "use": [0],
        "forward": [1],
        "back": [0],
        "left": [0],
        "right": [0],
        "jump": [0],
        "sneak": [0],
        "sprint": [0],
        "inventory": [0],
        "drop": [0],
        **{f"hotbar.{slot}": [0] for slot in range(1, 10)},
    }

    output = _decoded_policy_output(
        decoded,
        inference_ns=1,
        model_version="goal-policy",
        camera_scale=0.5,
    )

    assert output.keys == ("w",)
    assert output.buttons == ("left",)
    assert output.mouse_dx == -5
    assert output.mouse_dy == 5


def test_camera_envelope_saturates_without_replacing_learned_task(
    tmp_path: Path,
) -> None:
    config = _policy_config(tmp_path).model_copy(
        update={"camera_max_step": 3, "camera_pitch_limit": 5, "camera_recovery_release": 2}
    )
    client = TemporalPolicyClient(config=config, frame_provider=lambda: None)
    output = LearnedPolicyOutput(
        keys=("w",),
        buttons=("left",),
        mouse_dx=9,
        mouse_dy=9,
        inference_ns=1,
        model_version="official-v1",
    )

    first = client._output_action(output, sequence=1)
    second = client._output_action(output, sequence=2)

    assert (first.mouse_dx, first.mouse_dy) == (3, 3)
    assert (second.mouse_dx, second.mouse_dy) == (3, 2)
    assert second.keys_up == ()
    assert second.buttons_up == ()
    assert client._camera_recovery_active is False
    recovery = client._conditioned_intent(MotorIntent(skill_id="explore", mode="explore"))
    assert recovery["skill_id"] == "explore"
    assert recovery["interaction_id"] == -1

    upward = output.model_copy(update={"keys": (), "buttons": (), "mouse_dy": -9})
    client._output_action(upward, sequence=3)
    client._output_action(upward, sequence=4)
    assert client._estimated_pitch_units == -1
    assert client._camera_recovery_active is False


def test_async_policy_does_not_double_count_an_already_recorded_deadline_miss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _policy_config(tmp_path)
    frame = CapturedFrame(frame_id=1, captured_ns=1, width=1, height=1, bgra=b"\0" * 4)
    client = TemporalPolicyClient(config=config, frame_provider=lambda: frame)
    monkeypatch.setattr(client, "_ensure_started", lambda _size: None)

    def consume() -> dict[str, object]:
        client._consumed_miss_recorded = True
        return {
            "output": {
                "inference_ns": 50_000_000,
                "model_version": "official-v1",
            }
        }

    monkeypatch.setattr(client, "_consume_pending_response", consume)
    monkeypatch.setattr(client, "_submit", lambda _frame, _intent, _board: None)

    action = client.act(
        PerceptionBlackboard(),
        MotorIntent(skill_id="explore", mode="explore"),
        sequence=1,
    )

    assert action.mouse_dx == 0
    assert action.mouse_dy == 0
    assert client.metrics.deadline_misses == 0


def test_async_policy_preserves_held_state_only_inside_action_duration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = CapturedFrame(frame_id=1, captured_ns=1, width=1, height=1, bgra=b"\0" * 4)
    client = TemporalPolicyClient(
        config=_policy_config(tmp_path, deadline_ms=150),
        frame_provider=lambda: frame,
    )
    client._held_keys = {"w"}
    client._held_until_ns = time.monotonic_ns() + 40_000_000
    client._pending_request_id = "in-flight"
    client._pending_deadline_ns = time.monotonic_ns() + 100_000_000
    monkeypatch.setattr(client, "_ensure_started", lambda _size: None)
    monkeypatch.setattr(client, "_consume_pending_response", lambda: None)

    action = client.act(
        PerceptionBlackboard(),
        MotorIntent(skill_id="explore", mode="explore"),
        sequence=1,
    )

    assert action.keys_up == ()
    assert client._held_keys == {"w"}
    assert client.metrics.deadline_misses == 0


def test_slow_policy_uses_bounded_sample_and_hold_window(tmp_path: Path) -> None:
    config = _policy_config(tmp_path).model_copy(update={"action_hold_ms": 250})
    client = TemporalPolicyClient(config=config, frame_provider=lambda: None)
    output = LearnedPolicyOutput(
        keys=("w",),
        inference_ns=1,
        model_version="official-v1",
    )

    before = time.monotonic_ns()
    action = client._output_action(output, sequence=1)

    assert action.duration_ms == 50
    assert client._held_until_ns >= before + 249_000_000
    assert client._held_until_ns <= before + 260_000_000


def test_async_policy_releases_expired_action_while_inference_is_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = CapturedFrame(frame_id=1, captured_ns=1, width=1, height=1, bgra=b"\0" * 4)
    client = TemporalPolicyClient(
        config=_policy_config(tmp_path, deadline_ms=150),
        frame_provider=lambda: frame,
    )
    client._held_keys = {"shift", "w"}
    client._held_buttons = {"left"}
    client._held_until_ns = time.monotonic_ns() - 1
    client._pending_request_id = "in-flight"
    client._pending_deadline_ns = time.monotonic_ns() + 100_000_000
    monkeypatch.setattr(client, "_ensure_started", lambda _size: None)
    monkeypatch.setattr(client, "_consume_pending_response", lambda: None)

    action = client.act(
        PerceptionBlackboard(),
        MotorIntent(skill_id="explore", mode="explore"),
        sequence=1,
    )

    assert action.keys_up == ("shift", "w")
    assert action.buttons_up == ("left",)
    assert not client._held_keys
    assert not client._held_buttons


def test_learned_static_gui_scene_blocks_world_policy_actions() -> None:
    now = time.monotonic_ns()
    board = PerceptionBlackboard()
    board.publish(
        FrameState(
            frame_id=1,
            captured_ns=now,
            instance_id="bedrock:test",
            width=1280,
            height=635,
            facts=(
                PerceptionFact(
                    key="scene.playable",
                    value=False,
                    confidence=0.99,
                    observed_ns=now,
                    source="vlm:qwen3-vl:test",
                    expires_after_ms=10_000,
                ),
                PerceptionFact(
                    key="scene.observation_dhash",
                    value="0123456789abcdef",
                    confidence=1.0,
                    observed_ns=now,
                    source="vlm:qwen3-vl:test",
                    expires_after_ms=10_000,
                ),
                PerceptionFact(
                    key="frame.dhash",
                    value="0123456789abcdef",
                    confidence=1.0,
                    observed_ns=now,
                    source="bootstrap:image-signal:not-training-label",
                    expires_after_ms=10_000,
                ),
            ),
        )
    )

    assert _learned_scene_blocked(board)


def test_rocket_interaction_taxonomy_matches_published_control_contract() -> None:
    assert _rocket_interaction_id("attack") == 0
    assert _rocket_interaction_id("gather_wood") == 2
    assert _rocket_interaction_id("interact") == 3
    assert _rocket_interaction_id("craft_planks") == 4
    assert _rocket_interaction_id("hotbar") == 5
    assert _rocket_interaction_id("approach") == 6
    assert _rocket_interaction_id("explore") == -1


def test_rocket_grounding_mask_uses_observed_track_region() -> None:
    numpy = pytest.importorskip("numpy")
    track = {
        "region": {"x": 0.25, "y": 0.20, "width": 0.50, "height": 0.40},
    }

    mask = _track_mask(100, 200, track, numpy)

    assert mask.shape == (100, 200)
    assert int(mask.sum()) == 40 * 100
    assert mask[20, 50] == 1
    assert mask[59, 149] == 1
