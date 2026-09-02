from pathlib import Path
import time

import pytest

from minecraft_ai.config import PolicyConfig
from minecraft_ai.motor import MotorIntent
from minecraft_ai.perception import FrameState, PerceptionBlackboard, PerceptionFact
from minecraft_ai.platforms.bedrock_x11 import CapturedFrame
from minecraft_ai.policy_service import (
    LearnedPolicyOutput,
    TemporalPolicyClient,
    _decoded_policy_output,
    _intent_instruction,
    _learned_scene_blocked,
    _rocket_interaction_id,
    _track_mask,
    _validate_policy_config,
)


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


def test_camera_envelope_reconditions_learned_policy_without_scripted_motion(
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
    assert second.keys_up == ("w",)
    assert second.buttons_up == ("left",)
    assert client._camera_recovery_active is True
    recovery = client._conditioned_intent(MotorIntent(skill_id="explore", mode="explore"))
    assert "level horizon" in str(recovery["instruction"])

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


def test_async_policy_preserves_held_state_inside_inference_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = CapturedFrame(frame_id=1, captured_ns=1, width=1, height=1, bgra=b"\0" * 4)
    client = TemporalPolicyClient(
        config=_policy_config(tmp_path, deadline_ms=150),
        frame_provider=lambda: frame,
    )
    client._held_keys = {"w"}
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
