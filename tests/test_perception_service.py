from __future__ import annotations

import time

from minecraft_ai.models import ModelResponse
from minecraft_ai.perception import (
    ActivePerceptionQuery,
    FrameState,
    PerceptionBlackboard,
    PerceptionFact,
)
from minecraft_ai.perception_service import (
    ActiveVLMWorker,
    SemanticJob,
    SemanticObservation,
    frame_dhash,
    perceptual_hash_distance,
)
from minecraft_ai.platforms.bedrock_x11 import CapturedFrame


class _UnusedVisionModel:
    model_id = "learned-test-vlm"

    def inspect(self, prompt: str, *, image_bytes: bytes, mime_type: str) -> ModelResponse:
        raise AssertionError((prompt, image_bytes, mime_type))


class _StructuredVisionModel:
    model_id = "structured-test-vlm"

    def __init__(self) -> None:
        self.schema: dict[str, object] | None = None

    def inspect(self, prompt: str, *, image_bytes: bytes, mime_type: str) -> ModelResponse:
        raise AssertionError("unstructured inspection must not be used when schema support exists")

    def inspect_structured(
        self,
        prompt: str,
        *,
        image_bytes: bytes,
        mime_type: str,
        name: str,
        schema: dict[str, object],
    ) -> ModelResponse:
        assert "Every tracks entry must be an object" in prompt
        assert image_bytes.startswith(b"\x89PNG")
        assert mime_type == "image/png"
        assert name == "minecraft_semantic_observation"
        self.schema = schema
        return ModelResponse(
            text=(
                '{"scene_mode":"world","scene_playable":true,"uncertainty":0.1,'
                '"danger_immediate":false,"obstacle_ahead":false,"target_visible":true,'
                '"scene_summary":"Tree ahead","target_dx":0.0,"target_dy":0.0,'
                '"target_kind":"oak_log","target_mineable":true,"target_near":false,'
                '"inventory_logs":null,"inventory_planks":null,'
                '"inventory_crafting_table":null,"inventory_build_blocks":null,'
                '"player_submerged":null,"player_air_visible":null,"facts":{},'
                '"confidences":{},'
                '"tracks":[{"label":"oak_log","confidence":0.9,"x":0.5,"y":0.5,'
                '"width":0.2,"height":0.4}],"chat":[]}'
            ),
            model=self.model_id,
            latency_ms=12.0,
        )


def _frame(pixels: bytes, *, width: int = 9, height: int = 8) -> CapturedFrame:
    return CapturedFrame(
        frame_id=1,
        captured_ns=1,
        width=width,
        height=height,
        bgra=pixels,
    )


def _hash_fact(value: str) -> PerceptionFact:
    return PerceptionFact(
        key="frame.dhash",
        value=value,
        confidence=1.0,
        observed_ns=time.monotonic_ns(),
        source="bootstrap:image-signal:not-training-label",
        expires_after_ms=10_000,
    )


def test_frame_dhash_is_deterministic_and_tracks_visual_change() -> None:
    rising = b"".join(bytes((value, value, value, 255)) for _row in range(8) for value in range(9))
    falling = b"".join(
        bytes((value, value, value, 255)) for _row in range(8) for value in reversed(range(9))
    )

    first = frame_dhash(_frame(rising))
    second = frame_dhash(_frame(rising))
    changed = frame_dhash(_frame(falling))

    assert first == second
    assert perceptual_hash_distance(first, changed) == 64


def test_active_vlm_prefers_strict_structured_vision_contract() -> None:
    model = _StructuredVisionModel()
    worker = ActiveVLMWorker(model, PerceptionBlackboard(), "bedrock:test")
    frame = _frame(b"\0" * (9 * 8 * 4))
    observation, latency_ms = worker._inspect(
        SemanticJob(
            query=ActivePerceptionQuery(query_id="q-structured", question="find wood", frame_id=1),
            frame=frame,
            frame_dhash=frame_dhash(frame),
        )
    )

    assert model.schema is not None
    assert observation.canonical_facts()["target.kind"] == "oak_log"
    assert observation.tracks[0].label == "oak_log"
    assert latency_ms == 12.0


def test_slow_vlm_result_survives_frame_age_when_scene_is_visually_unchanged() -> None:
    board = PerceptionBlackboard()
    board.publish(
        FrameState(
            frame_id=500,
            captured_ns=500,
            instance_id="bedrock:test",
            width=9,
            height=8,
            facts=(_hash_fact("0123456789abcdef"),),
        )
    )
    worker = ActiveVLMWorker(_UnusedVisionModel(), board, "bedrock:test")
    job = SemanticJob(
        query=ActivePerceptionQuery(query_id="q1", question="scene", frame_id=1),
        frame=_frame(b"\0" * (9 * 8 * 4)),
        frame_dhash="0123456789abcdef",
    )

    worker._publish(
        job,
        SemanticObservation(
            scene_mode="menu",
            scene_playable=False,
            uncertainty=0.1,
            danger_immediate=False,
            obstacle_ahead=False,
            target_visible=False,
            scene_summary="Menu",
            facts={"scene.mode": "menu", "scene.playable": False},
            confidences={"scene.mode": 0.99, "scene.playable": 0.99},
        ),
    )

    assert board.fact("scene.mode") is not None
    assert board.fact("scene.playable") is not None
    assert worker.metrics.stale_rejections == 0


def test_slow_vlm_result_is_rejected_after_material_scene_change() -> None:
    board = PerceptionBlackboard()
    board.publish(
        FrameState(
            frame_id=500,
            captured_ns=500,
            instance_id="bedrock:test",
            width=9,
            height=8,
            facts=(_hash_fact("ffffffffffffffff"),),
        )
    )
    worker = ActiveVLMWorker(_UnusedVisionModel(), board, "bedrock:test")
    job = SemanticJob(
        query=ActivePerceptionQuery(query_id="q1", question="scene", frame_id=1),
        frame=_frame(b"\0" * (9 * 8 * 4)),
        frame_dhash="0000000000000000",
    )

    worker._publish(
        job,
        SemanticObservation(
            scene_mode="menu",
            scene_playable=False,
            uncertainty=0.1,
            danger_immediate=False,
            obstacle_ahead=False,
            target_visible=False,
            scene_summary="Menu",
            facts={"scene.mode": "menu", "scene.playable": False},
            confidences={"scene.mode": 0.99, "scene.playable": 0.99},
        ),
    )

    assert board.fact("scene.mode") is None
    assert worker.metrics.stale_rejections == 1
