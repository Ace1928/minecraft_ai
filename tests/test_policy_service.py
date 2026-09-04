import hashlib
import time
from pathlib import Path

import numpy
import pytest

from minecraft_ai.action_levels import ActionLevel
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
    GroundedTargetObservation,
    LearnedSceneObservation,
    LearnedPolicyOutput,
    GroundedPolicyRouter,
    TemporalPolicyClient,
    _PolicyRequestContext,
    _RocketTwoBackend,
    _SteveOneBackend,
    _VPTBackend,
    _apply_action_constraints,
    _apply_discrete_action_contract,
    _crop_bbox_to_full,
    _decoded_policy_output,
    _fast_scene_belief,
    _intent_camera_scale,
    _intent_camera_semantics,
    _apply_observed_scene_action_contract,
    _intent_instruction,
    _learned_scene_blocked,
    _mineclip_scene_belief,
    _rocket_action_contract,
    _rocket_interaction_id,
    _rocket_reference_frame,
    _track_mask,
    _validate_policy_config,
)
from minecraft_ai.safety import MotorAction
from minecraft_ai.trajectory import motor_condition_id


class _RoutingPolicy:
    def __init__(self, policy_id: str, *, key: str) -> None:
        self.policy_id = policy_id
        self.key = key
        self.calls = 0
        self.resets = 0
        self.warmups = 0
        self.reported_status: dict[str, object] = {}
        self.last_intent: MotorIntent | None = None

    def act(
        self,
        blackboard: PerceptionBlackboard,
        intent: MotorIntent,
        *,
        sequence: int,
    ) -> MotorAction:
        self.calls += 1
        self.last_intent = intent
        return MotorAction(sequence=sequence, keys_down=(self.key,))

    def reset(self) -> MotorAction:
        self.resets += 1
        return MotorAction(sequence=self.calls, keys_up=(self.key,))

    def warmup(self) -> None:
        self.warmups += 1

    def status(self) -> dict[str, object]:
        return {"policy_id": self.policy_id, **self.reported_status}


class _TargetFeedbackPolicy(_RoutingPolicy):
    def __init__(self, policy_id: str, *, key: str) -> None:
        super().__init__(policy_id, key=key)
        self.observation: GroundedTargetObservation | None = None

    def target_observation(self) -> GroundedTargetObservation | None:
        return self.observation


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


def _operator_tracked_board(
    *,
    age_ms: int,
    reference_dhash: str | None = None,
    current_dhash: str | None = None,
    reference_image_path: Path | None = None,
) -> PerceptionBlackboard:
    board = _tracked_board(age_ms=age_ms)
    current = board.latest()
    assert current is not None
    attributes = {"source": "operator", "grounding": "explicit-region"}
    if reference_dhash is not None:
        attributes["reference_dhash"] = reference_dhash
    if reference_image_path is not None:
        attributes["reference_image_path"] = str(reference_image_path)
        attributes["reference_image_sha256"] = "a" * 64
    track = current.tracks[0].model_copy(update={"attributes": attributes})
    board.upsert_semantic_track(instance_id=current.instance_id, track=track)
    if current_dhash is not None:
        board.merge_semantics(
            instance_id=current.instance_id,
            facts=(
                PerceptionFact(
                    key="frame.dhash",
                    value=current_dhash,
                    confidence=1.0,
                    observed_ns=time.monotonic_ns(),
                    source="bootstrap:test",
                    expires_after_ms=1000,
                ),
            ),
        )
    return board


def test_grounded_router_uses_rocket_only_as_grounding_observer() -> None:
    primary = _RoutingPolicy("steve", key="w")
    grounded = _RoutingPolicy("rocket", key="a")
    router = GroundedPolicyRouter(primary, grounded, max_track_age_ms=100)
    mine = MotorIntent(
        skill_id="mine",
        mode="mine",
        episode_id="mine-1",
        action_level=ActionLevel.GROUNDED,
    )

    first = router.act(PerceptionBlackboard(), mine, sequence=1)
    assert first.keys_down == ("w",)
    assert grounded.calls == 0

    second = router.act(_tracked_board(), mine, sequence=2)
    assert second.keys_down == ("w",)
    assert second.keys_up == ()
    assert grounded.calls == 1
    status = router.status()
    assert status["active_route"] == "semantic"
    assert status["grounding_role"] == "asynchronous-target-belief-only"
    assert status["grounding_discarded_actions"] == 1

    router.reset()
    third = router.act(
        _tracked_board(age_ms=500),
        mine.model_copy(update={"episode_id": "mine-2"}),
        sequence=4,
    )
    assert third.keys_down == ("w",)
    assert third.keys_up == ()
    assert router.status()["switches"] == 0

    explore = MotorIntent(
        skill_id="explore",
        mode="explore",
        episode_id="explore-1",
        action_level=ActionLevel.LATENT,
    )
    router.act(_tracked_board(), explore, sequence=5)
    assert grounded.calls == 1


def test_grounded_router_prewarms_every_learned_controller() -> None:
    primary = _RoutingPolicy("steve", key="w")
    grounded = _RoutingPolicy("rocket", key="a")
    raw_motion = _RoutingPolicy("vpt", key="space")
    router = GroundedPolicyRouter(primary, grounded, raw_motion=raw_motion)

    router.warmup()

    assert primary.warmups == 1
    assert grounded.warmups == 1
    assert raw_motion.warmups == 1


def test_grounded_router_binds_vpt_body_for_atomic_motion_episode() -> None:
    primary = _RoutingPolicy("steve", key="w")
    grounded = _RoutingPolicy("rocket", key="a")
    raw_motion = _RoutingPolicy("openai-vpt-1x", key="space")
    router = GroundedPolicyRouter(primary, grounded, raw_motion=raw_motion)
    motion = MotorIntent(
        skill_id="traverse_visible_obstacle",
        mode="traverse_obstacle",
        episode_id="motion-1",
        action_level=ActionLevel.MOTION,
    )

    first = router.act(PerceptionBlackboard(), motion, sequence=1)
    assert first.keys_down == ("space",)
    assert first.keys_up == ("w",)

    # A malformed mid-episode level change is observable but cannot hand the
    # physical body to a different recurrent expert.
    conflicting = router.act(
        PerceptionBlackboard(),
        motion.model_copy(update={"action_level": ActionLevel.LATENT}),
        sequence=2,
    )
    assert conflicting.keys_down == ("space",)
    assert primary.calls == 0

    semantic = router.act(
        PerceptionBlackboard(),
        MotorIntent(
            skill_id="explore_forward",
            mode="explore",
            episode_id="semantic-1",
            action_level=ActionLevel.LATENT,
        ),
        sequence=3,
    )
    assert semantic.keys_down == ("w",)
    assert semantic.keys_up == ("space",)
    status = router.status()
    assert status["active_route"] == "semantic"
    assert status["episode_bindings"] == 2
    assert status["episode_binding_conflicts"] == 1
    assert status["switches"] == 2
    assert status["raw_motion"] == {"policy_id": "openai-vpt-1x"}


def test_grounded_router_routes_blocking_gui_to_dedicated_learned_expert() -> None:
    primary = _RoutingPolicy("steve", key="w")
    grounded = _RoutingPolicy("rocket", key="a")
    gui = _RoutingPolicy("vpt-gui", key="e")
    router = GroundedPolicyRouter(primary, grounded, gui=gui)

    action = router.act(
        PerceptionBlackboard(),
        MotorIntent(
            skill_id="respawn_after_death",
            mode="death_gui",
            episode_id="respawn-1",
            action_level=ActionLevel.GUI,
        ),
        sequence=1,
    )

    assert action.keys_down == ("e",)
    assert action.keys_up == ("w",)
    assert router.status()["active_route"] == "gui"
    assert router.status()["gui"] == {"policy_id": "vpt-gui"}
    router.warmup()
    assert (primary.warmups, grounded.warmups, gui.warmups) == (1, 1, 1)


def test_grounded_router_excludes_observer_camera_from_physical_pitch(tmp_path: Path) -> None:
    config = _policy_config(tmp_path).model_copy(
        update={
            "camera_max_step": 100,
            "camera_pitch_limit": 100,
            "camera_recovery_release": 50,
        }
    )
    primary = TemporalPolicyClient(config=config, frame_provider=lambda: None)
    grounded = TemporalPolicyClient(config=config, frame_provider=lambda: None)
    raw_motion = TemporalPolicyClient(config=config, frame_provider=lambda: None)
    router = GroundedPolicyRouter(primary, grounded, raw_motion=raw_motion)

    primary._output_action(
        LearnedPolicyOutput(
            mouse_dy=30,
            inference_ns=1,
            model_version="primary",
        ),
        sequence=1,
    )
    raw_motion._output_action(
        LearnedPolicyOutput(
            mouse_dy=40,
            inference_ns=1,
            model_version="raw-motion",
        ),
        sequence=1,
    )

    assert primary.status()["estimated_pitch_units"] == 70
    assert raw_motion.status()["estimated_pitch_units"] == 70
    assert grounded.status()["estimated_pitch_units"] == 0

    discarded = grounded._output_action(
        LearnedPolicyOutput(
            mouse_dy=50,
            inference_ns=1,
            model_version="grounded",
        ),
        sequence=2,
    )

    assert discarded.mouse_dy == 50
    assert primary.status()["estimated_pitch_units"] == 70
    saturated = raw_motion._output_action(
        LearnedPolicyOutput(
            mouse_dy=50,
            inference_ns=1,
            model_version="raw-motion",
        ),
        sequence=2,
    )

    assert saturated.mouse_dy == 30
    assert primary.status()["estimated_pitch_units"] == 100
    assert router.status()["world_camera"] == {"estimated_pitch_units": 100}

    router.restore_world_camera_state(estimated_pitch_units=-37)

    assert primary.status()["estimated_pitch_units"] == -37
    assert raw_motion.status()["estimated_pitch_units"] == -37
    assert grounded.status()["estimated_pitch_units"] == 50
    assert grounded.status()["camera_envelope_saturated"] is False


def test_restored_off_center_camera_never_synthesizes_recovery_motion(
    tmp_path: Path,
) -> None:
    config = _policy_config(tmp_path).model_copy(
        update={
            "camera_max_step": 3,
            "camera_pitch_limit": 100,
            "camera_recovery_release": 10,
        }
    )
    client = TemporalPolicyClient(config=config, frame_provider=lambda: None)
    client.restore_world_camera_state(estimated_pitch_units=-16)

    assert client.status()["camera_envelope_saturated"] is False

    learned = LearnedPolicyOutput(
        keys=("w",),
        mouse_dx=2,
        mouse_dy=-20,
        inference_ns=1,
        model_version="official-v1",
    )
    first = client._output_action(learned, sequence=1)
    second = client._hold(sequence=2)

    assert (first.mouse_dx, first.mouse_dy) == (2, -3)
    assert first.keys_down == ("w",)
    assert (second.mouse_dx, second.mouse_dy) == (0, -3)
    assert client.status()["estimated_pitch_units"] == -22
    assert client.status()["camera_envelope_saturated"] is False
    assert client.status()["pending_camera"] == {"mouse_dx": 0, "mouse_dy": -14}


def test_grounded_router_merges_temporally_filtered_target_feedback() -> None:
    primary = _RoutingPolicy("steve", key="w")
    grounded = _TargetFeedbackPolicy("rocket", key="a")
    router = GroundedPolicyRouter(primary, grounded)
    board = _tracked_board()
    router.act(
        board,
        MotorIntent(
            skill_id="approach",
            mode="approach",
            episode_id="approach-1",
            action_level=ActionLevel.GROUNDED,
        ),
        sequence=1,
    )
    observed_ns = time.monotonic_ns()
    grounded.observation = GroundedTargetObservation(
        observed_ns=observed_ns,
        probability=0.95,
        point_yx=(0.5, 0.6),
        bbox_xyxy=(0.4, 0.2, 0.7, 0.8),
        model_version="rocket-test",
    )

    assert router.merge_perception(board)
    visible = board.fact("target.visible", now_ns=observed_ns)
    assert visible is not None and visible.value is True
    target_dx = board.fact("target.dx", now_ns=observed_ns)
    assert target_dx is not None and target_dx.value == pytest.approx(0.2)
    target_near = board.fact("target.near", now_ns=observed_ns)
    assert target_near is not None and target_near.value is True
    target_fraction = board.fact("target.screen_fraction", now_ns=observed_ns)
    assert target_fraction is not None and target_fraction.value == pytest.approx(0.18)
    target_proximity = board.fact("target.proximity", now_ns=observed_ns)
    assert target_proximity is not None and target_proximity.value == pytest.approx(1.0)
    updated = board.latest()
    assert updated is not None
    track = next(track for track in updated.tracks if track.track_id == "log-1")
    assert track.region.x == pytest.approx(0.4)
    assert track.region.y == pytest.approx(0.2)
    assert track.region.width == pytest.approx(0.3)
    assert track.region.height == pytest.approx(0.6)
    assert track.attributes["tracking_model_version"] == "rocket-test"
    assert not router.merge_perception(board)

    grounded.observation = grounded.observation.__class__(
        observed_ns=observed_ns + 1,
        probability=0.0,
        point_yx=None,
        bbox_xyxy=None,
        model_version="rocket-test",
    )
    assert router.merge_perception(board)
    after_one_miss = board.fact("target.visible", now_ns=observed_ns + 1)
    assert after_one_miss is not None and after_one_miss.value is True

    grounded.observation = grounded.observation.__class__(
        observed_ns=observed_ns + 2,
        probability=0.0,
        point_yx=None,
        bbox_xyxy=None,
        model_version="rocket-test",
    )
    assert router.merge_perception(board)
    after_two_misses = board.fact("target.visible", now_ns=observed_ns + 2)
    assert after_two_misses is not None and after_two_misses.value is False


def test_grounded_router_keeps_small_learned_target_box_outside_near_range() -> None:
    primary = _RoutingPolicy("steve", key="w")
    grounded = _TargetFeedbackPolicy("rocket", key="a")
    router = GroundedPolicyRouter(primary, grounded)
    board = _tracked_board()
    router.act(
        board,
        MotorIntent(
            skill_id="approach",
            mode="approach",
            episode_id="approach-1",
            action_level=ActionLevel.GROUNDED,
        ),
        sequence=1,
    )
    observed_ns = time.monotonic_ns()
    grounded.observation = GroundedTargetObservation(
        observed_ns=observed_ns,
        probability=0.95,
        point_yx=(0.5, 0.5),
        bbox_xyxy=(0.45, 0.45, 0.55, 0.55),
        model_version="rocket-test",
    )

    assert router.merge_perception(board)
    target_near = board.fact("target.near", now_ns=observed_ns)
    assert target_near is not None and target_near.value is False
    target_proximity = board.fact("target.proximity", now_ns=observed_ns)
    assert target_proximity is not None and target_proximity.value == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("probabilities", "mode", "playable"),
    (
        (
            {"world": 0.02, "inventory": 0.91, "chat": 0.04, "wall": 0.03},
            "inventory",
            False,
        ),
        (
            {"world": 0.14, "inventory": 0.27, "chat": 0.50, "wall": 0.09},
            "chat",
            False,
        ),
        (
            {"world": 0.21, "inventory": 0.14, "chat": 0.10, "wall": 0.55},
            "world",
            True,
        ),
        (
            {"world": 0.33, "inventory": 0.22, "chat": 0.25, "wall": 0.20},
            "world",
            True,
        ),
        (
            {"world": 0.24, "inventory": 0.26, "chat": 0.25, "wall": 0.25},
            "unknown",
            None,
        ),
    ),
)
def test_mineclip_scene_belief_is_conservative(
    probabilities: dict[str, float],
    mode: str,
    playable: bool | None,
) -> None:
    observed_mode, observed_playable, confidence, returned = _mineclip_scene_belief(probabilities)

    assert observed_mode == mode
    assert observed_playable is playable
    assert 0.0 <= confidence <= 1.0
    assert returned == probabilities


@pytest.mark.parametrize(
    ("probabilities", "mode", "playable"),
    (
        ({"world": 0.97, "inventory": 0.03}, "world", True),
        ({"world": 0.18, "inventory": 0.82}, "inventory", False),
        ({"world": 0.51, "inventory": 0.49}, "unknown", None),
    ),
)
def test_fast_scene_belief_respects_evidence_gate(
    probabilities: dict[str, float],
    mode: str,
    playable: bool | None,
) -> None:
    observed_mode, observed_playable, confidence, returned = _fast_scene_belief(
        probabilities,
        min_confidence=0.80,
    )

    assert observed_mode == mode
    assert observed_playable is playable
    assert 0.0 <= confidence <= 1.0
    assert returned == probabilities


def test_temporal_policy_merges_learned_scene_belief_once(tmp_path: Path) -> None:
    client = TemporalPolicyClient(config=_policy_config(tmp_path), frame_provider=lambda: None)
    board = _tracked_board()
    observed_ns = time.monotonic_ns()
    client._last_scene_observation = LearnedSceneObservation(
        observed_ns=observed_ns,
        mode="inventory",
        playable=False,
        confidence=0.91,
        class_probabilities={
            "world": 0.02,
            "inventory": 0.91,
            "chat": 0.04,
            "wall": 0.03,
        },
        model_version="steve-test/mineclip",
    )

    assert client.merge_perception(board)
    playable = board.fact("scene.playable", now_ns=observed_ns)
    assert playable is not None
    assert playable.value is False
    assert playable.confidence == pytest.approx(0.91)
    assert playable.source.endswith(":scene:steve-test/mineclip")
    assert not client.merge_perception(board)


def test_policy_crop_box_maps_back_to_wide_full_frame() -> None:
    mapped = _crop_bbox_to_full(1279, 635, (0.0, 0.25, 1.0, 0.75))

    assert mapped is not None
    x0, y0, x1, y1 = mapped
    assert x0 == pytest.approx(75 / 1279)
    assert x1 == pytest.approx((75 + 1129) / 1279)
    assert y0 == pytest.approx(0.25)
    assert y1 == pytest.approx(0.75)


def test_grounded_router_rejects_stale_unbound_operator_region() -> None:
    primary = _RoutingPolicy("steve", key="w")
    grounded = _RoutingPolicy("rocket", key="a")
    router = GroundedPolicyRouter(primary, grounded, max_track_age_ms=100)

    action = router.act(
        _operator_tracked_board(age_ms=10_000),
        MotorIntent(
            skill_id="mine",
            mode="mine",
            episode_id="mine-1",
            action_level=ActionLevel.GROUNDED,
        ),
        sequence=1,
    )

    assert action.keys_down == ("w",)
    assert router.status()["active_route"] == "semantic"
    assert router.status()["grounding_active"] is False


def test_grounded_router_holds_hash_bound_target_only_inside_active_option() -> None:
    primary = _RoutingPolicy("steve", key="w")
    grounded = _RoutingPolicy("rocket", key="a")
    router = GroundedPolicyRouter(primary, grounded, max_track_age_ms=100)
    mine = MotorIntent(
        skill_id="mine",
        mode="mine",
        episode_id="mine-1",
        action_level=ActionLevel.GROUNDED,
    )
    board = _operator_tracked_board(
        age_ms=10_000,
        reference_dhash="0123456789abcdef",
        current_dhash="0123456789abcdef",
    )

    admitted = router.act(board, mine, sequence=1)
    assert admitted.keys_down == ("w",)
    assert grounded.calls == 1

    changed = board.fact("frame.dhash", min_confidence=1.0)
    assert changed is not None
    board.merge_semantics(
        instance_id="bedrock:test",
        facts=(changed.model_copy(update={"value": "fedcba9876543210"}),),
    )
    held = router.act(board, mine, sequence=2)
    assert held.keys_down == ("w",)
    assert grounded.calls == 2

    router.reset()
    rejected = router.act(
        board,
        mine.model_copy(update={"episode_id": "mine-2"}),
        sequence=4,
    )
    assert rejected.keys_down == ("w",)


def test_grounded_router_admits_persisted_cross_view_reference(tmp_path: Path) -> None:
    reference = tmp_path / "operator-reference.jpg"
    reference.write_bytes(b"reference")
    primary = _RoutingPolicy("steve", key="w")
    grounded = _RoutingPolicy("rocket", key="a")
    router = GroundedPolicyRouter(primary, grounded, max_track_age_ms=100)

    action = router.act(
        _operator_tracked_board(
            age_ms=10_000,
            current_dhash="fedcba9876543210",
            reference_image_path=reference,
        ),
        MotorIntent(
            skill_id="approach",
            mode="approach",
            episode_id="approach-1",
            action_level=ActionLevel.GROUNDED,
        ),
        sequence=1,
    )

    assert action.keys_down == ("w",)
    status = router.status()
    assert status["active_route"] == "semantic"
    assert status["grounding_active"] is True
    assert status["grounded_track_id"] == "log-1"
    assert status["grounding_episode_id"] == "approach-1"
    assert primary.last_intent is not None
    assert primary.last_intent.target_track_id == "log-1"
    assert grounded.last_intent is not None
    assert grounded.last_intent.target_track_id == "log-1"


def test_rocket_reference_frame_verifies_persisted_artifact(tmp_path: Path) -> None:
    payload = b"verified-reference"
    reference = tmp_path / "operator-reference.jpg"
    reference.write_bytes(payload)
    expected = numpy.zeros((8, 12, 3), dtype=numpy.uint8)

    class _CV2:
        IMREAD_COLOR = 1

        @staticmethod
        def imread(path: str, mode: int) -> numpy.ndarray:
            assert path == str(reference)
            assert mode == _CV2.IMREAD_COLOR
            return expected

    track = {
        "attributes": {
            "reference_image_path": str(reference),
            "reference_image_sha256": hashlib.sha256(payload).hexdigest(),
        }
    }

    loaded = _rocket_reference_frame(track, numpy.ones((4, 4, 4)), _CV2)

    assert loaded is expected
    track["attributes"]["reference_image_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        _rocket_reference_frame(track, numpy.ones((4, 4, 4)), _CV2)


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
    with pytest.raises(ValueError, match="action_hold_ms"):
        _validate_policy_config(
            config.model_copy(update={"action_hold_ms": 200, "action_hold_max_ms": 100})
        )

    scene_model = tmp_path / "fast-scene.pt"
    scene_model.touch()
    with pytest.raises(ValueError, match="scene model configuration is incomplete"):
        _validate_policy_config(config.model_copy(update={"scene_model_path": str(scene_model)}))
    _validate_policy_config(
        config.model_copy(
            update={
                "provider": "minestudio-steve1",
                "scene_model_path": str(scene_model),
                "scene_model_sha256": "d" * 64,
                "scene_model_version": "bedrock-fast-scene-v1",
            }
        )
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


def test_option_condition_scale_overrides_policy_default() -> None:
    from minecraft_ai.policy_service import _intent_condition_scale

    assert _intent_condition_scale({"condition_scale": 6.0}, default=4.0) == 6.0
    assert _intent_condition_scale({}, default=4.0) == 4.0


class _ConditionPolicy:
    def __init__(self) -> None:
        self.prepared: list[tuple[str, float, bool]] = []
        self.initialized: list[dict[str, object]] = []

    def prepare_condition(
        self,
        instruction: dict[str, object],
        *,
        deterministic: bool,
    ) -> dict[str, object]:
        text = str(instruction["text"])
        scale = float(instruction["cond_scale"])
        self.prepared.append((text, scale, deterministic))
        return {"text": text, "cond_scale": scale}

    def initial_state(
        self,
        _batch_size: int,
        condition: dict[str, object],
    ) -> list[str]:
        self.initialized.append(condition)
        return [f"initial:{condition['text']}"]


def test_steve_goal_change_preserves_same_episode_recurrent_state() -> None:
    policy = _ConditionPolicy()
    backend = object.__new__(_SteveOneBackend)
    backend.policy = policy
    backend.deterministic_condition = True
    backend.hidden_state = ["episode-memory"]
    backend.condition = {"text": "explore", "cond_scale": 4.0}
    backend.instruction = "explore"
    backend.active_condition_scale = 4.0
    backend.discrete_actions_emitted = {"inventory"}

    backend._update_condition("jump forward", 6.0)

    assert backend.hidden_state == ["episode-memory"]
    assert policy.initialized == []
    assert policy.prepared == [("jump forward", 6.0, True)]
    assert backend.discrete_actions_emitted == set()


def test_steve_cfg_topology_change_reinitializes_recurrent_state() -> None:
    policy = _ConditionPolicy()
    backend = object.__new__(_SteveOneBackend)
    backend.policy = policy
    backend.deterministic_condition = True
    backend.hidden_state = ["paired-cfg-memory"]
    backend.condition = {"text": "explore", "cond_scale": 4.0}
    backend.instruction = "explore"
    backend.active_condition_scale = 4.0
    backend.discrete_actions_emitted = set()

    backend._update_condition("explore", 0.0)

    assert backend.hidden_state == ["initial:explore"]
    assert policy.initialized == [{"text": "explore", "cond_scale": 0.0}]


def test_rocket_new_grounding_resets_recurrent_state_and_previous_action() -> None:
    backend = object.__new__(_RocketTwoBackend)
    backend.numpy = numpy
    backend.hidden_state = ["previous-target-memory"]
    backend.grounding_signature = "log-1:2"
    backend.previous_action = {
        "camera": numpy.asarray([4.0, -2.0], dtype=numpy.float32),
        "jump": numpy.asarray(1, dtype=numpy.int64),
    }

    backend._reset_grounding_context("coal-2:2")

    assert backend.hidden_state is None
    assert backend.grounding_signature == "coal-2:2"
    assert numpy.array_equal(
        backend.previous_action["camera"],
        numpy.zeros(2, dtype=numpy.float32),
    )
    assert all(
        int(backend.previous_action[key.replace("_", ".")]) == 0
        for key in (
            "forward",
            "back",
            "left",
            "right",
            "inventory",
            "sprint",
            "sneak",
            "jump",
            "attack",
            "use",
            "hotbar_1",
            "hotbar_2",
            "hotbar_3",
            "hotbar_4",
            "hotbar_5",
            "hotbar_6",
            "hotbar_7",
            "hotbar_8",
            "hotbar_9",
        )
    )


def test_decoded_policy_camera_uses_independent_axis_calibration() -> None:
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
        camera_pitch_scale=0.3,
    )

    assert output.keys == ("w",)
    assert output.buttons == ("left",)
    assert output.mouse_dx == -5
    assert output.mouse_dy == 3


def test_default_camera_adapter_matches_minecraft_half_sensitivity() -> None:
    config = PolicyConfig()

    assert config.camera_scale == pytest.approx(1.0 / 0.15)
    assert config.effective_camera_pitch_scale == pytest.approx(1.0 / 0.15)
    assert config.gui_camera_scale == pytest.approx(1.0)
    assert config.camera_max_step == 12
    assert config.camera_pitch_limit == 300
    assert config.scene_probe_interval == 0
    assert config.scene_min_confidence == pytest.approx(0.80)


def test_bedrock_camera_adapter_accepts_empirical_per_axis_scales() -> None:
    config = PolicyConfig(camera_scale=47.96, camera_pitch_scale=66.0)

    assert config.camera_scale == pytest.approx(47.96)
    assert config.effective_camera_pitch_scale == pytest.approx(66.0)


def test_gui_cursor_uses_pixel_scale_instead_of_world_camera_calibration() -> None:
    gui = {"mode": "gui"}
    craft = {"mode": "craft_inventory"}
    close_inventory = {"mode": "close_inventory"}
    world = {"mode": "explore"}

    assert _intent_camera_semantics(gui) == "cursor"
    assert _intent_camera_semantics(craft) == "cursor"
    assert _intent_camera_semantics(close_inventory) == "cursor"
    assert _intent_camera_semantics({"mode": "death_gui"}) == "cursor"
    assert _intent_camera_semantics(world) == "world"
    assert _intent_camera_scale(gui, world_scale=47.96, gui_scale=1.0) == 1.0
    assert _intent_camera_scale(close_inventory, world_scale=47.96, gui_scale=1.0) == 1.0
    assert _intent_camera_scale(world, world_scale=47.96, gui_scale=1.0) == 47.96


def test_cursor_motion_does_not_corrupt_world_pitch_estimate(tmp_path: Path) -> None:
    config = _policy_config(tmp_path).model_copy(
        update={
            "camera_max_step": 3,
            "camera_pitch_limit": 5,
            "camera_recovery_release": 2,
        }
    )
    client = TemporalPolicyClient(config=config, frame_provider=lambda: None)
    world = LearnedPolicyOutput(
        mouse_dy=9,
        inference_ns=1,
        model_version="official-v1",
    )
    cursor = world.model_copy(update={"mouse_dx": 5, "mouse_dy": 5, "camera_semantics": "cursor"})

    world_action = client._output_action(world, sequence=1)
    cursor_action = client._output_action(cursor, sequence=2)
    cursor_remainder = client._hold(sequence=3)

    assert (world_action.mouse_dx, world_action.mouse_dy) == (0, 3)
    assert (cursor_action.mouse_dx, cursor_action.mouse_dy) == (3, 3)
    assert (cursor_remainder.mouse_dx, cursor_remainder.mouse_dy) == (2, 2)
    assert client._world_camera_state.estimated_pitch_units == 3
    assert client._pending_camera_semantics == "world"


def test_camera_envelope_clamps_without_replacing_learned_task(
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
    assert client._camera_envelope_saturated is True
    recovery = client._conditioned_intent(MotorIntent(skill_id="explore", mode="explore"))
    assert recovery["skill_id"] == "explore"
    assert recovery["interaction_id"] == -1

    saturated = client._output_action(output, sequence=3)
    assert (saturated.mouse_dx, saturated.mouse_dy) == (3, 0)
    assert saturated.keys_up == ()
    assert saturated.buttons_up == ()
    assert client._world_camera_state.estimated_pitch_units == 5
    assert client._camera_envelope_saturated is True

    upward = output.model_copy(update={"keys": (), "buttons": (), "mouse_dy": -9})
    client._output_action(upward, sequence=4)
    client._output_action(upward, sequence=5)
    assert client._world_camera_state.estimated_pitch_units == -1
    assert client._camera_envelope_saturated is False


def test_camera_accumulator_preserves_motion_across_motor_ticks(tmp_path: Path) -> None:
    config = _policy_config(tmp_path).model_copy(
        update={
            "camera_max_step": 3,
            "camera_pitch_limit": 100,
            "camera_recovery_release": 50,
        }
    )
    client = TemporalPolicyClient(config=config, frame_provider=lambda: None)
    output = LearnedPolicyOutput(
        mouse_dx=9,
        mouse_dy=-8,
        inference_ns=1,
        model_version="official-v1",
    )

    first = client._output_action(output, sequence=1)
    second = client._hold(sequence=2)
    third = client._hold(sequence=3)
    fourth = client._hold(sequence=4)

    assert [(item.mouse_dx, item.mouse_dy) for item in (first, second, third, fourth)] == [
        (3, -3),
        (3, -3),
        (3, -2),
        (0, 0),
    ]
    status = client.status()
    assert status["pending_camera"] == {"mouse_dx": 0, "mouse_dy": 0}
    assert status["predicted_camera_total"] == {"mouse_dx": 9, "mouse_dy": -8}
    assert status["emitted_camera_total"] == {"mouse_dx": 9, "mouse_dy": -8}


def test_next_recurrent_observation_waits_for_complete_camera_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = CapturedFrame(frame_id=1, captured_ns=1, width=1, height=1, bgra=b"\0" * 4)
    config = _policy_config(tmp_path).model_copy(
        update={
            "camera_max_step": 3,
            "camera_pitch_limit": 100,
            "camera_recovery_release": 50,
        }
    )
    client = TemporalPolicyClient(config=config, frame_provider=lambda: frame)
    responses: list[dict[str, object] | None] = [
        {
            "output": {
                "mouse_dx": 9,
                "inference_ns": 1,
                "model_version": "official-v1",
            }
        },
        None,
        None,
        None,
    ]
    submissions: list[int] = []
    monkeypatch.setattr(client, "_ensure_started", lambda _size: None)
    monkeypatch.setattr(client, "_consume_pending_response", lambda: responses.pop(0))
    monkeypatch.setattr(
        client,
        "_submit",
        lambda submitted, _intent, _board: submissions.append(submitted.frame_id),
    )
    intent = MotorIntent(skill_id="traverse", mode="traverse")
    board = PerceptionBlackboard()

    actions = [client.act(board, intent, sequence=sequence) for sequence in range(1, 5)]

    assert [action.mouse_dx for action in actions] == [3, 3, 3, 0]
    assert submissions == [1]
    assert client.metrics.camera_feedback_waits == 2


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


@pytest.mark.parametrize("provider", ("minestudio-steve1", "minestudio-rocket2"))
def test_slow_learned_policy_renews_key_and_button_state_without_gaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    clock_ns = [1_000_000_000]
    monkeypatch.setattr(
        "minecraft_ai.policy_service.time.monotonic_ns",
        lambda: clock_ns[0],
    )
    client = TemporalPolicyClient(
        config=_policy_config(tmp_path, deadline_ms=1_000).model_copy(
            update={"provider": provider}
        ),
        frame_provider=lambda: None,
    )
    client._output_action(
        LearnedPolicyOutput(
            keys=("space", "w"),
            buttons=("left",),
            inference_ns=180_000_000,
            model_version="official-v1",
        ),
        sequence=1,
    )
    client._pending_request_id = "in-flight"
    client._pending_deadline_ns = 2_000_000_000
    clock_ns[0] = 1_300_000_000

    action = client._hold(sequence=2)

    assert action.keys_up == ()
    assert action.buttons_up == ()
    assert client._held_keys == {"space", "w"}
    assert client._held_buttons == {"left"}
    status = client.status()["adaptive_action_hold"]
    assert isinstance(status, dict)
    assert status["latency_ema_ms"] == 180.0
    assert status["horizon_ms"] == 216
    assert status["renewals"] == 1


def test_expired_learned_state_releases_keys_and_buttons_without_pending_prediction(
    tmp_path: Path,
) -> None:
    client = TemporalPolicyClient(config=_policy_config(tmp_path), frame_provider=lambda: None)
    client._held_keys = {"shift", "w"}
    client._held_buttons = {"left"}
    client._held_until_ns = time.monotonic_ns() - 1

    action = client._hold(sequence=1)

    assert action.keys_up == ("shift", "w")
    assert action.buttons_up == ("left",)
    assert not client._held_keys
    assert not client._held_buttons


def test_async_policy_releases_continuous_button_at_request_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = CapturedFrame(frame_id=1, captured_ns=1, width=1, height=1, bgra=b"\0" * 4)
    client = TemporalPolicyClient(
        config=_policy_config(tmp_path, deadline_ms=150),
        frame_provider=lambda: frame,
    )
    client._held_buttons = {"left"}
    client._pending_request_id = "in-flight"
    client._pending_deadline_ns = time.monotonic_ns() - 1
    monkeypatch.setattr(client, "_ensure_started", lambda _size: None)
    monkeypatch.setattr(client, "_consume_pending_response", lambda: None)

    action = client.act(
        PerceptionBlackboard(),
        MotorIntent(skill_id="mine", mode="mine"),
        sequence=1,
    )

    assert action.buttons_up == ("left",)
    assert not client._held_buttons
    assert client.metrics.deadline_misses == 1
    hold_status = client.status()["adaptive_action_hold"]
    assert isinstance(hold_status, dict)
    assert hold_status["last_invalidation_reason"] == "request-deadline-expired"


def test_safety_scene_invalidation_releases_learned_state_immediately(
    tmp_path: Path,
) -> None:
    client = TemporalPolicyClient(config=_policy_config(tmp_path), frame_provider=lambda: None)
    client._held_keys = {"w"}
    client._held_buttons = {"left"}
    client._held_until_ns = time.monotonic_ns() + 1_000_000_000
    board = _tracked_board()
    board.merge_semantics(
        instance_id="bedrock:test",
        facts=(
            PerceptionFact(
                key="scene.playable",
                value=False,
                confidence=1.0,
                observed_ns=time.monotonic_ns(),
                source="safety:test",
                expires_after_ms=1_000,
            ),
        ),
    )

    action = client.act(
        board,
        MotorIntent(skill_id="explore", mode="explore"),
        sequence=1,
    )

    assert action.keys_up == ("w",)
    assert action.buttons_up == ("left",)
    assert not client._held_keys
    assert not client._held_buttons
    hold_status = client.status()["adaptive_action_hold"]
    assert isinstance(hold_status, dict)
    assert hold_status["last_invalidation_reason"] == "scene-blocked"


def test_policy_status_exposes_predicted_and_emitted_camera(tmp_path: Path) -> None:
    config = _policy_config(tmp_path).model_copy(update={"camera_max_step": 1})
    client = TemporalPolicyClient(config=config, frame_provider=lambda: None)
    output = LearnedPolicyOutput(
        keys=("w",),
        buttons=("left",),
        mouse_dx=9,
        mouse_dy=-9,
        inference_ns=1,
        model_version="official-v1",
    )

    client._output_action(output, sequence=1)
    status = client.status()

    assert status["button_zero_order_hold"] is True
    assert status["last_prediction"] == {
        "keys": ("w",),
        "buttons": ("left",),
        "mouse_dx": 9,
        "mouse_dy": -9,
        "camera_semantics": "world",
        "target_exists_probability": None,
        "target_point_yx": None,
        "target_bbox_xyxy": None,
        "scene_mode": None,
        "scene_playable": None,
        "scene_confidence": None,
        "scene_class_probabilities": {},
        "scene_model_version": None,
        "behavior_token": None,
        "latent_id": None,
        "suppressed_actions": (),
    }
    assert status["last_emitted_camera"] == {"mouse_dx": 1, "mouse_dy": -1}
    assert status["pending_camera"] == {"mouse_dx": 8, "mouse_dy": -8}
    assert status["predicted_camera_total"] == {"mouse_dx": 9, "mouse_dy": -9}
    assert status["emitted_camera_total"] == {"mouse_dx": 1, "mouse_dy": -1}
    assert status["accepted_predictions"] == 1
    assert status["learned_action_counts"] == {
        "attack": 1,
        "camera": 1,
        "forward": 1,
    }


def test_async_action_provenance_stays_bound_to_consumed_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = CapturedFrame(frame_id=1, captured_ns=1, width=1, height=1, bgra=b"\0" * 4)
    client = TemporalPolicyClient(
        config=_policy_config(tmp_path, deadline_ms=500),
        frame_provider=lambda: frame,
    )
    board = _tracked_board()
    intent = MotorIntent(
        skill_id="mine_visible_block",
        mode="mine",
        episode_id="episode-old",
        action_level=ActionLevel.GROUNDED,
        target_label="oak log",
    )
    condition = client._conditioned_intent(intent, board)
    client._consumed_request_context = _PolicyRequestContext(
        request_id="request-old",
        condition=condition,
        target_track_id="log-1",
        interaction_id=2,
    )
    responses: list[dict[str, object] | None] = [
        {
            "output": {
                "keys": ["w"],
                "inference_ns": 1,
                "model_version": "official-v1",
                "behavior_token": 41,
                "latent_id": "z_041",
            }
        },
        None,
    ]
    monkeypatch.setattr(client, "_ensure_started", lambda _size: None)
    monkeypatch.setattr(client, "_consume_pending_response", lambda: responses.pop(0))

    client.act(board, intent, sequence=1)
    first = client.status()["last_action_provenance"]
    assert isinstance(first, dict)
    assert first["action_kind"] == "prediction"
    assert first["request_id"] == "request-old"
    assert first["episode_id"] == "episode-old"
    assert first["action_level"] == "grounded"
    assert first["target_track_id"] == "log-1"
    assert first["interaction_id"] == 2
    assert first["behavior_token"] == 41
    assert first["latent_id"] == "z_041"
    assert first["condition"] == condition

    new_intent = intent.model_copy(update={"episode_id": "episode-new"})
    new_condition = client._conditioned_intent(new_intent, board)
    client._pending_request_id = "request-new"
    client._pending_request_context = _PolicyRequestContext(
        request_id="request-new",
        condition=new_condition,
        target_track_id="log-1",
        interaction_id=2,
    )
    client._pending_deadline_ns = time.monotonic_ns() + 100_000_000

    client.act(board, new_intent, sequence=2)
    held = client.status()["last_action_provenance"]
    assert isinstance(held, dict)
    assert held["action_kind"] == "prediction_hold"
    assert held["request_id"] == "request-old"
    assert held["episode_id"] == "episode-old"
    pending = client.status()["pending_request"]
    assert isinstance(pending, dict)
    assert pending["request_id"] == "request-new"
    assert pending["condition"] == new_condition

    client.reset()
    released = client.status()["last_action_provenance"]
    assert isinstance(released, dict)
    assert released["action_kind"] == "release"
    assert released["request_id"] is None
    assert released["condition"] is None


def test_router_reidentifies_exact_prediction_condition_for_bound_route() -> None:
    primary = _RoutingPolicy("steve", key="w")
    grounded = _RoutingPolicy("rocket", key="a")
    raw_motion = _RoutingPolicy("openai-vpt-1x", key="space")
    condition = MotorIntent(
        skill_id="traverse_visible_obstacle",
        mode="traverse_obstacle",
        episode_id="motion-1",
        action_level=ActionLevel.MOTION,
    ).model_dump(mode="json")
    raw_motion.reported_status["last_action_provenance"] = {
        "action_kind": "prediction",
        "prediction_id": "request-1",
        "request_id": "request-1",
        "condition_id": "direct-id",
        "condition": condition,
        "episode_id": "motion-1",
        "action_level": "motion",
        "target_track_id": None,
        "interaction_id": -1,
        "policy_id": raw_motion.policy_id,
        "model_version": "openai-vpt-foundation-1x",
        "route_id": "direct",
        "behavior_token": None,
        "latent_id": None,
    }
    router = GroundedPolicyRouter(primary, grounded, raw_motion=raw_motion)
    router.act(
        PerceptionBlackboard(),
        MotorIntent.model_validate(condition),
        sequence=1,
    )

    provenance = router.status()["last_action_provenance"]
    assert isinstance(provenance, dict)
    assert provenance["route_id"] == "raw_motion"
    assert provenance["request_id"] == "request-1"
    assert provenance["condition_id"] == motor_condition_id(
        condition,
        route_id="raw_motion",
        target_track_id=None,
    )


def test_policy_status_counts_learned_jump_before_state_hold_actions(tmp_path: Path) -> None:
    client = TemporalPolicyClient(config=_policy_config(tmp_path), frame_provider=lambda: None)
    output = LearnedPolicyOutput(
        keys=("ctrl", "space", "w"),
        inference_ns=1,
        model_version="official-v1",
    )

    client._output_action(output, sequence=1)
    client._output_action(output, sequence=2)
    status = client.status()

    assert status["accepted_predictions"] == 2
    assert status["learned_action_counts"] == {
        "forward": 2,
        "jump": 2,
        "sprint_jump": 2,
    }


def test_policy_status_counts_learned_inventory_toggle(tmp_path: Path) -> None:
    client = TemporalPolicyClient(config=_policy_config(tmp_path), frame_provider=lambda: None)

    client._output_action(
        LearnedPolicyOutput(
            keys=("e",),
            inference_ns=1,
            model_version="official-v1",
        ),
        sequence=1,
    )

    assert client.status()["learned_action_counts"] == {"inventory": 1}


def test_explicit_action_constraints_mask_only_prohibited_learned_bits() -> None:
    decoded = {
        "attack": numpy.asarray([1]),
        "drop": numpy.asarray([1]),
        "hotbar.1": numpy.asarray([0]),
        "hotbar.4": numpy.asarray([1]),
        "inventory": numpy.asarray([1]),
        "use": numpy.asarray([1]),
        "jump": numpy.asarray([1]),
        "forward": numpy.asarray([1]),
    }

    constrained, suppressed = _apply_action_constraints(
        decoded,
        {
            "parameters": {
                "allow_attack": False,
                "allow_drop": False,
                "allow_hotbar": False,
                "allow_inventory": False,
                "allow_use": False,
                "allow_jump": True,
            }
        },
    )

    assert int(constrained["attack"][0]) == 0
    assert int(constrained["drop"][0]) == 0
    assert int(constrained["hotbar.1"][0]) == 0
    assert int(constrained["hotbar.4"][0]) == 0
    assert int(constrained["inventory"][0]) == 0
    assert int(constrained["use"][0]) == 0
    assert int(constrained["jump"][0]) == 1
    assert int(constrained["forward"][0]) == 1
    assert int(decoded["attack"][0]) == 1
    assert suppressed == ("attack", "use", "drop", "inventory", "hotbar")


def test_vpt_backend_enforces_option_action_constraints() -> None:
    class _Tensor:
        def __init__(self, value: numpy.ndarray) -> None:
            self.value = value

        def to(self, _device: str) -> "_Tensor":
            return self

        def cpu(self) -> "_Tensor":
            return self

        def numpy(self) -> numpy.ndarray:
            return self.value

    class _Torch:
        @staticmethod
        def from_numpy(value: numpy.ndarray) -> _Tensor:
            return _Tensor(value)

        @staticmethod
        def inference_mode():
            from contextlib import nullcontext

            return nullcontext()

    class _CV2:
        INTER_LINEAR = 1

        @staticmethod
        def resize(value: numpy.ndarray, _size: tuple[int, int], *, interpolation: int):
            assert interpolation == _CV2.INTER_LINEAR
            return value

    decoded = {
        "camera": numpy.asarray([[0.0, 0.0]]),
        "attack": numpy.asarray([1]),
        "use": numpy.asarray([1]),
        "forward": numpy.asarray([1]),
        "back": numpy.asarray([0]),
        "left": numpy.asarray([0]),
        "right": numpy.asarray([0]),
        "jump": numpy.asarray([0]),
        "sneak": numpy.asarray([0]),
        "sprint": numpy.asarray([0]),
        "inventory": numpy.asarray([0]),
        "drop": numpy.asarray([0]),
        **{f"hotbar.{slot}": numpy.asarray([0]) for slot in range(1, 10)},
    }

    class _Policy:
        @staticmethod
        def act(_observation, _first, hidden_state, *, stochastic: bool):
            assert stochastic
            return (
                {
                    "buttons": _Tensor(numpy.asarray([0])),
                    "camera": _Tensor(numpy.asarray([[0.0, 0.0]])),
                },
                hidden_state,
                None,
            )

    class _Mapper:
        @staticmethod
        def to_factored(_raw):
            return decoded

    class _Transformer:
        @staticmethod
        def policy2env(value):
            return value

    backend = object.__new__(_VPTBackend)
    backend.numpy = numpy
    backend.cv2 = _CV2()
    backend.torch = _Torch()
    backend.policy = _Policy()
    backend.mapper = _Mapper()
    backend.transformer = _Transformer()
    backend.hidden_state = []
    backend.first = object()
    backend.device = "cpu"
    backend.stochastic = True
    backend.model_version = "vpt-test"
    backend.camera_scale = 1.0
    backend.camera_pitch_scale = 1.0
    backend.gui_camera_scale = 1.0

    output = backend.infer(
        numpy.zeros((16, 16, 4), dtype=numpy.uint8),
        {"mode": "traverse", "parameters": {"allow_attack": False, "allow_use": False}},
    )

    assert output.keys == ("w",)
    assert output.buttons == ()
    assert output.suppressed_actions == ("attack", "use")


def test_close_inventory_learned_toggle_is_one_event_until_option_reset() -> None:
    emitted: set[str] = set()
    decoded = {
        "inventory": numpy.asarray([1]),
        "camera": numpy.asarray([[4, -2]]),
    }
    intent = {"mode": "close_inventory"}

    first, first_suppressed = _apply_discrete_action_contract(decoded, intent, emitted)
    repeated, repeated_suppressed = _apply_discrete_action_contract(
        decoded,
        intent,
        emitted,
    )

    assert int(first["inventory"][0]) == 1
    assert first_suppressed == ()
    assert int(repeated["inventory"][0]) == 0
    assert int(repeated["camera"][0][0]) == 4
    assert repeated_suppressed == ("inventory:repeat",)
    assert int(decoded["inventory"][0]) == 1


def test_close_inventory_does_not_reopen_verified_playable_scene() -> None:
    decoded = {
        "inventory": numpy.asarray([1]),
        "camera": numpy.asarray([[4, -2]]),
    }
    world_scene = (
        "world",
        True,
        0.99,
        {"world": 0.99, "inventory": 0.01},
        "bedrock-fast-scene-v1",
    )

    constrained, suppressed = _apply_observed_scene_action_contract(
        decoded,
        {"mode": "close_inventory"},
        world_scene,
    )

    assert int(constrained["inventory"][0]) == 0
    assert int(constrained["camera"][0][0]) == 4
    assert suppressed == ("inventory:scene-playable",)
    assert int(decoded["inventory"][0]) == 1


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
    assert not _learned_scene_blocked(
        board,
        MotorIntent(skill_id="activate", mode="gui", instruction="click button"),
    )
    assert not _learned_scene_blocked(
        board,
        MotorIntent(
            skill_id="close_open_inventory",
            mode="close_inventory",
            instruction="close inventory",
        ),
    )


def test_rocket_interaction_taxonomy_matches_published_control_contract() -> None:
    assert _rocket_interaction_id("attack") == 0
    assert _rocket_interaction_id("gather_wood") == 2
    assert _rocket_interaction_id("interact") == 3
    assert _rocket_interaction_id("gui") == 3
    assert _rocket_interaction_id("craft_planks") == 4
    assert _rocket_interaction_id("hotbar") == 5
    assert _rocket_interaction_id("approach") == 6
    assert _rocket_interaction_id("explore") == -1
    assert _rocket_interaction_id("close_inventory") == -1


def test_rocket_action_contract_masks_unsupported_drop_bit() -> None:
    numpy = pytest.importorskip("numpy")
    decoded = {"drop": numpy.asarray([1]), "jump": numpy.asarray([1])}

    safe = _rocket_action_contract(decoded)

    assert int(safe["drop"][0]) == 0
    assert int(safe["jump"][0]) == 1
    assert int(decoded["drop"][0]) == 1


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
