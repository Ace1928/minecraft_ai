from __future__ import annotations

import time

import pytest

from minecraft_ai.outcome_verifier import (
    OutcomeKind,
    OutcomeSignal,
    OutcomeStatus,
    OutcomeVerifierConfig,
    TemporalOutcomeVerifier,
)
from minecraft_ai.perception import (
    FrameState,
    PerceptionBlackboard,
    PerceptionFact,
    ScreenRegion,
    Track,
)
from minecraft_ai.safety import MotorAction


_SECOND = 1_000_000_000
_HASH_B = "ffffffffffffffff"


def _fact(
    key: str,
    value: str | int | float | bool,
    *,
    observed_ns: int,
    source: str | None = None,
    confidence: float = 1.0,
) -> PerceptionFact:
    return PerceptionFact(
        key=key,
        value=value,
        confidence=confidence,
        observed_ns=observed_ns,
        source=source or ("bootstrap:test-hash" if "dhash" in key else "learned:test"),
        expires_after_ms=60_000,
    )


def _board(now_ns: int, *, frame_hash: str = "0000000000000000") -> PerceptionBlackboard:
    board = PerceptionBlackboard()
    board.publish(
        FrameState(
            frame_id=0,
            captured_ns=now_ns,
            instance_id="bedrock:test",
            width=1280,
            height=720,
        )
    )
    _publish_hashes(board, now_ns, frame_hash=frame_hash)
    return board


def _publish_hashes(
    board: PerceptionBlackboard,
    observed_ns: int,
    *,
    frame_hash: str,
    crosshair_hash: str | None = None,
    ui_hash: str | None = None,
) -> None:
    board.merge_semantics(
        instance_id="bedrock:test",
        facts=(
            _fact("frame.dhash", frame_hash, observed_ns=observed_ns),
            _fact(
                "frame.crosshair_dhash",
                frame_hash if crosshair_hash is None else crosshair_hash,
                observed_ns=observed_ns,
            ),
            _fact(
                "frame.ui_dhash",
                frame_hash if ui_hash is None else ui_hash,
                observed_ns=observed_ns,
            ),
        ),
    )


def _publish_target(
    board: PerceptionBlackboard,
    observed_ns: int,
    *,
    visible: bool,
    source: str = "learned:rocket:test",
    track_id: str = "test",
    label: str = "dirt",
    visible_confidence: float = 1.0,
    kind_confidence: float = 1.0,
    exists_probability: float | None = None,
    track_attributes: dict[str, str | int | float | bool] | None = None,
) -> None:
    attributes = dict(track_attributes or {})
    if not attributes:
        if source.startswith("operator:"):
            attributes = {"source": "operator"}
        else:
            attributes = {"tracking_source": source}
    probability = (0.95 if visible else 0.05) if exists_probability is None else exists_probability
    board.merge_semantics(
        instance_id="bedrock:test",
        facts=(
            _fact(
                "target.visible",
                visible,
                observed_ns=observed_ns,
                source=source,
                confidence=visible_confidence,
            ),
            _fact(
                "target.kind",
                label,
                observed_ns=observed_ns,
                source=source,
                confidence=kind_confidence,
            ),
            _fact(
                "target.exists_probability",
                probability,
                observed_ns=observed_ns,
                source=source,
            ),
        ),
        tracks=(
            Track(
                track_id=track_id,
                label=label,
                confidence=1.0,
                region=ScreenRegion(x=0.4, y=0.4, width=0.2, height=0.2),
                first_seen_ns=observed_ns,
                last_seen_ns=observed_ns,
                attributes=attributes,
            ),
        ),
    )


def test_observe_requires_an_active_run() -> None:
    now = time.monotonic_ns()
    verifier = TemporalOutcomeVerifier()

    with pytest.raises(RuntimeError, match="no active run"):
        verifier.observe(_board(now), now_ns=now)


def test_crack_animation_without_corroboration_is_only_progress() -> None:
    now = time.monotonic_ns()
    board = _board(now)
    _publish_target(board, now, visible=True)
    verifier = TemporalOutcomeVerifier()
    verifier.begin("mine-1", OutcomeKind.MINING, board, now_ns=now)

    verifier.observe(
        board,
        action=MotorAction(sequence=1, buttons_down=("left",)),
        now_ns=now,
    )
    verdict = None
    for index in range(1, 5):
        sample_ns = now + (400 + index * 50) * 1_000_000
        _publish_hashes(
            board,
            sample_ns,
            frame_hash="f0f0f0f0f0f0f0f0",
            crosshair_hash="ffffffffffffffff",
        )
        _publish_target(board, sample_ns, visible=True)
        verdict = verifier.observe(board, now_ns=sample_ns)

    assert verdict is not None
    assert verdict.status == OutcomeStatus.PROGRESS
    assert verdict.signal == OutcomeSignal.BLOCK_DAMAGE_PROGRESS
    assert verdict.terminal is False


def test_cracks_that_clear_after_release_do_not_become_block_break_success() -> None:
    now = time.monotonic_ns()
    board = _board(now)
    _publish_target(board, now, visible=True, source="operator:explicit-grounding:test")
    verifier = TemporalOutcomeVerifier()
    verifier.begin("mine-2", OutcomeKind.MINING, board, now_ns=now)
    verifier.observe(
        board,
        action=MotorAction(sequence=1, buttons_down=("left",)),
        now_ns=now,
    )
    for index in range(1, 4):
        sample_ns = now + (400 + index * 50) * 1_000_000
        _publish_hashes(
            board,
            sample_ns,
            frame_hash="ffffffffffffffff",
            crosshair_hash="ffffffffffffffff",
        )
        verifier.observe(board, now_ns=sample_ns)
    release_ns = now + 700_000_000
    verifier.observe(
        board,
        action=MotorAction(sequence=2, buttons_up=("left",)),
        now_ns=release_ns,
    )
    settled_ns = now + 1_000_000_000
    _publish_hashes(board, settled_ns, frame_hash="0000000000000000")

    verdict = verifier.observe(board, now_ns=settled_ns)

    assert verdict.status == OutcomeStatus.PENDING
    assert verdict.signal == OutcomeSignal.NONE


def test_complete_damage_cycle_verifies_same_material_replacement() -> None:
    now = time.monotonic_ns()
    board = _board(now, frame_hash="402420a0a2a22454")
    track_id = "operator:same-material"
    operator_source = f"operator:explicit-grounding:{track_id}"
    rocket_source = "learned:learned:minestudio-rocket2:test:aux-localization:not-training-label"
    _publish_target(
        board,
        now,
        visible=True,
        source=operator_source,
        track_id=track_id,
        track_attributes={"source": "operator", "grounding": "explicit-region"},
    )
    verifier = TemporalOutcomeVerifier()
    verifier.begin(
        "mine-same-material",
        OutcomeKind.MINING,
        board,
        now_ns=now,
        trusted_transition_source=rocket_source,
    )
    verifier.observe(
        board,
        action=MotorAction(sequence=1, buttons_down=("left",)),
        now_ns=now,
    )

    # These are the distinct crosshair phases from a real Bedrock dirt-break
    # trajectory. The target remains semantically visible because another dirt
    # block is directly behind the exact block that was attacked.
    damage_hashes = (
        "558c64a4a2a22050",
        "4a2a26a6a2a26050",
        "a34326a6a2b17874",
        "5d5a5e942c513530",
        "6b6d6a972e553d38",
    )
    for index, crosshair_hash in enumerate(damage_hashes, start=1):
        sample_ns = now + (400 + index * 100) * 1_000_000
        _publish_hashes(
            board,
            sample_ns,
            frame_hash=crosshair_hash,
            crosshair_hash=crosshair_hash,
        )
        _publish_target(
            board,
            sample_ns,
            visible=True,
            source=operator_source,
            track_id=track_id,
            exists_probability=0.98,
            track_attributes={"source": "operator", "grounding": "explicit-region"},
        )
        verdict = verifier.observe(board, now_ns=sample_ns)
        assert verdict.status != OutcomeStatus.SUCCEEDED

    release_ns = now + 1_100_000_000
    verifier.observe(
        board,
        action=MotorAction(sequence=2, buttons_up=("left",)),
        now_ns=release_ns,
    )
    replacement = "51535e942c513534"
    for index in range(3):
        settled_ns = now + (1_400 + index * 100) * 1_000_000
        _publish_hashes(
            board,
            settled_ns,
            frame_hash=replacement,
            crosshair_hash=replacement,
        )
        _publish_target(
            board,
            settled_ns,
            visible=True,
            source=rocket_source,
            track_id=track_id,
            exists_probability=0.98,
            track_attributes={"source": "operator", "tracking_source": rocket_source},
        )
        verdict = verifier.observe(board, now_ns=settled_ns)

    assert verdict.status == OutcomeStatus.SUCCEEDED
    assert verdict.signal == OutcomeSignal.BLOCK_BROKEN
    assert verdict.confidence == 0.88
    assert verdict.evidence_keys == (
        "frame.crosshair_dhash",
        "target.track_id",
        f"target.track_id={track_id}",
        f"target.binding_source={operator_source}",
        "target.exists_probability",
        f"target.tracking_source={rocket_source}",
    )
    assert "multiple damage phases" in verdict.reason


def test_incomplete_damage_cycle_cannot_verify_same_material_replacement() -> None:
    now = time.monotonic_ns()
    board = _board(now)
    _publish_target(board, now, visible=True)
    verifier = TemporalOutcomeVerifier()
    verifier.begin("mine-incomplete-cycle", OutcomeKind.MINING, board, now_ns=now)
    verifier.observe(
        board,
        action=MotorAction(sequence=1, buttons_down=("left",)),
        now_ns=now,
    )

    # A single large occlusion followed by changed stable pixels is not a
    # complete damage cycle and therefore remains fail-closed.
    burst_ns = now + 600_000_000
    _publish_hashes(
        board,
        burst_ns,
        frame_hash=_HASH_B,
        crosshair_hash=_HASH_B,
    )
    verifier.observe(board, now_ns=burst_ns)
    release_ns = now + 700_000_000
    verifier.observe(
        board,
        action=MotorAction(sequence=2, buttons_up=("left",)),
        now_ns=release_ns,
    )
    replacement = "0f0f0f0f0f0f0f0f"
    for index in range(3):
        settled_ns = now + (1_000 + index * 100) * 1_000_000
        _publish_hashes(
            board,
            settled_ns,
            frame_hash=replacement,
            crosshair_hash=replacement,
        )
        _publish_target(board, settled_ns, visible=True, exists_probability=0.98)
        verdict = verifier.observe(board, now_ns=settled_ns)

    assert verdict.status != OutcomeStatus.SUCCEEDED
    assert verdict.signal != OutcomeSignal.BLOCK_BROKEN


def test_damage_cycle_rejects_positive_observation_captured_before_release() -> None:
    now = time.monotonic_ns()
    board = _board(now, frame_hash="402420a0a2a22454")
    _publish_target(board, now, visible=True)
    verifier = TemporalOutcomeVerifier()
    verifier.begin("mine-stale-positive", OutcomeKind.MINING, board, now_ns=now)
    verifier.observe(
        board,
        action=MotorAction(sequence=1, buttons_down=("left",)),
        now_ns=now,
    )
    for index, crosshair_hash in enumerate(
        (
            "558c64a4a2a22050",
            "4a2a26a6a2a26050",
            "a34326a6a2b17874",
            "5d5a5e942c513530",
            "6b6d6a972e553d38",
        ),
        start=1,
    ):
        sample_ns = now + (400 + index * 100) * 1_000_000
        _publish_hashes(board, sample_ns, frame_hash=crosshair_hash)
        _publish_target(board, sample_ns, visible=True, exists_probability=0.98)
        verifier.observe(board, now_ns=sample_ns)
    verifier.observe(
        board,
        action=MotorAction(sequence=2, buttons_up=("left",)),
        now_ns=now + 1_100_000_000,
    )
    replacement = "51535e942c513534"
    for index in range(3):
        settled_ns = now + (1_400 + index * 100) * 1_000_000
        _publish_hashes(board, settled_ns, frame_hash=replacement)
        verdict = verifier.observe(board, now_ns=settled_ns)

    assert verdict.status != OutcomeStatus.SUCCEEDED
    assert verdict.signal != OutcomeSignal.BLOCK_BROKEN


def test_damage_phases_that_clear_to_baseline_remain_unverified() -> None:
    now = time.monotonic_ns()
    baseline = "402420a0a2a22454"
    board = _board(now, frame_hash=baseline)
    _publish_target(board, now, visible=True)
    verifier = TemporalOutcomeVerifier()
    verifier.begin("mine-cleared-cycle", OutcomeKind.MINING, board, now_ns=now)
    verifier.observe(
        board,
        action=MotorAction(sequence=1, buttons_down=("left",)),
        now_ns=now,
    )
    for index, crosshair_hash in enumerate(
        (
            "558c64a4a2a22050",
            "4a2a26a6a2a26050",
            "a34326a6a2b17874",
            "5d5a5e942c513530",
            "6b6d6a972e553d38",
        ),
        start=1,
    ):
        sample_ns = now + (400 + index * 100) * 1_000_000
        _publish_hashes(
            board,
            sample_ns,
            frame_hash=crosshair_hash,
            crosshair_hash=crosshair_hash,
        )
        verifier.observe(board, now_ns=sample_ns)
    verifier.observe(
        board,
        action=MotorAction(sequence=2, buttons_up=("left",)),
        now_ns=now + 900_000_000,
    )
    for index in range(3):
        settled_ns = now + (1_200 + index * 100) * 1_000_000
        _publish_hashes(board, settled_ns, frame_hash=baseline)
        _publish_target(board, settled_ns, visible=True, exists_probability=0.98)
        verdict = verifier.observe(board, now_ns=settled_ns)

    assert verdict.status != OutcomeStatus.SUCCEEDED
    assert verdict.signal != OutcomeSignal.BLOCK_BROKEN


def test_stable_replacement_plus_fresh_target_loss_verifies_block_break() -> None:
    now = time.monotonic_ns()
    board = _board(now)
    _publish_target(board, now, visible=True)
    verifier = TemporalOutcomeVerifier()
    verifier.begin("mine-3", OutcomeKind.MINING, board, now_ns=now)
    verifier.observe(
        board,
        action=MotorAction(sequence=1, buttons_down=("left",)),
        now_ns=now,
    )
    verdict = None
    for index in range(1, 4):
        sample_ns = now + (400 + index * 75) * 1_000_000
        _publish_hashes(
            board,
            sample_ns,
            frame_hash="ffffffffffffffff",
            crosshair_hash="ffffffffffffffff",
        )
        _publish_target(board, sample_ns, visible=False)
        verdict = verifier.observe(board, now_ns=sample_ns)

    assert verdict is not None
    assert verdict.status == OutcomeStatus.PROGRESS

    release_ns = now + 700_000_000
    verifier.observe(
        board,
        action=MotorAction(sequence=2, buttons_up=("left",)),
        now_ns=release_ns,
    )
    settled_ns = now + 1_000_000_000
    _publish_hashes(
        board,
        settled_ns,
        frame_hash="ffffffffffffffff",
        crosshair_hash="ffffffffffffffff",
    )
    _publish_target(board, settled_ns, visible=False)
    verdict = verifier.observe(board, now_ns=settled_ns)

    assert verdict.status == OutcomeStatus.SUCCEEDED
    assert verdict.signal == OutcomeSignal.BLOCK_BROKEN
    assert verdict.terminal is True
    assert "target.exists_probability" in verdict.evidence_keys
    assert "target.visible" not in verdict.evidence_keys


def test_delayed_bound_target_loss_still_verifies_with_live_rocket_latency() -> None:
    now = time.monotonic_ns()
    board = _board(now)
    _publish_target(board, now, visible=True)
    verifier = TemporalOutcomeVerifier()
    verifier.begin("mine-delayed", OutcomeKind.MINING, board, now_ns=now)
    verifier.observe(
        board,
        action=MotorAction(sequence=1, buttons_down=("left",)),
        now_ns=now,
    )
    for offset_ms in (450, 500):
        sample_ns = now + offset_ms * 1_000_000
        _publish_hashes(board, sample_ns, frame_hash=_HASH_B, crosshair_hash=_HASH_B)
        verifier.observe(board, now_ns=sample_ns)
    release_ns = now + 600_000_000
    verifier.observe(
        board,
        action=MotorAction(sequence=2, buttons_up=("left",)),
        now_ns=release_ns,
    )
    stable_ns = now + 700_000_000
    _publish_hashes(board, stable_ns, frame_hash=_HASH_B, crosshair_hash=_HASH_B)
    verifier.observe(board, now_ns=stable_ns)

    delayed_ns = now + 2_300_000_000
    _publish_hashes(board, delayed_ns, frame_hash=_HASH_B, crosshair_hash=_HASH_B)
    _publish_target(board, delayed_ns, visible=False)
    first_loss = verifier.observe(board, now_ns=delayed_ns)

    second_loss_ns = now + 4_000_000_000
    _publish_hashes(board, second_loss_ns, frame_hash=_HASH_B, crosshair_hash=_HASH_B)
    _publish_target(board, second_loss_ns, visible=False)
    verdict = verifier.observe(board, now_ns=second_loss_ns)

    assert first_loss.status != OutcomeStatus.SUCCEEDED
    assert verdict.status == OutcomeStatus.SUCCEEDED
    assert verdict.signal == OutcomeSignal.BLOCK_BROKEN


def test_live_operator_track_accepts_two_exact_rocket_negative_observations() -> None:
    now = time.monotonic_ns()
    board = _board(now)
    track_id = "operator:28ea"
    operator_source = f"operator:explicit-grounding:{track_id}"
    rocket_source = (
        "learned:learned:minestudio-rocket2:test:aux-localization:not-training-label"
    )
    _publish_target(
        board,
        now,
        visible=True,
        source=operator_source,
        track_id=track_id,
        label="stone",
        track_attributes={"source": "operator", "grounding": "explicit-region"},
    )
    verifier = TemporalOutcomeVerifier()
    verifier.begin(
        "mine-live-operator",
        OutcomeKind.MINING,
        board,
        now_ns=now,
        trusted_transition_source=rocket_source,
    )
    verifier.observe(
        board,
        action=MotorAction(sequence=1, buttons_down=("left",)),
        now_ns=now,
    )
    for offset_ms in (450, 500, 550):
        sample_ns = now + offset_ms * 1_000_000
        _publish_hashes(board, sample_ns, frame_hash=_HASH_B, crosshair_hash=_HASH_B)
        verifier.observe(board, now_ns=sample_ns)
    verifier.observe(
        board,
        action=MotorAction(sequence=2, buttons_up=("left",)),
        now_ns=now + 600_000_000,
    )

    first_ns = now + 2_300_000_000
    _publish_hashes(board, first_ns, frame_hash=_HASH_B, crosshair_hash=_HASH_B)
    _publish_target(
        board,
        first_ns,
        # The raw miss is strong even though the EMA still says visible.
        visible=True,
        source=rocket_source,
        track_id=track_id,
        label="stone",
        visible_confidence=0.56,
        kind_confidence=0.56,
        exists_probability=0.048361,
        track_attributes={"source": "operator", "tracking_source": rocket_source},
    )
    first_eval_ns = first_ns + 100_000_000
    board.merge_semantics(
        instance_id="bedrock:test",
        facts=(
            _fact(
                "target.visible",
                True,
                observed_ns=first_eval_ns,
                source=operator_source,
            ),
        ),
    )
    first = verifier.observe(board, now_ns=first_eval_ns)

    second_ns = now + 4_000_000_000
    _publish_hashes(board, second_ns, frame_hash=_HASH_B, crosshair_hash=_HASH_B)
    _publish_target(
        board,
        second_ns,
        visible=True,
        source=rocket_source,
        track_id=track_id,
        label="stone",
        visible_confidence=0.55,
        kind_confidence=0.55,
        exists_probability=0.03,
        track_attributes={"source": "operator", "tracking_source": rocket_source},
    )
    second_eval_ns = second_ns + 100_000_000
    board.merge_semantics(
        instance_id="bedrock:test",
        facts=(
            _fact(
                "target.visible",
                True,
                observed_ns=second_eval_ns,
                source=operator_source,
            ),
        ),
    )
    second = verifier.observe(board, now_ns=second_eval_ns)

    assert first.status != OutcomeStatus.SUCCEEDED
    assert second.status == OutcomeStatus.SUCCEEDED
    assert second.signal == OutcomeSignal.BLOCK_BROKEN
    assert "target.exists_probability" in second.evidence_keys
    assert f"target.track_id={track_id}" in second.evidence_keys
    assert f"target.tracking_source={rocket_source}" in second.evidence_keys


def test_operator_negative_rejects_self_consistent_foreign_source_and_weak_probability() -> None:
    now = time.monotonic_ns()
    board = _board(now)
    track_id = "operator:28ea"
    operator_source = f"operator:explicit-grounding:{track_id}"
    rocket_source = (
        "learned:learned:minestudio-rocket2:test:aux-localization:not-training-label"
    )
    foreign_source = "learned:foreign:self-consistent"
    _publish_target(
        board,
        now,
        visible=True,
        source=operator_source,
        track_id=track_id,
        label="stone",
        track_attributes={"source": "operator"},
    )
    verifier = TemporalOutcomeVerifier()
    verifier.begin(
        "mine-live-adversarial",
        OutcomeKind.MINING,
        board,
        now_ns=now,
        trusted_transition_source=rocket_source,
    )
    verifier.observe(
        board,
        action=MotorAction(sequence=1, buttons_down=("left",)),
        now_ns=now,
    )
    for offset_ms in (450, 500, 550):
        sample_ns = now + offset_ms * 1_000_000
        _publish_hashes(board, sample_ns, frame_hash=_HASH_B, crosshair_hash=_HASH_B)
        verifier.observe(board, now_ns=sample_ns)
    verifier.observe(
        board,
        action=MotorAction(sequence=2, buttons_up=("left",)),
        now_ns=now + 600_000_000,
    )

    for index, probability in enumerate((0.01, 0.02, 0.45, 0.40), start=1):
        sample_ns = now + (900 + index * 400) * 1_000_000
        observation_source = foreign_source if index <= 2 else rocket_source
        _publish_hashes(board, sample_ns, frame_hash=_HASH_B, crosshair_hash=_HASH_B)
        _publish_target(
            board,
            sample_ns,
            visible=False,
            source=observation_source,
            track_id=track_id,
            label="stone",
            visible_confidence=0.55,
            kind_confidence=0.55,
            exists_probability=probability,
            track_attributes={"source": "operator", "tracking_source": observation_source},
        )
        verdict = verifier.observe(board, now_ns=sample_ns)

    assert verdict.status != OutcomeStatus.SUCCEEDED
    assert verdict.signal != OutcomeSignal.BLOCK_BROKEN


def test_fresh_bound_high_probability_resets_consecutive_loss_samples() -> None:
    now = time.monotonic_ns()
    board = _board(now)
    _publish_target(board, now, visible=True)
    verifier = TemporalOutcomeVerifier()
    verifier.begin("mine-probability-reset", OutcomeKind.MINING, board, now_ns=now)
    verifier.observe(
        board,
        action=MotorAction(sequence=1, buttons_down=("left",)),
        now_ns=now,
    )
    for offset_ms in (450, 500, 550):
        sample_ns = now + offset_ms * 1_000_000
        _publish_hashes(board, sample_ns, frame_hash=_HASH_B, crosshair_hash=_HASH_B)
        verifier.observe(board, now_ns=sample_ns)
    verifier.observe(
        board,
        action=MotorAction(sequence=2, buttons_up=("left",)),
        now_ns=now + 600_000_000,
    )

    verdicts = []
    for offset_ms, probability in ((1_000, 0.05), (1_200, 0.90), (1_400, 0.04)):
        sample_ns = now + offset_ms * 1_000_000
        _publish_hashes(board, sample_ns, frame_hash=_HASH_B, crosshair_hash=_HASH_B)
        _publish_target(
            board,
            sample_ns,
            visible=True,
            exists_probability=probability,
        )
        verdicts.append(verifier.observe(board, now_ns=sample_ns))

    final_ns = now + 1_600_000_000
    _publish_hashes(board, final_ns, frame_hash=_HASH_B, crosshair_hash=_HASH_B)
    _publish_target(board, final_ns, visible=True, exists_probability=0.03)
    final = verifier.observe(board, now_ns=final_ns)

    assert all(item.status != OutcomeStatus.SUCCEEDED for item in verdicts)
    assert final.status == OutcomeStatus.SUCCEEDED
    assert final.signal == OutcomeSignal.BLOCK_BROKEN


def test_foreign_target_loss_and_inventory_change_cannot_verify_break() -> None:
    now = time.monotonic_ns()
    board = _board(now)
    _publish_target(board, now, visible=True, source="learned:target-a", track_id="a")
    board.merge_semantics(
        instance_id="bedrock:test",
        facts=(_fact("inventory.logs", 0, observed_ns=now, source="learned:inventory"),),
    )
    verifier = TemporalOutcomeVerifier()
    verifier.begin("mine-foreign", OutcomeKind.MINING, board, now_ns=now)
    verifier.observe(
        board,
        action=MotorAction(sequence=1, buttons_down=("left",)),
        now_ns=now,
    )
    for index in range(1, 4):
        sample_ns = now + (400 + index * 75) * 1_000_000
        _publish_hashes(board, sample_ns, frame_hash=_HASH_B, crosshair_hash=_HASH_B)
        _publish_target(
            board,
            sample_ns,
            visible=False,
            source="learned:target-b",
            track_id="b",
        )
        board.merge_semantics(
            instance_id="bedrock:test",
            facts=(
                _fact(
                    "inventory.logs",
                    index,
                    observed_ns=sample_ns,
                    source="learned:inventory",
                ),
            ),
        )
        verifier.observe(board, now_ns=sample_ns)
    release_ns = now + 700_000_000
    verifier.observe(
        board,
        action=MotorAction(sequence=2, buttons_up=("left",)),
        now_ns=release_ns,
    )
    settled_ns = now + 1_100_000_000
    _publish_hashes(board, settled_ns, frame_hash=_HASH_B, crosshair_hash=_HASH_B)

    verdict = verifier.observe(board, now_ns=settled_ns)

    assert verdict.status != OutcomeStatus.SUCCEEDED
    assert verdict.signal != OutcomeSignal.BLOCK_BROKEN


def test_operator_fact_absence_is_unknown_not_block_break() -> None:
    now = time.monotonic_ns()
    board = _board(now)
    _publish_target(
        board,
        now,
        visible=True,
        source="operator:explicit-grounding:test",
    )
    verifier = TemporalOutcomeVerifier()
    verifier.begin("mine-operator-absence", OutcomeKind.MINING, board, now_ns=now)
    verifier.observe(
        board,
        action=MotorAction(sequence=1, buttons_down=("left",)),
        now_ns=now,
    )
    for index in range(1, 4):
        sample_ns = now + (400 + index * 75) * 1_000_000
        _publish_hashes(board, sample_ns, frame_hash=_HASH_B, crosshair_hash=_HASH_B)
        verifier.observe(board, now_ns=sample_ns)
    verifier.observe(
        board,
        action=MotorAction(sequence=2, buttons_up=("left",)),
        now_ns=now + 700_000_000,
    )
    absent_ns = now + 61_000_000_000
    _publish_hashes(board, absent_ns, frame_hash=_HASH_B, crosshair_hash=_HASH_B)

    verdict = verifier.observe(board, now_ns=absent_ns)

    assert verdict.status != OutcomeStatus.SUCCEEDED
    assert verdict.signal != OutcomeSignal.BLOCK_BROKEN


def test_foreign_post_attack_broken_fact_is_not_target_evidence() -> None:
    now = time.monotonic_ns()
    board = _board(now)
    _publish_target(board, now, visible=True, source="learned:target-a", track_id="a")
    verifier = TemporalOutcomeVerifier()
    verifier.begin("mine-foreign-broken", OutcomeKind.MINING, board, now_ns=now)
    verifier.observe(
        board,
        action=MotorAction(sequence=1, buttons_down=("left",)),
        now_ns=now,
    )
    for index in range(1, 4):
        sample_ns = now + (400 + index * 75) * 1_000_000
        _publish_hashes(board, sample_ns, frame_hash=_HASH_B, crosshair_hash=_HASH_B)
        _publish_target(
            board,
            sample_ns,
            visible=True,
            source="learned:target-a",
            track_id="a",
        )
        board.merge_semantics(
            instance_id="bedrock:test",
            facts=(
                _fact(
                    "target.broken",
                    True,
                    observed_ns=sample_ns,
                    source="learned:target-b",
                ),
            ),
        )
        verdict = verifier.observe(board, now_ns=sample_ns)

    assert verdict.status != OutcomeStatus.SUCCEEDED
    assert verdict.signal != OutcomeSignal.BLOCK_BROKEN


def test_static_attack_reports_mining_stall() -> None:
    now = time.monotonic_ns()
    board = _board(now)
    _publish_target(board, now, visible=True)
    verifier = TemporalOutcomeVerifier()
    verifier.begin("mine-4", OutcomeKind.MINING, board, now_ns=now)
    verifier.observe(
        board,
        action=MotorAction(sequence=1, buttons_down=("left",)),
        now_ns=now,
    )
    verdict = None
    for index in range(1, 5):
        sample_ns = now + index * 500_000_000
        _publish_hashes(board, sample_ns, frame_hash="0000000000000000")
        verdict = verifier.observe(board, now_ns=sample_ns)

    assert verdict is not None
    assert verdict.status == OutcomeStatus.STALLED
    assert verdict.signal == OutcomeSignal.MINING_STALLED


def test_repeated_quiet_camera_motion_changes_report_traversal_progress() -> None:
    now = time.monotonic_ns()
    board = _board(now)
    verifier = TemporalOutcomeVerifier()
    verifier.begin("walk-1", OutcomeKind.TRAVERSAL, board, now_ns=now)
    verifier.observe(
        board,
        action=MotorAction(sequence=1, keys_down=("w",)),
        now_ns=now,
    )
    first_ns = now + 300_000_000
    _publish_hashes(
        board,
        first_ns,
        frame_hash="00ff00ff00ff00ff",
        crosshair_hash="00000000ffffffff",
    )
    verifier.observe(board, now_ns=first_ns)
    second_ns = now + 600_000_000
    _publish_hashes(
        board,
        second_ns,
        frame_hash="ff00ff00ff00ff00",
        crosshair_hash="ffffffff00000000",
    )

    verdict = verifier.observe(board, now_ns=second_ns)

    assert verdict.status == OutcomeStatus.PROGRESS
    assert verdict.signal == OutcomeSignal.LOCOMOTION_PROGRESS
    assert verdict.terminal is False


def test_static_world_while_movement_is_held_reports_traversal_stall() -> None:
    now = time.monotonic_ns()
    board = _board(now)
    verifier = TemporalOutcomeVerifier()
    verifier.begin("walk-2", OutcomeKind.TRAVERSAL, board, now_ns=now)
    verifier.observe(
        board,
        action=MotorAction(sequence=1, keys_down=("w",)),
        now_ns=now,
    )
    verdict = None
    for index in range(1, 5):
        sample_ns = now + index * 550_000_000
        _publish_hashes(board, sample_ns, frame_hash="0000000000000000")
        verdict = verifier.observe(board, now_ns=sample_ns)

    assert verdict is not None
    assert verdict.status == OutcomeStatus.STALLED
    assert verdict.signal == OutcomeSignal.LOCOMOTION_STALLED


def test_camera_motion_cannot_be_claimed_as_traversal_progress_or_stall() -> None:
    now = time.monotonic_ns()
    board = _board(now)
    verifier = TemporalOutcomeVerifier(
        OutcomeVerifierConfig(camera_quiet_ms=500, traversal_stall_ms=600)
    )
    verifier.begin("walk-3", OutcomeKind.TRAVERSAL, board, now_ns=now)
    verifier.observe(
        board,
        action=MotorAction(sequence=1, keys_down=("w",), mouse_dx=20),
        now_ns=now,
    )
    verdict = None
    for index in range(1, 5):
        sample_ns = now + index * 200_000_000
        _publish_hashes(
            board,
            sample_ns,
            frame_hash=("ffffffffffffffff" if index % 2 else "0000000000000000"),
        )
        verdict = verifier.observe(
            board,
            action=MotorAction(sequence=index + 1, mouse_dx=10),
            now_ns=sample_ns,
        )

    assert verdict is not None
    assert verdict.status == OutcomeStatus.PENDING
    assert verdict.signal == OutcomeSignal.NONE


def test_inventory_open_requires_input_stable_pixels_and_fresh_scene_fact() -> None:
    now = time.monotonic_ns()
    board = _board(now)
    board.merge_semantics(
        instance_id="bedrock:test",
        facts=(_fact("scene.mode", "world", observed_ns=now),),
    )
    verifier = TemporalOutcomeVerifier()
    verifier.begin("gui-1", OutcomeKind.INVENTORY_OPEN, board, now_ns=now)
    verifier.observe(
        board,
        action=MotorAction(sequence=1, keys_down=("e",)),
        now_ns=now,
    )
    verdict = None
    for index in range(1, 3):
        sample_ns = now + index * 150_000_000
        _publish_hashes(
            board,
            sample_ns,
            frame_hash="ffffffffffffffff",
            ui_hash="ffffffffffffffff",
        )
        board.merge_semantics(
            instance_id="bedrock:test",
            facts=(_fact("scene.mode", "inventory", observed_ns=sample_ns),),
        )
        verdict = verifier.observe(board, now_ns=sample_ns)

    assert verdict is not None
    assert verdict.status == OutcomeStatus.SUCCEEDED
    assert verdict.signal == OutcomeSignal.INVENTORY_OPENED


def test_pixel_change_without_inventory_state_is_not_success() -> None:
    now = time.monotonic_ns()
    board = _board(now)
    verifier = TemporalOutcomeVerifier()
    verifier.begin("gui-2", OutcomeKind.INVENTORY_OPEN, board, now_ns=now)
    verifier.observe(
        board,
        action=MotorAction(sequence=1, keys_down=("e",)),
        now_ns=now,
    )
    verdict = None
    for index in range(1, 3):
        sample_ns = now + index * 150_000_000
        _publish_hashes(
            board,
            sample_ns,
            frame_hash="ffffffffffffffff",
            ui_hash="ffffffffffffffff",
        )
        verdict = verifier.observe(board, now_ns=sample_ns)

    assert verdict is not None
    assert verdict.status == OutcomeStatus.PROGRESS
    assert verdict.signal == OutcomeSignal.GUI_TRANSITION_PROGRESS


def test_inventory_close_accepts_fresh_playable_world_after_stable_change() -> None:
    now = time.monotonic_ns()
    board = _board(now, frame_hash="ffffffffffffffff")
    board.merge_semantics(
        instance_id="bedrock:test",
        facts=(_fact("scene.mode", "inventory", observed_ns=now),),
    )
    verifier = TemporalOutcomeVerifier()
    verifier.begin("gui-3", OutcomeKind.INVENTORY_CLOSE, board, now_ns=now)
    verifier.observe(
        board,
        action=MotorAction(sequence=1, keys_down=("e",)),
        now_ns=now,
    )
    verdict = None
    for index in range(1, 3):
        sample_ns = now + index * 150_000_000
        _publish_hashes(board, sample_ns, frame_hash="0000000000000000")
        board.merge_semantics(
            instance_id="bedrock:test",
            facts=(
                _fact(
                    "scene.playable",
                    True,
                    observed_ns=sample_ns,
                    source="learned:scene:test",
                ),
            ),
        )
        verdict = verifier.observe(board, now_ns=sample_ns)

    assert verdict is not None
    assert verdict.status == OutcomeStatus.SUCCEEDED
    assert verdict.signal == OutcomeSignal.INVENTORY_CLOSED
