from __future__ import annotations

from dataclasses import asdict

import pytest

import minecraft_ai.outcome_verifier as verifier_module
from minecraft_ai.outcome_verifier import (
    OutcomeKind,
    OutcomeSignal,
    OutcomeStatus,
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


_START = 10_000_000_000
_MS = 1_000_000
_SOURCE = "learned:test-release"


def _fact(key: str, value: str | bool | float, now: int) -> PerceptionFact:
    return PerceptionFact(
        key=key,
        value=value,
        confidence=1.0,
        observed_ns=now,
        source=_SOURCE,
        expires_after_ms=60_000,
    )


def _sample(
    board: PerceptionBlackboard,
    now: int,
    *,
    pixels: str = "0000000000000000",
    luma: int = 32,
    broken: bool = False,
    absent: bool = False,
) -> None:
    board.merge_semantics(
        instance_id="bedrock:release-test",
        facts=(
            _fact("frame.dhash", pixels, now),
            _fact("frame.ui_dhash", pixels, now),
            _fact("frame.crosshair_dhash", pixels, now),
            _fact("frame.crosshair_luma_grid", f"{luma:02x}" * 64, now),
            _fact("target.visible", True, now),
            _fact("target.exists_probability", 0.05 if absent else 0.95, now),
            _fact("target.broken", broken, now),
        ),
        tracks=(
            Track(
                track_id="release-target",
                label="dirt",
                confidence=1.0,
                region=ScreenRegion(x=0.4, y=0.4, width=0.2, height=0.2),
                first_seen_ns=_START,
                last_seen_ns=now,
                attributes={"tracking_source": _SOURCE},
            ),
        ),
    )


def _begin(kind: OutcomeKind) -> tuple[TemporalOutcomeVerifier, PerceptionBlackboard]:
    board = PerceptionBlackboard()
    board.publish(
        FrameState(
            frame_id=0,
            captured_ns=_START,
            instance_id="bedrock:release-test",
            width=1280,
            height=720,
        )
    )
    _sample(board, _START)
    verifier = TemporalOutcomeVerifier()
    verifier.begin("preserve-this-run", kind, board, now_ns=_START)
    return verifier, board


def test_release_without_run_does_not_create_an_outcome_contract() -> None:
    verifier = TemporalOutcomeVerifier()

    assert verifier.notify_inputs_released(now_ns=_START) is None

    assert verifier.active_run_id is None
    assert verifier._state is None
    assert not verifier._held_keys
    assert not verifier._held_buttons


def test_release_closes_mining_once_without_reading_or_replacing_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier, board = _begin(OutcomeKind.MINING)
    verifier.observe(
        board,
        action=MotorAction(sequence=1, buttons_down=("left", "right")),
        now_ns=_START,
    )
    for index, pixels in enumerate(("00000000ffffffff", "ffffffffffffffff"), start=1):
        observed = _START + (400 + 100 * index) * _MS
        _sample(board, observed, pixels=pixels, absent=True)
        verifier.observe(board, now_ns=observed)
    state = verifier._state
    assert isinstance(state, verifier_module._MiningState)
    before = asdict(state)
    assert before["damage_phase_hashes"]
    assert before["candidate_samples"] > 0
    assert before["target_loss_samples"] > 0

    def forbidden_observation(*args: object, **kwargs: object) -> None:
        raise AssertionError("release notification must not observe stale pixels")

    monkeypatch.setattr(PerceptionBlackboard, "fact", forbidden_observation)
    monkeypatch.setattr(PerceptionBlackboard, "latest", forbidden_observation)
    released = _START + 700 * _MS

    assert verifier.notify_inputs_released(now_ns=released) is None

    expected = dict(before)
    expected.update(
        attack_released_ns=released,
        candidate_hash=None,
        candidate_samples=0,
        luma_candidate=None,
        luma_candidate_samples=0,
        target_loss_samples=0,
        target_loss_source=None,
    )
    assert asdict(state) == expected
    assert verifier._state is state
    assert verifier.active_run_id == "preserve-this-run"
    assert verifier._started_ns == _START
    assert not verifier._held_buttons
    assert not verifier._held_keys

    verifier.notify_inputs_released(now_ns=released + 5_000 * _MS)
    assert asdict(state) == expected


def test_short_attack_cannot_gain_duration_while_released() -> None:
    verifier, board = _begin(OutcomeKind.MINING)
    verifier.observe(
        board,
        action=MotorAction(sequence=1, buttons_down=("left",)),
        now_ns=_START,
    )
    verifier.notify_inputs_released(now_ns=_START + 200 * _MS)

    # Even a fresh, exact-bound break fact cannot turn a 200 ms attack into the
    # required 400 ms attack because the camera was unavailable for seconds.
    for index in range(3):
        observed = _START + (2_000 + index * 100) * _MS
        _sample(board, observed, pixels="ffffffffffffffff", broken=True)
        result = verifier.observe(board, now_ns=observed)
        assert result.status == OutcomeStatus.PENDING
        assert result.signal == OutcomeSignal.NONE


def test_repeated_release_keeps_new_post_release_samples_and_original_cutoff() -> None:
    verifier, board = _begin(OutcomeKind.MINING)
    verifier.observe(
        board,
        action=MotorAction(sequence=1, buttons_down=("left",)),
        now_ns=_START,
    )
    verifier.notify_inputs_released(now_ns=_START + 500 * _MS)
    _sample(board, _START + 900 * _MS, pixels="ffffffffffffffff", luma=128, absent=True)
    verifier.observe(board, now_ns=_START + 900 * _MS)
    assert isinstance(verifier._state, verifier_module._MiningState)
    before = asdict(verifier._state)
    assert before["candidate_samples"] == 1
    assert before["luma_candidate_samples"] == 1

    verifier.notify_inputs_released(now_ns=_START + 1_000 * _MS)

    assert asdict(verifier._state) == before


def test_mining_release_does_not_invent_attack_and_allows_a_later_press() -> None:
    verifier, board = _begin(OutcomeKind.MINING)
    assert isinstance(verifier._state, verifier_module._MiningState)
    before = asdict(verifier._state)

    verifier.notify_inputs_released(now_ns=_START + 500 * _MS)

    assert asdict(verifier._state) == before
    verifier.observe(
        board,
        action=MotorAction(sequence=1, buttons_down=("left",)),
        now_ns=_START + 800 * _MS,
    )
    assert verifier._state.attack_started_ns == _START + 800 * _MS
    assert verifier._state.attack_released_ns is None


@pytest.mark.parametrize(
    "kind",
    (
        OutcomeKind.INVENTORY_OPEN,
        OutcomeKind.INVENTORY_CLOSE,
        OutcomeKind.CRAFTING,
        OutcomeKind.RESOURCE_ACQUISITION,
    ),
)
def test_release_preserves_inventory_interaction_and_transition_evidence(kind: OutcomeKind) -> None:
    verifier, board = _begin(kind)
    verifier.observe(
        board,
        action=MotorAction(sequence=1, keys_down=("e",), buttons_down=("left",)),
        now_ns=_START,
    )
    _sample(board, _START + 100 * _MS, pixels="ffffffffffffffff")
    verifier.observe(board, now_ns=_START + 100 * _MS)
    assert isinstance(verifier._state, verifier_module._InventoryState)
    before = asdict(verifier._state)
    assert before["interaction_started_ns"] == _START
    assert before["candidate_samples"] == 1

    assert verifier.notify_inputs_released(now_ns=_START + 200 * _MS) is None

    assert asdict(verifier._state) == before
    assert verifier.active_run_id == "preserve-this-run"
    assert not verifier._held_keys
    assert not verifier._held_buttons


def test_traversal_release_bounds_duration_and_preserves_observed_progress() -> None:
    verifier, board = _begin(OutcomeKind.TRAVERSAL)
    verifier.observe(board, action=MotorAction(sequence=1, keys_down=("w",)), now_ns=_START)
    _sample(board, _START + 500 * _MS, luma=96)
    verifier.observe(board, now_ns=_START + 500 * _MS)
    state = verifier._state
    assert isinstance(state, verifier_module._TraversalState)
    assert state.progress_samples == 1
    baseline = state.luma_grid

    verifier.notify_inputs_released(now_ns=_START + 750 * _MS)

    assert state.progress_samples == 1
    assert state.luma_grid == baseline
    assert state.commanded_movement_ns == 750 * _MS
    assert state.lifetime_commanded_movement_ns == 750 * _MS
    assert not verifier._held_keys
    after_release = asdict(state)
    verifier.notify_inputs_released(now_ns=_START + 4_000 * _MS)
    assert asdict(state) == after_release

    # A fresh image after a long stale gap is a comparison boundary, not a
    # movement sample; nor may the released interval inflate commanded time.
    _sample(board, _START + 5_000 * _MS, luma=224)
    resumed = verifier.observe(board, now_ns=_START + 5_000 * _MS)
    assert resumed.status == OutcomeStatus.PENDING
    assert resumed.signal == OutcomeSignal.NONE
    assert state.progress_samples == 1
    assert state.commanded_movement_ns == 750 * _MS
    assert state.lifetime_commanded_movement_ns == 750 * _MS

    verifier.observe(
        board,
        action=MotorAction(sequence=2, keys_down=("w",)),
        now_ns=_START + 5_100 * _MS,
    )
    _sample(board, _START + 5_300 * _MS, luma=128)
    result = verifier.observe(board, now_ns=_START + 5_300 * _MS)
    assert result.signal == OutcomeSignal.LOCOMOTION_PROGRESS
    assert state.lifetime_commanded_movement_ns == 950 * _MS


def test_stale_observation_does_not_consume_traversal_release_boundary() -> None:
    verifier, board = _begin(OutcomeKind.TRAVERSAL)
    verifier.observe(board, action=MotorAction(sequence=1, keys_down=("w",)), now_ns=_START)
    verifier.notify_inputs_released(now_ns=_START + 500 * _MS)
    state = verifier._state
    assert isinstance(state, verifier_module._TraversalState)

    result = verifier.observe(board, now_ns=_START + 600 * _MS)

    assert result.status == OutcomeStatus.PENDING
    assert state.release_luma_after_ns == _START + 500 * _MS
    assert state.lifetime_commanded_movement_ns == 500 * _MS


def test_unseen_pre_release_pixels_cannot_consume_traversal_release_boundary() -> None:
    verifier, board = _begin(OutcomeKind.TRAVERSAL)
    verifier.observe(board, action=MotorAction(sequence=1, keys_down=("w",)), now_ns=_START)
    verifier.notify_inputs_released(now_ns=_START + 500 * _MS)
    state = verifier._state
    assert isinstance(state, verifier_module._TraversalState)
    baseline = state.luma_grid
    # A delayed semantic update may be newer than the last observation while
    # still describing pixels captured before the release request.
    _sample(board, _START + 400 * _MS, luma=96)

    result = verifier.observe(board, now_ns=_START + 600 * _MS)

    assert result.status == OutcomeStatus.PENDING
    assert state.release_luma_after_ns == _START + 500 * _MS
    assert state.luma_grid == baseline
    assert state.progress_samples == 0
    assert state.lifetime_commanded_movement_ns == 500 * _MS


def test_traversal_cutoff_never_rewinds_existing_observation_or_adds_negative_time() -> None:
    verifier, board = _begin(OutcomeKind.TRAVERSAL)
    verifier.observe(board, action=MotorAction(sequence=1, keys_down=("w",)), now_ns=_START)
    verifier.observe(board, now_ns=_START + 500 * _MS)
    state = verifier._state
    assert isinstance(state, verifier_module._TraversalState)

    verifier.notify_inputs_released(now_ns=_START + 400 * _MS)

    assert state.last_observe_ns == _START + 500 * _MS
    assert state.commanded_movement_ns == 500 * _MS
    assert state.lifetime_commanded_movement_ns == 500 * _MS


def test_traversal_nonmovement_release_does_not_create_commanded_time() -> None:
    verifier, board = _begin(OutcomeKind.TRAVERSAL)
    verifier.observe(
        board,
        action=MotorAction(sequence=1, keys_down=("shift",), buttons_down=("right",)),
        now_ns=_START,
    )
    assert isinstance(verifier._state, verifier_module._TraversalState)
    before = asdict(verifier._state)

    verifier.notify_inputs_released(now_ns=_START + 500 * _MS)

    assert asdict(verifier._state) == before
    assert not verifier._held_keys
    assert not verifier._held_buttons


def test_default_release_timestamp_uses_monotonic_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    verifier, board = _begin(OutcomeKind.MINING)
    verifier.observe(
        board,
        action=MotorAction(sequence=1, buttons_down=("left",)),
        now_ns=_START,
    )
    monkeypatch.setattr(verifier_module.time, "monotonic_ns", lambda: _START + 500 * _MS)

    verifier.notify_inputs_released()

    assert isinstance(verifier._state, verifier_module._MiningState)
    assert verifier._state.attack_released_ns == _START + 500 * _MS


@pytest.mark.parametrize("invalid", (-1, True, 1.5, "42"))
def test_invalid_release_timestamp_does_not_partially_clear_inputs(invalid: object) -> None:
    verifier, board = _begin(OutcomeKind.MINING)
    verifier.observe(
        board,
        action=MotorAction(sequence=1, buttons_down=("left",)),
        now_ns=_START,
    )
    assert isinstance(verifier._state, verifier_module._MiningState)
    before = asdict(verifier._state)

    with pytest.raises(ValueError, match="release time"):
        verifier.notify_inputs_released(now_ns=invalid)  # type: ignore[arg-type]

    assert asdict(verifier._state) == before
    assert verifier._held_buttons == {"left"}
