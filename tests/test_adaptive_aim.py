from __future__ import annotations

import math
import time

import pytest

from minecraft_ai.control.adaptive_aim import AdaptiveMiningAim
from minecraft_ai.mining_control import MiningLeaseGuard
from minecraft_ai.motor import MotorIntent
from minecraft_ai.perception import ScreenRegion
from minecraft_ai.safety import MotorAction
from minecraft_ai.skills import SkillFailureCode
from test_mining_control import _ROCKET_SOURCE, _fact, _mining_board


def _step(aim, center, tick, **kwargs):
    return aim.step(target_id=kwargs.pop("target_id", "block"), center=center,
                    aligned=(False, False), observed_ns=tick, now_ns=tick, **kwargs)


@pytest.mark.parametrize("sensitivity", [0.0002, 0.0005, 0.002, 0.01])
def test_feedback_converges_across_camera_sensitivities(sensitivity: float) -> None:
    aim = AdaptiveMiningAim()
    center = [0.2, 0.7]
    counts = []
    for tick in range(1, 21):
        command = aim.step(target_id="block", center=tuple(center),
                           aligned=tuple(abs(c - 0.5) < 0.01 for c in center),
                           observed_ns=tick, now_ns=tick)
        counts.append(command)
        center = [c - sensitivity * d for c, d in zip(center, command, strict=True)]
        assert all(abs(d) <= 256 for d in command)
    assert max(abs(c - 0.5) for c in center) < 0.01
    if sensitivity < 0.002:
        assert any(abs(dx) > 12 for dx, _ in counts)


def test_axes_calibrate_independently() -> None:
    aim = AdaptiveMiningAim()
    center = [0.2, 0.7]
    for tick in range(1, 21):
        dx, dy = _step(aim, tuple(center), tick)
        center[0] -= 0.0002 * dx
        center[1] -= 0.005 * dy
    assert abs(center[0] - 0.5) < 0.01
    assert abs(center[1] - 0.5) < 0.01


def test_stale_future_and_replayed_feedback_does_not_issue_camera_commands() -> None:
    aim = AdaptiveMiningAim()
    assert _step(aim, (0.2, 0.5), 10) == (-12, 0)
    assert _step(aim, (0.21, 0.5), 10) == (0, 0)
    assert _step(aim, (0.21, 0.5), 9) == (0, 0)
    assert _step(aim, (0.21, 0.5), 9, target_id="stale-other-target") == (0, 0)
    assert aim.step(target_id="block", center=(0.21, 0.5), aligned=(False, True),
                    observed_ns=12, now_ns=11) == (0, 0)
    assert _step(aim, (0.21, 0.5), 13)[0] < -12


def test_target_switch_and_reset_discard_calibration() -> None:
    aim = AdaptiveMiningAim()
    _step(aim, (0.2, 0.5), 1)
    assert _step(aim, (0.21, 0.5), 2)[0] < -12
    assert _step(aim, (0.2, 0.5), 3, target_id="different") == (-12, 0)
    aim.reset()
    assert _step(aim, (0.2, 0.5), 4) == (-12, 0)


def test_locomotion_and_wrong_direction_do_not_teach_larger_gain() -> None:
    aim = AdaptiveMiningAim()
    _step(aim, (0.2, 0.5), 1, stationary=False)
    assert _step(aim, (0.21, 0.5), 2) == (-12, 0)
    assert _step(aim, (0.20, 0.5), 3) == (-12, 0)


@pytest.mark.parametrize("value", [math.nan, math.inf, -0.1, 1.1])
def test_invalid_centers_cannot_emit_input(value: float) -> None:
    assert _step(AdaptiveMiningAim(), (value, 0.5), 1) == (0, 0)


@pytest.mark.parametrize("value", [0, 4097, True, 1.5])
def test_camera_bound_validation(value) -> None:
    with pytest.raises(ValueError):
        AdaptiveMiningAim(max_step=value)


def _operator_board(now: int, center: float):
    return _mining_board(
        now_ns=now, kind="dirt", item=None, include_selected_slot=False,
        track_region=ScreenRegion(x=center - 0.025, y=0.35, width=0.05, height=0.3),
        track_attributes={"source": "operator", "tracking_source": _ROCKET_SOURCE,
                          "target_exists_probability": 0.96},
        track_seen_ns=now, target_source=_ROCKET_SOURCE,
        extra_facts=(_fact("target.reference_available", True, now_ns=now,
                           source="operator:cross-view-reference:target:one"),),
    )


def test_live_guard_learns_aim_then_requires_fresh_centered_evidence_for_attack() -> None:
    now = time.monotonic_ns()
    guard = MiningLeaseGuard()
    intent = MotorIntent(skill_id="mine_visible_block", mode="mine", episode_id="acquire",
                         target_label="dirt", parameters={"target": "dirt"})
    center = 0.2
    commands = []
    for i in range(20):
        tick = now + i * 200_000_000
        result = guard.inspect(MotorAction(sequence=i + 1, buttons_down=("left",)),
                               _operator_board(tick, center), intent, now_ns=tick)
        assert result.failure_code is None
        if result.action.buttons_down:
            assert abs(center - 0.5) <= 0.025
            assert result.action.mouse_dx == result.action.mouse_dy == 0
            break
        commands.append(result.action.mouse_dx)
        assert result.synthetic
        center -= 0.0002 * result.action.mouse_dx
    else:
        pytest.fail("adaptive acquisition did not center the target within the deadline")
    assert any(abs(dx) > 12 for dx in commands)


def test_no_camera_response_still_hits_absolute_acquisition_deadline() -> None:
    now = time.monotonic_ns()
    guard = MiningLeaseGuard()
    intent = MotorIntent(skill_id="mine_visible_block", mode="mine", episode_id="acquire",
                         target_label="dirt", parameters={"target": "dirt"})
    for i in range(7):
        tick = now + i * 1_000_000_000
        result = guard.inspect(MotorAction(sequence=i + 1, buttons_down=("left",)),
                               _operator_board(tick, 0.2), intent, now_ns=tick)
    assert result.failure_code == SkillFailureCode.MINING_ACQUISITION_TIMEOUT
    assert result.force_release_left
