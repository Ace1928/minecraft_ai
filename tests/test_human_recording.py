from __future__ import annotations

from minecraft_ai.human_recording import (
    HumanInputAccumulator,
    HumanInputEvent,
    HumanInputKind,
    XInput2StreamParser,
    normalize_keysym,
)


class _Keys:
    def __call__(self, keycode: int) -> str | None:
        return {25: "w", 37: "ctrl", 65: "space"}.get(keycode)


def test_xinput2_parser_preserves_relative_mouse_and_raw_keys() -> None:
    parser = XInput2StreamParser()
    lines = (
        "EVENT type 13 (RawKeyPress)\n",
        "    device: 3 (9)\n",
        "    detail: 65\n",
        "    valuators:\n",
        "\n",
        "EVENT type 17 (RawMotion)\n",
        "    detail: 0\n",
        "    valuators:\n",
        "          0: 3.00 (3.00)\n",
        "          1: -2.00 (-2.00)\n",
        "\n",
    )
    events = tuple(event for line in lines for event in parser.feed(line))

    assert events[0].kind == HumanInputKind.KEY_PRESS
    assert events[0].detail == 65
    assert events[1].kind == HumanInputKind.MOTION
    assert (events[1].dx, events[1].dy) == (3.0, -2.0)


def test_accumulator_records_transitions_jump_and_camera_without_repeat() -> None:
    accumulator = HumanInputAccumulator(_Keys())  # type: ignore[arg-type]
    for detail in (25, 37, 65, 65):
        accumulator.ingest(HumanInputEvent(HumanInputKind.KEY_PRESS, detail=detail))
    accumulator.ingest(HumanInputEvent(HumanInputKind.MOTION, dx=3.4, dy=-2.2))

    first, accepted_ns = accumulator.snapshot(0, duration_ms=50)
    second, _ = accumulator.snapshot(1, duration_ms=50)

    assert first.keys_down == ("ctrl", "space", "w")
    assert (first.mouse_dx, first.mouse_dy) == (3, -2)
    assert first.duration_ms == 50
    assert accepted_ns > 0
    assert not second.keys_down
    assert not second.keys_up
    assert (second.mouse_dx, second.mouse_dy) == (0, 0)

    accumulator.ingest(HumanInputEvent(HumanInputKind.KEY_RELEASE, detail=65))
    released, _ = accumulator.snapshot(2, duration_ms=50)
    assert released.keys_up == ("space",)


def test_keysym_normalization_matches_motor_vocabulary() -> None:
    assert normalize_keysym("Control_L") == "ctrl"
    assert normalize_keysym("Shift_R") == "shift"
    assert normalize_keysym("Return") == "enter"
    assert normalize_keysym("w") == "w"
    assert normalize_keysym(None) is None
