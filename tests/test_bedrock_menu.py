from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from minecraft_ai.bedrock_menu import (
    BedrockMenuNavigator,
    ConfiguredServer,
    MenuNavigationError,
    MenuStage,
    NestedXTestMenuInput,
    OcrLine,
    TesseractMenuTextReader,
    _parse_tesseract_tsv,
    classify_menu_stage,
    load_configured_local_server,
)
from minecraft_ai.platforms.bedrock_x11 import CapturedFrame


def _frame(frame_id: int, *, bgra: bytes = b"") -> CapturedFrame:
    return CapturedFrame(
        frame_id=frame_id,
        captured_ns=frame_id,
        width=1000,
        height=600,
        bgra=bgra,
    )


def _lines(*values: str) -> tuple[OcrLine, ...]:
    return tuple(
        OcrLine(text=value, left=300, top=120 + index * 60, width=400, height=40)
        for index, value in enumerate(values)
    )


def _pixel_frame(
    frame_id: int,
    *,
    width: int,
    height: int,
    green_box: tuple[int, int, int, int] | None = None,
) -> CapturedFrame:
    pixels = bytearray(b"\x00\x00\x00\xff" * width * height)
    if green_box is not None:
        left, top, right, bottom = green_box
        for y in range(top, bottom + 1):
            for x in range(left, right + 1):
                offset = (y * width + x) * 4
                pixels[offset : offset + 4] = bytes((39, 133, 60, 255))
    return CapturedFrame(frame_id, frame_id, width, height, bytes(pixels))


def test_tesseract_coordinates_are_mapped_back_from_scaled_input() -> None:
    payload = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\t"
        "width\theight\tconf\ttext\n"
        "5\t1\t1\t1\t1\t1\t940\t880\t204\t80\t96\tEidos\n"
        "5\t1\t1\t1\t1\t2\t1160\t880\t190\t80\t95\tLocal\n"
        "5\t1\t1\t1\t1\t3\t1360\t880\t300\t80\t94\tBedrock\n"
    )

    assert _parse_tesseract_tsv(payload, coordinate_scale=2) == (
        OcrLine(
            text="Eidos Local Bedrock",
            left=470,
            top=440,
            width=360,
            height=40,
            confidence=95.0,
        ),
    )


class _SequenceCapture:
    def __init__(self, frames: list[CapturedFrame]) -> None:
        self.frames = iter(frames)

    def capture(self) -> CapturedFrame:
        return next(self.frames)


class _MappedTextReader:
    def __init__(self, lines: dict[int, tuple[OcrLine, ...]]) -> None:
        self.lines = lines

    def read(self, frame: CapturedFrame) -> tuple[OcrLine, ...]:
        return self.lines.get(frame.frame_id, ())


class _RecordingClicks:
    def __init__(self) -> None:
        self.clicks: list[tuple[int, int, int]] = []

    def click(self, frame: CapturedFrame, x: int, y: int) -> None:
        self.clicks.append((frame.frame_id, x, y))


def _away_lines() -> tuple[OcrLine, ...]:
    return tuple(
        OcrLine(text, left=410, top=400 + index * 36, width=300, height=20)
        for index, text in enumerate(
            ("You've been away for a bit.", "Press any button to jump back", "into the game.")
        )
    )


def test_away_overlay_takes_priority_over_background_world_hud() -> None:
    assert classify_menu_stage(
        _frame(1),
        _away_lines(),
        lan_name="BedrockConnect",
        server_name="Eidos Local Bedrock",
        hud_detector=lambda _frame: True,
    ) == MenuStage.AWAY


@pytest.mark.parametrize("missing_index", [0, 1, 2])
def test_partial_away_text_does_not_authorize_wake(missing_index: int) -> None:
    assert classify_menu_stage(
        _frame(1),
        tuple(line for index, line in enumerate(_away_lines()) if index != missing_index),
        lan_name="BedrockConnect",
        server_name="Eidos Local Bedrock",
        hud_detector=lambda _frame: False,
    ) == MenuStage.UNKNOWN


def test_away_phrases_in_upper_left_chat_do_not_authorize_wake() -> None:
    lines = tuple(
        OcrLine(line.text, 10, 30 + index * 25, 300, 20)
        for index, line in enumerate(_away_lines())
    )
    assert classify_menu_stage(
        _frame(1),
        lines,
        lan_name="BedrockConnect",
        server_name="Eidos Local Bedrock",
        hud_detector=lambda _frame: True,
    ) == MenuStage.IN_WORLD


def test_low_confidence_away_anchor_does_not_authorize_wake() -> None:
    lines = list(_away_lines())
    line = lines[1]
    lines[1] = OcrLine(line.text, line.left, line.top, line.width, line.height, 59.9)
    assert classify_menu_stage(
        _frame(1),
        tuple(lines),
        lan_name="BedrockConnect",
        server_name="Eidos Local Bedrock",
        hud_detector=lambda _frame: False,
    ) == MenuStage.UNKNOWN


@pytest.mark.parametrize("complete_crop", [False, True])
def test_away_ocr_crop_maps_original_coordinates_and_requires_complete_anchors(
    monkeypatch: pytest.MonkeyPatch, complete_crop: bool,
) -> None:
    image_sizes: list[tuple[int, int]] = []
    crop_lines = tuple(
        OcrLine(line.text, 10, 5 + index * 30, 300, 20)
        for index, line in enumerate(_away_lines())
        if complete_crop or index != 1
    )
    full_lines = _lines("Minecraft", "Play", "Settings")
    reader = TesseractMenuTextReader(executable="unused-test-tesseract")

    def read_image(image: Any) -> tuple[OcrLine, ...]:
        image_sizes.append(image.size)
        return crop_lines if len(image_sizes) == 1 else full_lines

    monkeypatch.setattr(reader, "_read_image", read_image)
    result = reader.read(_pixel_frame(1, width=1000, height=600))

    assert image_sizes[0] == (320, 102)
    if complete_crop:
        assert image_sizes == [(320, 102)]
        assert result == tuple(
            OcrLine(line.text, line.left + 400, line.top + 402, line.width, line.height)
            for line in crop_lines
        )
    else:
        assert image_sizes == [(320, 102), (1000, 600)]
        assert result == full_lines


def test_away_wakes_once_and_observes_through_away_and_unknown_until_hud() -> None:
    clicks = _RecordingClicks()
    navigator = BedrockMenuNavigator(
        capture=_SequenceCapture([_frame(index) for index in range(1, 5)]),
        text_reader=_MappedTextReader({1: _away_lines(), 2: _away_lines()}),
        click_backend=clicks,
        lan_name="BedrockConnect",
        server=ConfiguredServer("Eidos Local Bedrock", "192.168.4.166", 19133),
        max_retries=5,
        poll_interval_s=0.0,
        sleep=lambda _seconds: None,
        hud_detector=lambda frame: frame.frame_id in {1, 2, 4},
    )

    result = navigator.run()

    assert result.actions == 1
    assert result.visited[0] == MenuStage.AWAY
    assert result.visited[-1] == MenuStage.IN_WORLD
    assert clicks.clicks == [(1, *_away_lines()[1].center)]


def test_away_timeout_does_not_repeat_wake_even_with_multiple_retries() -> None:
    now = 0.0

    def advance(seconds: float) -> None:
        nonlocal now
        now += seconds

    frames = [_frame(index) for index in range(1, 11)]
    clicks = _RecordingClicks()
    navigator = BedrockMenuNavigator(
        capture=_SequenceCapture(frames),
        text_reader=_MappedTextReader({frame.frame_id: _away_lines() for frame in frames}),
        click_backend=clicks,
        lan_name="BedrockConnect",
        server=ConfiguredServer("Eidos Local Bedrock", "192.168.4.166", 19133),
        max_retries=5,
        timeout_s=1.0,
        response_timeout_s=0.25,
        poll_interval_s=0.1,
        clock=lambda: now,
        sleep=advance,
        hud_detector=lambda _frame: True,
    )

    with pytest.raises(MenuNavigationError):
        navigator.run()

    assert 0.25 <= now <= 1.0
    assert clicks.clicks == [(1, *_away_lines()[1].center)]


@pytest.mark.parametrize("initially_permitted", [False, True])
def test_away_recovery_honors_interlock_before_wake_and_while_waiting(
    initially_permitted: bool,
) -> None:
    permitted = initially_permitted

    def latch(_seconds: float) -> None:
        nonlocal permitted
        permitted = False

    clicks = _RecordingClicks()
    navigator = BedrockMenuNavigator(
        capture=_SequenceCapture([_frame(1), _frame(2)]),
        text_reader=_MappedTextReader({1: _away_lines()}),
        click_backend=clicks,
        lan_name="BedrockConnect",
        server=ConfiguredServer("Eidos Local Bedrock", "192.168.4.166", 19133),
        sleep=latch,
        hud_detector=lambda _frame: True,
        input_permitted=lambda: permitted,
    )

    with pytest.raises(MenuNavigationError, match="interlock is not clear"):
        navigator.run()

    assert clicks.clicks == (
        [(1, *_away_lines()[1].center)] if initially_permitted else []
    )


def test_away_requires_fresh_post_wake_hud_not_replayed_capture() -> None:
    class _SequenceReader:
        def __init__(self) -> None:
            self.read_count = 0

        def read(self, frame: CapturedFrame) -> tuple[OcrLine, ...]:
            self.read_count += 1
            return _away_lines() if self.read_count == 1 else ()

    reader = _SequenceReader()
    clicks = _RecordingClicks()
    navigator = BedrockMenuNavigator(
        capture=_SequenceCapture([_frame(1), _frame(1), _frame(2)]),
        text_reader=reader,
        click_backend=clicks,
        lan_name="BedrockConnect",
        server=ConfiguredServer("Eidos Local Bedrock", "192.168.4.166", 19133),
        poll_interval_s=0.0,
        sleep=lambda _seconds: None,
        hud_detector=lambda _frame: True,
    )

    result = navigator.run()

    assert result.actions == 1
    assert reader.read_count == 3
    assert clicks.clicks == [(1, *_away_lines()[1].center)]


def test_menu_navigator_follows_only_screenshot_confirmed_transitions() -> None:
    frames = [_frame(index) for index in range(1, 6)]
    reader = _MappedTextReader(
        {
            1: _lines("Welcome to Minecraft", "Skip for now"),
            2: _lines("Minecraft", "Play", "Settings", "Marketplace"),
            3: _lines("Play", "Worlds", "LAN Games", "BedrockConnect"),
            4: _lines("ServerList", "Connect to a Server", "Eidos Local Bedrock"),
        }
    )
    clicks = _RecordingClicks()
    navigator = BedrockMenuNavigator(
        capture=_SequenceCapture(frames),
        text_reader=reader,
        click_backend=clicks,
        lan_name="BedrockConnect",
        server=ConfiguredServer("Eidos Local Bedrock", "192.168.4.166", 19133),
        poll_interval_s=0.0,
        sleep=lambda _seconds: None,
        hud_detector=lambda frame: frame.frame_id == 5,
    )

    result = navigator.run()

    assert result.visited == (
        MenuStage.STARTUP_POPUP,
        MenuStage.TITLE,
        MenuStage.PLAY,
        MenuStage.BEDROCK_CONNECT,
        MenuStage.IN_WORLD,
    )
    assert result.actions == 4
    assert [click[0] for click in clicks.clicks] == [1, 2, 3, 4]
    assert result.payload()["status"] == "in-world"


def test_menu_navigator_sends_nothing_on_unknown_initial_screen() -> None:
    clicks = _RecordingClicks()
    navigator = BedrockMenuNavigator(
        capture=_SequenceCapture([_frame(1)]),
        text_reader=_MappedTextReader({1: _lines("mystery screen")}),
        click_backend=clicks,
        lan_name="BedrockConnect",
        server=ConfiguredServer("Eidos Local Bedrock", "192.168.4.166", 19133),
        poll_interval_s=0.0,
        sleep=lambda _seconds: None,
        hud_detector=lambda _frame: False,
    )

    with pytest.raises(MenuNavigationError, match="no input sent"):
        navigator.run()

    assert clicks.clicks == []


def test_menu_navigator_rechecks_emergency_interlock_between_transitions() -> None:
    permitted = True

    class _LatchAfterClick(_RecordingClicks):
        def click(self, frame: CapturedFrame, x: int, y: int) -> None:
            nonlocal permitted
            super().click(frame, x, y)
            permitted = False

    clicks = _LatchAfterClick()
    navigator = BedrockMenuNavigator(
        capture=_SequenceCapture([_frame(1), _frame(2)]),
        text_reader=_MappedTextReader(
            {
                1: _lines("Minecraft", "Play", "Settings"),
                2: _lines("Play", "Worlds", "LAN Games", "BedrockConnect"),
            }
        ),
        click_backend=clicks,
        lan_name="BedrockConnect",
        server=ConfiguredServer("Eidos Local Bedrock", "192.168.4.166", 19133),
        poll_interval_s=0.0,
        sleep=lambda _seconds: None,
        hud_detector=lambda _frame: False,
        input_permitted=lambda: permitted,
    )

    with pytest.raises(MenuNavigationError, match="interlock is not clear"):
        navigator.run()

    assert len(clicks.clicks) == 1


def test_menu_navigator_rechecks_interlock_while_loading() -> None:
    permitted = True

    def latch_during_wait(_seconds: float) -> None:
        nonlocal permitted
        permitted = False

    clicks = _RecordingClicks()
    navigator = BedrockMenuNavigator(
        capture=_SequenceCapture([_frame(1), _frame(2)]),
        text_reader=_MappedTextReader({1: _lines("Loading resource packs")}),
        click_backend=clicks,
        lan_name="BedrockConnect",
        server=ConfiguredServer("Eidos Local Bedrock", "192.168.4.166", 19133),
        poll_interval_s=0.1,
        sleep=latch_during_wait,
        hud_detector=lambda _frame: False,
        input_permitted=lambda: permitted,
    )

    with pytest.raises(MenuNavigationError, match="interlock is not clear"):
        navigator.run()

    assert clicks.clicks == []


def test_menu_navigator_stops_after_confirmed_error_transition() -> None:
    clicks = _RecordingClicks()
    navigator = BedrockMenuNavigator(
        capture=_SequenceCapture([_frame(1), _frame(2)]),
        text_reader=_MappedTextReader(
            {
                1: _lines("Minecraft", "Play", "Settings"),
                2: _lines("Unable to connect to world", "Back to menu"),
            }
        ),
        click_backend=clicks,
        lan_name="BedrockConnect",
        server=ConfiguredServer("Eidos Local Bedrock", "192.168.4.166", 19133),
        poll_interval_s=0.0,
        sleep=lambda _seconds: None,
        hud_detector=lambda _frame: False,
    )

    with pytest.raises(MenuNavigationError, match="Bedrock error after title"):
        navigator.run()

    assert len(clicks.clicks) == 1


def test_in_world_hud_needs_no_menu_input() -> None:
    clicks = _RecordingClicks()
    navigator = BedrockMenuNavigator(
        capture=_SequenceCapture([_frame(1)]),
        text_reader=_MappedTextReader({}),
        click_backend=clicks,
        lan_name="BedrockConnect",
        server=ConfiguredServer("Eidos Local Bedrock", "192.168.4.166", 19133),
        hud_detector=lambda _frame: True,
    )

    assert navigator.run().actions == 0
    assert clicks.clicks == []


def test_error_text_takes_priority_over_background_hud() -> None:
    frame = _frame(1)
    stage = classify_menu_stage(
        frame,
        _lines("Unable to connect to world", "Reconnect"),
        lan_name="BedrockConnect",
        server_name="Eidos Local Bedrock",
        hud_detector=lambda _frame: True,
    )

    assert stage == MenuStage.ERROR


def test_play_page_accepts_real_truncated_lan_label_with_tab_anchors() -> None:
    stage = classify_menu_stage(
        _frame(1),
        _lines(
            "PLAY",
            "Servers",
            "Realms",
            "Worlds",
            "Import world",
            "Survival",
            "BedrockConne...",
            "Join To Open S...",
        ),
        lan_name="BedrockConnect",
        server_name="Eidos Local Bedrock",
        hud_detector=lambda _frame: False,
    )

    assert stage == MenuStage.PLAY


def test_loading_page_tolerates_bedrock_font_ocr_error() -> None:
    stage = classify_menu_stage(
        _frame(1),
        _lines(
            "Generating Horld",
            "Appearance is taking a long time to load, Proceeding",
            "with world generation",
        ),
        lan_name="BedrockConnect",
        server_name="Eidos Local Bedrock",
        hud_detector=lambda _frame: False,
    )

    assert stage == MenuStage.LOADING


def test_bedrock_connect_external_server_transition_is_loading() -> None:
    assert classify_menu_stage(
        _frame(1),
        _lines("Connecting to external server"),
        lan_name="BedrockConnect",
        server_name="Eidos Local Bedrock",
        hud_detector=lambda _frame: False,
    ) == MenuStage.LOADING


def test_title_fallback_ignores_left_play_now_and_clicks_central_green_play() -> None:
    title = _pixel_frame(
        1,
        width=1920,
        height=1020,
        green_box=(668, 400, 1248, 510),
    )
    title_lines = (
        OcrLine("Play Mow!", 173, 498, 158, 28),
        OcrLine("tings", 880, 568, 156, 32),
        OcrLine("Harketp", 840, 824, 236, 32),
    )
    frames = [title, _frame(2), _frame(3), _frame(4)]
    clicks = _RecordingClicks()
    navigator = BedrockMenuNavigator(
        capture=_SequenceCapture(frames),
        text_reader=_MappedTextReader(
            {
                1: title_lines,
                2: _lines("Play", "Worlds", "LAN Games", "BedrockConnect"),
                3: _lines("ServerList", "Eidos Local Bedrock"),
            }
        ),
        click_backend=clicks,
        lan_name="BedrockConnect",
        server=ConfiguredServer("Eidos Local Bedrock", "192.168.4.166", 19133),
        poll_interval_s=0.0,
        sleep=lambda _seconds: None,
        hud_detector=lambda frame: frame.frame_id == 4,
    )

    result = navigator.run()

    assert result.visited == (
        MenuStage.TITLE,
        MenuStage.PLAY,
        MenuStage.BEDROCK_CONNECT,
        MenuStage.IN_WORLD,
    )
    assert clicks.clicks[0] == (1, 958, 455)
    assert clicks.clicks[0][1] > 665


def test_title_fallback_requires_both_visual_control_and_menu_anchors() -> None:
    frame_without_green = _pixel_frame(1, width=1920, height=1020)
    frame_without_anchors = _pixel_frame(
        2,
        width=1920,
        height=1020,
        green_box=(668, 400, 1248, 510),
    )
    fragmentary_anchors = (
        OcrLine("Play Mow!", 173, 498, 158, 28),
        OcrLine("tings", 880, 568, 156, 32),
        OcrLine("Harketp", 840, 824, 236, 32),
    )

    assert classify_menu_stage(
        frame_without_green,
        fragmentary_anchors,
        lan_name="BedrockConnect",
        server_name="Eidos Local Bedrock",
        hud_detector=lambda _frame: False,
    ) == MenuStage.UNKNOWN
    assert classify_menu_stage(
        frame_without_anchors,
        (OcrLine("Play Mow!", 173, 498, 158, 28),),
        lan_name="BedrockConnect",
        server_name="Eidos Local Bedrock",
        hud_detector=lambda _frame: False,
    ) == MenuStage.UNKNOWN


def test_death_recovery_uses_visual_respawn_anchor_when_ocr_misses_label() -> None:
    width, height = 1000, 600
    pixels = bytearray(b"\x00\x00\x00\xff" * width * height)
    for y in range(380, 431):
        for x in range(330, 671):
            offset = (y * width + x) * 4
            pixels[offset : offset + 4] = bytes((39, 133, 60, 255))
    frames = [_frame(1, bgra=bytes(pixels)), _frame(2)]
    clicks = _RecordingClicks()
    navigator = BedrockMenuNavigator(
        capture=_SequenceCapture(frames),
        text_reader=_MappedTextReader({1: _lines("TOU DIED!", "Player was slain by Zombie")}),
        click_backend=clicks,
        lan_name="BedrockConnect",
        server=ConfiguredServer("Eidos Local Bedrock", "192.168.4.166", 19133),
        poll_interval_s=0.0,
        sleep=lambda _seconds: None,
        hud_detector=lambda frame: frame.frame_id == 2,
    )

    result = navigator.run()

    assert result.visited == (MenuStage.DEATH, MenuStage.IN_WORLD)
    assert result.actions == 1
    assert clicks.clicks == [(1, 500, 405)]


def test_death_recovery_fails_closed_without_visual_or_ocr_respawn_target() -> None:
    clicks = _RecordingClicks()
    navigator = BedrockMenuNavigator(
        capture=_SequenceCapture([_frame(1, bgra=b"\x00\x00\x00\xff" * 1000 * 600)]),
        text_reader=_MappedTextReader({1: _lines("YOU DIED!", "Player was slain")}),
        click_backend=clicks,
        lan_name="BedrockConnect",
        server=ConfiguredServer("Eidos Local Bedrock", "192.168.4.166", 19133),
        poll_interval_s=0.0,
        sleep=lambda _seconds: None,
        hud_detector=lambda _frame: True,
    )

    with pytest.raises(MenuNavigationError, match="not visually confirmed"):
        navigator.run()

    assert clicks.clicks == []


class _FakeXTest:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, int, int]] = []

    def fake_input(self, display: Any, event_type: int, detail: int) -> None:
        self.calls.append((display, event_type, detail))


def test_nested_menu_click_translates_drawable_point_into_private_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    translated: list[tuple[Any, int, int]] = []
    warped: list[tuple[int, int]] = []
    focused: list[tuple[int, int]] = []
    sleeps: list[float] = []
    monkeypatch.setattr("minecraft_ai.bedrock_menu.time.sleep", sleeps.append)
    window = SimpleNamespace(
        get_geometry=lambda: SimpleNamespace(width=1000, height=600),
        get_attributes=lambda: SimpleNamespace(),
        set_input_focus=lambda revert, when: focused.append((revert, when)),
    )

    class _Root:
        def translate_coords(self, source: Any, x: int, y: int) -> Any:
            translated.append((source, x, y))
            return SimpleNamespace(x=x + 40, y=y + 30)

        def warp_pointer(self, x: int, y: int) -> None:
            warped.append((x, y))

    root = _Root()
    display = SimpleNamespace(
        screen=lambda: SimpleNamespace(root=root),
        sync=lambda: None,
    )
    xtest = _FakeXTest()
    backend = object.__new__(NestedXTestMenuInput)
    backend._window = window
    backend._display = display
    backend._xtest = xtest
    backend._input_permitted = lambda: True
    backend._x = SimpleNamespace(
        RevertToParent=1,
        CurrentTime=0,
        ButtonPress=4,
        ButtonRelease=5,
    )

    backend.click(_frame(1), 500, 405)

    assert translated == [(window, 500, 405)]
    assert warped == [(540, 435)]
    assert focused == [(1, 0)]
    assert xtest.calls == [(display, 4, 1), (display, 5, 1)]
    assert sleeps == [0.03, 0.075]


def test_nested_menu_click_rechecks_interlock_at_button_press(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    permitted = True

    def sleep(_seconds: float) -> None:
        nonlocal permitted
        permitted = False

    monkeypatch.setattr("minecraft_ai.bedrock_menu.time.sleep", sleep)
    window = SimpleNamespace(
        get_geometry=lambda: SimpleNamespace(width=1000, height=600),
        get_attributes=lambda: SimpleNamespace(),
        set_input_focus=lambda _revert, _when: None,
    )
    root = SimpleNamespace(
        translate_coords=lambda _source, x, y: SimpleNamespace(x=x, y=y),
        warp_pointer=lambda _x, _y: None,
    )
    display = SimpleNamespace(
        screen=lambda: SimpleNamespace(root=root),
        sync=lambda: None,
    )
    xtest = _FakeXTest()
    backend = object.__new__(NestedXTestMenuInput)
    backend._window = window
    backend._display = display
    backend._xtest = xtest
    backend._input_permitted = lambda: permitted
    backend._x = SimpleNamespace(
        RevertToParent=1,
        CurrentTime=0,
        ButtonPress=4,
        ButtonRelease=5,
    )

    with pytest.raises(MenuNavigationError, match="no click was sent"):
        backend.click(_frame(1), 500, 405)

    assert xtest.calls == []


def test_menu_navigator_waits_through_transient_unknown_transition_frame() -> None:
    frames = [_frame(index) for index in range(1, 6)]
    reader = _MappedTextReader(
        {
            1: _lines("Minecraft", "Play", "Settings"),
            2: _lines("partially rendered artwork"),
            3: _lines("Play", "Worlds", "LAN Games", "BedrockConnect"),
            4: _lines("ServerList", "Eidos Local Bedrock"),
        }
    )
    clicks = _RecordingClicks()
    navigator = BedrockMenuNavigator(
        capture=_SequenceCapture(frames),
        text_reader=reader,
        click_backend=clicks,
        lan_name="BedrockConnect",
        server=ConfiguredServer("Eidos Local Bedrock", "192.168.4.166", 19133),
        poll_interval_s=0.0,
        sleep=lambda _seconds: None,
        hud_detector=lambda frame: frame.frame_id == 5,
    )

    result = navigator.run()

    assert result.visited == (
        MenuStage.TITLE,
        MenuStage.PLAY,
        MenuStage.BEDROCK_CONNECT,
        MenuStage.IN_WORLD,
    )
    assert result.actions == 3
    assert [click[0] for click in clicks.clicks] == [1, 3, 4]


def test_menu_navigator_accepts_unique_substantial_configured_server_prefix() -> None:
    frames = [_frame(index) for index in range(1, 3)]
    clicks = _RecordingClicks()
    navigator = BedrockMenuNavigator(
        capture=_SequenceCapture(frames),
        text_reader=_MappedTextReader(
            {
                1: _lines("Server List", "Exit List", "Eidos Local", "Hive"),
            }
        ),
        click_backend=clicks,
        lan_name="BedrockConnect",
        server=ConfiguredServer("Eidos Local Bedrock", "192.168.4.166", 19133),
        poll_interval_s=0.0,
        sleep=lambda _seconds: None,
        hud_detector=lambda frame: frame.frame_id == 2,
    )

    result = navigator.run()

    assert result.actions == 1
    assert result.visited == (MenuStage.BEDROCK_CONNECT, MenuStage.IN_WORLD)
    assert clicks.clicks == [(1, 500, 260)]


def test_load_configured_local_server_selects_unique_private_target(tmp_path: Path) -> None:
    path = tmp_path / "custom_servers.json"
    path.write_text(
        json.dumps(
            [
                {"name": "Public", "address": "example.com", "port": 19132},
                {
                    "name": "Eidos Local Bedrock",
                    "address": "192.168.4.166",
                    "port": 19133,
                },
            ]
        ),
        encoding="utf-8",
    )

    assert load_configured_local_server(path) == ConfiguredServer(
        "Eidos Local Bedrock",
        "192.168.4.166",
        19133,
    )


def test_load_configured_local_server_requires_name_when_ambiguous(tmp_path: Path) -> None:
    path = tmp_path / "custom_servers.json"
    path.write_text(
        json.dumps(
            [
                {"name": "One", "address": "10.0.0.1", "port": 19132},
                {"name": "Two", "address": "10.0.0.2", "port": 19133},
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(MenuNavigationError, match="exactly one"):
        load_configured_local_server(path)
    assert load_configured_local_server(path, requested_name="Two").name == "Two"
