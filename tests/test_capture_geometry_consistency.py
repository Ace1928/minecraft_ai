from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from minecraft_ai.platforms.bedrock_x11 import (
    IsolatedX11Capture,
    IsolatedX11InputBackend,
    IsolationError,
    _wine_content_rect,
)


def _content_display(
    *, x: int = 0, y: int = 26, width: int = 1920, height: int = 1054,
) -> tuple[Any, list[dict[str, int]]]:
    changes: list[dict[str, int]] = []
    client = SimpleNamespace(
        get_geometry=lambda: SimpleNamespace(width=width, height=height),
        configure=lambda **kwargs: changes.append(kwargs),
    )
    minecraft = SimpleNamespace(
        get_wm_name=lambda: "Minecraft",
        query_tree=lambda: SimpleNamespace(children=[client]),
        configure=lambda **kwargs: changes.append(kwargs),
    )
    target = SimpleNamespace(
        get_wm_name=lambda: "Wine Desktop",
        query_tree=lambda: SimpleNamespace(children=[minecraft]),
        translate_coords=lambda *_args: SimpleNamespace(x=x, y=y),
    )
    return SimpleNamespace(
        create_resource_object=lambda *_args: target,
        sync=lambda: changes.append({"sync": 1}),
    ), changes


def test_content_resolution_preserves_a_smaller_contained_drawable() -> None:
    display, changes = _content_display(width=1280, height=720)

    assert _wine_content_rect(display, 2, 1920, 1080) == (0, 26, 1280, 720)
    assert changes == []


@pytest.mark.parametrize("x,width,height", [(-4, 1920, 1054), (0, 1921, 1054), (0, 1920, 1080)])
def test_clipped_content_is_rejected_without_reposition_or_resize(
    x: int, width: int, height: int,
) -> None:
    display, changes = _content_display(x=x, width=width, height=height)

    with pytest.raises(IsolationError, match="clipped"):
        _wine_content_rect(display, 2, 1920, 1080)

    assert changes == []


@pytest.mark.parametrize("missing", ["minecraft", "drawable", "query_failure"])
def test_unavailable_wine_content_never_falls_back_to_parent(missing: str) -> None:
    display, changes = _content_display()
    target = display.create_resource_object("window", 2)
    if missing == "minecraft":
        target.query_tree = lambda: SimpleNamespace(children=[])
    elif missing == "drawable":
        target.query_tree().children[0].query_tree = lambda: SimpleNamespace(children=[])
    else:
        def failed_query() -> Any:
            raise RuntimeError("drawable disappeared")
        target.query_tree = failed_query

    with pytest.raises(IsolationError, match="cannot resolve"):
        _wine_content_rect(display, 2, 1920, 1080)

    assert changes == []


def test_direct_game_window_still_uses_its_complete_frame() -> None:
    display, _ = _content_display()
    target = display.create_resource_object("window", 2)
    target.get_wm_name = lambda: "Minecraft"
    target.query_tree = lambda: SimpleNamespace(children=[])

    assert _wine_content_rect(display, 2, 1920, 1080) is None


def _capture() -> tuple[IsolatedX11Capture, Any]:
    capture = object.__new__(IsolatedX11Capture)
    capture.target_window_id = 2
    capture.display_name = ":private-test-only"
    capture._X = SimpleNamespace(ZPixmap=2)
    capture._mss_module = None
    capture._frame_id = 0
    window = SimpleNamespace(get_image=lambda *_args: SimpleNamespace(data=bytes(range(64))))
    capture._display = SimpleNamespace(create_resource_object=lambda *_args: window)
    capture._bounds = lambda: {"left": 0, "top": 0, "width": 4, "height": 4}  # type: ignore[method-assign]
    capture._content_rect = lambda *_args: (1, 1, 2, 2)  # type: ignore[method-assign]
    return capture, window


def test_capture_uses_same_pre_and_post_geometry_and_acquisition_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture, window = _capture()
    order: list[str] = []

    def content(*_args: int) -> tuple[int, int, int, int]:
        order.append("geometry")
        return 1, 1, 2, 2

    def pixels(*_args: int) -> Any:
        order.append("pixels")
        return SimpleNamespace(data=bytes(range(64)))

    capture._content_rect = content  # type: ignore[method-assign]
    window.get_image = pixels
    monkeypatch.setattr("minecraft_ai.platforms.bedrock_x11.time.monotonic_ns", lambda: 123)

    frame = capture.capture()

    assert order == ["geometry", "pixels", "geometry"]
    assert (frame.width, frame.height, frame.frame_id, frame.captured_ns) == (2, 2, 1, 123)
    assert frame.bgra == bytes(range(20, 28)) + bytes(range(36, 44))


@pytest.mark.parametrize("changed", ["bounds", "content", "content_disappeared"])
def test_capture_rejects_geometry_changes_without_publishing_a_frame(changed: str) -> None:
    capture, window = _capture()

    def pixels(*_args: int) -> Any:
        if changed == "bounds":
            capture._bounds = lambda: {"left": 1, "top": 0, "width": 4, "height": 4}  # type: ignore[method-assign]
        else:
            capture._content_rect = lambda *_args: (0, 1, 2, 2) if changed == "content" else None  # type: ignore[method-assign]
        return SimpleNamespace(data=bytes(range(64)))

    window.get_image = pixels

    with pytest.raises(IsolationError, match="geometry changed"):
        capture.capture()

    assert capture._frame_id == 0


@pytest.mark.parametrize("length", [1, 60, 68])
def test_capture_rejects_wrong_bgra_length(length: int) -> None:
    capture, window = _capture()
    window.get_image = lambda *_args: SimpleNamespace(data=bytes(length))

    with pytest.raises(IsolationError, match="incomplete BGRA"):
        capture.capture()

    assert capture._frame_id == 0


def test_gui_cursor_rejects_clipped_content_without_warping() -> None:
    display, changes = _content_display(x=-4)
    target = display.create_resource_object("window", 2)
    target.get_geometry = lambda: SimpleNamespace(width=1920, height=1080)
    warps: list[tuple[int, int]] = []
    display.screen = lambda: SimpleNamespace(root=SimpleNamespace(
        warp_pointer=lambda x, y: warps.append((x, y)),
    ))
    backend = object.__new__(IsolatedX11InputBackend)
    backend._require_input_isolation = lambda: None  # type: ignore[method-assign]
    backend._display = display
    backend._targeted = False
    backend.target_window_id = 2
    backend._input_window_id = 3

    with pytest.raises(IsolationError, match="clipped"):
        backend._position_pointer_in_game(0.5, 0.5)

    assert changes == []
    assert warps == []
