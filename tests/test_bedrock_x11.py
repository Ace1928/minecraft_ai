from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from minecraft_ai.platforms.bedrock_x11 import (
    IsolationError,
    _client_fit_dimensions,
    _contained_content_rect,
    _resolve_minecraft_input_window,
    _window_is_descendant_or_same,
)


@dataclass
class _Tree:
    parent: _Window
    children: list[_Window] = field(default_factory=list)


@dataclass
class _Window:
    id: int
    name: str | None = None
    wm_class: tuple[str, ...] = ()
    tree: _Tree | None = None
    width: int = 1
    height: int = 1

    def get_wm_name(self) -> str | None:
        return self.name

    def get_wm_class(self) -> tuple[str, ...]:
        return self.wm_class

    def query_tree(self) -> _Tree:
        assert self.tree is not None
        return self.tree

    def get_geometry(self) -> _Window:
        return self


@dataclass
class _Display:
    windows: dict[int, _Window]

    def create_resource_object(self, resource: str, window_id: int) -> _Window:
        assert resource == "window"
        return self.windows[window_id]


def _wine_tree() -> tuple[_Display, _Window, _Window, _Window]:
    root = _Window(1)
    desktop = _Window(2, "Wine Desktop", ("explorer.exe", "explorer.exe"))
    ime = _Window(5, "Default IME", ("minecraft.windows.exe",))
    minecraft = _Window(
        3,
        "Minecraft",
        ("minecraft.windows.exe", "minecraft.windows.exe"),
        width=1280,
        height=720,
    )
    drawable = _Window(4)
    root.tree = _Tree(parent=root, children=[desktop])
    desktop.tree = _Tree(parent=root, children=[ime, minecraft])
    ime.tree = _Tree(parent=desktop)
    minecraft.tree = _Tree(parent=desktop, children=[drawable])
    drawable.tree = _Tree(parent=minecraft)
    display = _Display({item.id: item for item in (root, desktop, ime, minecraft, drawable)})
    return display, desktop, minecraft, drawable


def test_resolve_minecraft_input_window_preserves_wine_capture_parent() -> None:
    display, desktop, minecraft, _ = _wine_tree()

    resolved = _resolve_minecraft_input_window(display, desktop.id)

    assert resolved is minecraft


def test_focus_descendant_is_accepted_as_part_of_minecraft_subtree() -> None:
    _, _, minecraft, drawable = _wine_tree()

    assert _window_is_descendant_or_same(minecraft, minecraft.id)
    assert _window_is_descendant_or_same(drawable, minecraft.id)
    assert not _window_is_descendant_or_same(minecraft, 999)


def test_complete_wine_drawable_is_accepted_for_capture() -> None:
    rect = _contained_content_rect(
        parent_width=1908,
        parent_height=1047,
        x=0,
        y=26,
        content_width=1908,
        content_height=1021,
    )

    assert rect == (0, 26, 1908, 1021)


def test_wine_drawable_fit_preserves_the_complete_hud_area() -> None:
    assert _client_fit_dimensions(
        parent_width=1842,
        parent_height=1018,
        x=0,
        y=26,
    ) == (1842, 992)


def test_wine_drawable_fit_rejects_an_origin_outside_the_parent() -> None:
    with pytest.raises(IsolationError, match="origin is outside"):
        _client_fit_dimensions(
            parent_width=1842,
            parent_height=1018,
            x=1842,
            y=26,
        )


def test_clipped_wine_drawable_is_rejected_before_capture() -> None:
    with pytest.raises(IsolationError, match="client drawable is clipped"):
        _contained_content_rect(
            parent_width=1279,
            parent_height=661,
            x=0,
            y=26,
            content_width=1280,
            content_height=694,
        )
