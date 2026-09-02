from __future__ import annotations

from dataclasses import dataclass, field

from minecraft_ai.platforms.bedrock_x11 import (
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

    def get_wm_name(self) -> str | None:
        return self.name

    def get_wm_class(self) -> tuple[str, ...]:
        return self.wm_class

    def query_tree(self) -> _Tree:
        assert self.tree is not None
        return self.tree


@dataclass
class _Display:
    windows: dict[int, _Window]

    def create_resource_object(self, resource: str, window_id: int) -> _Window:
        assert resource == "window"
        return self.windows[window_id]


def _wine_tree() -> tuple[_Display, _Window, _Window, _Window]:
    root = _Window(1)
    desktop = _Window(2, "Wine Desktop", ("explorer.exe", "explorer.exe"))
    minecraft = _Window(
        3,
        "Minecraft",
        ("minecraft.windows.exe", "minecraft.windows.exe"),
    )
    drawable = _Window(4)
    root.tree = _Tree(parent=root, children=[desktop])
    desktop.tree = _Tree(parent=root, children=[minecraft])
    minecraft.tree = _Tree(parent=desktop, children=[drawable])
    drawable.tree = _Tree(parent=minecraft)
    display = _Display(
        {item.id: item for item in (root, desktop, minecraft, drawable)}
    )
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
