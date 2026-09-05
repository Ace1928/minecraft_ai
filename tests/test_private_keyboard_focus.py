from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from minecraft_ai.platforms.bedrock_x11 import IsolatedX11InputBackend, IsolationError


def _backend(*, focus_id: int = 4, child_mask: int = 32768) -> tuple[Any, Any]:
    state = SimpleNamespace(focus_id=focus_id, changes=[], syncs=0, permitted=True, bounce=None)
    windows: dict[int, Any] = {}
    for window_id, parent_id, mask in [(1, 1, 0), (2, 1, 0), (3, 2, 3), (4, 3, child_mask)]:
        def set_focus(_revert: int, _time: int, *, target: int = window_id) -> None:
            state.changes.append(target)
            state.focus_id = target if state.bounce is None else state.bounce

        windows[window_id] = SimpleNamespace(
            id=window_id,
            query_tree=lambda parent=parent_id: SimpleNamespace(parent=windows[parent]),
            get_attributes=lambda selected_mask=mask: SimpleNamespace(
                map_state=2, all_event_masks=selected_mask,
            ),
            set_input_focus=set_focus,
        )
    state.windows = windows
    backend = object.__new__(IsolatedX11InputBackend)
    backend._require_input_isolation = lambda: None  # type: ignore[method-assign]
    backend._targeted = False
    backend.target_window_id = 2
    backend._input_window_id = 3
    backend._input_permitted = lambda: state.permitted
    backend._x = SimpleNamespace(KeyPressMask=1, KeyReleaseMask=2, RevertToParent=2, CurrentTime=0)
    backend._display = SimpleNamespace(
        get_input_focus=lambda: SimpleNamespace(focus=windows[state.focus_id]),
        create_resource_object=lambda _kind, window_id: windows[window_id],
        sync=lambda: setattr(state, "syncs", state.syncs + 1),
    )
    return backend, state


@pytest.mark.parametrize("mask", [0, 1, 2, 32768])
def test_render_only_or_partial_key_child_restores_verified_input_parent(mask: int) -> None:
    backend, state = _backend(child_mask=mask)

    backend._ensure_input_focus()

    assert state.changes == [3]
    assert state.focus_id == 3
    assert state.syncs == 1


@pytest.mark.parametrize("focus_id", [2, 3, 4])
def test_valid_desktop_parent_and_keyboard_interested_child_preserve_focus(focus_id: int) -> None:
    backend, state = _backend(focus_id=focus_id, child_mask=3)

    backend._ensure_input_focus()

    assert state.changes == []
    assert state.syncs == 0


def test_focus_can_bounce_to_known_wine_desktop_after_parent_selection() -> None:
    backend, state = _backend()
    state.bounce = 2

    backend._ensure_input_focus()

    assert state.changes == [3]
    assert state.focus_id == 2


def test_unrelated_private_focus_preserves_existing_desktop_restoration_policy() -> None:
    backend, state = _backend(focus_id=1)

    backend._ensure_input_focus()

    assert state.changes == [2]


@pytest.mark.parametrize("window_id", [2, 3, 4])
def test_unreadable_focus_attributes_fail_closed_without_focus_change(window_id: int) -> None:
    backend, state = _backend(focus_id=2 if window_id == 2 else 4)

    def unknown_attributes() -> Any:
        raise RuntimeError("attributes unavailable")

    state.windows[window_id].get_attributes = unknown_attributes

    with pytest.raises(IsolationError, match="cannot focus"):
        backend._ensure_input_focus()

    assert state.changes == []


@pytest.mark.parametrize("attributes", [
    SimpleNamespace(map_state=2),
    SimpleNamespace(map_state=2, all_event_masks=None),
    SimpleNamespace(map_state=0, all_event_masks=3),
    SimpleNamespace(map_state=2, all_event_masks=32768),
])
def test_unverified_input_parent_never_receives_focus(attributes: Any) -> None:
    backend, state = _backend()
    state.windows[3].get_attributes = lambda: attributes

    with pytest.raises(IsolationError, match="cannot focus"):
        backend._ensure_input_focus()

    assert state.changes == []


def test_rejected_focus_change_is_not_retried_or_accepted() -> None:
    backend, state = _backend()
    state.bounce = 4

    with pytest.raises(IsolationError, match="cannot focus"):
        backend._ensure_input_focus()

    assert state.changes == [3]


def test_interlock_prevents_private_focus_change() -> None:
    backend, state = _backend()
    state.permitted = False

    with pytest.raises(IsolationError, match="cannot focus"):
        backend._ensure_input_focus()

    assert state.changes == []


def test_explicit_host_debug_never_reads_or_changes_focus() -> None:
    backend, state = _backend()
    backend._targeted = True
    backend._display.get_input_focus = lambda: pytest.fail("host focus must not be inspected")

    backend._ensure_input_focus()

    assert state.changes == []
