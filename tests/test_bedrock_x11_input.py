from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import pytest

from minecraft_ai.mining_control import MiningLeaseGuard
from minecraft_ai.motor import MotorIntent
from minecraft_ai.perception import PerceptionBlackboard
from minecraft_ai.platforms.bedrock_x11 import IsolatedX11InputBackend
from minecraft_ai.safety import MotorAction, MotorLease, MotorRejected


class _FakeXTest:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, int, int]] = []

    def fake_input(self, display: Any, event_type: int, keycode: int) -> None:
        self.calls.append((display, event_type, keycode))


def _key_routing_backend(*, host_targeted: bool) -> tuple[IsolatedX11InputBackend, _FakeXTest]:
    backend = object.__new__(IsolatedX11InputBackend)
    fake_xtest = _FakeXTest()
    backend._targeted = host_targeted
    backend._display = object()
    backend._x = SimpleNamespace(KeyPress=2, KeyRelease=3)
    backend._xtest = fake_xtest
    return backend, fake_xtest


def test_isolated_display_keyboard_uses_private_xserver_xtest() -> None:
    backend, fake_xtest = _key_routing_backend(host_targeted=False)
    targeted: list[tuple[int, bool]] = []
    backend._targeted_key = lambda keycode, down: targeted.append((keycode, down))  # type: ignore[method-assign]

    backend._send_key(38, down=True)
    backend._send_key(38, down=False)

    assert fake_xtest.calls == [
        (backend._display, backend._x.KeyPress, 38),
        (backend._display, backend._x.KeyRelease, 38),
    ]
    assert targeted == []


def test_host_debug_keyboard_retains_window_targeted_xsend_event() -> None:
    backend, fake_xtest = _key_routing_backend(host_targeted=True)
    targeted: list[tuple[int, bool]] = []
    backend._targeted_key = lambda keycode, down: targeted.append((keycode, down))  # type: ignore[method-assign]

    backend._send_key(38, down=True)
    backend._send_key(38, down=False)

    assert targeted == [(38, True), (38, False)]
    assert fake_xtest.calls == []


def test_absolute_cursor_maps_cropped_frame_coordinates_to_wine_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    translated: list[tuple[Any, int, int]] = []
    warped: list[tuple[int, int]] = []
    window = SimpleNamespace(get_geometry=lambda: SimpleNamespace(width=1928, height=1088))
    root = SimpleNamespace(
        translate_coords=lambda item, x, y: (
            translated.append((item, x, y))
            or SimpleNamespace(x=x + 100, y=y + 200)
        ),
        warp_pointer=lambda x, y: warped.append((x, y)),
    )
    display = SimpleNamespace(
        create_resource_object=lambda _kind, _window_id: window,
        screen=lambda: SimpleNamespace(root=root),
        sync=lambda: None,
    )
    backend = object.__new__(IsolatedX11InputBackend)
    backend._targeted = False
    backend.target_window_id = 42
    backend._input_window_id = 99
    backend._display = display
    monkeypatch.setattr(
        "minecraft_ai.platforms.bedrock_x11._wine_content_rect",
        lambda *_args: (4, 30, 1920, 1054),
    )

    backend._position_pointer_in_game(0.25, 0.25)

    assert translated == [(window, 484, 293)]
    assert warped == [(584, 493)]


def test_atomic_gui_tap_leaves_bedrock_backend_and_guard_state_in_agreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = object.__new__(IsolatedX11InputBackend)
    fake_xtest = _FakeXTest()
    backend._targeted = False
    backend._display = SimpleNamespace(sync=lambda: None)
    backend._x = SimpleNamespace(
        KeyPress=2,
        KeyRelease=3,
        ButtonPress=4,
        ButtonRelease=5,
    )
    backend._xtest = fake_xtest
    backend._relative_mouse = SimpleNamespace(move=lambda _x, _y: None)
    backend._lease = MotorLease(
        lease_id="lease",
        session_id="session",
        target_instance="bedrock:test",
        backend_id=backend.backend_id,
        expires_monotonic_ns=time.monotonic_ns() + 1_000_000_000,
        allowed_actions=frozenset({"keyboard", "button", "mouse"}),
        max_action_duration_ms=250,
        first_sequence=0,
    )
    backend.target_window_id = 42
    backend._input_window_id = 42
    backend._host_monitor_binding = None
    backend._input_permitted = lambda: True
    backend._held_keys = set()
    backend._held_buttons = set()
    backend.probe_target = lambda: True  # type: ignore[method-assign]
    backend._ensure_input_focus = lambda: None  # type: ignore[method-assign]
    backend._park_pointer_in_game = lambda: None  # type: ignore[method-assign]
    backend._position_pointer_in_game = lambda _x, _y: None  # type: ignore[method-assign]
    backend._keycode = lambda _key: 26  # type: ignore[method-assign]

    sleeps: list[float] = []
    monkeypatch.setattr("minecraft_ai.platforms.bedrock_x11.time.sleep", sleeps.append)
    action = MotorAction(
        sequence=1,
        keys_down=("e",),
        keys_up=("e",),
        buttons_down=("right",),
        buttons_up=("right",),
        cursor_x=0.25,
        cursor_y=0.25,
        camera_semantics="cursor",
        duration_ms=50,
    )
    guard = MiningLeaseGuard()
    decision = guard.inspect(
        action,
        PerceptionBlackboard(),
        MotorIntent(
            skill_id="craft_wood_planks",
            mode="craft_planks",
            episode_id="craft:one",
        ),
        now_ns=time.monotonic_ns(),
    )

    backend.apply(decision.action)

    assert guard.held_keys == tuple(backend.held_keys) == ()
    assert guard.held_buttons == tuple(backend.held_buttons) == ()
    assert [event_type for _display, event_type, _detail in fake_xtest.calls] == [
        backend._x.KeyPress,
        backend._x.ButtonPress,
        backend._x.KeyRelease,
        backend._x.ButtonRelease,
    ]
    assert sleeps == [0.05]


def test_apply_rechecks_interlock_before_positive_events_but_keeps_releases() -> None:
    backend = object.__new__(IsolatedX11InputBackend)
    fake_xtest = _FakeXTest()
    permitted = True
    mouse_moves: list[tuple[int, int]] = []
    pointer_parks = 0

    def park_then_latch() -> None:
        nonlocal permitted, pointer_parks
        pointer_parks += 1
        if pointer_parks == 2:
            permitted = False

    backend._targeted = False
    backend._display = SimpleNamespace(sync=lambda: None)
    backend._x = SimpleNamespace(
        KeyPress=2,
        KeyRelease=3,
        ButtonPress=4,
        ButtonRelease=5,
    )
    backend._xtest = fake_xtest
    backend._relative_mouse = SimpleNamespace(
        move=lambda x, y: mouse_moves.append((x, y))
    )
    backend._lease = MotorLease(
        lease_id="lease",
        session_id="session",
        target_instance="bedrock:test",
        backend_id=backend.backend_id,
        expires_monotonic_ns=time.monotonic_ns() + 1_000_000_000,
        allowed_actions=frozenset({"keyboard", "button", "mouse"}),
        max_action_duration_ms=250,
        first_sequence=0,
    )
    backend.target_window_id = 42
    backend._host_monitor_binding = None
    backend._input_permitted = lambda: permitted
    backend._held_keys = {"a"}
    backend._held_buttons = {"right"}
    backend.probe_target = lambda: True  # type: ignore[method-assign]
    backend._ensure_input_focus = lambda: None  # type: ignore[method-assign]
    backend._park_pointer_in_game = park_then_latch  # type: ignore[method-assign]
    backend._keycode = lambda key: {"a": 38, "w": 25}[key]  # type: ignore[method-assign]

    action = MotorAction(
        sequence=0,
        keys_up=("a",),
        buttons_up=("right",),
        mouse_dx=3,
        keys_down=("w",),
        buttons_down=("left",),
    )
    with pytest.raises(MotorRejected, match="interlock"):
        backend.apply(action)

    assert (backend._display, backend._x.KeyRelease, 38) in fake_xtest.calls
    assert (backend._display, backend._x.ButtonRelease, 3) in fake_xtest.calls
    assert all(
        event_type not in {backend._x.KeyPress, backend._x.ButtonPress}
        for _display, event_type, _detail in fake_xtest.calls
    )
    assert pointer_parks == 2
    assert mouse_moves == []
    assert backend._held_keys == set()
    assert backend._held_buttons == set()


@pytest.mark.parametrize(
    "action",
    (
        MotorAction(sequence=0, mouse_dx=1),
        MotorAction(sequence=0, keys_down=("w",)),
        MotorAction(sequence=0, buttons_down=("left",)),
    ),
)
def test_apply_blocks_each_positive_event_when_interlock_is_latched(
    action: MotorAction,
) -> None:
    backend = object.__new__(IsolatedX11InputBackend)
    fake_xtest = _FakeXTest()
    mouse_moves: list[tuple[int, int]] = []
    backend._targeted = False
    backend._display = SimpleNamespace(sync=lambda: None)
    backend._x = SimpleNamespace(
        KeyPress=2,
        KeyRelease=3,
        ButtonPress=4,
        ButtonRelease=5,
    )
    backend._xtest = fake_xtest
    backend._relative_mouse = SimpleNamespace(
        move=lambda x, y: mouse_moves.append((x, y))
    )
    backend._lease = MotorLease(
        lease_id="lease",
        session_id="session",
        target_instance="bedrock:test",
        backend_id=backend.backend_id,
        expires_monotonic_ns=time.monotonic_ns() + 1_000_000_000,
        allowed_actions=frozenset({"keyboard", "button", "mouse"}),
        max_action_duration_ms=250,
        first_sequence=0,
    )
    backend.target_window_id = 42
    backend._host_monitor_binding = None
    backend._input_permitted = lambda: False
    backend._held_keys = set()
    backend._held_buttons = set()
    backend.probe_target = lambda: True  # type: ignore[method-assign]
    backend._ensure_input_focus = lambda: None  # type: ignore[method-assign]
    backend._park_pointer_in_game = lambda: None  # type: ignore[method-assign]
    backend._keycode = lambda _key: 25  # type: ignore[method-assign]

    with pytest.raises(MotorRejected, match="interlock"):
        backend.apply(action)

    assert fake_xtest.calls == []
    assert mouse_moves == []


def test_release_all_ignores_latched_positive_input_interlock() -> None:
    backend = object.__new__(IsolatedX11InputBackend)
    fake_xtest = _FakeXTest()
    sent_keys: list[bool] = []
    backend._display = SimpleNamespace(sync=lambda: None)
    backend._x = SimpleNamespace(ButtonRelease=5)
    backend._xtest = fake_xtest
    backend._input_permitted = lambda: False
    backend._held_keys = {"w"}
    backend._held_buttons = {"left"}
    backend.release_count = 0
    backend._ensure_input_focus = lambda: None  # type: ignore[method-assign]
    backend._park_pointer_in_game = lambda: None  # type: ignore[method-assign]
    backend._keycode = lambda _key: 25  # type: ignore[method-assign]
    backend._send_key = lambda _keycode, down: sent_keys.append(down)  # type: ignore[method-assign]

    backend.release_all()

    assert sent_keys and not any(sent_keys)
    assert fake_xtest.calls
    assert all(event_type == backend._x.ButtonRelease for _, event_type, _ in fake_xtest.calls)
    assert backend._held_keys == set()
    assert backend._held_buttons == set()
    assert backend.release_count == 1
