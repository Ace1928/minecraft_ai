from __future__ import annotations

import time
import struct
from types import SimpleNamespace
from typing import Any

import pytest

from minecraft_ai.mining_control import MiningLeaseGuard
from minecraft_ai.motor import MotorIntent
from minecraft_ai.perception import PerceptionBlackboard
from minecraft_ai.platforms.bedrock_x11 import IsolatedX11InputBackend
from minecraft_ai.safety import InputRouteUnavailable, MotorAction, MotorLease, MotorRejected


class _FakeXTest:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, int, int]] = []
        self.motions: list[tuple[int, int]] = []

    def fake_input(self, display: Any, event_type: int, keycode: int = 0, **kwargs: int) -> None:
        self.calls.append((display, event_type, kwargs.get("detail", keycode)))
        if event_type == 6:
            self.motions.append((kwargs["x"], kwargs["y"]))


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
    backend._park_pointer_in_game = lambda: True  # type: ignore[method-assign]
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

    def park_then_latch() -> bool:
        nonlocal permitted, pointer_parks
        pointer_parks += 1
        if pointer_parks == 1:
            permitted = False
        return True

    backend._targeted = False
    backend._display = SimpleNamespace(sync=lambda: None)
    backend._x = SimpleNamespace(
        KeyPress=2,
        KeyRelease=3,
        ButtonPress=4,
        ButtonRelease=5,
    )
    backend._xtest = fake_xtest
    fake_xtest.motions = mouse_moves
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
    assert pointer_parks == 1
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
    fake_xtest.motions = mouse_moves
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
    backend._park_pointer_in_game = lambda: True  # type: ignore[method-assign]
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
    backend._park_pointer_in_game = lambda: False  # type: ignore[method-assign]
    backend._keycode = lambda _key: 25  # type: ignore[method-assign]
    backend._send_key = lambda _keycode, down: sent_keys.append(down)  # type: ignore[method-assign]

    backend.release_all()

    assert sent_keys and not any(sent_keys)
    assert fake_xtest.calls
    assert all(event_type == backend._x.ButtonRelease for _, event_type, _ in fake_xtest.calls)
    assert backend._held_keys == set()
    assert backend._held_buttons == set()
    assert backend.release_count == 1


def _pointer_parking_backend() -> tuple[IsolatedX11InputBackend, dict[str, Any]]:
    state: dict[str, Any] = {
        "point": (1012, 748), "warps": [], "relative": [], "keys": [],
        "query_failure": False,
    }

    class Window:
        def __init__(self, identity: int, rect: tuple[int, int, int, int]) -> None:
            self.id = identity
            self.rect = rect
            self.children: list[Window] = []
            self.viewable = True
            self.geometry_failure = False

        def get_geometry(self) -> Any:
            if self.geometry_failure:
                raise RuntimeError("unavailable geometry")
            return SimpleNamespace(width=self.rect[2], height=self.rect[3])

        def get_attributes(self) -> Any:
            return SimpleNamespace(map_state=2 if self.viewable else 0, win_class=1)

        def query_tree(self) -> Any:
            return SimpleNamespace(children=self.children)

        def query_pointer(self) -> Any:
            if state["query_failure"]:
                raise RuntimeError("unavailable pointer")
            path = state["hit_path"]
            next_index = path.index(self) + 1 if self in path else len(path)
            child = path[next_index] if next_index < len(path) else 0
            return SimpleNamespace(
                same_screen=True, root_x=state["point"][0], root_y=state["point"][1],
                child=child,
            )

    root = Window(1, (0, 0, 1920, 1080))
    desktop = Window(2, (0, 0, 1920, 1080))
    wrapper = Window(3, (-4, -4, 1928, 1088))
    client = Window(4, (0, 26, 1920, 1054))
    overlay = Window(5, (0, 26, 1920, 1054))
    root.children = [desktop]
    desktop.children = [wrapper, overlay]
    wrapper.children = [client]
    state.update(root=root, desktop=desktop, wrapper=wrapper, client=client, overlay=overlay)
    state["hit_path"] = [root, desktop, wrapper, client]
    windows = {w.id: w for w in (root, desktop, wrapper, client, overlay)}

    def warp(x: int, y: int) -> None:
        state["warps"].append((x, y))
        state["point"] = (x, y)

    root.warp_pointer = warp  # type: ignore[attr-defined]
    root.translate_coords = lambda window, x, y: SimpleNamespace(  # type: ignore[attr-defined]
        x=window.rect[0] + x, y=window.rect[1] + y,
    )
    backend = object.__new__(IsolatedX11InputBackend)
    backend._display = SimpleNamespace(
        screen=lambda: SimpleNamespace(root=root),
        create_resource_object=lambda _kind, identity: windows[identity],
        sync=lambda: None,
    )
    backend._targeted = False
    backend.target_window_id = desktop.id
    backend._input_window_id = wrapper.id
    backend._host_monitor_binding = None
    backend._input_permitted = lambda: True
    backend._held_keys = set()
    backend._held_buttons = set()
    backend.release_count = 0
    backend._x = SimpleNamespace(
        KeyPress=2, KeyRelease=3, ButtonPress=4, ButtonRelease=5, MotionNotify=6, NONE=0,
    )
    backend._xtest = _FakeXTest()
    state["relative"] = backend._xtest.motions
    backend._lease = MotorLease(
        lease_id="lease", session_id="session", target_instance="bedrock:test",
        backend_id=backend.backend_id,
        expires_monotonic_ns=time.monotonic_ns() + 1_000_000_000,
        allowed_actions=frozenset({"keyboard", "button", "mouse"}),
        max_action_duration_ms=250, first_sequence=0,
    )
    backend._ensure_input_focus = lambda: None  # type: ignore[method-assign]
    backend._keycode = lambda _key: 25  # type: ignore[method-assign]
    return backend, state


def test_pointer_already_in_exact_game_client_does_not_warp() -> None:
    backend, state = _pointer_parking_backend()
    assert backend._park_pointer_in_game()
    assert state["warps"] == []


def test_in_client_recentering_between_pointer_queries_does_not_warp_or_reject() -> None:
    backend, state = _pointer_parking_backend()
    path = state["hit_path"]
    for index, window in enumerate(path):
        window.query_pointer = lambda index=index: SimpleNamespace(
            same_screen=True,
            root_x=1012 if index == 0 else 960,
            root_y=748 if index == 0 else 553,
            child=path[index + 1] if index + 1 < len(path) else 0,
        )

    backend.apply(MotorAction(sequence=0, mouse_dx=7, mouse_dy=-9))

    assert state["relative"] == [(7, -9)]
    assert state["warps"] == []


@pytest.mark.parametrize("uncertainty", ["outside", "other_screen", "wrong_subtree"])
def test_pointer_change_during_chain_does_not_preserve_stale_client_routing(
    uncertainty: str,
) -> None:
    backend, state = _pointer_parking_backend()
    desktop = state["desktop"]
    desktop.query_pointer = lambda: SimpleNamespace(
        same_screen=uncertainty != "other_screen",
        root_x=1920 if uncertainty == "outside" else 960,
        root_y=553,
        child=state["overlay"] if uncertainty == "wrong_subtree" else state["wrapper"],
    )

    with pytest.raises(InputRouteUnavailable, match="pointer routing could not be verified"):
        backend.apply(MotorAction(sequence=0, mouse_dx=7, mouse_dy=-9))

    assert state["relative"] == []
    assert state["warps"] == []


def test_pointer_outside_game_cannot_authorize_implicit_repositioning() -> None:
    backend, state = _pointer_parking_backend()
    state["point"] = (10, 5)
    assert not backend._park_pointer_in_game()
    assert state["warps"] == []


@pytest.mark.parametrize("hit", ["overlay", "desktop", "wrapper"])
def test_pointer_bounds_without_exact_game_hit_do_not_authorize_input(hit: str) -> None:
    backend, state = _pointer_parking_backend()
    path = [state["root"], state["desktop"]]
    if hit != "desktop":
        path.append(state[hit])
    state["hit_path"] = path
    assert not backend._park_pointer_in_game()
    assert state["warps"] == []


def test_pointer_hit_cycle_is_bounded_and_does_not_authorize_skip() -> None:
    backend, state = _pointer_parking_backend()
    state["root"].query_pointer = lambda: SimpleNamespace(
        same_screen=True, root_x=1012, root_y=748, child=state["root"],
    )
    assert not backend._park_pointer_in_game()
    assert state["warps"] == []


def test_unknown_pointer_position_never_falls_back_to_absolute_warp() -> None:
    backend, state = _pointer_parking_backend()
    state["query_failure"] = True
    assert not backend._park_pointer_in_game()
    assert state["warps"] == []


@pytest.mark.parametrize("failure", ["pointer", "geometry", "unmapped", "ambiguous"])
def test_pointer_inspection_failures_never_block_release_sweep(failure: str) -> None:
    backend, state = _pointer_parking_backend()
    if failure == "pointer":
        state["query_failure"] = True
    elif failure == "geometry":
        state["client"].geometry_failure = True
    elif failure == "unmapped":
        state["client"].viewable = False
    else:
        state["wrapper"].children.append(state["overlay"])
    backend._input_permitted = lambda: False
    backend._held_keys = {"w"}
    backend._held_buttons = {"left"}

    backend.release_all()

    releases = [(event, detail) for _display, event, detail in backend._xtest.calls]
    assert (backend._x.KeyRelease, 25) in releases
    assert {detail for event, detail in releases if event == backend._x.ButtonRelease} == {1, 2, 3}
    assert all(event in {backend._x.KeyRelease, backend._x.ButtonRelease} for event, _ in releases)
    assert backend.held_keys == backend.held_buttons == frozenset()
    assert state["warps"] == []


def test_relative_motion_remains_exact_without_extra_absolute_warp() -> None:
    backend, state = _pointer_parking_backend()
    backend.apply(MotorAction(sequence=0, mouse_dx=7, mouse_dy=-9))
    assert state["relative"] == [(7, -9)]
    assert state["warps"] == []


@pytest.mark.parametrize("positive", ["motion", "button"])
@pytest.mark.parametrize("uncertainty", ["geometry", "ambiguous", "pointer", "overlay"])
def test_unverified_pointer_routing_blocks_positive_input_but_keeps_releases(
    positive: str, uncertainty: str,
) -> None:
    backend, state = _pointer_parking_backend()
    if uncertainty == "geometry":
        state["client"].geometry_failure = True
    elif uncertainty == "ambiguous":
        state["wrapper"].children.append(state["overlay"])
    elif uncertainty == "pointer":
        state["query_failure"] = True
    else:
        state["hit_path"] = [state["root"], state["desktop"], state["overlay"]]
    backend._held_keys = {"a"}
    backend._held_buttons = {"right"}
    action = MotorAction(
        sequence=0, keys_up=("a",), buttons_up=("right",),
        mouse_dx=7 if positive == "motion" else 0,
        buttons_down=("left",) if positive == "button" else (),
    )

    with pytest.raises(InputRouteUnavailable, match="pointer routing could not be verified"):
        backend.apply(action)

    assert state["relative"] == []
    events = [event for _display, event, _detail in backend._xtest.calls]
    assert backend._x.KeyRelease in events
    assert backend._x.ButtonRelease in events
    assert backend._x.KeyPress not in events
    assert backend._x.ButtonPress not in events
    assert backend.held_keys == backend.held_buttons == frozenset()


@pytest.mark.parametrize("positive", ["motion", "button"])
def test_outside_pointer_rejects_positive_input_without_warp(positive: str) -> None:
    backend, state = _pointer_parking_backend()
    state["point"] = (10, 5)
    with pytest.raises(InputRouteUnavailable, match="pointer routing could not be verified"):
        backend.apply(MotorAction(
            sequence=0, mouse_dx=7 if positive == "motion" else 0,
            buttons_down=("left",) if positive == "button" else (),
        ))
    assert state["warps"] == []
    assert state["relative"] == []
    assert all(event != backend._x.ButtonPress for _, event, _ in backend._xtest.calls)


@pytest.mark.parametrize("dx,dy", [(7, -9), (-4096, 4096), (4096, -4096)])
def test_relative_wire_uses_same_connection_as_releases_and_presses(dx: int, dy: int) -> None:
    # Exercise the installed optional Xlib serializer, never a real Display or
    # X server. Other routing/interlock tests use the platform-independent fake.
    xtest = pytest.importorskip("Xlib.ext.xtest")
    x = pytest.importorskip("Xlib.X")
    wire: list[bytes | str] = []

    class Connection:
        def get_extension_major(self, name: str) -> int:
            assert name == "XTEST"
            return 200  # Synthetic extension opcode; no server query.

        def send_request(self, request: Any, _want_error: bool) -> None:
            wire.append(request._binary)

    backend, _state = _pointer_parking_backend()
    backend._display = SimpleNamespace(display=Connection(), sync=lambda: wire.append("sync"))
    backend._xtest = xtest
    backend._x = x
    backend.probe_target = lambda: True  # type: ignore[method-assign]
    backend._park_pointer_in_game = lambda: True  # type: ignore[method-assign]
    backend._keycode = lambda key: {"a": 38, "w": 25}[key]  # type: ignore[method-assign]

    backend.apply(MotorAction(
        sequence=0, keys_up=("a",), buttons_up=("right",),
        mouse_dx=dx, mouse_dy=dy, keys_down=("w",), buttons_down=("left",),
    ))

    assert wire[-1] == "sync"
    requests = [packet for packet in wire if isinstance(packet, bytes)]
    assert len(wire) == len(requests) + 1
    assert [(packet[4], packet[5]) for packet in requests] == [
        (x.KeyRelease, 38), (x.ButtonRelease, 3), (x.MotionNotify, 1),
        (x.KeyPress, 25), (x.ButtonPress, 1),
    ]
    motion = requests[2]
    assert len(motion) == 36
    assert motion[:2] == bytes((200, 2))
    assert struct.unpack("=II", motion[8:16]) == (x.CurrentTime, x.NONE)
    assert struct.unpack("=hh", motion[24:28]) == (dx, dy)


@pytest.mark.parametrize("pause_after_enter", [False, True])
def test_chat_transition_waits_follow_same_connection_flushes(
    pause_after_enter: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, _state = _pointer_parking_backend()
    backend._held_keys = {"w"}
    keycodes = {"t": 28, "a": 38, "enter": 36, "escape": 9, "w": 25}
    backend._keycode = lambda key: keycodes.get(key, 24)  # type: ignore[method-assign]
    queued: list[tuple[int, int]] = []
    delivered: list[tuple[int, int]] = []
    waited_after: list[tuple[int, int]] = []
    permitted = True

    def send(_display: Any, kind: int, key: int) -> None:
        queued.append((kind, key))

    def sync() -> None:
        delivered.extend(queued)
        queued.clear()

    def sleep(seconds: float) -> None:
        nonlocal permitted
        assert seconds == 0.04
        assert not queued, "a UI delay must start after its events are flushed"
        waited_after.append(delivered[-1])
        if pause_after_enter and len(waited_after) == 2:
            permitted = False

    backend._xtest = SimpleNamespace(fake_input=send)
    backend._display.sync = sync
    monkeypatch.setattr("minecraft_ai.platforms.bedrock_x11.time.sleep", sleep)
    if pause_after_enter:
        with pytest.raises(RuntimeError, match="chat input interlock"):
            backend.type_chat("a", input_permitted=lambda: permitted)
    else:
        backend.type_chat("a", input_permitted=lambda: permitted)

    assert waited_after == [
        (backend._x.KeyRelease, keycodes[key])
        for key in (["t", "enter"] if pause_after_enter else ["t", "enter", "escape"])
    ]
    positive = [key for kind, key in delivered if kind == backend._x.KeyPress]
    expected = ["t", "a", "enter"]
    if not pause_after_enter:
        expected.extend(("escape", "escape", "w"))
    assert positive == [keycodes[key] for key in expected]
    assert backend.held_keys == (frozenset() if pause_after_enter else frozenset({"w"}))
    assert not queued


def test_host_debug_parking_behavior_is_not_changed_by_private_routing_gate() -> None:
    backend, state = _pointer_parking_backend()
    backend._targeted = True
    state["wrapper"].geometry_failure = True
    backend.apply(MotorAction(sequence=0, mouse_dx=7, mouse_dy=-9))
    assert state["relative"] == [(7, -9)]
    assert state["warps"] == []


@pytest.mark.parametrize("position", [(10, 10), (960, 553)])
@pytest.mark.parametrize("query_failure", [False, True])
def test_button_release_and_release_all_never_move_pointer(
    position: tuple[int, int], query_failure: bool,
) -> None:
    backend, state = _pointer_parking_backend()
    state["point"] = position
    state["query_failure"] = query_failure
    backend._input_permitted = lambda: False
    backend._held_buttons = {"left"}
    backend.apply(MotorAction(sequence=0, buttons_up=("left",)))
    backend.release_all()
    assert state["warps"] == []
    assert backend.held_buttons == frozenset()
    assert sum(event == backend._x.ButtonRelease for _d, event, _b in backend._xtest.calls) == 4


def test_gui_target_remains_explicit_and_release_does_not_recenter_it() -> None:
    backend, state = _pointer_parking_backend()
    targets = []

    def position(x: float, y: float) -> None:
        targets.append((x, y))
        state["point"] = (480, 289)

    backend._position_pointer_in_game = position  # type: ignore[method-assign]
    backend.apply(MotorAction(sequence=0, cursor_x=0.25, cursor_y=0.25,
                              camera_semantics="cursor", buttons_down=("left",)))
    backend.apply(MotorAction(sequence=1, buttons_up=("left",)))
    assert targets == [(0.25, 0.25)]
    assert state["point"] == (480, 289)
    assert state["warps"] == []
