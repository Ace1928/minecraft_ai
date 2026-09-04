from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from minecraft_ai.platforms.bedrock_x11 import IsolatedX11InputBackend


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
