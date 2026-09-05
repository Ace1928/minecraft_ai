from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import pytest

from minecraft_ai.platforms import bedrock_session, bedrock_x11
from minecraft_ai.platforms.bedrock_x11 import IsolatedX11InputBackend, IsolationError
from minecraft_ai.safety import MotorAction, MotorLease


def _lease() -> MotorLease:
    return MotorLease(
        lease_id="isolation-test",
        session_id="session",
        target_instance="bedrock:test",
        backend_id=IsolatedX11InputBackend.backend_id,
        expires_monotonic_ns=time.monotonic_ns() + 4_000_000_000,
        allowed_actions=frozenset({"keyboard", "button", "mouse"}),
        max_action_duration_ms=250,
        first_sequence=0,
    )


def _backend(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    """Mock X delivery, preserving the backend's actual isolation delegation."""
    state = SimpleNamespace(
        session=SimpleNamespace(display=":72"),
        failure=None,
        loads=0,
        validations=[],
        events=[],
        syncs=0,
        focus_changes=[],
        positions=[],
        after_event=lambda _kind, _detail: None,
    )

    def load() -> Any:
        state.loads += 1
        return state.session

    def validate(session: Any) -> None:
        state.validations.append(session)
        if state.failure is not None:
            raise state.failure

    def emit(_display: Any, kind: int, detail: int = 0, **kwargs: int) -> None:
        state.events.append((kind, detail, kwargs))
        state.after_event(kind, detail)

    monkeypatch.setattr(bedrock_session.BedrockSession, "load", staticmethod(load))
    monkeypatch.setattr(bedrock_session, "require_autonomous_input_isolation", validate)
    backend = object.__new__(IsolatedX11InputBackend)
    backend.display_name = ":72.0"
    backend._targeted = False
    backend.target_window_id = 2
    backend._input_window_id = 3
    backend._host_monitor_binding = None
    backend._input_permitted = lambda: True
    backend._held_keys = set()
    backend._held_buttons = set()
    backend.release_count = 0
    backend._lease = _lease()
    backend._x = SimpleNamespace(
        KeyPress=2, KeyRelease=3, ButtonPress=4, ButtonRelease=5,
        MotionNotify=6, NONE=0, KeyPressMask=1, KeyReleaseMask=2,
        RevertToParent=2, CurrentTime=0,
    )
    backend._display = SimpleNamespace(
        sync=lambda: setattr(state, "syncs", state.syncs + 1),
    )
    backend._xtest = SimpleNamespace(fake_input=emit)
    backend._keycode = lambda key: {"w": 25, "a": 38}.get(key, 40)
    backend.probe_target = lambda: True
    backend._ensure_input_focus = lambda: None
    backend._park_pointer_in_game = lambda: True
    backend._position_pointer_in_game = lambda x, y: state.positions.append((x, y))
    return backend, state


def _reject(state: Any, reason: str) -> None:
    if reason == "display-mismatch":
        state.session = SimpleNamespace(display=":73")
    elif reason == "legacy":
        state.failure = IsolationError("legacy session has an unverified host input seat")
    else:
        state.failure = IsolationError("live compositor process identity was replaced")


def test_isolation_reloads_and_delegates_for_each_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, state = _backend(monkeypatch)
    backend._require_input_isolation()
    assert state.validations == [state.session]
    state.failure = IsolationError("process replaced after the first check")

    with pytest.raises(IsolationError, match="process replaced"):
        backend._require_input_isolation()

    assert state.loads == 2
    assert state.validations == [state.session, state.session]


@pytest.mark.parametrize("reason", ["display-mismatch", "legacy", "replaced-process"])
def test_constructor_rejects_unverified_session_before_opening_x(
    monkeypatch: pytest.MonkeyPatch, reason: str,
) -> None:
    _backend_unused, state = _backend(monkeypatch)
    _reject(state, reason)
    monkeypatch.setattr(
        bedrock_x11.importlib, "import_module",
        lambda _name: pytest.fail("X libraries must not be opened before isolation admission"),
    )

    with pytest.raises(IsolationError):
        IsolatedX11InputBackend(":72", host_display=":0", target_window_id=2)


@pytest.mark.parametrize("error", [OSError("missing"), ValueError("invalid"), TypeError("bad")])
def test_unreadable_session_becomes_attributable_isolation_rejection(
    monkeypatch: pytest.MonkeyPatch, error: Exception,
) -> None:
    backend, _state = _backend(monkeypatch)

    def fail_load() -> Any:
        raise error

    monkeypatch.setattr(bedrock_session.BedrockSession, "load", staticmethod(fail_load))
    with pytest.raises(IsolationError, match="could not be verified") as caught:
        backend._require_input_isolation()
    assert caught.value.__cause__ is error


@pytest.mark.parametrize("reason", ["display-mismatch", "legacy", "replaced-process"])
def test_unverified_session_cannot_bind_a_lease(
    monkeypatch: pytest.MonkeyPatch, reason: str,
) -> None:
    backend, state = _backend(monkeypatch)
    backend._lease = None
    _reject(state, reason)
    with pytest.raises(IsolationError):
        backend.bind_lease(_lease())
    assert backend._lease is None
    assert state.events == []


@pytest.mark.parametrize("reason", ["display-mismatch", "legacy", "replaced-process"])
@pytest.mark.parametrize("action", [
    MotorAction(sequence=1, keys_down=("w",)),
    MotorAction(sequence=1, buttons_down=("left",)),
    MotorAction(sequence=1, mouse_dx=8),
    MotorAction(sequence=1, cursor_x=0.25, cursor_y=0.5, camera_semantics="cursor"),
])
def test_no_positive_event_can_bypass_managed_isolation(
    monkeypatch: pytest.MonkeyPatch, reason: str, action: MotorAction,
) -> None:
    backend, state = _backend(monkeypatch)
    _reject(state, reason)
    with pytest.raises(IsolationError):
        backend.apply(action)
    assert state.events == []
    assert state.positions == []
    assert backend.held_keys == backend.held_buttons == frozenset()


def test_private_focus_change_requires_current_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, state = _backend(monkeypatch)
    state.failure = IsolationError("compositor was replaced")
    current = SimpleNamespace(id=4)
    parent = SimpleNamespace(
        id=3,
        set_input_focus=lambda *_args: state.focus_changes.append(3),
    )
    backend._display.get_input_focus = lambda: SimpleNamespace(focus=current)
    backend._display.create_resource_object = lambda *_args: parent
    monkeypatch.setattr(
        bedrock_x11, "_private_focus_routes_keyboard", lambda window, **_kwargs: window.id == 3,
    )
    monkeypatch.setattr(bedrock_x11, "_window_is_descendant_or_same", lambda *_args: True)
    with pytest.raises(IsolationError, match="cannot focus"):
        IsolatedX11InputBackend._ensure_input_focus(backend)
    assert state.focus_changes == []
    assert state.events == []


@pytest.mark.parametrize("reason", ["display-mismatch", "legacy", "replaced-process"])
def test_release_only_action_still_releases_without_changing_focus(
    monkeypatch: pytest.MonkeyPatch, reason: str,
) -> None:
    backend, state = _backend(monkeypatch)
    backend._held_keys = {"w"}
    backend._held_buttons = {"left"}
    backend._ensure_input_focus = lambda: pytest.fail("release must not restore focus")
    _reject(state, reason)

    backend.apply(MotorAction(sequence=1, keys_up=("w",), buttons_up=("left",)))

    assert [(kind, detail) for kind, detail, _kwargs in state.events] == [(3, 25), (5, 1)]
    assert backend.held_keys == backend.held_buttons == frozenset()
    assert state.loads == 0


def test_direct_key_press_rechecks_isolation_while_release_is_unconditional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, state = _backend(monkeypatch)
    state.failure = IsolationError("compositor was replaced")
    with pytest.raises(IsolationError):
        backend._send_key(25, down=True)
    backend._send_key(25, down=False)
    assert [(kind, detail) for kind, detail, _kwargs in state.events] == [(3, 25)]


def test_release_all_survives_failed_isolation_and_focus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, state = _backend(monkeypatch)
    state.failure = IsolationError("compositor was replaced")
    backend._held_keys = {"w"}
    backend._held_buttons = {"left"}
    backend._ensure_input_focus = backend._require_input_isolation

    backend.release_all()

    assert state.events
    assert all(kind in {3, 5} for kind, _detail, _kwargs in state.events)
    assert any(kind == 3 and detail == 25 for kind, detail, _kwargs in state.events)
    assert any(kind == 5 and detail == 1 for kind, detail, _kwargs in state.events)
    assert backend.held_keys == backend.held_buttons == frozenset()
    assert backend.release_count == 1
    assert state.syncs == 1


def _prepare_chat(backend: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    backend._tap_key = lambda _key: None
    backend._type_ascii = lambda _char: None
    backend._held_keys = {"a", "w"}
    backend._held_buttons = {"left"}
    monkeypatch.setattr(bedrock_x11.time, "sleep", lambda _seconds: None)


def test_chat_does_not_restore_holds_after_isolation_is_lost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, state = _backend(monkeypatch)
    _prepare_chat(backend, monkeypatch)

    def type_char(_char: str) -> None:
        state.failure = IsolationError("isolation lost during chat")

    backend._type_ascii = type_char
    backend.type_chat("x")

    assert all(kind in {3, 5} for kind, _detail, _kwargs in state.events)
    assert backend.held_keys == backend.held_buttons == frozenset()


@pytest.mark.parametrize("body_fails", [False, True])
@pytest.mark.parametrize("restore_kind", ["key", "button"])
def test_chat_partial_hold_restore_is_released_and_preserves_body_error(
    monkeypatch: pytest.MonkeyPatch, body_fails: bool, restore_kind: str,
) -> None:
    backend, state = _backend(monkeypatch)
    _prepare_chat(backend, monkeypatch)
    press_kind, release_kind, released_detail = 2, 3, 38
    if restore_kind == "button":
        backend._held_keys.clear()
        backend._held_buttons = {"left", "right"}
        press_kind, release_kind, released_detail = 4, 5, 1
    body_error = RuntimeError("original chat submission failed")

    def type_char(_char: str) -> None:
        if body_fails:
            raise body_error

    def after_event(kind: int, _detail: int) -> None:
        if kind == press_kind:
            state.failure = IsolationError("isolation lost after the first restored hold")

    backend._type_ascii = type_char
    state.after_event = after_event
    with pytest.raises(RuntimeError) as caught:
        backend.type_chat("x")

    if body_fails:
        assert caught.value is body_error
    else:
        assert isinstance(caught.value, IsolationError)
    press_indices = [
        i for i, (kind, _detail, _kwargs) in enumerate(state.events) if kind == press_kind
    ]
    assert len(press_indices) == 1
    after_press = state.events[press_indices[0] + 1:]
    assert any(
        kind == release_kind and detail == released_detail for kind, detail, _kwargs in after_press
    )
    assert not any(kind in {2, 4, 6} for kind, _detail, _kwargs in after_press)
    assert backend.held_keys == backend.held_buttons == frozenset()
    assert backend.release_count == 2


def test_chat_cleanup_sync_failure_does_not_replace_original_body_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, state = _backend(monkeypatch)
    _prepare_chat(backend, monkeypatch)
    body_error = RuntimeError("original chat submission failed")

    def type_char(_char: str) -> None:
        raise body_error

    def sync() -> None:
        state.syncs += 1
        if state.syncs >= 3:
            raise OSError("X connection lost during hold restoration")

    backend._type_ascii = type_char
    backend._display.sync = sync
    with pytest.raises(RuntimeError) as caught:
        backend.type_chat("x")

    assert caught.value is body_error
    assert backend.held_keys == backend.held_buttons == frozenset()
    assert backend.release_count >= 2


def test_explicit_host_debug_does_not_claim_managed_private_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, state = _backend(monkeypatch)
    backend._targeted = True
    state.failure = IsolationError("unverified private session")
    backend._require_input_isolation()
    assert state.loads == 0
    assert state.validations == []
