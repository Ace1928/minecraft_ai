from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from minecraft_ai.safety import MotorAction, MotorLease
from minecraft_ai.platforms.bedrock_x11 import _x11_keysym_name
from minecraft_ai.supervisor import Supervisor


@dataclass
class ChatBackend:
    backend_id: str = "chat-test"
    live_capable: bool = True
    messages: list[str] = field(default_factory=list)
    held_keys: set[str] = field(default_factory=set)
    held_buttons: set[str] = field(default_factory=set)
    release_count: int = 0
    lease: MotorLease | None = None
    fail_chat: bool = False

    def bind_lease(self, lease: MotorLease) -> None:
        self.lease = lease

    def clear_lease(self) -> None:
        self.lease = None

    def apply(self, action: MotorAction) -> None:
        assert self.lease is not None
        self.held_keys.update(action.keys_down)
        self.held_keys.difference_update(action.keys_up)

    def release_all(self) -> None:
        self.held_keys.clear()
        self.held_buttons.clear()
        self.release_count += 1

    def type_chat(self, text: str, *, input_permitted=lambda: True) -> None:
        if not input_permitted():
            raise RuntimeError("chat interlock blocked")
        if self.fail_chat:
            raise RuntimeError("backend failure")
        self.messages.append(text)


def _running_supervisor(backend: ChatBackend) -> tuple[Supervisor, str]:
    supervisor = Supervisor()
    supervisor.start()
    supervisor.replace_backend(backend)
    lease_info = supervisor.arm("bedrock:test")
    lease_id = str(lease_info["lease_id"])
    supervisor.activate()
    return supervisor, lease_id


def test_chat_requires_running_live_lease() -> None:
    backend = ChatBackend()
    supervisor, lease_id = _running_supervisor(backend)
    result = supervisor.send_chat(lease_id, "hello world")
    assert result == {"sent": True, "characters": 11}
    assert backend.messages == ["hello world"]


def test_chat_wrong_lease_fails_closed() -> None:
    backend = ChatBackend()
    supervisor, _ = _running_supervisor(backend)
    with pytest.raises(RuntimeError):
        supervisor.send_chat("wrong", "hello")
    assert supervisor.status()["state"] == "FAILSAFE"
    assert backend.lease is None


def test_chat_backend_fault_enters_failsafe() -> None:
    backend = ChatBackend(fail_chat=True)
    supervisor, lease_id = _running_supervisor(backend)
    with pytest.raises(RuntimeError):
        supervisor.send_chat(lease_id, "hello")
    assert supervisor.status()["state"] == "FAILSAFE"
    assert backend.lease is None


def test_chat_rejects_control_characters_without_backend_call() -> None:
    backend = ChatBackend()
    supervisor, lease_id = _running_supervisor(backend)
    with pytest.raises(ValueError):
        supervisor.send_chat(lease_id, "bad\nmessage")
    assert backend.messages == []
    assert supervisor.status()["state"] == "RUNNING"


def test_printable_chat_punctuation_maps_to_x11_keysyms() -> None:
    expected = {
        ".": "period",
        ",": "comma",
        "'": "apostrophe",
        "-": "minus",
        "=": "equal",
        "[": "bracketleft",
        "]": "bracketright",
        "\\": "backslash",
        ";": "semicolon",
        "`": "grave",
        "/": "slash",
    }
    assert {char: _x11_keysym_name(char) for char in expected} == expected
