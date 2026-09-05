from __future__ import annotations

import pytest

from minecraft_ai.platforms.bedrock_session import _headless_compositor_environment


def test_headless_environment_removes_inherited_wayland_server_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An inherited single-client transport must not override the private socket."""
    monkeypatch.setenv("WAYLAND_SERVER_SOCKET", "37")
    monkeypatch.setenv("MINECRAFT_AI_BENIGN_SETTING", "preserved")

    environment = _headless_compositor_environment()

    # Do not expose unrelated environment values in assertion diagnostics.
    inherited_transport_present = "WAYLAND_SERVER_SOCKET" in environment
    assert not inherited_transport_present
    assert environment["MINECRAFT_AI_BENIGN_SETTING"] == "preserved"
