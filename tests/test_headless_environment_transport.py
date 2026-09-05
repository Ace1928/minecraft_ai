from __future__ import annotations

import pytest

from minecraft_ai.platforms.bedrock_session import _headless_compositor_environment


@pytest.mark.parametrize("transport", ["WAYLAND_SERVER_SOCKET", "LIBEI_SOCKET"])
def test_headless_environment_removes_inherited_input_transport(
    monkeypatch: pytest.MonkeyPatch, transport: str,
) -> None:
    """An inherited single-client transport must not override the private socket."""
    monkeypatch.setenv(transport, "37")
    monkeypatch.setenv("MINECRAFT_AI_BENIGN_SETTING", "preserved")

    environment = _headless_compositor_environment()

    # Do not expose unrelated environment values in assertion diagnostics.
    inherited_transport_present = transport in environment
    assert not inherited_transport_present
    assert environment["MINECRAFT_AI_BENIGN_SETTING"] == "preserved"
