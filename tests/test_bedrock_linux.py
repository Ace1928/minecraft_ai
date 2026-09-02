from __future__ import annotations

import json
from pathlib import Path

import pytest

from minecraft_ai.platforms.bedrock_linux import _read_install_record
from minecraft_ai.platforms.bedrock_session import BedrockSession, bedrock_session_alive
from minecraft_ai.platforms.bedrock_x11 import (
    IsolationError,
    _crop_bgra,
    _wine_relative_motion_target,
    require_isolated_display,
)


def test_install_record_provides_exact_bedrock_build(tmp_path: Path) -> None:
    root = tmp_path / "games" / "release" / "1.21.100"
    root.mkdir(parents=True)
    (root / "Minecraft.Windows.exe").write_bytes(b"marker")
    (root / ".bedrock-on-linux-install.json").write_text(
        json.dumps({"edition": "release", "version": "1.21.100"}),
        encoding="utf-8",
    )
    build = _read_install_record(root)
    assert build is not None
    assert build.edition_id == "release"
    assert build.version == "1.21.100"
    assert build.game_root == root


def test_managed_directory_is_safe_version_fallback(tmp_path: Path) -> None:
    root = tmp_path / "games" / "preview" / "1.22.0.20"
    root.mkdir(parents=True)
    (root / "Minecraft.Windows.exe").write_bytes(b"marker")
    build = _read_install_record(root)
    assert build is not None
    assert build.edition_id == "preview"
    assert build.version == "1.22.0.20"


def test_isolated_backend_refuses_host_x_server() -> None:
    with pytest.raises(IsolationError):
        require_isolated_display(":0.0", host_display=":0")


def test_different_x_server_is_accepted() -> None:
    require_isolated_display(":71", host_display=":0")


def test_wine_client_crop_removes_window_decoration_without_reordering_pixels() -> None:
    pixels = bytes(range(4 * 4 * 3))
    cropped = _crop_bgra(pixels, source_width=4, rect=(1, 1, 2, 2))
    assert cropped == pixels[20:28] + pixels[36:44]


def test_wine_grab_bridge_preserves_yaw_and_inverts_bedrock_pitch() -> None:
    # VPT positive pitch means look down. The managed Wine client requires a
    # negative physical Y delta to produce that same camera motion.
    assert _wine_relative_motion_target(640, 317, 20, 10) == (660, 307)
    assert _wine_relative_motion_target(640, 317, -20, -10) == (620, 327)


def test_nested_session_is_not_alive_when_launcher_exited(tmp_path: Path, monkeypatch) -> None:
    session = BedrockSession(
        display=":71",
        host_display=":0",
        xserver_pid=100,
        launcher_pid=200,
        width=1280,
        height=720,
        created_ns=1,
        launcher_command=("bedrock-on-linux", "play"),
        mode="weston",
    )
    monkeypatch.setattr(
        "minecraft_ai.platforms.bedrock_session._pid_alive",
        lambda pid: pid == 100,
    )
    socket_path = tmp_path / "X71"
    socket_path.touch()
    monkeypatch.setattr(
        "minecraft_ai.platforms.bedrock_session._x_socket",
        lambda _display: socket_path,
    )

    assert not bedrock_session_alive(session)
