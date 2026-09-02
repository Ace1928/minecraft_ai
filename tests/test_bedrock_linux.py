from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from minecraft_ai.platforms.bedrock_linux import _read_install_record
from minecraft_ai.platforms.bedrock_session import (
    DEFAULT_BEDROCK_HEIGHT,
    DEFAULT_BEDROCK_WIDTH,
    BedrockSession,
    _weston_command,
    bedrock_session_alive,
    launch_isolated_bedrock_session,
)
from minecraft_ai.platforms.bedrock_x11 import (
    IsolationError,
    _crop_bgra,
    _x11_keysym_name,
    _wine_relative_motion_delta,
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


def test_managed_session_defaults_preserve_full_hd_hud_surface() -> None:
    signature = inspect.signature(launch_isolated_bedrock_session)

    assert DEFAULT_BEDROCK_WIDTH == 1920
    assert DEFAULT_BEDROCK_HEIGHT == 1080
    assert signature.parameters["width"].default == DEFAULT_BEDROCK_WIDTH
    assert signature.parameters["height"].default == DEFAULT_BEDROCK_HEIGHT
    assert signature.parameters["fullscreen"].default is True


def test_weston_defaults_to_host_fullscreen_for_complete_bedrock_hud(
    tmp_path: Path,
) -> None:
    command = _weston_command(
        weston="/usr/bin/weston",
        wayland_socket="minecraft-ai-test",
        width=1920,
        height=1080,
        fullscreen=True,
        compositor_log=tmp_path / "weston.log",
    )

    assert "--fullscreen" in command
    assert "--width=1920" in command
    assert "--height=1080" in command


def test_weston_windowed_escape_hatch_is_explicit(tmp_path: Path) -> None:
    command = _weston_command(
        weston="/usr/bin/weston",
        wayland_socket="minecraft-ai-test",
        width=1280,
        height=720,
        fullscreen=False,
        compositor_log=tmp_path / "weston.log",
    )

    assert "--fullscreen" not in command


def test_isolated_backend_refuses_host_x_server() -> None:
    with pytest.raises(IsolationError):
        require_isolated_display(":0.0", host_display=":0")


def test_different_x_server_is_accepted() -> None:
    require_isolated_display(":71", host_display=":0")


def test_wine_client_crop_removes_window_decoration_without_reordering_pixels() -> None:
    pixels = bytes(range(4 * 4 * 3))
    cropped = _crop_bgra(pixels, source_width=4, rect=(1, 1, 2, 2))
    assert cropped == pixels[20:28] + pixels[36:44]


def test_wine_grab_bridge_preserves_relative_vpt_camera_deltas() -> None:
    # XTEST uses relative motion when root is X.NONE. VPT and Bedrock both use
    # positive pitch-down and positive yaw-right, so no absolute pointer
    # coordinates or sign inversion belongs in the adapter.
    assert _wine_relative_motion_delta(20, 10) == (20, 10)
    assert _wine_relative_motion_delta(-20, -10) == (-20, -10)


def test_bedrock_menu_navigation_uses_x11_arrow_keysyms() -> None:
    assert _x11_keysym_name("up") == "Up"
    assert _x11_keysym_name("down") == "Down"
    assert _x11_keysym_name("left") == "Left"
    assert _x11_keysym_name("right") == "Right"


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
