from __future__ import annotations

import json
from pathlib import Path

import pytest

from minecraft_ai.platforms.bedrock_linux import _read_install_record
from minecraft_ai.platforms.bedrock_x11 import IsolationError, require_isolated_display


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
