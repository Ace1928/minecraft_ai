from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BedrockBuild:
    edition_id: str
    version: str
    game_root: Path
    source: str


@dataclass(frozen=True)
class BedrockLinuxInstall:
    data_dir: Path
    wine_prefix: Path
    launcher_command: str | None
    source: str
    selected_build: BedrockBuild | None = None


@dataclass(frozen=True)
class BedrockLinuxInstance:
    pid: int
    executable: str
    wine_prefix: Path | None
    build: BedrockBuild | None = None

    @property
    def instance_id(self) -> str:
        prefix = str(self.wine_prefix) if self.wine_prefix is not None else "unknown-prefix"
        build = self.build.version if self.build is not None else "unknown-version"
        return f"bedrock-wine:{self.pid}:{build}:{prefix}"


def _xdg_data_home() -> Path:
    raw = os.environ.get("XDG_DATA_HOME", "").strip()
    return Path(raw).expanduser() if raw else Path.home() / ".local" / "share"


def _read_install_record(game_root: Path) -> BedrockBuild | None:
    metadata = game_root / ".bedrock-on-linux-install.json"
    try:
        raw = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        raw = None
    if isinstance(raw, dict):
        version = str(raw.get("version", "")).strip()
        edition = str(raw.get("edition", "release")).strip() or "release"
        if version:
            return BedrockBuild(
                edition_id=edition,
                version=version,
                game_root=game_root,
                source="bedrock-on-linux-install-record",
            )

    # Managed BedrockOnLinux layout is games/<edition>/<version>. This is a
    # fallback for installs predating the metadata record.
    parts = game_root.parts
    if len(parts) >= 2 and (game_root / "Minecraft.Windows.exe").exists():
        return BedrockBuild(
            edition_id=parts[-2],
            version=parts[-1],
            game_root=game_root,
            source="managed-directory-layout",
        )
    return None


def _selected_build(data_dir: Path) -> BedrockBuild | None:
    content = data_dir / "content"
    try:
        resolved = content.resolve(strict=True)
    except OSError:
        return None
    root = resolved if resolved.is_dir() else resolved.parent
    return _read_install_record(root)


def discover_bedrock_linux_install() -> BedrockLinuxInstall | None:
    """Discover a BedrockOnLinux-style WineGDK installation without importing it."""

    override = os.environ.get("BOL_HOME", "").strip()
    data_dir = Path(override).expanduser() if override else _xdg_data_home() / "bedrock-on-linux"
    prefix_override = os.environ.get("BOL_WINEPREFIX", "").strip()
    wine_prefix = (
        Path(prefix_override).expanduser()
        if prefix_override
        else data_dir / "compatdata" / "pfx"
    )

    launcher: str | None = None
    for candidate in (
        Path.home() / ".local" / "bin" / "bedrock-on-linux",
        Path("/usr/local/bin/bedrock-on-linux"),
        Path("/usr/bin/bedrock-on-linux"),
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            launcher = str(candidate)
            break

    if not data_dir.exists() and launcher is None:
        return None
    return BedrockLinuxInstall(
        data_dir=data_dir,
        wine_prefix=wine_prefix,
        launcher_command=launcher,
        source="bedrock-on-linux",
        selected_build=_selected_build(data_dir),
    )


def _process_prefix(pid_dir: Path) -> Path | None:
    try:
        fields = (pid_dir / "environ").read_bytes().split(b"\0")
    except OSError:
        return None
    for field in fields:
        if field.startswith(b"WINEPREFIX="):
            value = os.fsdecode(field.split(b"=", 1)[1]).strip()
            return Path(value) if value else None
    return None


def find_bedrock_linux_instances() -> list[BedrockLinuxInstance]:
    """Find running Windows Bedrock processes hosted by Wine/Proton via /proc."""

    proc = Path("/proc")
    if not proc.is_dir():
        return []
    install = discover_bedrock_linux_install()
    result: list[BedrockLinuxInstance] = []
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ")
        except OSError:
            continue
        if b"Minecraft.Windows.exe" not in cmdline:
            continue
        try:
            executable = os.fsdecode((entry / "exe").resolve())
        except OSError:
            executable = "Minecraft.Windows.exe"
        result.append(
            BedrockLinuxInstance(
                pid=int(entry.name),
                executable=executable,
                wine_prefix=_process_prefix(entry)
                or (install.wine_prefix if install is not None else None),
                build=install.selected_build if install is not None else None,
            )
        )
    return sorted(result, key=lambda item: item.pid)
