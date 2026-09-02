from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BedrockLinuxInstall:
    data_dir: Path
    wine_prefix: Path
    launcher_command: str | None
    source: str


@dataclass(frozen=True)
class BedrockLinuxInstance:
    pid: int
    executable: str
    wine_prefix: Path | None

    @property
    def instance_id(self) -> str:
        prefix = str(self.wine_prefix) if self.wine_prefix is not None else "unknown-prefix"
        return f"bedrock-wine:{self.pid}:{prefix}"


def _xdg_data_home() -> Path:
    raw = os.environ.get("XDG_DATA_HOME", "").strip()
    return Path(raw).expanduser() if raw else Path.home() / ".local" / "share"


def discover_bedrock_linux_install() -> BedrockLinuxInstall | None:
    """Discover a BedrockOnLinux-style WineGDK installation without importing it."""

    override = os.environ.get("BOL_HOME", "").strip()
    data_dir = (
        Path(override).expanduser()
        if override
        else _xdg_data_home() / "bedrock-on-linux"
    )
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
    )


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
                wine_prefix=install.wine_prefix if install is not None else None,
            )
        )
    return sorted(result, key=lambda item: item.pid)
