from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .client import BridgeEndpoint


@dataclass(frozen=True)
class DiscoveredBridge:
    descriptor_path: Path
    endpoint: BridgeEndpoint
    edition: Literal["java", "bedrock"]
    version: str
    process_id: int | None


class BridgeDiscoveryError(RuntimeError):
    pass


def discover_bridge_descriptors(game_dir: str | Path) -> tuple[DiscoveredBridge, ...]:
    """Read scoped bridge descriptors without connecting or granting authority."""
    root = Path(game_dir).expanduser()
    bridge_dir = root / ".minecraft-ai"
    if not bridge_dir.is_dir():
        return ()

    discovered: list[DiscoveredBridge] = []
    for path in sorted(bridge_dir.glob("bridge-*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                continue
            edition_raw = raw.get("edition")
            if edition_raw not in {"java", "bedrock"}:
                continue
            version_raw = raw.get("version")
            if not isinstance(version_raw, str) or not version_raw:
                continue
            process_raw = raw.get("process_id")
            process_id = int(process_raw) if process_raw is not None else None
            if process_id is not None and process_id <= 0:
                continue
            endpoint = BridgeEndpoint(
                host=str(raw["host"]),
                port=int(raw["port"]),
                token=str(raw["token"]),
                instance_id=str(raw["instance_id"]),
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        discovered.append(
            DiscoveredBridge(
                descriptor_path=path,
                endpoint=endpoint,
                edition=edition_raw,
                version=version_raw,
                process_id=process_id,
            )
        )
    return tuple(discovered)


def select_bridge(
    game_dir: str | Path,
    *,
    instance_id: str | None = None,
    edition: Literal["java", "bedrock"] | None = None,
    version: str | None = None,
) -> DiscoveredBridge:
    """Select exactly one bridge, failing on ambiguity rather than guessing."""
    candidates = list(discover_bridge_descriptors(game_dir))
    if instance_id is not None:
        candidates = [item for item in candidates if item.endpoint.instance_id == instance_id]
    if edition is not None:
        candidates = [item for item in candidates if item.edition == edition]
    if version is not None:
        candidates = [item for item in candidates if item.version == version]

    if not candidates:
        raise BridgeDiscoveryError("no scoped Minecraft bridge matches the requested identity")
    if len(candidates) > 1:
        identities = ", ".join(item.endpoint.instance_id for item in candidates)
        raise BridgeDiscoveryError(
            "multiple scoped Minecraft bridges match; choose an explicit instance_id: " + identities
        )
    return candidates[0]
