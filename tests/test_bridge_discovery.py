from __future__ import annotations

import json
from pathlib import Path

import pytest

from minecraft_ai.bridge import (
    BridgeDiscoveryError,
    discover_bridge_descriptors,
    select_bridge,
)


def _descriptor(
    game_dir: Path,
    *,
    instance_id: str,
    version: str = "26.2",
    port: int = 12345,
) -> Path:
    path = game_dir / ".minecraft-ai" / f"bridge-{instance_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "host": "127.0.0.1",
                "port": port,
                "token": "t" * 32,
                "instance_id": instance_id,
                "edition": "java",
                "version": version,
                "process_id": 1234,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_discovery_preserves_instance_version_metadata(tmp_path: Path) -> None:
    _descriptor(tmp_path, instance_id="instance-one")

    found = discover_bridge_descriptors(tmp_path)

    assert len(found) == 1
    assert found[0].endpoint.instance_id == "instance-one"
    assert found[0].edition == "java"
    assert found[0].version == "26.2"
    assert found[0].process_id == 1234


def test_select_bridge_refuses_ambiguous_instance(tmp_path: Path) -> None:
    _descriptor(tmp_path, instance_id="instance-one", port=12345)
    _descriptor(tmp_path, instance_id="instance-two", port=12346)

    with pytest.raises(BridgeDiscoveryError, match="multiple"):
        select_bridge(tmp_path, edition="java", version="26.2")

    selected = select_bridge(tmp_path, instance_id="instance-two")
    assert selected.endpoint.port == 12346


def test_invalid_or_partial_descriptor_is_ignored(tmp_path: Path) -> None:
    valid = _descriptor(tmp_path, instance_id="valid-instance")
    broken = valid.parent / "bridge-broken.json"
    broken.write_text('{"host":"127.0.0.1"}', encoding="utf-8")

    found = discover_bridge_descriptors(tmp_path)

    assert tuple(item.endpoint.instance_id for item in found) == ("valid-instance",)


def test_no_match_fails_explicitly(tmp_path: Path) -> None:
    _descriptor(tmp_path, instance_id="instance-one")
    with pytest.raises(BridgeDiscoveryError, match="no scoped Minecraft bridge"):
        select_bridge(tmp_path, instance_id="missing-instance")
