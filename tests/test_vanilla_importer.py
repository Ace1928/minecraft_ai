from __future__ import annotations

import json
from pathlib import Path

from minecraft_ai.knowledge import Edition, GameVersion
from minecraft_ai.knowledge.importers import import_java_datapack


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_import_java_recipe_and_advancement(tmp_path: Path) -> None:
    _write(
        tmp_path / "data" / "minecraft" / "recipe" / "crafting_table.json",
        {
            "type": "minecraft:crafting_shaped",
            "pattern": ["##", "##"],
            "key": {"#": {"tag": "minecraft:planks"}},
            "result": {"id": "minecraft:crafting_table", "count": 1},
        },
    )
    _write(
        tmp_path / "data" / "minecraft" / "advancement" / "story" / "mine_stone.json",
        {
            "parent": "minecraft:story/root",
            "criteria": {"get_stone": {"trigger": "minecraft:inventory_changed"}},
        },
    )

    version = GameVersion(edition=Edition.JAVA, version_id="1.test")
    graph = import_java_datapack(tmp_path, version)

    process = "process:recipe:minecraft:crafting_table"
    assert process in graph.nodes
    outputs = graph.outgoing(process)
    assert any(edge.target == "item:minecraft:crafting_table" for edge in outputs)

    requirements = graph.requirements(process)
    assert len(requirements) == 4
    assert all(edge.target == "tag:minecraft:planks" for edge in requirements)
    assert all(edge.provenance.version_key == version.key for edge in requirements)

    advancement = "advancement:minecraft:story/mine_stone"
    parent = "advancement:minecraft:story/root"
    assert advancement in graph.nodes
    assert parent in graph.prerequisite_closure(advancement)


def test_importer_accepts_legacy_recipes_directory(tmp_path: Path) -> None:
    _write(
        tmp_path / "data" / "minecraft" / "recipes" / "stick.json",
        {
            "type": "minecraft:crafting_shaped",
            "pattern": ["#", "#"],
            "key": {"#": {"item": "minecraft:oak_planks"}},
            "result": {"item": "minecraft:stick", "count": 4},
        },
    )
    version = GameVersion(edition=Edition.JAVA, version_id="legacy-test")
    graph = import_java_datapack(tmp_path, version)
    process = "process:recipe:minecraft:stick"
    ways = graph.ways_to_obtain("item:minecraft:stick")
    assert any(edge.source == process and edge.quantity == 4 for edge in ways)
