from __future__ import annotations

import json
from pathlib import Path

from minecraft_ai.knowledge import Edition, GameVersion
from minecraft_ai.knowledge.importers import import_java_datapack
from minecraft_ai.knowledge.queries import acquisition_methods, tag_members


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_tag_members_expand_into_recipe_acquisition_choices(tmp_path: Path) -> None:
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
        tmp_path / "data" / "minecraft" / "tags" / "item" / "planks.json",
        {
            "replace": False,
            "values": ["minecraft:oak_planks", "minecraft:spruce_planks"],
        },
    )

    graph = import_java_datapack(
        tmp_path,
        GameVersion(edition=Edition.JAVA, version_id="tag-test"),
    )

    assert tag_members(graph, "tag:minecraft:planks") == (
        "item:minecraft:oak_planks",
        "item:minecraft:spruce_planks",
    )

    methods = acquisition_methods(graph, "item:minecraft:crafting_table")
    assert len(methods) == 1
    method = methods[0]
    assert method.process == "process:recipe:minecraft:crafting_table"
    assert len(method.requirements) == 4
    expected = (
        "item:minecraft:oak_planks",
        "item:minecraft:spruce_planks",
    )
    assert all(choice.alternatives == expected for choice in method.requirements)


def test_nested_tags_resolve_to_concrete_members(tmp_path: Path) -> None:
    _write(
        tmp_path / "data" / "minecraft" / "tags" / "item" / "logs.json",
        {
            "replace": False,
            "values": ["#minecraft:oak_logs", "minecraft:spruce_log"],
        },
    )
    _write(
        tmp_path / "data" / "minecraft" / "tags" / "item" / "oak_logs.json",
        {"replace": False, "values": ["minecraft:oak_log", "minecraft:oak_wood"]},
    )

    graph = import_java_datapack(
        tmp_path,
        GameVersion(edition=Edition.JAVA, version_id="nested-tag-test"),
    )

    assert tag_members(graph, "tag:minecraft:logs") == (
        "item:minecraft:oak_log",
        "item:minecraft:oak_wood",
        "item:minecraft:spruce_log",
    )
