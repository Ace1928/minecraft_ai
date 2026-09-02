from __future__ import annotations

import json
from pathlib import Path

from minecraft_ai.knowledge import Edition, GameVersion
from minecraft_ai.knowledge.importers import import_minecraft_data
from minecraft_ai.knowledge.queries import acquisition_methods


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_bedrock_importer_resolves_inherited_domains(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write(
        data / "dataPaths.json",
        {
            "bedrock": {
                "1.99.1": {
                    "items": "bedrock/base",
                    "blocks": "bedrock/base",
                    "recipes": "bedrock/recipes-v2",
                }
            }
        },
    )
    _write(
        data / "bedrock/base/items.json",
        [
            {"id": 1, "name": "oak_planks", "displayName": "Oak Planks", "stackSize": 64},
            {"id": 2, "name": "stick", "displayName": "Stick", "stackSize": 64},
        ],
    )
    _write(data / "bedrock/base/blocks.json", [{"id": 5, "name": "oak_planks"}])
    _write(
        data / "bedrock/recipes-v2/recipes.json",
        {
            "10": {
                "type": "crafting_table",
                "name": "minecraft:sticks",
                "ingredients": [{"name": "oak_planks", "count": 2}],
                "output": [{"name": "stick", "count": 4}],
            }
        },
    )

    version = GameVersion(edition=Edition.BEDROCK, version_id="1.99.1")
    graph = import_minecraft_data(tmp_path, version)

    assert "item:minecraft:oak_planks" in graph.nodes
    assert "block:minecraft:oak_planks" in graph.nodes
    methods = acquisition_methods(graph, "item:minecraft:stick")
    assert len(methods) == 1
    assert methods[0].output_quantity == 4
    assert methods[0].requirements[0].alternatives == ("item:minecraft:oak_planks",)
    assert methods[0].requirements[0].quantity == 2
    assert graph.validate() == []


def test_importer_refuses_unmapped_exact_version(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write(data / "dataPaths.json", {"bedrock": {"1.0": {}}})
    version = GameVersion(edition=Edition.BEDROCK, version_id="2.0")
    try:
        import_minecraft_data(tmp_path, version)
    except RuntimeError as exc:
        assert "does not map exact" in str(exc)
    else:
        raise AssertionError("unmapped versions must fail loudly")
