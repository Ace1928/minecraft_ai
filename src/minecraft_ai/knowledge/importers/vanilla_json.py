from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..model import (
    EdgeKind,
    GameVersion,
    KnowledgeEdge,
    KnowledgeGraph,
    KnowledgeNode,
    NodeKind,
    Provenance,
)


class VanillaDataError(RuntimeError):
    pass


def import_java_datapack(root: str | Path, version: GameVersion) -> KnowledgeGraph:
    """Import exact-version Java generated/data-pack JSON into a typed graph.

    This first importer intentionally covers recipes and advancements only. Loot,
    tags, registries, trades and reports are separate import passes so source
    provenance stays explicit and each domain can be validated independently.
    """
    if version.edition.value != "java":
        raise VanillaDataError("Java data-pack importer requires a Java GameVersion")
    root_path = Path(root)
    data_dir = root_path / "data"
    if not data_dir.exists():
        raise VanillaDataError(f"missing data directory: {data_dir}")

    graph = KnowledgeGraph(version)
    _import_recipes(data_dir, graph)
    _import_advancements(data_dir, graph)
    return graph


def _import_recipes(data_dir: Path, graph: KnowledgeGraph) -> None:
    for path in sorted(data_dir.glob("*/recipe/**/*.json")) + sorted(
        data_dir.glob("*/recipes/**/*.json")
    ):
        payload = _load_json(path)
        recipe_id = _resource_id(path, "recipe")
        if recipe_id is None:
            recipe_id = _resource_id(path, "recipes")
        if recipe_id is None:
            continue
        process_id = f"process:recipe:{recipe_id}"
        graph.add_node(
            KnowledgeNode(
                node_id=process_id,
                kind=NodeKind.PROCESS,
                name=f"recipe {recipe_id}",
                metadata={"recipe_type": str(payload.get("type", "unknown"))},
            )
        )
        provenance = _provenance(path, graph.version)
        for output_id, output_count in _recipe_outputs(payload):
            item_node = _item_node_id(output_id)
            _ensure_item(graph, item_node, output_id)
            graph.add_edge(
                KnowledgeEdge(
                    edge_id=f"{process_id}:produces:{item_node}",
                    source=process_id,
                    target=item_node,
                    kind=_recipe_output_kind(str(payload.get("type", ""))),
                    quantity=output_count,
                    provenance=provenance,
                )
            )
        for index, alternatives in enumerate(_recipe_ingredients(payload)):
            group = f"{process_id}:ingredient:{index}"
            for alternative_index, ingredient_id in enumerate(alternatives):
                ingredient_node = _item_node_id(ingredient_id)
                _ensure_item(graph, ingredient_node, ingredient_id)
                graph.add_edge(
                    KnowledgeEdge(
                        edge_id=f"{process_id}:requires:{index}:{alternative_index}:{ingredient_node}",
                        source=process_id,
                        target=ingredient_node,
                        kind=(
                            EdgeKind.ALTERNATIVE_REQUIRES
                            if len(alternatives) > 1
                            else EdgeKind.REQUIRES
                        ),
                        alternative_group=group if len(alternatives) > 1 else None,
                        provenance=provenance,
                    )
                )


def _import_advancements(data_dir: Path, graph: KnowledgeGraph) -> None:
    paths = sorted(data_dir.glob("*/advancement/**/*.json")) + sorted(
        data_dir.glob("*/advancements/**/*.json")
    )
    for path in paths:
        payload = _load_json(path)
        advancement_id = _resource_id(path, "advancement")
        if advancement_id is None:
            advancement_id = _resource_id(path, "advancements")
        if advancement_id is None:
            continue
        node_id = f"advancement:{advancement_id}"
        graph.add_node(
            KnowledgeNode(
                node_id=node_id,
                kind=NodeKind.ADVANCEMENT,
                name=advancement_id,
            )
        )
        parent = payload.get("parent")
        if isinstance(parent, str) and parent:
            parent_id = f"advancement:{parent}"
            if parent_id not in graph.nodes:
                graph.add_node(
                    KnowledgeNode(
                        node_id=parent_id,
                        kind=NodeKind.ADVANCEMENT,
                        name=parent,
                    )
                )
            graph.add_edge(
                KnowledgeEdge(
                    edge_id=f"{node_id}:parent:{parent_id}",
                    source=node_id,
                    target=parent_id,
                    kind=EdgeKind.ADVANCEMENT_REQUIRES,
                    provenance=_provenance(path, graph.version),
                )
            )


def _recipe_output_kind(recipe_type: str) -> EdgeKind:
    lowered = recipe_type.lower()
    if (
        "smelting" in lowered
        or "blasting" in lowered
        or "smoking" in lowered
        or "campfire" in lowered
    ):
        return EdgeKind.COOKS
    if "smithing" in lowered:
        return EdgeKind.SMITHS
    return EdgeKind.CRAFTS


def _recipe_outputs(payload: dict[str, Any]) -> list[tuple[str, float]]:
    raw = payload.get("result")
    if raw is None:
        return []
    if isinstance(raw, str):
        return [(raw, 1.0)]
    if isinstance(raw, dict):
        item = raw.get("id", raw.get("item"))
        if isinstance(item, str):
            count_raw = raw.get("count", 1)
            count = float(count_raw) if isinstance(count_raw, (int, float)) else 1.0
            return [(item, max(1.0, count))]
    return []


def _recipe_ingredients(payload: dict[str, Any]) -> list[list[str]]:
    ingredients: list[list[str]] = []
    raw_ingredients = payload.get("ingredients")
    if isinstance(raw_ingredients, list):
        for entry in raw_ingredients:
            alternatives = _ingredient_ids(entry)
            if alternatives:
                ingredients.append(alternatives)
    key = payload.get("key")
    pattern = payload.get("pattern")
    if isinstance(key, dict) and isinstance(pattern, list):
        counts: dict[str, int] = {}
        for row in pattern:
            if not isinstance(row, str):
                continue
            for symbol in row:
                if symbol != " ":
                    counts[symbol] = counts.get(symbol, 0) + 1
        for symbol, count in sorted(counts.items()):
            alternatives = _ingredient_ids(key.get(symbol))
            for _ in range(count):
                if alternatives:
                    ingredients.append(alternatives)
    return ingredients


def _ingredient_ids(raw: Any) -> list[str]:
    if isinstance(raw, list):
        result: list[str] = []
        for entry in raw:
            result.extend(_ingredient_ids(entry))
        return _dedupe(result)
    if isinstance(raw, str):
        return [raw]
    if not isinstance(raw, dict):
        return []
    item = raw.get("item", raw.get("id"))
    if isinstance(item, str):
        return [item]
    tag = raw.get("tag")
    if isinstance(tag, str):
        return [f"#{tag}"]
    items = raw.get("items")
    if isinstance(items, list):
        return _dedupe([str(item_id) for item_id in items if isinstance(item_id, str)])
    return []


def _item_node_id(resource_id: str) -> str:
    if resource_id.startswith("#"):
        return f"tag:{resource_id[1:]}"
    return f"item:{resource_id}"


def _ensure_item(graph: KnowledgeGraph, node_id: str, resource_id: str) -> None:
    if node_id in graph.nodes:
        return
    graph.add_node(
        KnowledgeNode(
            node_id=node_id,
            kind=NodeKind.TAG if resource_id.startswith("#") else NodeKind.ITEM,
            name=resource_id,
        )
    )


def _resource_id(path: Path, marker: str) -> str | None:
    parts = path.parts
    try:
        data_index = len(parts) - 1 - list(reversed(parts)).index("data")
        marker_index = parts.index(marker, data_index + 2)
    except ValueError:
        return None
    namespace = parts[data_index + 1]
    relative = Path(*parts[marker_index + 1 :]).with_suffix("").as_posix()
    return f"{namespace}:{relative}"


def _provenance(path: Path, version: GameVersion) -> Provenance:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return Provenance(
        source_type="vanilla_data",
        source_id=path.as_posix(),
        version_key=version.key,
        checksum=f"sha256:{digest}",
        confidence=1.0,
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VanillaDataError(f"failed to parse {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise VanillaDataError(f"expected JSON object: {path}")
    return payload


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))
