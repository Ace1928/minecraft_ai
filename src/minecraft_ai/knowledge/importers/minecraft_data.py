from __future__ import annotations

import hashlib
import json
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


class MinecraftDataError(RuntimeError):
    pass


def import_minecraft_data(root: str | Path, version: GameVersion) -> KnowledgeGraph:
    """Import normalized PrismarineJS minecraft-data for one exact version.

    minecraft-data deliberately reuses unchanged domain files across game
    versions. `data/dataPaths.json` is therefore the authority for resolving
    each domain rather than assuming every file lives in the target version
    directory.
    """

    root_path = Path(root)
    data_root = root_path / "data" if (root_path / "data").is_dir() else root_path
    paths_file = data_root / "dataPaths.json"
    if not paths_file.is_file():
        raise MinecraftDataError(f"missing minecraft-data dataPaths.json: {paths_file}")
    paths_raw = _load_json(paths_file)
    edition_key = "bedrock" if version.edition.value == "bedrock" else "pc"
    edition_paths = paths_raw.get(edition_key)
    if not isinstance(edition_paths, dict):
        raise MinecraftDataError(f"minecraft-data contains no {edition_key!r} mapping")
    mapping = edition_paths.get(version.version_id)
    if not isinstance(mapping, dict):
        raise MinecraftDataError(
            f"minecraft-data does not map exact {version.edition.value} version {version.version_id}"
        )

    graph = KnowledgeGraph(version)
    _import_nodes(data_root, mapping, graph, "items", NodeKind.ITEM)
    _import_nodes(data_root, mapping, graph, "blocks", NodeKind.BLOCK)
    _import_nodes(data_root, mapping, graph, "entities", NodeKind.ENTITY)
    _import_nodes(data_root, mapping, graph, "biomes", NodeKind.BIOME)
    _import_nodes(data_root, mapping, graph, "enchantments", NodeKind.ENCHANTMENT)
    _import_recipes(data_root, mapping, graph)
    return graph


def resolved_domain_path(
    data_root: Path,
    mapping: dict[str, Any],
    domain: str,
) -> Path | None:
    location = mapping.get(domain)
    if not isinstance(location, str) or not location.strip():
        return None
    candidate = data_root / location / f"{domain}.json"
    return candidate if candidate.is_file() else None


def _import_nodes(
    data_root: Path,
    mapping: dict[str, Any],
    graph: KnowledgeGraph,
    domain: str,
    kind: NodeKind,
) -> None:
    path = resolved_domain_path(data_root, mapping, domain)
    if path is None:
        return
    raw = _load_json_value(path)
    records: list[Any]
    if isinstance(raw, list):
        records = raw
    elif isinstance(raw, dict):
        records = list(raw.values())
    else:
        return
    for record in records:
        if not isinstance(record, dict):
            continue
        _add_record_node(graph, record, kind)
        variations = record.get("variations")
        if isinstance(variations, list):
            for variation in variations:
                if isinstance(variation, dict):
                    _add_record_node(graph, variation, kind)


def _add_record_node(graph: KnowledgeGraph, record: dict[str, Any], kind: NodeKind) -> None:
    raw_name = record.get("name")
    if not isinstance(raw_name, str) or not raw_name.strip():
        return
    resource = _resource_name(raw_name)
    node_id = _node_id(kind, resource)
    metadata: dict[str, str | int | float | bool] = {}
    display_name = record.get("displayName")
    if isinstance(display_name, str):
        metadata["display_name"] = display_name
    stack_size = record.get("stackSize")
    if isinstance(stack_size, int):
        metadata["stack_size"] = stack_size
    numeric_id = record.get("id")
    if isinstance(numeric_id, int):
        metadata["numeric_id"] = numeric_id
    graph.add_node(
        KnowledgeNode(
            node_id=node_id,
            kind=kind,
            name=resource,
            metadata=metadata,
        )
    )


def _import_recipes(
    data_root: Path,
    mapping: dict[str, Any],
    graph: KnowledgeGraph,
) -> None:
    path = resolved_domain_path(data_root, mapping, "recipes")
    if path is None:
        return
    raw = _load_json_value(path)
    if isinstance(raw, dict):
        recipes = list(raw.items())
    elif isinstance(raw, list):
        recipes = [(str(index), value) for index, value in enumerate(raw)]
    else:
        return
    provenance = _provenance(path, graph.version)
    for fallback_id, recipe in recipes:
        if not isinstance(recipe, dict):
            continue
        recipe_name_raw = recipe.get("name")
        recipe_name = recipe_name_raw if isinstance(recipe_name_raw, str) else fallback_id
        process_id = f"process:recipe:{recipe_name}"
        recipe_type_raw = recipe.get("type")
        recipe_type = recipe_type_raw if isinstance(recipe_type_raw, str) else "unknown"
        graph.add_node(
            KnowledgeNode(
                node_id=process_id,
                kind=NodeKind.PROCESS,
                name=recipe_name,
                metadata={"recipe_type": recipe_type},
            )
        )
        ingredients = recipe.get("ingredients")
        if isinstance(ingredients, list):
            for index, ingredient in enumerate(ingredients):
                for alternative_index, (name, quantity) in enumerate(
                    _ingredient_alternatives(ingredient)
                ):
                    item_id = _ensure_item(graph, name)
                    group = f"{process_id}:ingredient:{index}"
                    alternatives = _ingredient_alternatives(ingredient)
                    graph.add_edge(
                        KnowledgeEdge(
                            edge_id=(
                                f"{process_id}:requires:{index}:{alternative_index}:{item_id}"
                            ),
                            source=process_id,
                            target=item_id,
                            kind=(
                                EdgeKind.ALTERNATIVE_REQUIRES
                                if len(alternatives) > 1
                                else EdgeKind.REQUIRES
                            ),
                            quantity=max(1.0, quantity),
                            alternative_group=group if len(alternatives) > 1 else None,
                            provenance=provenance,
                        )
                    )
        outputs = recipe.get("output")
        if isinstance(outputs, list):
            for output_index, output in enumerate(outputs):
                parsed = _named_count(output)
                if parsed is None:
                    continue
                name, quantity = parsed
                item_id = _ensure_item(graph, name)
                graph.add_edge(
                    KnowledgeEdge(
                        edge_id=f"{process_id}:output:{output_index}:{item_id}",
                        source=process_id,
                        target=item_id,
                        kind=_recipe_output_kind(recipe_type),
                        quantity=max(1.0, quantity),
                        provenance=provenance,
                    )
                )


def _ingredient_alternatives(raw: Any) -> list[tuple[str, float]]:
    if isinstance(raw, list):
        result: list[tuple[str, float]] = []
        for entry in raw:
            parsed = _named_count(entry)
            if parsed is not None:
                result.append(parsed)
        return result
    parsed = _named_count(raw)
    return [] if parsed is None else [parsed]


def _named_count(raw: Any) -> tuple[str, float] | None:
    if isinstance(raw, str):
        return (_resource_name(raw), 1.0)
    if not isinstance(raw, dict):
        return None
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    count_raw = raw.get("count", 1)
    quantity = float(count_raw) if isinstance(count_raw, (int, float)) else 1.0
    return (_resource_name(name), quantity)


def _recipe_output_kind(recipe_type: str) -> EdgeKind:
    lowered = recipe_type.lower()
    if "smith" in lowered:
        return EdgeKind.SMITHS
    if any(token in lowered for token in ("furnace", "smelt", "cook", "campfire")):
        return EdgeKind.COOKS
    return EdgeKind.CRAFTS


def _ensure_item(graph: KnowledgeGraph, name: str) -> str:
    resource = _resource_name(name)
    item_id = _node_id(NodeKind.ITEM, resource)
    if item_id not in graph.nodes:
        graph.add_node(KnowledgeNode(node_id=item_id, kind=NodeKind.ITEM, name=resource))
    return item_id


def _resource_name(name: str) -> str:
    return name if ":" in name else f"minecraft:{name}"


def _node_id(kind: NodeKind, resource: str) -> str:
    return f"{kind.value}:{resource}"


def _provenance(path: Path, version: GameVersion) -> Provenance:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return Provenance(
        source_type="minecraft_data",
        source_id=str(path),
        version_key=version.key,
        checksum=digest,
        confidence=0.95,
    )


def _load_json(path: Path) -> dict[str, Any]:
    raw = _load_json_value(path)
    if not isinstance(raw, dict):
        raise MinecraftDataError(f"expected JSON object: {path}")
    return raw


def _load_json_value(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise MinecraftDataError(f"failed to read {path}: {exc}") from exc
