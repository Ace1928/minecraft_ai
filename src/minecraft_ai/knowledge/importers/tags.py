from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..model import EdgeKind, KnowledgeEdge, KnowledgeGraph, KnowledgeNode, NodeKind, Provenance


class TagDataError(RuntimeError):
    pass


def enrich_java_tags(root: str | Path, graph: KnowledgeGraph) -> None:
    """Add exact-version item/block tag membership to an existing Java graph."""
    root_path = Path(root)
    data_dir = root_path / "data"
    if not data_dir.exists():
        raise TagDataError(f"missing data directory: {data_dir}")

    patterns = (
        ("*/tags/item/**/*.json", NodeKind.ITEM, "item"),
        ("*/tags/items/**/*.json", NodeKind.ITEM, "items"),
        ("*/tags/block/**/*.json", NodeKind.BLOCK, "block"),
        ("*/tags/blocks/**/*.json", NodeKind.BLOCK, "blocks"),
    )
    seen: set[Path] = set()
    for pattern, member_kind, marker in patterns:
        for path in sorted(data_dir.glob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            _import_tag(path, graph, member_kind=member_kind, marker=marker)


def _import_tag(
    path: Path,
    graph: KnowledgeGraph,
    *,
    member_kind: NodeKind,
    marker: str,
) -> None:
    payload = _load_json(path)
    tag_resource = _tag_resource_id(path, marker)
    if tag_resource is None:
        return
    tag_node_id = f"tag:{tag_resource}"
    _ensure_node(graph, tag_node_id, NodeKind.TAG, tag_resource)
    provenance = _provenance(path, graph)

    values = payload.get("values", [])
    if not isinstance(values, list):
        raise TagDataError(f"tag values must be a list: {path}")
    for index, raw in enumerate(values):
        resource_id = _tag_value_id(raw)
        if resource_id is None:
            continue
        if resource_id.startswith("#"):
            child_resource = resource_id[1:]
            child_node_id = f"tag:{child_resource}"
            _ensure_node(graph, child_node_id, NodeKind.TAG, child_resource)
        else:
            child_resource = resource_id
            child_node_id = f"{member_kind.value}:{child_resource}"
            _ensure_node(graph, child_node_id, member_kind, child_resource)
        graph.add_edge(
            KnowledgeEdge(
                edge_id=f"tag-member:{tag_resource}:{index}:{child_node_id}",
                source=child_node_id,
                target=tag_node_id,
                kind=EdgeKind.MEMBER_OF,
                provenance=provenance,
            )
        )


def _tag_value_id(raw: Any) -> str | None:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        value = raw.get("id")
        if isinstance(value, str):
            return value
    return None


def _tag_resource_id(path: Path, marker: str) -> str | None:
    parts = path.parts
    try:
        data_index = len(parts) - 1 - list(reversed(parts)).index("data")
        tags_index = parts.index("tags", data_index + 2)
        marker_index = parts.index(marker, tags_index + 1)
    except ValueError:
        return None
    namespace = parts[data_index + 1]
    relative = Path(*parts[marker_index + 1 :]).with_suffix("").as_posix()
    return f"{namespace}:{relative}"


def _ensure_node(
    graph: KnowledgeGraph,
    node_id: str,
    kind: NodeKind,
    resource_id: str,
) -> None:
    if node_id not in graph.nodes:
        graph.add_node(KnowledgeNode(node_id=node_id, kind=kind, name=resource_id))


def _provenance(path: Path, graph: KnowledgeGraph) -> Provenance:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return Provenance(
        source_type="vanilla_data",
        source_id=path.as_posix(),
        version_key=graph.version.key,
        checksum=f"sha256:{digest}",
        confidence=1.0,
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TagDataError(f"failed to parse {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TagDataError(f"expected JSON object: {path}")
    return payload
