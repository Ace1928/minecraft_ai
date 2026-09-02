"""Exact-version knowledge importers."""

from pathlib import Path

from ..model import GameVersion, KnowledgeGraph
from .minecraft_data import MinecraftDataError, import_minecraft_data
from .tags import TagDataError, enrich_java_tags
from .vanilla_json import VanillaDataError, import_java_datapack as _import_java_datapack


def import_java_datapack(root: str | Path, version: GameVersion) -> KnowledgeGraph:
    """Compile supported exact-version Java data domains into one graph."""
    graph = _import_java_datapack(root, version)
    enrich_java_tags(root, graph)
    return graph


__all__ = [
    "MinecraftDataError",
    "TagDataError",
    "VanillaDataError",
    "enrich_java_tags",
    "import_java_datapack",
    "import_minecraft_data",
]
