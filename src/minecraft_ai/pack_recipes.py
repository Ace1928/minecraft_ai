"""Hash-pinned, read-only Minecraft pack recipe lookup."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .wiki import WikiEvidence
from .pack_scope import ActiveRecipeIdentity, active_recipe_identity
from .perception import CognitionReadView

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_CATALOG_BYTES = 10 * 1024 * 1024
# Capture unsupported identity suffixes too, so exact lookup refuses the whole
# token instead of accepting a valid prefix (e.g. beacon@2.9 or beacon:9).
_ITEM_KEY = re.compile(r"(?<![a-z0-9_./:@-])[a-z0-9_.-]+:[a-z0-9_./:@-]+")
_RECIPE_INTENT = re.compile(
    r"(?<![a-z0-9_./:@-])(craft(?:ing)?|make|recipe|create)(?![a-z0-9_./:@-])"
)


def _normalize(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value.casefold())
    ascii_text = "".join(char for char in folded if not unicodedata.combining(char))
    return " ".join(re.findall(r"[a-z0-9]+", ascii_text))


@dataclass(frozen=True, slots=True)
class PackRecipeAnswer:
    evidence: WikiEvidence
    chat_reply: str
    identity: ActiveRecipeIdentity | None = None


class PackRecipeCatalog:
    """Small query surface over one exact server-exported recipe snapshot.

    The catalog never reads server credentials, contacts the server, or edits a
    world. Its SHA-256 must be pinned by the local Minecraft runtime config.
    """

    def __init__(self, payload: dict[str, Any], sha256: str) -> None:
        if (
            type(payload) is not dict
            or payload.get("schema_version") != 1
            or payload.get("world") != "PokemonFamily"
            or type(payload.get("bds_version")) is not str
            or type(payload.get("revision")) is not str
            or _SHA256.fullmatch(payload["revision"]) is None
            or type(payload.get("items")) is not dict
            or type(payload.get("recipes")) is not dict
            or type(payload.get("warnings")) is not list
            or payload["warnings"]
            or type(sha256) is not str
            or _SHA256.fullmatch(sha256) is None
        ):
            raise ValueError("invalid or warning-bearing family pack recipe catalog")
        self._payload = payload
        self.sha256 = sha256
        self.version_id = payload["bds_version"]

    def active_scope_status(self, view: CognitionReadView) -> str:
        return self._scope_status(active_recipe_identity(view))

    def _scope_status(self, active: ActiveRecipeIdentity | None) -> str:
        configured = self._payload.get("configured_recipe_identity")
        if (
            type(configured) is not dict
            or configured.get("state") != "configured_only"
            or type(configured.get("scope_sha256")) is not str
            or _SHA256.fullmatch(configured["scope_sha256"]) is None
            or active is None
        ):
            return "unknown"
        if (
            active.world != self._payload["world"]
            or active.bds_version != self.version_id
            or active.catalog_sha256 != self.sha256
            or active.scope_sha256 != configured["scope_sha256"]
        ):
            return "mismatch"
        return "verified"

    def active_identity(self, view: CognitionReadView) -> ActiveRecipeIdentity | None:
        active = active_recipe_identity(view)
        return active if self._scope_status(active) == "verified" else None

    @staticmethod
    def is_recipe_query(query: str) -> bool:
        return _RECIPE_INTENT.search(query.casefold()) is not None

    def lookup_live(
        self, query: str, view: CognitionReadView,
    ) -> PackRecipeAnswer | None:
        """Only a fresh matching active identity admits configured recipe text."""
        identity = self.active_identity(view)
        if identity is None:
            return None
        answer = self.lookup(query, game_version=identity.bds_version)
        if answer is None:
            return None
        return PackRecipeAnswer(answer.evidence, answer.chat_reply, identity)

    @classmethod
    def load(cls, path: str | Path, expected_sha256: str) -> "PackRecipeCatalog":
        selected = Path(path)
        if not selected.is_absolute() or selected.resolve(strict=True) != selected:
            raise ValueError("pack recipe catalog path must be canonical and absolute")
        before = selected.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or before.st_size <= 0
            or before.st_size > _MAX_CATALOG_BYTES
        ):
            raise ValueError("pack recipe catalog must be an owned, bounded regular file")
        digest = hashlib.sha256()
        fd = os.open(selected, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            opened = os.fstat(fd)
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                raise ValueError("pack recipe catalog changed while opening")
            with os.fdopen(fd, "rb", closefd=False) as stream:
                raw = stream.read(_MAX_CATALOG_BYTES + 1)
                digest.update(raw)
        finally:
            os.close(fd)
        if (
            len(raw) != before.st_size
            or len(raw) > _MAX_CATALOG_BYTES
            or digest.hexdigest() != expected_sha256
        ):
            raise ValueError("pack recipe catalog digest or size mismatch")
        value = json.loads(raw)
        return cls(value, expected_sha256)

    def lookup(self, query: str, *, game_version: str) -> PackRecipeAnswer | None:
        if type(query) is not str or not query.strip() or game_version != self.version_id:
            return None
        intent_match = _RECIPE_INTENT.search(query.casefold())
        if intent_match is None:
            return None
        # Keep the identity-bearing target separate from ingredient/context
        # clauses. An unknown namespace/variant must never become a same-named
        # item from another pack, even when that item is craftable.
        target_clause = query.casefold()[intent_match.end():]
        target_clause = re.sub(
            r"^\s*for(?![a-z0-9_./:@-])", "", target_clause, count=1,
        )
        target_clause = re.split(
            r"(?<![a-z0-9_./:@-])(?:with|using|from|for)(?![a-z0-9_./:@-])",
            target_clause, maxsplit=1,
        )[0]
        prefix_before_intent = query.casefold()[:intent_match.start()]
        if re.match(
            r"^\s*(?:with|using|from|for)(?![a-z0-9_./:@-])", prefix_before_intent,
        ):
            prefix_before_intent = ""  # An ingredient prelude is not the output.
        explicit_targets = {
            match.group().rstrip(".")
            for clause in (prefix_before_intent, target_clause)
            for match in _ITEM_KEY.finditer(clause)
        }
        if len(explicit_targets) > 1:
            return None
        explicit_target = next(iter(explicit_targets), None)
        normalized = _normalize(query)
        compact = normalized.replace(" ", "")
        intent_position = len(_normalize(query[:intent_match.end()]))
        items = self._payload["items"]
        recipes = self._payload["recipes"]
        candidates: list[tuple[int, int, int, str, dict[str, Any]]] = []
        for item_id, item in items.items():
            if explicit_target is not None and item_id != explicit_target:
                continue
            if type(item) is not dict or item.get("craftable") is not True:
                continue
            name = item.get("name")
            if type(name) is not str:
                continue
            name_norm = _normalize(name)
            id_norm = _normalize(item_id.split(":", 1)[-1].replace("_", " "))
            variants = (
                {_normalize(item_id)} if explicit_target is not None
                else {name_norm, f"{name_norm}s", id_norm, f"{id_norm}s"}
            )
            phrases = [
                (match.start(), match.end() - match.start())
                for variant in variants
                if variant
                for match in re.finditer(
                    r"(?<![a-z0-9])" + re.escape(variant) + r"(?![a-z0-9])",
                    normalized,
                )
            ]
            if not phrases:
                for variant in variants:
                    compact_variant = variant.replace(" ", "")
                    if compact_variant and compact_variant in compact:
                        phrases.append((compact.index(compact_variant), 0))
            if phrases:
                position, exact_length = min(
                    phrases,
                    key=lambda phrase: (abs(phrase[0] - intent_position), -phrase[1]),
                )
                before_intent = int(position < intent_position)
                candidates.append(
                    (before_intent, abs(position - intent_position), -exact_length, item_id, item)
                )
        if not candidates:
            return None
        # In questions that name both a result and its ingredients, prefer the
        # first exact item phrase after "craft", "make", or "recipe". If phrases
        # start at that same position, retain the complete compound item name.
        candidates.sort(key=lambda candidate: candidate[:4])
        if len(candidates) > 1 and candidates[0][:3] == candidates[1][:3]:
            return None  # Equal display-name matches need an exact item key.
        _before_intent, _distance, _length, selected_id, selected_item = candidates[0]
        recipe_ids = selected_item.get("recipes")
        if type(recipe_ids) is not list:
            return None
        matches = [
            recipes[key] for key in recipe_ids[:3] if key in recipes and type(recipes[key]) is dict
        ]
        if not matches:
            return None
        rendered: list[str] = []
        sources: list[str] = []
        for recipe in matches:
            line = self._render_recipe(recipe, items, output_key=selected_id)
            if line is not None:
                rendered.append(line)
                source = recipe.get("source")
                if type(source) is dict and type(source.get("pack")) is str:
                    sources.append(source["pack"])
        if not rendered:
            return None
        pack = sources[0] if sources else "active pack JSON recipe"
        extract = (
            f"{pack}; Bedrock {self.version_id}; pack revision {self._payload['revision'][:16]}. "
            + " ".join(rendered)
        )
        evidence = WikiEvidence(
            title=f"{selected_item['name']} recipe — family pack",
            extract=extract[:600],
            retrieved_ns=time.time_ns(),
            query=query[:120],
            version_key=f"bedrock:{self.version_id}:pack:{self._payload['revision']}",
            confidence=1.0,
        )
        # Keep the in-game line short and source-exact; metadata stays in cognition.
        reply = self._ascii_chat_projection(rendered[0])
        if reply is None or len(reply) > 150:
            return None
        return PackRecipeAnswer(evidence=evidence, chat_reply=reply)

    @staticmethod
    def _ascii_chat_projection(line: str) -> str | None:
        """Project Latin accents/grid blanks into the existing ASCII actuator.

        Original names and recipe text remain untouched in the catalog and
        reference evidence. Unsupported characters refuse delivery rather than
        disappearing from an item identifier or changing recipe quantities.
        """
        # Canonical decomposition supports accents only; compatibility
        # transliteration (ligatures, full-width digits, etc.) is not admitted.
        decomposed = unicodedata.normalize("NFD", line.replace("·", "."))
        reply = "".join(char for char in decomposed if not unicodedata.combining(char))
        if not reply or any(not 32 <= ord(char) <= 126 for char in reply):
            return None
        return reply

    def mentions_pack_content(self, query: str) -> bool:
        """A vanilla wiki cannot establish mechanics of a locally installed item."""
        if any(
            not match.group().startswith("minecraft:")
            or match.group().rstrip(".") not in self._payload["items"]
            for match in _ITEM_KEY.finditer(query.casefold())
        ):
            return True
        normalized = _normalize(query)
        for item_id, item in self._payload["items"].items():
            if item_id.startswith("minecraft:") or type(item) is not dict:
                continue
            name = item.get("name")
            if type(name) is not str:
                continue
            names = {_normalize(name), _normalize(item_id.split(":", 1)[-1])}
            if any(re.search(r"(?<![a-z0-9])" + re.escape(value) + r"s?(?![a-z0-9])",
                             normalized) for value in names if value):
                return True
        return False

    @staticmethod
    def _render_recipe(
        recipe: dict[str, Any], items: dict[str, Any], *, output_key: str | None = None,
    ) -> str | None:
        ingredients = recipe.get("ingredients")
        outputs = recipe.get("outputs")
        stations = recipe.get("stations")
        if (
            type(ingredients) is not list
            or not ingredients
            or type(outputs) is not list
            or not outputs
            or type(stations) is not list
            or not stations
            or any(type(station) is not str for station in stations)
            or any(type(item) is not dict for item in ingredients + outputs)
            or any(not station for station in stations)
        ):
            return None

        def reference_key(item: dict[str, Any]) -> str | None:
            # The export's key distinguishes ingredient tags and legacy item
            # metadata. Falling back from a declared key would change the
            # recipe (for example, Awkward Potion becomes generic Potion).
            key = item.get("key", item.get("id"))
            if type(key) is not str:
                return None
            if "key" not in item and (
                item.get("tag") is True or item.get("data") not in (None, 0)
            ):
                return None
            return key

        def named(item: dict[str, Any]) -> str | None:
            key = reference_key(item)
            record = items.get(key) if key is not None else None
            name = record.get("name") if type(record) is dict else None
            return name if type(name) is str else None

        ingredient_rows = []
        for ingredient in ingredients:
            name = named(ingredient)
            count = ingredient.get("count")
            if name is None or type(count) is not int or count <= 0:
                return None
            ingredient_rows.append((name, count, reference_key(ingredient)))
        if output_key is None:
            output = outputs[0]
        else:
            matching_outputs = [item for item in outputs if reference_key(item) == output_key]
            if len(matching_outputs) != 1:
                return None
            output = matching_outputs[0]
        output_name = named(output)
        output_count = output.get("count")
        if output_name is None or type(output_count) is not int or output_count <= 0:
            return None
        station = (
            "crafting table"
            if "crafting_table" in stations
            else ", ".join(str(item).replace("_", " ") for item in stations[:2])
        )
        arrangement = ""
        grid = recipe.get("grid")
        if (
            type(grid) is list
            and grid
            and len(ingredient_rows) == 2
            and all(type(row) is list for row in grid)
        ):
            key_to_name = {key: name for name, _count, key in ingredient_rows}
            rendered_rows = []
            for row in grid:
                rendered_cells = []
                for cell in row:
                    if cell is None:
                        rendered_cells.append("·")
                        continue
                    if type(cell) is not dict or reference_key(cell) not in key_to_name:
                        return None
                    rendered_cells.append(key_to_name[reference_key(cell)][:1].upper())
                rendered_rows.append(" ".join(rendered_cells))
            arrangement = " Pattern " + " / ".join(rendered_rows) + "."
        joined = " and ".join(f"{count} {name}" for name, count, _item_id in ingredient_rows)
        plural_output = output_name + (
            "s" if output_count != 1 and not output_name.casefold().endswith("s") else ""
        )
        line = f"At a {station}, use {joined} to make {output_count} {plural_output}."
        if arrangement and len(line) + len(arrangement) <= 150:
            line = (
                f"At a {station}, use {joined} to make {output_count} {plural_output}.{arrangement}"
            )
        return line
