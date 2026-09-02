from __future__ import annotations

import hashlib
import importlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .knowledge import GameVersion


class WikiEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str
    extract: str
    page_id: int | None = None
    revision_id: int | None = None
    url: str | None = None
    retrieved_ns: int
    query: str
    version_key: str
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)


@dataclass
class WikiService:
    """Optional explanatory retrieval layer; never an authoritative graph writer."""

    cache_dir: Path
    api_url: str = "https://minecraft.wiki/api.php"
    timeout_s: float = 15.0

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def search(
        self,
        query: str,
        version: GameVersion,
        *,
        limit: int = 5,
        max_age_s: float = 7 * 86_400,
    ) -> tuple[WikiEvidence, ...]:
        locked_query = f"{query} {version.edition.value} {version.version_id}"
        cache_path = self._cache_path(locked_query, version)
        cached = self._read_cache(cache_path, max_age_s=max_age_s)
        if cached is not None:
            return cached[:limit]
        evidence = self._fetch(locked_query, version, limit=limit)
        self._write_cache(cache_path, evidence)
        return evidence

    def _fetch(
        self,
        query: str,
        version: GameVersion,
        *,
        limit: int,
    ) -> tuple[WikiEvidence, ...]:
        try:
            httpx = importlib.import_module("httpx")
        except ImportError as exc:
            raise RuntimeError("install minecraft-ai[knowledge] for online wiki retrieval") from exc
        params = {
            "action": "query",
            "format": "json",
            "formatversion": 2,
            "generator": "search",
            "gsrsearch": query,
            "gsrlimit": max(1, min(limit, 20)),
            "prop": "extracts|info|revisions",
            "exintro": 1,
            "explaintext": 1,
            "inprop": "url",
            "rvprop": "ids",
        }
        with httpx.Client(timeout=self.timeout_s, follow_redirects=True) as client:
            response = client.get(self.api_url, params=params)
            response.raise_for_status()
            raw = response.json()
        if not isinstance(raw, dict):
            return ()
        query_obj = raw.get("query")
        if not isinstance(query_obj, dict):
            return ()
        pages = query_obj.get("pages")
        if not isinstance(pages, list):
            return ()
        now = time.time_ns()
        results: list[WikiEvidence] = []
        for page in pages:
            if not isinstance(page, dict):
                continue
            title = page.get("title")
            extract = page.get("extract")
            if not isinstance(title, str) or not isinstance(extract, str):
                continue
            revisions = page.get("revisions")
            revision_id: int | None = None
            if isinstance(revisions, list) and revisions and isinstance(revisions[0], dict):
                raw_revision = revisions[0].get("revid")
                if isinstance(raw_revision, int):
                    revision_id = raw_revision
            page_id = page.get("pageid") if isinstance(page.get("pageid"), int) else None
            url = page.get("fullurl") if isinstance(page.get("fullurl"), str) else None
            results.append(
                WikiEvidence(
                    title=title,
                    extract=extract[:5000],
                    page_id=page_id,
                    revision_id=revision_id,
                    url=url,
                    retrieved_ns=now,
                    query=query,
                    version_key=version.key,
                )
            )
        return tuple(results)

    def _cache_path(self, query: str, version: GameVersion) -> Path:
        digest = hashlib.sha256(f"{version.key}\0{query}".encode()).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _read_cache(
        self,
        path: Path,
        *,
        max_age_s: float,
    ) -> tuple[WikiEvidence, ...] | None:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            modified_age = max(0.0, time.time() - path.stat().st_mtime)
        except (OSError, UnicodeError, ValueError):
            return None
        if modified_age > max_age_s or not isinstance(raw, list):
            return None
        try:
            return tuple(WikiEvidence.model_validate(item) for item in raw)
        except ValueError:
            return None

    def _write_cache(self, path: Path, evidence: tuple[WikiEvidence, ...]) -> None:
        payload = [item.model_dump(mode="json") for item in evidence]
        staged = path.with_name(f".{path.name}.tmp")
        staged.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        staged.replace(path)
