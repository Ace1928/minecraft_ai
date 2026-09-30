"""Bounded public Minecraft references through the shared ERAIS search owner."""

from __future__ import annotations

import http.client
import json
import re
import time
from dataclasses import dataclass
from urllib.parse import urlencode, urlsplit

from .native_world_model import _read_private_file
from .wiki import WikiEvidence

_SUBJECTS = (
    "crafting table", "stone pickaxe", "wooden pickaxe", "iron pickaxe", "copper ore",
    "iron ore", "oak tree", "oak log", "coal ore", "copper ingot", "iron ingot",
    "furnace", "smelting", "hunger", "food", "shelter", "creeper", "zombie", "spider",
    "daylight", "night", "wood", "planks", "stick", "bed", "water", "coal", "copper",
)
_PACK_WORDS = re.compile(
    r"\b(?:pok[eé]mon|pok[eé]dex|pok[eé]\s*ball|apricorn|cobbledrock|cobblemon|"
    r"great\s*ball|ultra\s*ball|master\s*ball|starter|evolution|evolve)s?\b", re.IGNORECASE,
)
_QUESTION = re.compile(
    r"^(?:what|how|where|why|which|is|are|can|does|search|look\s+up|explain)\b",
    re.IGNORECASE,
)
_VERSION = re.compile(r"[0-9]+(?:\.[0-9]+){1,3}\Z")
_MAX_RESPONSE_BYTES = 8192


def minecraft_public_topic(query: str, game_version: str) -> str | None:
    """Export only known public subject/intent words, never chat or identities."""
    if (
        type(query) is not str or not 1 <= len(query) <= 280
        or type(game_version) is not str or _VERSION.fullmatch(game_version) is None
        or _QUESTION.search(query.strip()) is None or _PACK_WORDS.search(query)
    ):
        return None
    lower = query.casefold()
    subject = next((word for word in _SUBJECTS if re.search(
        r"\b" + re.escape(word) + r"s?\b", lower,
    )), None)
    if subject is None:
        return None
    intent = (
        "crafting recipe" if re.search(r"\b(?:craft|make|recipe)\b", lower)
        else "locations" if re.search(r"\b(?:where|find)\b", lower)
        else "mechanics"
    )
    # A wiki is general reference material. Exact patch numbers suppress useful
    # articles; the evidence retains the target version and a general-wiki tier.
    return f"Minecraft Wiki Bedrock {subject} {intent}"


@dataclass
class WorldMinecraftSearch:
    """One fixed authenticated search route, called inside the cognition worker."""

    token_file: str
    timeout_s: float = 5.0

    def search(
        self, query: str, *, game_version: str, deadline_ns: int | None = None,
    ) -> tuple[WikiEvidence, ...]:
        topic = minecraft_public_topic(query, game_version)
        if topic is None:
            return ()
        cutoff = time.monotonic_ns() + int(min(max(self.timeout_s, 0.01), 5.0) * 1e9)
        if deadline_ns is not None:
            cutoff = min(cutoff, deadline_ns)
        connection: http.client.HTTPConnection | None = None
        response: http.client.HTTPResponse | None = None
        try:
            def remaining() -> float:
                seconds = (cutoff - time.monotonic_ns()) / 1e9
                if seconds <= 0:
                    raise TimeoutError("Minecraft search budget expired")
                if connection is not None and connection.sock is not None:
                    connection.sock.settimeout(seconds)
                elif response is not None and response.fp is not None:
                    # HTTPConnection detaches a Connection: close response's
                    # socket; bound each read through its still-open file owner.
                    response_socket = getattr(getattr(response.fp, "raw", None), "_sock", None)
                    if response_socket is not None:
                        response_socket.settimeout(seconds)
                return seconds

            token = _read_private_file(self.token_file, limit=512).decode("ascii").strip()
            if not 32 <= len(token) <= 512 or any(not 33 <= ord(char) <= 126 for char in token):
                return ()
            body = urlencode({
                "q": topic, "format": "json", "categories": "general", "language": "en",
                "safesearch": "1",
            }).encode()
            connection = http.client.HTTPConnection("127.0.0.1", 8889, timeout=remaining())
            connection.request("POST", "/search", body=body, headers={
                "Authorization": "Bearer " + token,
                "Content-Type": "application/x-www-form-urlencoded",
                "Connection": "close",
            })
            remaining()
            response = connection.getresponse()
            if response.status != 200:
                return ()
            raw = bytearray()
            while True:
                remaining()
                chunk = response.read1(min(4096, _MAX_RESPONSE_BYTES + 1 - len(raw)))
                if not chunk:
                    break
                raw.extend(chunk)
                if len(raw) > _MAX_RESPONSE_BYTES:
                    return ()
            remaining()
            payload = json.loads(raw)
            rows = payload.get("results") if type(payload) is dict else None
            if type(rows) is not list:
                return ()
            evidence = []
            for row in rows[:3]:
                if type(row) is not dict:
                    continue
                url, title, content = row.get("url"), row.get("title"), row.get("content")
                if type(url) is not str or type(title) is not str or type(content) is not str:
                    continue
                parsed = urlsplit(url)
                if (
                    parsed.scheme != "https" or parsed.hostname != "minecraft.wiki"
                    or parsed.username is not None or parsed.password is not None
                    or parsed.port not in {None, 443} or not content.strip()
                ):
                    continue
                evidence.append(WikiEvidence(
                    title=title[:120], extract=content[:480] + (
                        " General Minecraft wiki reference; this does not verify pack overrides."
                    ), url=url[:240], query=topic, retrieved_ns=time.time_ns(),
                    version_key=f"bedrock:{game_version}:general-wiki", confidence=0.65,
                ))
            return tuple(evidence)
        except (OSError, ValueError, UnicodeError, http.client.HTTPException):
            return ()
        finally:
            if connection is not None:
                connection.close()
