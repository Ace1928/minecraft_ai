from __future__ import annotations

import threading
import time
from collections.abc import Iterable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PerceptSource(StrEnum):
    FAST_VISION = "fast_vision"
    HUD = "hud"
    CHAT = "chat"
    GUI = "gui"
    VLM = "vlm"
    MOTOR = "motor"
    AUDIO = "audio"
    SYSTEM = "system"


class PerceptKind(StrEnum):
    PLAYER = "player"
    HUD = "hud"
    HOTBAR = "hotbar"
    GUI = "gui"
    CHAT = "chat"
    CROSSHAIR = "crosshair"
    OBJECT = "object"
    TERRAIN = "terrain"
    EVENT = "event"
    MOTION = "motion"
    WORLD = "world"


class PerceptFact(BaseModel):
    """One typed observation with explicit freshness and provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(min_length=1, max_length=256)
    kind: PerceptKind
    source: PerceptSource
    value: Any
    confidence: float = Field(ge=0.0, le=1.0)
    sequence: int = Field(ge=0)
    observed_monotonic_ns: int = Field(gt=0)
    ttl_ms: int = Field(default=500, ge=1, le=300_000)
    provenance: str | None = Field(default=None, max_length=1024)

    def age_ms(self, now_ns: int | None = None) -> float:
        now = time.monotonic_ns() if now_ns is None else now_ns
        return max(0.0, (now - self.observed_monotonic_ns) / 1_000_000.0)

    def stale(self, now_ns: int | None = None) -> bool:
        return self.age_ms(now_ns) >= self.ttl_ms


class BlackboardSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int = Field(ge=0)
    captured_monotonic_ns: int = Field(gt=0)
    facts: tuple[PerceptFact, ...]

    def by_key(self) -> dict[str, PerceptFact]:
        return {fact.key: fact for fact in self.facts}


class PerceptionBlackboard:
    """Thread-safe latest-fact store shared by perception, skills and cognition.

    Higher sequence wins. For equal sequence, newer observation wins. Stale facts
    are never returned by default, so a slow VLM cannot keep an old claim alive
    after fast perception has lost the target.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._facts: dict[str, PerceptFact] = {}
        self._snapshot_sequence = 0

    def publish(self, fact: PerceptFact) -> bool:
        with self._lock:
            current = self._facts.get(fact.key)
            if current is not None:
                if fact.sequence < current.sequence:
                    return False
                if (
                    fact.sequence == current.sequence
                    and fact.observed_monotonic_ns <= current.observed_monotonic_ns
                ):
                    return False
            self._facts[fact.key] = fact
            return True

    def publish_many(self, facts: Iterable[PerceptFact]) -> int:
        accepted = 0
        for fact in facts:
            if self.publish(fact):
                accepted += 1
        return accepted

    def get(
        self,
        key: str,
        *,
        min_confidence: float = 0.0,
        include_stale: bool = False,
        now_ns: int | None = None,
    ) -> PerceptFact | None:
        with self._lock:
            fact = self._facts.get(key)
            if fact is None or fact.confidence < min_confidence:
                return None
            if not include_stale and fact.stale(now_ns):
                return None
            return fact

    def snapshot(
        self,
        *,
        kinds: Iterable[PerceptKind] | None = None,
        sources: Iterable[PerceptSource] | None = None,
        min_confidence: float = 0.0,
        include_stale: bool = False,
        now_ns: int | None = None,
    ) -> BlackboardSnapshot:
        captured = time.monotonic_ns() if now_ns is None else now_ns
        allowed_kinds = None if kinds is None else frozenset(kinds)
        allowed_sources = None if sources is None else frozenset(sources)
        with self._lock:
            facts = tuple(
                sorted(
                    (
                        fact
                        for fact in self._facts.values()
                        if fact.confidence >= min_confidence
                        and (allowed_kinds is None or fact.kind in allowed_kinds)
                        and (allowed_sources is None or fact.source in allowed_sources)
                        and (include_stale or not fact.stale(captured))
                    ),
                    key=lambda fact: fact.key,
                )
            )
            self._snapshot_sequence += 1
            sequence = self._snapshot_sequence
        return BlackboardSnapshot(
            sequence=sequence,
            captured_monotonic_ns=captured,
            facts=facts,
        )

    def expire(self, *, now_ns: int | None = None) -> tuple[str, ...]:
        now = time.monotonic_ns() if now_ns is None else now_ns
        with self._lock:
            stale_keys = tuple(sorted(key for key, fact in self._facts.items() if fact.stale(now)))
            for key in stale_keys:
                del self._facts[key]
            return stale_keys

    def clear_source(self, source: PerceptSource) -> int:
        with self._lock:
            keys = [key for key, fact in self._facts.items() if fact.source == source]
            for key in keys:
                del self._facts[key]
            return len(keys)
