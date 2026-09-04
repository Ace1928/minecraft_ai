from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class MemoryKind(StrEnum):
    WORKING = "working"
    EPISODIC = "episodic"
    SPATIAL = "spatial"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    SOCIAL = "social"
    FAILURE = "failure"


class MemoryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    memory_id: str
    kind: MemoryKind
    text: str
    created_ns: int
    updated_ns: int
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    goal_tags: tuple[str, ...] = ()
    entity_tags: tuple[str, ...] = ()
    location_key: str | None = None
    source: str = "agent"
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


@dataclass
class MemoryStore:
    records: dict[str, MemoryRecord] = field(default_factory=dict)

    def upsert(self, record: MemoryRecord) -> None:
        previous = self.records.get(record.memory_id)
        if previous is not None and record.updated_ns < previous.updated_ns:
            raise ValueError("memory updates cannot move backwards in time")
        self.records[record.memory_id] = record

    def remove(self, memory_id: str) -> None:
        self.records.pop(memory_id, None)

    def retrieve(
        self,
        *,
        kinds: set[MemoryKind] | None = None,
        goal_tags: set[str] | None = None,
        entity_tags: set[str] | None = None,
        location_key: str | None = None,
        limit: int = 20,
        now_ns: int | None = None,
    ) -> list[MemoryRecord]:
        if limit < 1:
            return []
        # Durable memory timestamps use Unix time so recency remains meaningful
        # after a process restart or host reboot. Runtime deadlines use a
        # monotonic clock elsewhere and must never be stored in these fields.
        now = time.time_ns() if now_ns is None else now_ns
        scored: list[tuple[float, MemoryRecord]] = []
        for record in self.records.values():
            if (
                record.source == "runtime:verified-skill-outcome"
                and record.metadata.get("context_key") == "explore-keepalive"
                and record.metadata.get("outcome") == "timed_out"
                and record.metadata.get("skill_id")
                in {
                    "explore_forward",
                    "traverse_level_ground",
                    "traverse_visible_obstacle",
                }
            ):
                # Older runtimes persisted normal bounded-chunk expiry as a
                # failure. Retain the audit record but never suggest it as a
                # learned failure to current planning.
                continue
            if kinds is not None and record.kind not in kinds:
                continue
            goal_overlap = len(set(record.goal_tags) & (goal_tags or set()))
            entity_overlap = len(set(record.entity_tags) & (entity_tags or set()))
            location_bonus = (
                1.0 if location_key is not None and record.location_key == location_key else 0.0
            )
            age_s = max(0.0, (now - record.updated_ns) / 1e9)
            recency = math.exp(-age_s / 86_400.0)
            score = (
                record.importance * 1.5
                + record.confidence
                + goal_overlap * 0.8
                + entity_overlap * 0.6
                + location_bonus
                + recency * 0.5
            )
            scored.append((score, record))
        scored.sort(key=lambda pair: (pair[0], pair[1].updated_ns), reverse=True)
        return [record for _, record in scored[:limit]]
