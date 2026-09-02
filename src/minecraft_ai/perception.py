from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field


class ScreenRegion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(gt=0.0, le=1.0)
    height: float = Field(gt=0.0, le=1.0)


class Track(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    track_id: str
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    region: ScreenRegion
    first_seen_ns: int
    last_seen_ns: int
    attributes: dict[str, str | int | float | bool] = Field(default_factory=dict)


class ChatLine(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    speaker: str | None = None
    observed_ns: int
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class PlayerEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    health: float | None = Field(default=None, ge=0.0)
    hunger: float | None = Field(default=None, ge=0.0)
    air: float | None = Field(default=None, ge=0.0)
    selected_slot: int | None = Field(default=None, ge=0, le=8)
    gui_mode: str = "world"


class PerceptionFact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str
    value: str | int | float | bool
    confidence: float = Field(ge=0.0, le=1.0)
    observed_ns: int
    source: str
    expires_after_ms: int = Field(default=1000, ge=1)

    def fresh(self, now_ns: int | None = None) -> bool:
        now = time.monotonic_ns() if now_ns is None else now_ns
        return now - self.observed_ns <= self.expires_after_ms * 1_000_000


class FrameState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    frame_id: int = Field(ge=0)
    captured_ns: int
    instance_id: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    player: PlayerEstimate = Field(default_factory=PlayerEstimate)
    tracks: tuple[Track, ...] = ()
    chat: tuple[ChatLine, ...] = ()
    facts: tuple[PerceptionFact, ...] = ()


T = TypeVar("T")


@dataclass
class RingBuffer(Generic[T]):
    capacity: int
    _items: deque[T] = field(init=False)

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError("capacity must be positive")
        self._items = deque(maxlen=self.capacity)

    def append(self, item: T) -> None:
        self._items.append(item)

    def latest(self) -> T | None:
        return self._items[-1] if self._items else None

    def snapshot(self) -> tuple[T, ...]:
        return tuple(self._items)


@dataclass
class PerceptionBlackboard:
    """Thread-safe-at-service-boundary typed state store.

    Services should publish immutable FrameState snapshots rather than sharing
    mutable detector internals. Stale semantic facts are filtered on read.
    """

    frame_capacity: int = 90
    _frames: RingBuffer[FrameState] = field(init=False)
    _facts: dict[str, PerceptionFact] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._frames = RingBuffer(self.frame_capacity)

    def publish(self, frame: FrameState) -> None:
        previous = self._frames.latest()
        if previous is not None:
            if frame.instance_id != previous.instance_id:
                raise ValueError("perception instance identity changed")
            if frame.frame_id <= previous.frame_id:
                raise ValueError("frame ids must increase monotonically")
            if frame.captured_ns <= previous.captured_ns:
                raise ValueError("frame timestamps must increase monotonically")
        self._frames.append(frame)
        for fact in frame.facts:
            existing = self._facts.get(fact.key)
            if existing is None or fact.observed_ns >= existing.observed_ns:
                self._facts[fact.key] = fact

    def latest(self) -> FrameState | None:
        return self._frames.latest()

    def frames(self) -> tuple[FrameState, ...]:
        return self._frames.snapshot()

    def fact(self, key: str, *, min_confidence: float = 0.0) -> PerceptionFact | None:
        fact = self._facts.get(key)
        if fact is None or fact.confidence < min_confidence or not fact.fresh():
            return None
        return fact

    def fresh_facts(self, *, min_confidence: float = 0.0) -> dict[str, PerceptionFact]:
        return {
            key: fact
            for key, fact in self._facts.items()
            if fact.confidence >= min_confidence and fact.fresh()
        }


class ActivePerceptionQuery(BaseModel):
    """Narrow semantic question submitted to a slower VLM service."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_id: str
    question: str = Field(min_length=1, max_length=1024)
    skill_id: str | None = None
    frame_id: int = Field(ge=0)
    deadline_ms: int = Field(default=750, ge=50, le=10_000)
    output_keys: tuple[str, ...] = ()
