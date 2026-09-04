from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field


class ScreenRegion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(gt=0.0, le=1.0)
    height: float = Field(gt=0.0, le=1.0)


class EvidenceRegion(StrEnum):
    """Stable screen partitions used to ground semantic observations."""

    WORLD = "world"
    HUD = "hud"
    HOTBAR = "hotbar"
    CHAT = "chat"
    GUI = "gui"


class PerceptionQueryMode(StrEnum):
    """Typed VLM routes; free-form question text never selects a safety contract."""

    GROUNDED = "grounded"
    CROSSHAIR_BLOCK = "crosshair_block"


class PerceptionEvidence(BaseModel):
    """Content-addressed reference to the exact pixels supporting a claim.

    The pixels remain in the synchronized trajectory/frame store.  This small
    manifest is safe to keep on the realtime blackboard and lets downstream
    planners distinguish visible evidence from an unsupported model assertion.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(min_length=1, max_length=256)
    frame_id: int = Field(ge=0)
    captured_ns: int = Field(gt=0)
    region_kind: EvidenceRegion
    region: ScreenRegion
    pixel_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    crop_width: int = Field(gt=0)
    crop_height: int = Field(gt=0)


class Track(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    track_id: str
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    region: ScreenRegion
    first_seen_ns: int
    last_seen_ns: int
    attributes: dict[str, str | int | float | bool] = Field(default_factory=dict)
    evidence_refs: tuple[str, ...] = ()


class ChatLine(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    speaker: str | None = None
    observed_ns: int
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence_refs: tuple[str, ...] = ()


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
    evidence_refs: tuple[str, ...] = ()

    def fresh(self, now_ns: int | None = None) -> bool:
        now = time.monotonic_ns() if now_ns is None else now_ns
        age_ns = now - self.observed_ns
        return 0 <= age_ns <= self.expires_after_ms * 1_000_000


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
    evidence: tuple[PerceptionEvidence, ...] = ()


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
    """Thread-safe typed state store shared by fast capture and semantic workers."""

    frame_capacity: int = 90
    _frames: RingBuffer[FrameState] = field(init=False)
    _facts: dict[str, PerceptionFact] = field(default_factory=dict)
    _semantic_tracks: tuple[Track, ...] = ()
    _chat: tuple[ChatLine, ...] = ()
    _evidence: dict[str, PerceptionEvidence] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)

    def __post_init__(self) -> None:
        self._frames = RingBuffer(self.frame_capacity)

    def publish(self, frame: FrameState) -> None:
        """Publish a real captured frame; capture sequence is never VLM-generated."""
        with self._lock:
            previous = self._frames.latest()
            if previous is not None:
                if frame.instance_id != previous.instance_id:
                    raise ValueError("perception instance identity changed")
                if frame.frame_id <= previous.frame_id:
                    raise ValueError("frame ids must increase monotonically")
                if frame.captured_ns <= previous.captured_ns:
                    raise ValueError("frame timestamps must increase monotonically")
            self._validate_evidence_refs_locked(
                frame.facts,
                frame.tracks,
                frame.chat,
                additional_evidence=frame.evidence,
            )
            self._frames.append(frame)
            self._merge_evidence_locked(frame.evidence)
            self._merge_facts_locked(frame.facts)
            if frame.tracks:
                self._semantic_tracks = frame.tracks
            if frame.chat:
                self._chat = frame.chat

    def merge_semantics(
        self,
        *,
        instance_id: str,
        facts: tuple[PerceptionFact, ...] = (),
        tracks: tuple[Track, ...] = (),
        chat: tuple[ChatLine, ...] = (),
        evidence: tuple[PerceptionEvidence, ...] = (),
    ) -> bool:
        """Merge late semantic results without fabricating capture timestamps.

        Returns False when the originating instance is no longer current.
        """
        with self._lock:
            latest = self._frames.latest()
            if latest is None or latest.instance_id != instance_id:
                return False
            self._validate_evidence_refs_locked(
                facts,
                tracks,
                chat,
                additional_evidence=evidence,
            )
            self._merge_evidence_locked(evidence)
            self._merge_facts_locked(facts)
            if tracks:
                newest_track_ns = max(track.last_seen_ns for track in tracks)
                current_track_ns = max(
                    (track.last_seen_ns for track in self._semantic_tracks),
                    default=-1,
                )
                if newest_track_ns >= current_track_ns:
                    self._semantic_tracks = tracks
            if chat:
                combined = {f"{line.observed_ns}:{line.text}": line for line in self._chat}
                for line in chat:
                    combined[f"{line.observed_ns}:{line.text}"] = line
                self._chat = tuple(
                    sorted(combined.values(), key=lambda line: line.observed_ns)[-100:]
                )
            return True

    def _validate_evidence_refs_locked(
        self,
        facts: tuple[PerceptionFact, ...],
        tracks: tuple[Track, ...],
        chat: tuple[ChatLine, ...],
        *,
        additional_evidence: tuple[PerceptionEvidence, ...] = (),
    ) -> None:
        known_evidence = set(self._evidence)
        known_evidence.update(item.evidence_id for item in additional_evidence)
        dangling = {
            evidence_id
            for fact in facts
            for evidence_id in fact.evidence_refs
            if evidence_id not in known_evidence
        }
        dangling.update(
            evidence_id
            for track in tracks
            for evidence_id in track.evidence_refs
            if evidence_id not in known_evidence
        )
        dangling.update(
            evidence_id
            for line in chat
            for evidence_id in line.evidence_refs
            if evidence_id not in known_evidence
        )
        if dangling:
            raise ValueError(
                "perception observations reference unknown evidence: " + ", ".join(sorted(dangling))
            )

    def upsert_semantic_track(self, *, instance_id: str, track: Track) -> bool:
        """Add or replace one durable semantic track without discarding peers."""
        with self._lock:
            latest = self._frames.latest()
            if latest is None or latest.instance_id != instance_id:
                return False
            self._validate_evidence_refs_locked((), (track,), ())
            tracks = {existing.track_id: existing for existing in self._semantic_tracks}
            tracks[track.track_id] = track
            self._semantic_tracks = tuple(tracks.values())
            return True

    def remove_semantic_track(self, track_id: str) -> bool:
        """Remove one semantic track, returning whether it was present."""
        with self._lock:
            retained = tuple(track for track in self._semantic_tracks if track.track_id != track_id)
            removed = len(retained) != len(self._semantic_tracks)
            self._semantic_tracks = retained
            return removed

    def remove_semantic_facts(
        self,
        keys: tuple[str, ...],
        *,
        expected_source: str,
    ) -> tuple[str, ...]:
        """Remove only transaction-owned facts without touching newer producers."""

        removed: list[str] = []
        with self._lock:
            for key in keys:
                fact = self._facts.get(key)
                if fact is not None and fact.source == expected_source:
                    del self._facts[key]
                    removed.append(key)
        return tuple(removed)

    def _merge_facts_locked(self, facts: tuple[PerceptionFact, ...]) -> None:
        for fact in facts:
            existing = self._facts.get(fact.key)
            if existing is None or fact.observed_ns >= existing.observed_ns:
                self._facts[fact.key] = fact

    def _merge_evidence_locked(self, evidence: tuple[PerceptionEvidence, ...]) -> None:
        for item in evidence:
            self._evidence[item.evidence_id] = item
        # Evidence is only a compact manifest, but bound it so a long-running
        # process cannot grow without limit if semantic queries are frequent.
        if len(self._evidence) > 512:
            retained = sorted(
                self._evidence.values(),
                key=lambda item: (item.captured_ns, item.evidence_id),
                reverse=True,
            )[:512]
            self._evidence = {item.evidence_id: item for item in retained}

    def latest(self) -> FrameState | None:
        with self._lock:
            frame = self._frames.latest()
            if frame is None:
                return None
            facts = tuple(self.fresh_facts().values())
            referenced = {evidence_id for fact in facts for evidence_id in fact.evidence_refs}
            referenced.update(
                evidence_id
                for track in self._semantic_tracks
                for evidence_id in track.evidence_refs
            )
            referenced.update(
                evidence_id for line in self._chat for evidence_id in line.evidence_refs
            )
            return frame.model_copy(
                update={
                    "tracks": self._semantic_tracks or frame.tracks,
                    "chat": self._chat,
                    "facts": facts,
                    "evidence": tuple(
                        self._evidence[evidence_id]
                        for evidence_id in sorted(referenced)
                        if evidence_id in self._evidence
                    ),
                }
            )

    def raw_latest(self) -> FrameState | None:
        with self._lock:
            return self._frames.latest()

    def frames(self) -> tuple[FrameState, ...]:
        with self._lock:
            return self._frames.snapshot()

    def fact(
        self, key: str, *, min_confidence: float = 0.0, now_ns: int | None = None
    ) -> PerceptionFact | None:
        with self._lock:
            fact = self._facts.get(key)
            if fact is None or fact.confidence < min_confidence or not fact.fresh(now_ns=now_ns):
                return None
            return fact

    def fresh_facts(self, *, min_confidence: float = 0.0) -> dict[str, PerceptionFact]:
        with self._lock:
            return {
                key: fact
                for key, fact in self._facts.items()
                if fact.confidence >= min_confidence and fact.fresh()
            }


class ActivePerceptionQuery(BaseModel):
    """Narrow semantic question submitted to a slower VLM service."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_id: str
    mode: PerceptionQueryMode = PerceptionQueryMode.GROUNDED
    question: str = Field(min_length=1, max_length=1024)
    skill_id: str | None = None
    frame_id: int = Field(ge=0)
    deadline_ms: int = Field(default=750, ge=50, le=10_000)
    output_keys: tuple[str, ...] = ()
