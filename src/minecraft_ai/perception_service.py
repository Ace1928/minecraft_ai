from __future__ import annotations

import importlib
import io
import json
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from .models import VisionLanguageModel
from .perception import (
    ActivePerceptionQuery,
    ChatLine,
    FrameState,
    PerceptionBlackboard,
    PerceptionFact,
    ScreenRegion,
    Track,
)
from .platforms.bedrock_x11 import CapturedFrame


class CaptureSource(Protocol):
    def capture(self) -> CapturedFrame: ...

    def close(self) -> None: ...


class FastPerception(Protocol):
    """Realtime visual inference boundary.

    Learned implementations may set ``training_label_eligible`` only after
    passing the perception promotion gate.
    """

    @property
    def model_id(self) -> str: ...

    @property
    def training_label_eligible(self) -> bool: ...

    def infer(self, frame: CapturedFrame) -> tuple[PerceptionFact, ...]: ...


class SemanticTrack(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(gt=0.0, le=1.0)
    height: float = Field(gt=0.0, le=1.0)


class SemanticObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    facts: dict[str, str | int | float | bool] = Field(default_factory=dict)
    confidences: dict[str, float] = Field(default_factory=dict)
    tracks: tuple[SemanticTrack, ...] = ()
    chat: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticJob:
    query: ActivePerceptionQuery
    frame: CapturedFrame


@dataclass
class ActiveVLMWorker:
    model: VisionLanguageModel
    blackboard: PerceptionBlackboard
    instance_id: str
    queue_size: int = 4
    _jobs: queue.Queue[SemanticJob | None] = field(init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)

    def __post_init__(self) -> None:
        self._jobs = queue.Queue(maxsize=self.queue_size)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="minecraft-ai-vlm", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._jobs.put_nowait(None)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def submit(self, job: SemanticJob) -> bool:
        """Drop stale semantic work instead of blocking realtime capture."""
        if self._stop.is_set():
            return False
        try:
            self._jobs.put_nowait(job)
            return True
        except queue.Full:
            try:
                self._jobs.get_nowait()
            except queue.Empty:
                return False
            try:
                self._jobs.put_nowait(job)
                return True
            except queue.Full:
                return False

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            if job is None:
                return
            try:
                observation = self._inspect(job)
                self._publish(job, observation)
            except Exception:
                # Semantic VLM failure must never terminate capture or motor control.
                continue

    def _inspect(self, job: SemanticJob) -> SemanticObservation:
        png = _bgra_to_png(job.frame)
        prompt = (
            "Inspect this Minecraft Bedrock screenshot. Answer only JSON matching "
            "{facts:object, confidences:object, tracks:array, chat:string[]}. Always include "
            "scene.mode as one of world, gui, loading, menu, death, unknown; scene.playable "
            "as a boolean; and perception.uncertainty from 0 to 1. In world mode report only "
            "visible HUD state, immediate hazards, terrain affordances, and task-relevant "
            "targets. Only report target.visible=true with target.dx and target.dy in -1..1 "
            "when a target is visibly localized. Never infer hidden inventory, coordinates, "
            "seed, biome, identity, or time of day. Track coordinates are normalized 0..1. "
            f"Question: {job.query.question}"
        )
        response = self.model.inspect(prompt, image_bytes=png, mime_type="image/png")
        raw_text = _strip_code_fence(response.text)
        return SemanticObservation.model_validate(json.loads(raw_text))

    def _publish(self, job: SemanticJob, observation: SemanticObservation) -> None:
        latest = self.blackboard.raw_latest()
        if latest is None or latest.instance_id != self.instance_id:
            return
        # If a very old semantic job survived long enough to be irrelevant,
        # discard it instead of contaminating the current tactical state.
        if latest.frame_id - job.query.frame_id > 120:
            return
        now = time.monotonic_ns()
        facts = tuple(
            PerceptionFact(
                key=key,
                value=value,
                confidence=max(0.0, min(1.0, observation.confidences.get(key, 0.7))),
                observed_ns=now,
                source=f"vlm:{self.model.model_id}:{job.query.query_id}",
                expires_after_ms=max(250, job.query.deadline_ms * 3),
            )
            for key, value in observation.facts.items()
        )
        tracks = tuple(
            Track(
                track_id=f"vlm:{job.query.query_id}:{index}",
                label=item.label,
                confidence=item.confidence,
                region=ScreenRegion(
                    x=item.x,
                    y=item.y,
                    width=item.width,
                    height=item.height,
                ),
                first_seen_ns=now,
                last_seen_ns=now,
            )
            for index, item in enumerate(observation.tracks)
        )
        chat = tuple(
            ChatLine(text=text, observed_ns=now, confidence=0.7) for text in observation.chat
        )
        self.blackboard.merge_semantics(
            instance_id=self.instance_id,
            facts=facts,
            tracks=tracks,
            chat=chat,
        )


@dataclass(frozen=True)
class BootstrapFastPerception:
    """Observable image statistics for smoke tests, never semantic ground truth.

    Semantic RGB guesses were intentionally removed: color is not a reliable
    block, target, obstacle, biome, scene, or time-of-day classifier.
    """

    model_id: str = "bootstrap-rgb-v1"
    training_label_eligible: bool = False

    def infer(self, frame: CapturedFrame) -> tuple[PerceptionFact, ...]:
        if not frame.bgra or frame.width <= 0 or frame.height <= 0:
            return ()
        now = time.monotonic_ns()
        pixel_count = frame.width * frame.height
        sample_stride = max(4, pixel_count // 4096) * 4
        luma_total = 0.0
        samples = 0
        for offset in range(0, len(frame.bgra) - 3, sample_stride):
            blue, green, red = frame.bgra[offset : offset + 3]
            luma_total += 0.299 * red + 0.587 * green + 0.114 * blue
            samples += 1
        mean_luma = luma_total / max(1, samples)
        source = f"bootstrap:{self.model_id}:not-training-label"
        values: tuple[tuple[str, float | bool, float, int], ...] = (
            ("image.mean_luma", round(mean_luma / 255.0, 4), 1.0, 500),
            ("perception.bootstrap_active", True, 1.0, 500),
        )
        return tuple(
            PerceptionFact(
                key=key,
                value=value,
                confidence=confidence,
                observed_ns=now,
                source=source,
                expires_after_ms=expires_after_ms,
            )
            for key, value, confidence, expires_after_ms in values
        )


@dataclass
class RealtimePerceptionService:
    capture_source: CaptureSource
    blackboard: PerceptionBlackboard
    instance_id: str
    target_hz: float = 20.0
    stale_frame_ms: int = 500
    active_vlm: ActiveVLMWorker | None = None
    fast_perception: FastPerception | None = field(default_factory=BootstrapFastPerception)
    _last_frame_ns: int | None = field(default=None, init=False)
    _last_capture: CapturedFrame | None = field(default=None, init=False)

    @property
    def last_capture(self) -> CapturedFrame | None:
        return self._last_capture

    def capture_once(self) -> FrameState:
        captured = self.capture_source.capture()
        if self._last_frame_ns is not None and captured.captured_ns <= self._last_frame_ns:
            raise RuntimeError("capture timestamps are not monotonic")
        self._last_frame_ns = captured.captured_ns
        self._last_capture = captured
        previous = self.blackboard.raw_latest()
        frame_id = previous.frame_id + 1 if previous is not None else 0
        state = FrameState(
            frame_id=frame_id,
            captured_ns=captured.captured_ns,
            instance_id=self.instance_id,
            width=captured.width,
            height=captured.height,
        )
        self.blackboard.publish(state)

        if self.fast_perception is not None:
            facts = self.fast_perception.infer(captured)
            if facts:
                self.blackboard.merge_semantics(instance_id=self.instance_id, facts=facts)
        return state

    def request_semantics(
        self,
        query: ActivePerceptionQuery,
        frame: CapturedFrame | None = None,
    ) -> bool:
        if self.active_vlm is None:
            return False
        selected = self._last_capture if frame is None else frame
        if selected is None:
            return False
        return self.active_vlm.submit(SemanticJob(query=query, frame=selected))

    def stale(self, now_ns: int | None = None) -> bool:
        latest = self.blackboard.raw_latest()
        if latest is None:
            return True
        now = time.monotonic_ns() if now_ns is None else now_ns
        return now - latest.captured_ns > self.stale_frame_ms * 1_000_000

    def close(self) -> None:
        if self.active_vlm is not None:
            self.active_vlm.stop()
        self.capture_source.close()


def _bgra_to_png(frame: CapturedFrame) -> bytes:
    try:
        image_module = importlib.import_module("PIL.Image")
    except ImportError as exc:
        raise RuntimeError("install minecraft-ai[vision] for VLM image encoding") from exc
    image = image_module.frombytes(
        "RGBA",
        (frame.width, frame.height),
        frame.bgra,
        "raw",
        "BGRA",
    )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def _strip_code_fence(text: str) -> str:
    candidate = text.strip()
    if not candidate.startswith("```"):
        return candidate
    lines = candidate.splitlines()
    if lines:
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()
