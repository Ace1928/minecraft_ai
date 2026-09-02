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
            "Answer only JSON matching {facts:object, confidences:object, tracks:array, "
            "chat:string[]}. Coordinates are normalized 0..1. Do not infer hidden state. "
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
            ChatLine(text=text, observed_ns=now, confidence=0.7)
            for text in observation.chat
        )
        self.blackboard.merge_semantics(
            instance_id=self.instance_id,
            facts=facts,
            tracks=tracks,
            chat=chat,
        )


@dataclass
class RealtimePerceptionService:
    capture_source: CaptureSource
    blackboard: PerceptionBlackboard
    instance_id: str
    target_hz: float = 20.0
    stale_frame_ms: int = 500
    active_vlm: ActiveVLMWorker | None = None
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

        # Fast 20Hz visual perception feature extraction
        self._extract_fast_visual_features(captured)
        return state

    def _extract_fast_visual_features(self, frame: CapturedFrame) -> None:
        if not frame.bgra or frame.width <= 0 or frame.height <= 0:
            return
        now = time.monotonic_ns()
        w = frame.width
        h = frame.height
        cx = w // 2
        cy = h // 2

        # Fast sampling of center region (crosshair / block in front)
        center_samples = 0
        brown_count = 0
        gray_count = 0
        green_count = 0
        row_bytes = w * 4

        # Sample a 30x30 region at screen center
        for dy in range(-15, 15, 3):
            sy = cy + dy
            if sy < 0 or sy >= h:
                continue
            row_off = sy * row_bytes
            for dx in range(-15, 15, 3):
                sx = cx + dx
                if sx < 0 or sx >= w:
                    continue
                off = row_off + sx * 4
                b = frame.bgra[off]
                g = frame.bgra[off + 1]
                r = frame.bgra[off + 2]
                center_samples += 1
                # Wood/bark (brownish: R > B, R > G or R ~ G > B)
                if r > 60 and g > 40 and r >= g and r > b + 15:
                    brown_count += 1
                # Stone (grayish: R ~ G ~ B)
                elif abs(r - g) < 20 and abs(g - b) < 20 and r > 50 and r < 180:
                    gray_count += 1
                # Vegetation (greenish: G > R + 15 and G > B + 15)
                elif g > r + 15 and g > b + 15:
                    green_count += 1

        # Sample upper vs lower screen for sky/horizon pitch detection and day/night cycle
        sky_samples = 0
        upper_sky = 0
        lower_sky = 0
        total_sky_brightness = 0.0

        # Sample top 15% and bottom 15%
        for dx in range(w // 4, 3 * w // 4, w // 10):
            # Top sample
            off_top = (h // 8) * row_bytes + dx * 4
            b_t, g_t, r_t = frame.bgra[off_top], frame.bgra[off_top + 1], frame.bgra[off_top + 2]
            # Bottom sample
            off_bot = (7 * h // 8) * row_bytes + dx * 4
            b_b, g_b, r_b = frame.bgra[off_bot], frame.bgra[off_bot + 1], frame.bgra[off_bot + 2]
            
            sky_samples += 1
            lum_t = 0.299 * r_t + 0.587 * g_t + 0.114 * b_t
            total_sky_brightness += lum_t

            if b_t > 150 and g_t > 130 and r_t > 110: # Sky/cloud brightness
                upper_sky += 1
            if b_b > 150 and g_b > 130 and r_b > 110:
                lower_sky += 1

        looking_at_sky = bool(sky_samples > 0 and upper_sky >= (sky_samples * 0.7) and lower_sky >= (sky_samples * 0.5))
        avg_brightness = total_sky_brightness / max(1, sky_samples)
        
        # Day / Dusk / Night cycle estimation
        if avg_brightness > 135:
            time_of_day = "day"
        elif avg_brightness > 75:
            time_of_day = "dusk"
        else:
            time_of_day = "night"

        # Check for underwater immersion (cyan/blue high, red very low across screen)
        is_underwater = bool(b_t > 120 and g_t > 100 and r_t < 40 and b_b > 120 and r_b < 40)

        # Obstacle proximity check directly below crosshair (foot/block barrier)
        obstacle_samples = 0
        solid_count = 0
        for dy in range(20, 50, 5):
            sy = cy + dy
            if sy >= h:
                continue
            row_off = sy * row_bytes
            for dx in range(-10, 10, 5):
                sx = cx + dx
                if sx < 0 or sx >= w:
                    continue
                off = row_off + sx * 4
                b_o, g_o, r_o = frame.bgra[off], frame.bgra[off + 1], frame.bgra[off + 2]
                obstacle_samples += 1
                if (r_o + g_o + b_o) > 60: # Solid non-black block
                    solid_count += 1

        obstacle_ahead = bool(obstacle_samples > 0 and solid_count >= (obstacle_samples * 0.75))

        if center_samples > 0:
            target_visible = (brown_count + gray_count + green_count) > (center_samples * 0.25)
            target_mineable = (brown_count + gray_count) > (center_samples * 0.20)
            
            self.blackboard.merge_semantics(
                instance_id=self.instance_id,
                facts=(
                    PerceptionFact(
                        key="target.visible",
                        value=target_visible,
                        confidence=0.85,
                        observed_ns=now,
                        source="heuristic-vision-20hz",
                        expires_after_ms=300,
                    ),
                    PerceptionFact(
                        key="target.mineable",
                        value=target_mineable,
                        confidence=0.85,
                        observed_ns=now,
                        source="heuristic-vision-20hz",
                        expires_after_ms=300,
                    ),
                    PerceptionFact(
                        key="pitch.looking_up",
                        value=looking_at_sky,
                        confidence=0.80,
                        observed_ns=now,
                        source="heuristic-vision-20hz",
                        expires_after_ms=300,
                    ),
                    PerceptionFact(
                        key="environment.time_of_day",
                        value=time_of_day,
                        confidence=0.75,
                        observed_ns=now,
                        source="heuristic-vision-20hz",
                        expires_after_ms=5000,
                    ),
                    PerceptionFact(
                        key="environment.underwater",
                        value=is_underwater,
                        confidence=0.90,
                        observed_ns=now,
                        source="heuristic-vision-20hz",
                        expires_after_ms=500,
                    ),
                    PerceptionFact(
                        key="obstacle.ahead",
                        value=obstacle_ahead,
                        confidence=0.80,
                        observed_ns=now,
                        source="heuristic-vision-20hz",
                        expires_after_ms=300,
                    ),
                ),
            )

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
