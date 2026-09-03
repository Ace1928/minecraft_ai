from __future__ import annotations

import importlib
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field

from .grounded_perception import (
    GroundedPerceptionHarness,
    GroundedPerceptionRepairError,
    GroundedPerceptionReport,
)
from .models import VisionLanguageModel
from .perception import (
    ActivePerceptionQuery,
    ChatLine,
    FrameState,
    PerceptionBlackboard,
    PerceptionEvidence,
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
    evidence_id: str | None = Field(default=None, max_length=256)


class SemanticObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scene_mode: Literal["world", "gui", "loading", "menu", "death", "unknown"] | None = None
    scene_playable: bool | None = None
    uncertainty: float = Field(default=1.0, ge=0.0, le=1.0)
    danger_immediate: bool | None = None
    obstacle_ahead: bool | None = None
    target_visible: bool | None = None
    scene_summary: str = Field(
        default="No claims established from visible pixel evidence.",
        min_length=1,
        max_length=2048,
    )
    target_dx: float | None = Field(default=None, ge=-1.0, le=1.0)
    target_dy: float | None = Field(default=None, ge=-1.0, le=1.0)
    target_kind: str | None = Field(default=None, max_length=128)
    target_mineable: bool | None = None
    target_near: bool | None = None
    inventory_logs: int | None = Field(default=None, ge=0)
    inventory_planks: int | None = Field(default=None, ge=0)
    inventory_crafting_table: int | None = Field(default=None, ge=0)
    inventory_build_blocks: int | None = Field(default=None, ge=0)
    player_submerged: bool | None = None
    player_air_visible: bool | None = None
    facts: dict[str, str | int | float | bool] = Field(default_factory=dict)
    confidences: dict[str, float] = Field(default_factory=dict)
    evidence: tuple[PerceptionEvidence, ...] = ()
    evidence_refs: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    rejection_count: int = Field(default=0, ge=0)
    unknown_claim_count: int = Field(default=0, ge=0)
    prose_rejected: bool = False
    tracks: tuple[SemanticTrack, ...] = ()
    chat: tuple[str, ...] = ()
    chat_evidence_ids: tuple[str, ...] = ()

    def canonical_facts(self) -> dict[str, str | int | float | bool]:
        values = dict(self.facts)
        values.update(
            {
                "perception.uncertainty": self.uncertainty,
                "scene.summary": self.scene_summary,
            }
        )
        optional = {
            "scene.mode": self.scene_mode,
            "scene.playable": self.scene_playable,
            "danger.immediate": self.danger_immediate,
            "obstacle.ahead": self.obstacle_ahead,
            "target.visible": self.target_visible,
            "target.dx": self.target_dx,
            "target.dy": self.target_dy,
            "target.kind": self.target_kind,
            "target.mineable": self.target_mineable,
            "target.near": self.target_near,
            "inventory.logs": self.inventory_logs,
            "inventory.planks": self.inventory_planks,
            "inventory.crafting_table": self.inventory_crafting_table,
            "inventory.build_blocks": self.inventory_build_blocks,
            "player.submerged": self.player_submerged,
            "player.air_visible": self.player_air_visible,
        }
        values.update({key: value for key, value in optional.items() if value is not None})
        return values


def _semantic_observation(report: GroundedPerceptionReport) -> SemanticObservation:
    """Convert only validated/cited claims into the legacy runtime surface."""
    values = report.observed_values()
    mapped_keys = {
        "scene.mode",
        "scene.playable",
        "danger.immediate",
        "obstacle.ahead",
        "target.visible",
        "target.dx",
        "target.dy",
        "target.kind",
        "target.mineable",
        "target.near",
        "inventory.logs",
        "inventory.planks",
        "inventory.crafting_table",
        "inventory.build_blocks",
        "player.submerged",
        "player.air_visible",
    }
    extra_facts = {key: value for key, value in values.items() if key not in mapped_keys}
    scene_mode = cast(
        Literal["world", "gui", "loading", "menu", "death", "unknown"] | None,
        values.get("scene.mode"),
    )
    evidence_refs = report.evidence_by_key()
    summary_evidence = tuple(
        dict.fromkeys(
            evidence_id for references in evidence_refs.values() for evidence_id in references
        )
    )
    if summary_evidence:
        evidence_refs["scene.summary"] = summary_evidence
    return SemanticObservation(
        scene_mode=scene_mode,
        scene_playable=cast(bool | None, values.get("scene.playable")),
        uncertainty=report.uncertainty,
        danger_immediate=cast(bool | None, values.get("danger.immediate")),
        obstacle_ahead=cast(bool | None, values.get("obstacle.ahead")),
        target_visible=cast(bool | None, values.get("target.visible")),
        scene_summary=report.deterministic_summary,
        target_dx=cast(float | None, values.get("target.dx")),
        target_dy=cast(float | None, values.get("target.dy")),
        target_kind=cast(str | None, values.get("target.kind")),
        target_mineable=cast(bool | None, values.get("target.mineable")),
        target_near=cast(bool | None, values.get("target.near")),
        inventory_logs=cast(int | None, values.get("inventory.logs")),
        inventory_planks=cast(int | None, values.get("inventory.planks")),
        inventory_crafting_table=cast(int | None, values.get("inventory.crafting_table")),
        inventory_build_blocks=cast(int | None, values.get("inventory.build_blocks")),
        player_submerged=cast(bool | None, values.get("player.submerged")),
        player_air_visible=cast(bool | None, values.get("player.air_visible")),
        facts=extra_facts,
        confidences=report.confidence_by_key(),
        evidence=report.evidence,
        evidence_refs=evidence_refs,
        rejection_count=len(report.rejections),
        unknown_claim_count=sum(claim.status != "observed" for claim in report.claims),
        prose_rejected=bool(report.model_summary and not report.summary_accepted),
        tracks=tuple(
            SemanticTrack(
                label=item.label,
                confidence=item.confidence,
                x=item.x,
                y=item.y,
                width=item.width,
                height=item.height,
                evidence_id=item.evidence_id,
            )
            for item in report.tracks
        ),
        chat=tuple(item.text for item in report.chat),
        chat_evidence_ids=tuple(item.evidence_id for item in report.chat),
    )


@dataclass(frozen=True)
class SemanticJob:
    query: ActivePerceptionQuery
    frame: CapturedFrame
    frame_dhash: str
    ui_dhash: str = ""


@dataclass
class ActiveVLMMetrics:
    requests: int = 0
    completed: int = 0
    failures: int = 0
    queue_replacements: int = 0
    busy_rejections: int = 0
    stale_rejections: int = 0
    last_latency_ms: float = 0.0
    last_frame_age: int = 0
    last_hash_distance: int | None = None
    last_error: str | None = None
    last_fact_keys: tuple[str, ...] = ()
    claim_rejections: int = 0
    prose_rejections: int = 0
    unknown_claims: int = 0
    schema_repair_attempts: int = 0
    schema_repair_successes: int = 0
    schema_repair_failures: int = 0


@dataclass
class ActiveVLMWorker:
    model: VisionLanguageModel
    blackboard: PerceptionBlackboard
    instance_id: str
    queue_size: int = 1
    metrics: ActiveVLMMetrics = field(default_factory=ActiveVLMMetrics, init=False)
    _jobs: queue.Queue[SemanticJob | None] = field(init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _busy: threading.Event = field(default_factory=threading.Event, init=False)
    _admission_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _work_admitted: bool = field(default=False, init=False)

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
        with self._admission_lock:
            if self._stop.is_set():
                return False
            self.metrics.requests += 1
            # Admission covers both the queued and executing states. Without
            # this atomic flag, the worker can dequeue a job just before it
            # marks itself busy and a second caller can fill the queue in that
            # narrow window, reproducing the cognition-starvation bug.
            if self._work_admitted:
                self.metrics.busy_rejections += 1
                return False
            self._work_admitted = True
            try:
                self._jobs.put_nowait(job)
                return True
            except queue.Full:
                self._work_admitted = False
                return False

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            if job is None:
                return
            self._busy.set()
            try:
                observation, latency_ms = self._inspect(job)
                self.metrics.completed += 1
                self.metrics.last_latency_ms = latency_ms
                self.metrics.last_error = None
                self.metrics.last_fact_keys = tuple(sorted(observation.canonical_facts()))
                self.metrics.claim_rejections += observation.rejection_count
                self.metrics.prose_rejections += int(observation.prose_rejected)
                self.metrics.unknown_claims += observation.unknown_claim_count
                self._publish(job, observation)
            except Exception as exc:
                # Semantic VLM failure must never terminate capture or motor control.
                self.metrics.failures += 1
                self.metrics.last_error = f"{type(exc).__name__}: {exc}"
                continue
            finally:
                self._busy.clear()
                with self._admission_lock:
                    self._work_admitted = False

    def available(self) -> bool:
        """Return whether a fresh semantic job can start without queueing."""
        with self._admission_lock:
            return not self._stop.is_set() and not self._work_admitted

    def _inspect(self, job: SemanticJob) -> tuple[SemanticObservation, float]:
        try:
            result = GroundedPerceptionHarness(self.model).inspect_detailed(
                job.frame,
                frame_id=job.query.frame_id,
                question=job.query.question,
                output_keys=job.query.output_keys,
            )
        except GroundedPerceptionRepairError:
            self.metrics.schema_repair_attempts += 1
            self.metrics.schema_repair_failures += 1
            raise
        if result.schema_repaired:
            self.metrics.schema_repair_attempts += 1
            self.metrics.schema_repair_successes += 1
        return _semantic_observation(result.report), result.latency_ms

    def _publish(self, job: SemanticJob, observation: SemanticObservation) -> None:
        latest = self.blackboard.raw_latest()
        if latest is None or latest.instance_id != self.instance_id:
            return
        frame_age = latest.frame_id - job.query.frame_id
        current_hash_fact = self.blackboard.fact("frame.dhash", min_confidence=1.0)
        current_ui_hash_fact = self.blackboard.fact("frame.ui_dhash", min_confidence=1.0)
        current_hash = (
            current_hash_fact.value
            if current_hash_fact is not None and isinstance(current_hash_fact.value, str)
            else None
        )
        hash_distance = (
            perceptual_hash_distance(job.frame_dhash, current_hash)
            if current_hash is not None
            else None
        )
        current_ui_hash = (
            current_ui_hash_fact.value
            if current_ui_hash_fact is not None and isinstance(current_ui_hash_fact.value, str)
            else None
        )
        ui_hash_distance = (
            perceptual_hash_distance(job.ui_dhash, current_ui_hash)
            if job.ui_dhash and current_ui_hash is not None
            else None
        )
        self.metrics.last_frame_age = frame_age
        self.metrics.last_hash_distance = hash_distance
        # Slow semantics remain valid for an unchanged static GUI or view. A
        # numerically old result from a changed scene must not control tactics.
        if frame_age > 120 and (
            hash_distance is None
            or hash_distance > 6
            or (ui_hash_distance is not None and ui_hash_distance > 3)
        ):
            self.metrics.stale_rejections += 1
            return
        now = time.monotonic_ns()
        facts = tuple(
            PerceptionFact(
                key=key,
                value=value,
                confidence=max(0.0, min(1.0, observation.confidences.get(key, 0.7))),
                observed_ns=now,
                source=f"vlm:{self.model.model_id}:{job.query.query_id}",
                expires_after_ms=max(15_000, job.query.deadline_ms * 3),
                evidence_refs=observation.evidence_refs.get(key, ()),
            )
            for key, value in observation.canonical_facts().items()
        ) + (
            PerceptionFact(
                key="scene.observation_dhash",
                value=job.frame_dhash,
                confidence=1.0,
                observed_ns=now,
                source=f"vlm:{self.model.model_id}:{job.query.query_id}",
                expires_after_ms=120_000,
            ),
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
                evidence_refs=() if item.evidence_id is None else (item.evidence_id,),
            )
            for index, item in enumerate(observation.tracks)
        )
        chat = tuple(
            ChatLine(
                text=text,
                observed_ns=now,
                confidence=0.7,
                evidence_refs=(observation.chat_evidence_ids[index],)
                if index < len(observation.chat_evidence_ids)
                else (),
            )
            for index, text in enumerate(observation.chat)
        )
        self.blackboard.merge_semantics(
            instance_id=self.instance_id,
            facts=facts,
            tracks=tracks,
            chat=chat,
            evidence=observation.evidence,
        )

    def status(self) -> dict[str, object]:
        thread = self._thread
        return {
            "model_id": self.model.model_id,
            "thread_alive": bool(thread is not None and thread.is_alive()),
            "requests": self.metrics.requests,
            "completed": self.metrics.completed,
            "failures": self.metrics.failures,
            "queue_replacements": self.metrics.queue_replacements,
            "busy_rejections": self.metrics.busy_rejections,
            "busy": self._busy.is_set(),
            "pending_requests": self._jobs.qsize(),
            "stale_rejections": self.metrics.stale_rejections,
            "last_latency_ms": round(self.metrics.last_latency_ms, 3),
            "last_frame_age": self.metrics.last_frame_age,
            "last_hash_distance": self.metrics.last_hash_distance,
            "last_error": self.metrics.last_error,
            "last_fact_keys": list(self.metrics.last_fact_keys),
            "claim_rejections": self.metrics.claim_rejections,
            "prose_rejections": self.metrics.prose_rejections,
            "unknown_claims": self.metrics.unknown_claims,
            "schema_repair_attempts": self.metrics.schema_repair_attempts,
            "schema_repair_successes": self.metrics.schema_repair_successes,
            "schema_repair_failures": self.metrics.schema_repair_failures,
        }


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
        bootstrap_source = f"bootstrap:{self.model_id}:not-training-label"
        values: tuple[tuple[str, str | bool, float, int], ...] = (
            ("perception.bootstrap_active", True, 1.0, 500),
            ("frame.dhash", frame_dhash(frame), 1.0, 500),
            ("frame.ui_dhash", frame_region_dhash(frame, y_start=0.0, y_end=0.18), 1.0, 500),
        )
        if bedrock_ui_chrome_present(frame):
            # A deterministic negative-only interlock is appropriate here: it
            # never labels world semantics or training data, but it prevents a
            # world policy from typing/moving inside Bedrock's full-screen UI.
            values = (
                *values,
                ("scene.playable", False, 0.99, 250),
                ("scene.ui_overlay", True, 0.99, 250),
            )
        facts = [
            PerceptionFact(
                key=key,
                value=value,
                confidence=confidence,
                observed_ns=now,
                source=bootstrap_source,
                expires_after_ms=expires_after_ms,
            )
            for key, value, confidence, expires_after_ms in values
        ]
        safety_source = "safety:bedrock-hud-v1:not-training-label"
        safety_values: tuple[tuple[str, str | int | float | bool], ...] = ()
        death_screen = bedrock_death_screen_present(frame)
        ui_overlay = bedrock_ui_chrome_present(frame)
        if death_screen:
            safety_values = (
                ("scene.playable", False),
                ("scene.ui_overlay", True),
                ("scene.mode", "death"),
                ("scene.death", True),
            )
        else:
            air_bubbles = bedrock_air_bubbles(frame)
            if air_bubbles is not None:
                safety_values = (
                    ("player.air_visible", True),
                    ("player.air_bubbles", air_bubbles),
                    ("player.air_fraction", air_bubbles / 10.0),
                    ("player.submerged", True),
                    ("environment.underwater", True),
                    ("danger.immediate", True),
                    ("danger.drowning", True),
                )
            elif not ui_overlay:
                # In a playable HUD, disappearance of a previously visible air
                # meter is the observable surface transition needed to terminate
                # a learned water-escape option. Do not clear generic danger;
                # another perception source may still observe a different hazard.
                safety_values = (
                    ("player.air_visible", False),
                    ("player.submerged", False),
                    ("environment.underwater", False),
                    ("danger.drowning", False),
                )
        facts.extend(
            PerceptionFact(
                key=key,
                value=value,
                confidence=0.995,
                observed_ns=now,
                source=safety_source,
                expires_after_ms=250,
            )
            for key, value in safety_values
        )
        return tuple(facts)


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
        return self.active_vlm.submit(
            SemanticJob(
                query=query,
                frame=selected,
                frame_dhash=frame_dhash(selected),
                ui_dhash=frame_region_dhash(selected, y_start=0.0, y_end=0.18),
            )
        )

    def semantic_available(self) -> bool:
        """Return whether active perception can accept a fresh frame now."""
        return self.active_vlm is not None and self.active_vlm.available()

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


def frame_dhash(frame: CapturedFrame) -> str:
    """Return a compact image-similarity signal without assigning semantics."""
    return frame_region_dhash(frame, y_start=0.0, y_end=1.0)


def frame_region_dhash(
    frame: CapturedFrame,
    *,
    y_start: float,
    y_end: float,
) -> str:
    """Return a dHash for one normalized horizontal band of a captured frame."""
    if not frame.bgra or frame.width < 2 or frame.height < 1:
        return "0" * 16
    if not 0.0 <= y_start < y_end <= 1.0:
        raise ValueError("normalized dHash band must satisfy 0 <= start < end <= 1")
    source = memoryview(frame.bgra)
    comparisons = 0
    bit = 1
    for row in range(8):
        normalized_y = y_start + (row + 0.5) * (y_end - y_start) / 8
        y = min(frame.height - 1, int(normalized_y * frame.height))
        previous: int | None = None
        for column in range(9):
            x = min(frame.width - 1, int((column + 0.5) * frame.width / 9))
            offset = (y * frame.width + x) * 4
            blue, green, red = source[offset : offset + 3]
            luma = 29 * int(blue) + 150 * int(green) + 77 * int(red)
            if previous is not None:
                if previous > luma:
                    comparisons |= bit
                bit <<= 1
            previous = luma
    return f"{comparisons:016x}"


def bedrock_ui_chrome_present(frame: CapturedFrame) -> bool:
    """Detect Bedrock's bright full-width top UI chrome as a motor interlock.

    This intentionally has no positive world classification and is never an
    eligible training label. It only blocks input during an obvious overlay.
    """
    if not frame.bgra or frame.width < 32 or frame.height < 32:
        return False
    source = memoryview(frame.bgra)
    band_height = max(1, int(frame.height * 0.05))
    bright = 0
    sampled = 0
    step = max(1, frame.width // 256)
    for y in range(0, band_height, 2):
        for x in range(0, frame.width, step):
            offset = (y * frame.width + x) * 4
            blue, green, red = source[offset : offset + 3]
            luma = 29 * int(blue) + 150 * int(green) + 77 * int(red)
            bright += int(luma > 180 * 256)
            sampled += 1
    return sampled > 0 and bright / sampled >= 0.90


def bedrock_survival_hud_present(frame: CapturedFrame) -> bool:
    """Verify an in-world survival HUD before supervisory camera calibration.

    This is a conservative actuator interlock, not semantic perception or a
    training label. Requiring both the red heart bank and neutral hotbar frame
    prevents calibration motion from being emitted over menus or loading UI.
    """
    if not frame.bgra or frame.width < 320 or frame.height < 180:
        return False
    pixels = _numpy_bgra(frame)
    heart_ratio = _hud_palette_ratio(
        frame,
        x_start=0.29,
        x_end=0.49,
        y_start=0.82,
        y_end=0.91,
        palette="heart",
        pixels=pixels,
    )
    hotbar_ratio = _hud_palette_ratio(
        frame,
        x_start=0.28,
        x_end=0.72,
        y_start=0.89,
        y_end=1.0,
        palette="hotbar",
        pixels=pixels,
    )
    return heart_ratio >= 0.02 and hotbar_ratio >= 0.03


def _hud_palette_ratio(
    frame: CapturedFrame,
    *,
    x_start: float,
    x_end: float,
    y_start: float,
    y_end: float,
    palette: Literal["heart", "hotbar"],
    pixels: Any | None,
) -> float:
    x0, x1 = int(frame.width * x_start), int(frame.width * x_end)
    y0, y1 = int(frame.height * y_start), int(frame.height * y_end)
    if pixels is not None:
        roi = pixels[y0:y1, x0:x1, :3]
        blue, green, red = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]
        if palette == "heart":
            selected = (red >= 175) & (red >= green * 1.35) & (red >= blue * 1.25)
        else:
            high = roi.max(axis=2).astype("int16")
            low = roi.min(axis=2).astype("int16")
            selected = (high - low <= 18) & (red >= 85) & (red <= 235)
        return float(selected.mean()) if selected.size else 0.0
    source = memoryview(frame.bgra)
    matched = 0
    sampled = 0
    for y in range(y0, y1):
        for x in range(x0, x1):
            offset = (y * frame.width + x) * 4
            blue, green, red = (int(value) for value in source[offset : offset + 3])
            if palette == "heart":
                selected = red >= 175 and red >= green * 1.35 and red >= blue * 1.25
            else:
                selected = max(blue, green, red) - min(blue, green, red) <= 18 and (
                    85 <= red <= 235
                )
            matched += int(selected)
            sampled += 1
    return matched / sampled if sampled else 0.0


def bedrock_air_bubbles(frame: CapturedFrame) -> int | None:
    """Read Bedrock's rendered air HUD without inferring world semantics.

    The palette and normalized HUD band were calibrated against raw 1279x635
    trajectory frames. Filled bubbles contribute 64 matching pixels at that
    resolution; scaling the area keeps the count stable at other resolutions.
    This is a safety observation and is never eligible as a training label.
    """
    if not frame.bgra or frame.width < 64 or frame.height < 64:
        return None
    x_start = int(frame.width * 0.50)
    x_end = int(frame.width * 0.66)
    y_start = int(frame.height * 0.90)
    y_end = int(frame.height * 0.97)
    pixels = _numpy_bgra(frame)
    if pixels is not None:
        roi = pixels[y_start:y_end, x_start:x_end, :3]
        blue, green, red = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]
        matched = int(((blue >= 230) & (green >= 110) & (green <= 210) & (red <= 100)).sum())
    else:
        source = memoryview(frame.bgra)
        matched = 0
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                offset = (y * frame.width + x) * 4
                blue, green, red = source[offset : offset + 3]
                matched += int(int(blue) >= 230 and 110 <= int(green) <= 210 and int(red) <= 100)
    area_scale = (frame.width / 1279.0) * (frame.height / 635.0)
    if matched < max(8, round(32 * area_scale)):
        return None
    pixels_per_bubble = max(1.0, 64.0 * area_scale)
    return max(1, min(10, round(matched / pixels_per_bubble)))


def bedrock_death_screen_present(frame: CapturedFrame) -> bool:
    """Detect the paired primary/secondary controls on Bedrock's death screen.

    Requiring both the green primary control and gray secondary control makes
    this a conservative negative-only scene interlock. It does not identify a
    clickable point or emit an action, and it is not a training label.
    """
    if not frame.bgra or frame.width < 64 or frame.height < 64:
        return False
    pixels = _numpy_bgra(frame)
    primary_ratio = _region_palette_ratio(
        frame,
        x_start=0.39,
        x_end=0.61,
        y_start=0.77,
        y_end=0.81,
        palette="primary",
        pixels=pixels,
    )
    secondary_ratio = _region_palette_ratio(
        frame,
        x_start=0.39,
        x_end=0.61,
        y_start=0.85,
        y_end=0.89,
        palette="secondary",
        pixels=pixels,
    )
    return primary_ratio >= 0.70 and secondary_ratio >= 0.75


def _region_palette_ratio(
    frame: CapturedFrame,
    *,
    x_start: float,
    x_end: float,
    y_start: float,
    y_end: float,
    palette: Literal["primary", "secondary"],
    pixels: Any | None = None,
) -> float:
    x0, x1 = int(frame.width * x_start), int(frame.width * x_end)
    y0, y1 = int(frame.height * y_start), int(frame.height * y_end)
    if pixels is not None:
        roi = pixels[y0:y1, x0:x1, :3]
        blue, green, red = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]
        if palette == "primary":
            selected = (green >= 90) & (green >= red * 1.18) & (green >= blue * 1.05)
        else:
            high = roi.max(axis=2).astype("int16")
            low = roi.min(axis=2).astype("int16")
            selected = (high - low <= 18) & (red >= 100) & (red <= 230)
        return float(selected.mean()) if selected.size else 0.0
    source = memoryview(frame.bgra)
    matched = 0
    sampled = 0
    for y in range(y0, y1):
        for x in range(x0, x1):
            offset = (y * frame.width + x) * 4
            blue, green, red = (int(value) for value in source[offset : offset + 3])
            if palette == "primary":
                selected = green >= 90 and green >= red * 1.18 and green >= blue * 1.05
            else:
                selected = max(red, green, blue) - min(red, green, blue) <= 18 and 100 <= red <= 230
            matched += int(selected)
            sampled += 1
    return matched / sampled if sampled else 0.0


def _numpy_bgra(frame: CapturedFrame) -> Any | None:
    try:
        numpy = importlib.import_module("numpy")
    except ImportError:
        return None
    return numpy.frombuffer(frame.bgra, dtype=numpy.uint8).reshape(
        frame.height,
        frame.width,
        4,
    )


def perceptual_hash_distance(first: str, second: str) -> int:
    if len(first) != 16 or len(second) != 16:
        raise ValueError("frame dHash values must contain exactly 16 hexadecimal characters")
    try:
        return (int(first, 16) ^ int(second, 16)).bit_count()
    except ValueError as exc:
        raise ValueError("frame dHash values must be hexadecimal") from exc
