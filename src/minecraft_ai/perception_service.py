from __future__ import annotations

import importlib
import math
import queue
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field

from .grounded_perception import (
    CROSSHAIR_BLOCK_FAST_SOURCE,
    GroundedPerceptionHarness,
    GroundedPerceptionRepairError,
    GroundedPerceptionReport,
    crosshair_block_region,
    crosshair_block_rgb_grid,
    crosshair_block_rgb_grid_distance,
    crosshair_block_visually_equivalent,
)
from .models import VisionLanguageModel
from .perception import (
    ActivePerceptionQuery,
    ChatLine,
    FrameState,
    PerceptionBlackboard,
    PerceptionEvidence,
    PerceptionFact,
    PerceptionQueryMode,
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


BEDROCK_INVENTORY_ZERO_SOURCE = (
    "deterministic:bedrock-1.26.45.1-classic-inventory-v1:not-training-label"
)
BEDROCK_HOTBAR_LOG_COUNT_SOURCE = (
    "deterministic:bedrock-1.26.45.1-classic-hud-hotbar-oak-logs-v2:not-training-label"
)
BEDROCK_HUD_SAFETY_SOURCE = "safety:bedrock-hud-v1:not-training-label"


@dataclass(frozen=True)
class BedrockInventorySlotObservation:
    """Fail-closed result from the calibrated classic survival inventory grid."""

    occupied_slots: tuple[str, ...]
    known_non_wood_slots: tuple[str, ...]

    @property
    def wood_absence_certified(self) -> bool:
        return self.occupied_slots == self.known_non_wood_slots


class SemanticTrack(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(gt=0.0, le=1.0)
    height: float = Field(gt=0.0, le=1.0)
    evidence_id: str | None = Field(default=None, max_length=256)
    attributes: dict[str, str | int | float | bool] = Field(default_factory=dict)


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
    crosshair_block_dhash: str = ""
    crosshair_block_rgb_grid: str = ""


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
    last_crosshair_hash_distance: int | None = None
    last_crosshair_rgb_distance: float | None = None
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
            harness = GroundedPerceptionHarness(self.model)
            if job.query.mode == PerceptionQueryMode.CROSSHAIR_BLOCK:
                result = harness.inspect_crosshair_block_detailed(
                    job.frame,
                    frame_id=job.query.frame_id,
                )
            else:
                result = harness.inspect_detailed(
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
        current_crosshair_hash_fact = self.blackboard.fact(
            "frame.crosshair_block_dhash", min_confidence=1.0
        )
        current_crosshair_hash = (
            current_crosshair_hash_fact.value
            if current_crosshair_hash_fact is not None
            and isinstance(current_crosshair_hash_fact.value, str)
            else None
        )
        crosshair_hash_distance = (
            perceptual_hash_distance(job.crosshair_block_dhash, current_crosshair_hash)
            if job.crosshair_block_dhash and current_crosshair_hash is not None
            else None
        )
        current_crosshair_rgb_fact = self.blackboard.fact(
            "frame.crosshair_block_rgb_grid", min_confidence=1.0
        )
        current_crosshair_rgb_grid = (
            current_crosshair_rgb_fact.value
            if current_crosshair_rgb_fact is not None
            and isinstance(current_crosshair_rgb_fact.value, str)
            else None
        )
        measured_crosshair_rgb_distance = (
            crosshair_block_rgb_grid_distance(
                job.crosshair_block_rgb_grid,
                current_crosshair_rgb_grid,
            )
            if job.crosshair_block_rgb_grid and current_crosshair_rgb_grid is not None
            else None
        )
        crosshair_rgb_distance = (
            measured_crosshair_rgb_distance
            if measured_crosshair_rgb_distance is not None
            and math.isfinite(measured_crosshair_rgb_distance)
            else None
        )
        self.metrics.last_frame_age = frame_age
        self.metrics.last_hash_distance = hash_distance
        self.metrics.last_crosshair_hash_distance = crosshair_hash_distance
        self.metrics.last_crosshair_rgb_distance = crosshair_rgb_distance
        # Slow semantics remain valid for an unchanged static GUI or view. A
        # numerically old result from a changed scene must not control tactics.
        inventory_scoped = bool(
            job.query.skill_id == "craft_wood_planks"
            or any(key.startswith("inventory.") for key in job.query.output_keys)
        )
        crosshair_block_scoped = job.query.mode == PerceptionQueryMode.CROSSHAIR_BLOCK
        overlay = self.blackboard.fact(
            "scene.inventory_overlay",
            min_confidence=0.99,
        )
        inventory_still_visible = bool(
            overlay is not None
            and overlay.value is True
            and overlay.source.startswith("safety:bedrock-hud-v1:")
        )
        stale = bool(
            (
                crosshair_block_scoped
                and not (
                    current_crosshair_hash is not None
                    and current_crosshair_rgb_grid is not None
                    and crosshair_block_visually_equivalent(
                        job.crosshair_block_dhash,
                        current_crosshair_hash,
                        job.crosshair_block_rgb_grid,
                        current_crosshair_rgb_grid,
                    )
                )
            )
            or (
                not crosshair_block_scoped
                and inventory_scoped
                and (
                    hash_distance is None
                    or hash_distance > 6
                    or not inventory_still_visible
                )
            )
            or (
                not crosshair_block_scoped
                and not inventory_scoped
                and frame_age > 120
                and (
                    hash_distance is None
                    or hash_distance > 6
                    or (ui_hash_distance is not None and ui_hash_distance > 3)
                )
            )
        )
        if stale:
            self.metrics.stale_rejections += 1
            return
        now = time.monotonic_ns()
        source = f"vlm:{self.model.model_id}:{job.query.query_id}"
        fact_values = (
            {
                key: value
                for key, value in observation.facts.items()
                if key == "recovery.crosshair.block"
            }
            if crosshair_block_scoped
            else observation.canonical_facts()
        )
        facts = tuple(
            PerceptionFact(
                key=key,
                value=value,
                confidence=max(0.0, min(1.0, observation.confidences.get(key, 0.7))),
                observed_ns=now,
                source=source,
                expires_after_ms=max(15_000, job.query.deadline_ms * 3),
                evidence_refs=observation.evidence_refs.get(key, ()),
            )
            for key, value in fact_values.items()
        )
        hash_values = (
            (
                ("recovery.crosshair.frame_dhash", job.frame_dhash),
                ("recovery.crosshair.observation_dhash", job.crosshair_block_dhash),
            )
            if crosshair_block_scoped
            else (("scene.observation_dhash", job.frame_dhash),)
        )
        facts += tuple(
            PerceptionFact(
                key=key,
                value=value,
                confidence=1.0,
                observed_ns=now,
                source=source,
                expires_after_ms=120_000,
            )
            for key, value in hash_values
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
                attributes=item.attributes,
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
            "last_crosshair_hash_distance": self.metrics.last_crosshair_hash_distance,
            "last_crosshair_rgb_distance": self.metrics.last_crosshair_rgb_distance,
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
    _hotbar_geometry_cache: dict[tuple[int, int], tuple[int, int]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def infer(self, frame: CapturedFrame) -> tuple[PerceptionFact, ...]:
        if not frame.bgra or frame.width <= 0 or frame.height <= 0:
            return ()
        now = time.monotonic_ns()
        inventory_overlay = bedrock_inventory_overlay_present(frame)
        ui_overlay = (
            inventory_overlay
            or bedrock_away_overlay_present(frame)
            or _bedrock_top_ui_chrome_present(frame)
        )
        bootstrap_source = (
            CROSSHAIR_BLOCK_FAST_SOURCE
            if self.model_id == "bootstrap-rgb-v1"
            else f"bootstrap:{self.model_id}:not-training-label"
        )
        values: tuple[tuple[str, str | bool, float, int], ...] = (
            ("perception.bootstrap_active", True, 1.0, 500),
            ("frame.dhash", frame_dhash(frame), 1.0, 500),
            ("frame.ui_dhash", frame_region_dhash(frame, y_start=0.0, y_end=0.18), 1.0, 500),
            (
                "frame.crosshair_dhash",
                frame_region_dhash(
                    frame,
                    x_start=0.38,
                    x_end=0.62,
                    y_start=0.34,
                    y_end=0.66,
                ),
                1.0,
                500,
            ),
            (
                "frame.crosshair_luma_grid",
                frame_region_luma_grid(
                    frame,
                    x_start=0.38,
                    x_end=0.62,
                    y_start=0.34,
                    y_end=0.66,
                ),
                1.0,
                500,
            ),
            (
                "frame.crosshair_block_dhash",
                crosshair_block_dhash(frame),
                1.0,
                500,
            ),
            (
                "frame.crosshair_block_rgb_grid",
                crosshair_block_rgb_grid(frame),
                1.0,
                500,
            ),
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
        safety_source = BEDROCK_HUD_SAFETY_SOURCE
        safety_values: tuple[tuple[str, str | int | float | bool], ...] = ()
        death_screen = bedrock_death_screen_present(frame)
        in_world_hud = False
        if death_screen:
            safety_values = (
                ("scene.playable", False),
                ("scene.ui_overlay", True),
                ("scene.mode", "death"),
                ("scene.death", True),
            )
        elif ui_overlay:
            # This is a deterministic, negative-only actuator interlock. It
            # does not name the screen or create a training label; it only
            # establishes that world controls are unsafe on the current frame.
            safety_values = (
                ("scene.playable", False),
                ("scene.ui_overlay", True),
            )
            if inventory_overlay:
                safety_values = (*safety_values, ("scene.inventory_overlay", True))
        else:
            in_world_hud = bedrock_in_world_hud_present(frame)
            if in_world_hud:
                # Unlike generic color/texture heuristics, Bedrock's calibrated
                # survival/creative HUD is direct fast evidence that a GUI
                # toggle returned to the playable world.
                safety_values = (
                    ("scene.playable", True),
                    ("scene.ui_overlay", False),
                    ("scene.mode", "world"),
                )
            air_bubbles = bedrock_air_bubbles(frame)
            if air_bubbles is not None:
                safety_values = (
                    *safety_values,
                    ("player.air_visible", True),
                    ("player.air_bubbles", air_bubbles),
                    ("player.air_fraction", air_bubbles / 10.0),
                    ("player.submerged", True),
                    ("environment.underwater", True),
                    ("danger.immediate", True),
                    ("danger.drowning", True),
                )
            else:
                # In a playable HUD, disappearance of a previously visible air
                # meter is the observable surface transition needed to terminate
                # a learned water-escape option. Do not clear generic danger;
                # another perception source may still observe a different hazard.
                safety_values = (
                    *safety_values,
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
        if not ui_overlay and in_world_hud:
            pixels = _numpy_bgra(frame)
            geometry = self._hotbar_geometry_cache.get((frame.width, frame.height))
            if geometry is not None and not _classic_hotbar_geometry_matches(
                pixels,
                geometry,
            ):
                self._hotbar_geometry_cache.pop((frame.width, frame.height), None)
                geometry = None
            if geometry is None and pixels is not None:
                geometry = _classic_hotbar_geometry(pixels)
                if geometry is not None:
                    self._hotbar_geometry_cache[(frame.width, frame.height)] = geometry
            hotbar_logs = (
                None
                if pixels is None or geometry is None
                else _classic_hotbar_log_count(pixels, geometry=geometry)
            )
            if hotbar_logs is not None:
                facts.append(
                    PerceptionFact(
                        key="inventory.hotbar.logs",
                        value=hotbar_logs,
                        confidence=0.995,
                        # Bind possession to these captured pixels, not the
                        # later inference clock used by diagnostic facts.
                        observed_ns=frame.captured_ns,
                        source=BEDROCK_HOTBAR_LOG_COUNT_SOURCE,
                        expires_after_ms=250,
                    )
                )
        if inventory_overlay:
            inventory_slots = _bedrock_inventory_slot_observation(frame)
            if inventory_slots is not None and inventory_slots.wood_absence_certified:
                facts.extend(
                    PerceptionFact(
                        key=key,
                        value=0,
                        confidence=0.995,
                        observed_ns=now,
                        source=BEDROCK_INVENTORY_ZERO_SOURCE,
                        expires_after_ms=250,
                    )
                    for key in ("inventory.logs", "inventory.planks")
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
                crosshair_block_dhash=crosshair_block_dhash(selected),
                crosshair_block_rgb_grid=crosshair_block_rgb_grid(selected),
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


def crosshair_block_dhash(frame: CapturedFrame) -> str:
    """Hash the exact recovery classifier crop, allowing unrelated world animation."""

    region = crosshair_block_region(frame.width, frame.height)
    return frame_region_dhash(
        frame,
        x_start=region.x,
        x_end=region.x + region.width,
        y_start=region.y,
        y_end=region.y + region.height,
    )


def frame_region_dhash(
    frame: CapturedFrame,
    *,
    x_start: float = 0.0,
    x_end: float = 1.0,
    y_start: float,
    y_end: float,
) -> str:
    """Return a dHash for one normalized rectangle of a captured frame."""
    if not frame.bgra or frame.width < 2 or frame.height < 1:
        return "0" * 16
    if not 0.0 <= x_start < x_end <= 1.0 or not 0.0 <= y_start < y_end <= 1.0:
        raise ValueError("normalized dHash rectangle must satisfy 0 <= start < end <= 1")
    source = memoryview(frame.bgra)
    comparisons = 0
    bit = 1
    for row in range(8):
        normalized_y = y_start + (row + 0.5) * (y_end - y_start) / 8
        y = min(frame.height - 1, int(normalized_y * frame.height))
        previous: int | None = None
        for column in range(9):
            normalized_x = x_start + (column + 0.5) * (x_end - x_start) / 9
            x = min(frame.width - 1, int(normalized_x * frame.width))
            offset = (y * frame.width + x) * 4
            blue, green, red = source[offset : offset + 3]
            luma = 29 * int(blue) + 150 * int(green) + 77 * int(red)
            if previous is not None:
                if previous > luma:
                    comparisons |= bit
                bit <<= 1
            previous = luma
    return f"{comparisons:016x}"


def frame_region_luma_grid(
    frame: CapturedFrame,
    *,
    x_start: float = 0.0,
    x_end: float = 1.0,
    y_start: float,
    y_end: float,
) -> str:
    """Return an 8x8 absolute-luma grid for one normalized frame region.

    dHash intentionally discards absolute brightness.  This companion signal
    lets temporal outcome checks distinguish a crack animation that clears
    back to the same block from stable replacement pixels after a block break.
    """
    if not frame.bgra or frame.width < 1 or frame.height < 1:
        return "0" * 128
    if not 0.0 <= x_start < x_end <= 1.0 or not 0.0 <= y_start < y_end <= 1.0:
        raise ValueError("normalized luma rectangle must satisfy 0 <= start < end <= 1")
    source = memoryview(frame.bgra)
    samples = bytearray()
    for row in range(8):
        normalized_y = y_start + (row + 0.5) * (y_end - y_start) / 8
        y = min(frame.height - 1, int(normalized_y * frame.height))
        for column in range(8):
            normalized_x = x_start + (column + 0.5) * (x_end - x_start) / 8
            x = min(frame.width - 1, int(normalized_x * frame.width))
            offset = (y * frame.width + x) * 4
            blue, green, red = source[offset : offset + 3]
            samples.append(
                (29 * int(blue) + 150 * int(green) + 77 * int(red) + 128) // 256
            )
    return samples.hex()


def bedrock_ui_chrome_present(frame: CapturedFrame) -> bool:
    """Detect conservative Bedrock UI chrome as a motor interlock.

    This intentionally has no positive world classification and is never an
    eligible training label. It only blocks input during an obvious overlay.
    """
    return (
        _bedrock_top_ui_chrome_present(frame)
        or bedrock_inventory_overlay_present(frame)
        or bedrock_away_overlay_present(frame)
    )


def _bedrock_top_ui_chrome_present(frame: CapturedFrame) -> bool:
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
    return bool(sampled > 0 and bright / sampled >= 0.90)


# This deliberately narrow hotbar reader is pinned to Bedrock 1.26.45.1's
# classic HUD at the installed 4-pixel UI scale. Its oak-log appearance was
# calibrated from full-frame SHA-256
# 8030dcaadb885e78249874bd349f674f54e28c3afb54168f8093a961a207fcf2;
# the corresponding vanilla side/top textures hash to
# 7abc3068a65c71784b14562ba254e3420c6d6301c8eb33d29fd25084f7e95c86 and
# b955d6f267fc9a5b34aaa6ebae0db54b0bad45b4d46c2dbc80b904e79e3347d4.
# Dirt negatives and digit calibration include full-frame hashes
# b52d54b4304b36139734409cb9d332f09dd01be1fa97de4d9b7bfdd50719bed1
# and 78269c29397779015f18001ac0199b64cd4fc4b99fe4374093a49bac2a8a3edd.
# The 0-9 bitmaps are the first five ink columns and seven ink rows from the
# installed version's ``font/default8.png`` (SHA-256
# 8fb37d2fc1e61ab0cf4bdbb315435bce1da425ea8471a0b2ca815d5a7b74e193).
# Counts use that fixed game glyph atlas and layout; this is not general OCR.
# Only compact statistics live in source; operator screenshots remain outside
# the repository and are never treated as training labels.
_CLASSIC_HOTBAR_RAIL_MIN_LENGTH = 628
_CLASSIC_HOTBAR_RAIL_MAX_LENGTH = 636
_CLASSIC_HOTBAR_FIRST_SLOT_FROM_RAIL = 94
_CLASSIC_HOTBAR_SLOT_PITCH = 80
# Opaque perimeter cells of vanilla selected_hotbar_slot.png, SHA-256
# 2322d7bcd5897e324ff56d0df840aa9ff027160a6cd58c90633cfb9331676286.
# The installed 24x24 asset matches the selected fourth slot exactly at 4x
# scale in raw-frame SHA-256
# 20d6ed9445f6659324c03e4fb6a52dc43e17b6ce27466e2893900d43343b74c2.
# Order: top/bottom inner rows, then left/right inner columns (corners once).
_CLASSIC_HOTBAR_SELECTED_RGB = bytes.fromhex(
    "fffffffdfdfdf6f6f6f7f9f7ddf0d8fcfefcd5e8d0d5e8d0cee1c9d4e7cfd5e8d0daedd5d5e8d0d5e8d0"
    "d8ebd3d5e8d0cde0c8d5e8d0d5e8d0d5e8d0d5e8d0a1b29da1b29d5f6d5c5f6d5c647261647261596756"
    "5c6a595f6d5c5f6d5c6674636775645f6d5c5f6d5c5f6d5c5f6d5c5e6c5b6674635f6d5c6e7c6b5f6d5c"
    "566453606e5dfcfcfcf7f9f7ffffffd5e8d0d5e8d0d5e8d0caddc5d5e8d0d6e9d1d5e8d0daedd5d5e8d0"
    "dcefd7d5e8d0dcefd7caddc5d3e6ced5e8d0d6e9d1d5e8d05e6c5b5f6d5c5f6d5c6674636573625f6d5c"
    "5f6d5c5f6d5c5664536876655a68575f6d5c5f6d5c586655606e5d6573625f6d5c5563525f6d5c5f6d5c"
)
_CLASSIC_HOTBAR_LOG_RGB_5X13 = bytes.fromhex(
    "383d19685834937648ac8a56b28f58a98953a1824ba5854fad8f56a3844f91744563562f3e3f1c"
    "625030977d4d9c7c47b3915aa1814ba4854ea5864fa3824ba68750b3915ba98751a4855377623c"
    "4a3921524124795f389a7e4bac8c55ae8c57a5834ca78751aa8a53a483508066404a3a20392c19"
    "3a2c1a46371f4d3b214c3b2079603a9c7f4dac8c58a586527f653e4e3c232c220f33291b372b1c"
    "4939204a391f47372242341d524127584424715c39503b21322616392c1b342718322719342719"
)
_CLASSIC_HOTBAR_DIRT_RGB_5X13 = bytes.fromhex(
    "32261059432897745874563c92694779543b815b3f996e4f7e583c906844886b50735539423216"
    "8f6a4b7c5b3f8e6547855f4384654f8862427a573c7e573a8f6b4d7d583d785338906748815d42"
    "63442b73543d87603f845c3d8863458a63438e66449c72538162468a6247856043583d284a3424"
    "5c43316549325f48366a4d357f5c41795538865f4089614674543c5f48344431213e2d1e423123"
    "5c422d60422b60432e74533963432b75543b7c5b40654a32412c1d4434294a36283e2d1f402d1e"
)
_CLASSIC_HOTBAR_DIGITS_5X7 = {
    0: (".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."),
    1: ("..#..", ".##..", "..#..", "..#..", "..#..", "..#..", "#####"),
    2: (".###.", "#...#", "....#", "..##.", ".#...", "#...#", "#####"),
    3: (".###.", "#...#", "....#", "..##.", "....#", "#...#", ".###."),
    4: ("...##", "..#.#", ".#..#", "#...#", "#####", "....#", "....#"),
    5: ("#####", "#....", "####.", "....#", "....#", "#...#", ".###."),
    6: ("..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###."),
    7: ("#####", "#...#", "....#", "...#.", "..#..", "..#..", "..#.."),
    8: (".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."),
    9: (".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##.."),
}
_CLASSIC_HOTBAR_MAX_VERIFIED_COUNT = 16


def bedrock_hotbar_log_count(frame: CapturedFrame) -> int | None:
    """Read exact visible oak-log stacks from the pinned classic hotbar.

    This is a template observer, not OCR and not a generic item classifier.
    Every slot must be a calibrated oak log, calibrated dirt, or certified empty;
    unknown items (including other log species) make the entire count unknown.
    A recognized log needs either no count glyph (one) or a pinned font match
    from 2 through 16. This counts only the visible hotbar, never hidden inventory. Any UI
    scale, geometry, icon, or count uncertainty returns ``None``.
    """
    if not frame.bgra or frame.width < 1280 or frame.height < 700:
        return None
    if not bedrock_in_world_hud_present(frame):
        return None
    pixels = _numpy_bgra(frame)
    if pixels is None:
        return None
    geometry = _classic_hotbar_geometry(pixels)
    if geometry is None:
        return None
    return _classic_hotbar_log_count(pixels, geometry=geometry)


def _classic_hotbar_log_count(
    pixels: Any,
    *,
    geometry: tuple[int, int],
) -> int | None:
    first_slot_x, rail_y = geometry
    total = 0
    for slot in range(9):
        slot_x = first_slot_x + slot * _CLASSIC_HOTBAR_SLOT_PITCH
        classification = _classic_hotbar_slot_kind(pixels, slot_x=slot_x, rail_y=rail_y)
        if classification == "ambiguous":
            return None
        if classification != "log":
            continue
        count = _classic_hotbar_stack_count(pixels, slot_x=slot_x, rail_y=rail_y)
        if count is None:
            return None
        total += count
    return total


def _classic_hotbar_geometry_matches(
    pixels: Any | None,
    geometry: tuple[int, int],
) -> bool:
    """Cheaply validate cached rail geometry against the current frame."""

    if pixels is None:
        return False
    numpy = importlib.import_module("numpy")
    first_slot_x, rail_y = geometry
    height, width = pixels.shape[:2]
    rail_x = first_slot_x + _CLASSIC_HOTBAR_FIRST_SLOT_FROM_RAIL
    rail_width = (_CLASSIC_HOTBAR_SLOT_PITCH * 8) - 8
    if (
        rail_x < 1
        or rail_y < 1
        or rail_y + 4 > height
        or rail_x + rail_width >= width
    ):
        return False
    rail = pixels[rail_y : rail_y + 4, rail_x : rail_x + rail_width, :3].astype(
        numpy.int16
    )
    high = rail.max(axis=2)
    low = rail.min(axis=2)
    neutral = (high - low <= 18) & (rail[:, :, 2] >= 85) & (rail[:, :, 2] <= 235)
    before = pixels[rail_y : rail_y + 4, rail_x - 1, :3].astype(numpy.int16)
    after = pixels[rail_y : rail_y + 4, rail_x + rail_width, :3].astype(numpy.int16)
    ordinary_rail = bool(
        float(neutral.mean()) >= 0.98
        and not _classic_hotbar_neutral_column(before)
        and not _classic_hotbar_neutral_column(after)
        and _classic_hotbar_slot_dividers_match(pixels, geometry)
    )
    return ordinary_rail or _classic_hotbar_selected_geometry_matches(pixels, geometry)


@lru_cache(maxsize=1)
def _classic_hotbar_selected_template() -> tuple[Any, Any, Any]:
    numpy = importlib.import_module("numpy")
    points = (
        [(1, x) for x in range(1, 23)]
        + [(22, x) for x in range(1, 23)]
        + [(y, 1) for y in range(2, 22)]
        + [(y, 22) for y in range(2, 22)]
    )
    rows = numpy.asarray([y * 4 for y, _x in points])[:, None, None]
    columns = numpy.asarray([x * 4 for _y, x in points])[:, None, None]
    rows = rows + numpy.arange(4)[None, :, None]
    columns = columns + numpy.arange(4)[None, None, :]
    rgb = numpy.frombuffer(_CLASSIC_HOTBAR_SELECTED_RGB, dtype=numpy.uint8)
    return rows, columns, rgb.reshape(84, 1, 1, 3).astype(numpy.int16)


def _classic_hotbar_selected_geometry_matches(
    pixels: Any,
    geometry: tuple[int, int],
) -> bool:
    """Validate the pinned selection frame where it interrupts normal slot rails."""

    numpy = importlib.import_module("numpy")
    first_slot_x, rail_y = geometry
    height, width = pixels.shape[:2]
    if first_slot_x < 2 or rail_y < 8 or first_slot_x + 734 > width or rail_y + 84 > height:
        return False
    rows, columns, reference = _classic_hotbar_selected_template()
    for selected in range(9):
        selection_x = first_slot_x - 2 + selected * _CLASSIC_HOTBAR_SLOT_PITCH
        patch = pixels[rail_y - 8 + rows, selection_x + columns, :3]
        difference = numpy.abs(patch[..., ::-1].astype(numpy.int16) - reference)
        if int(difference.max()) > 2:
            continue
        rail_x = first_slot_x + 6
        rail = pixels[rail_y : rail_y + 4, rail_x : rail_x + 720, :3].astype(numpy.int16)
        outside = (
            (numpy.arange(720) + rail_x < selection_x)
            | (numpy.arange(720) + rail_x >= selection_x + 96)
        )
        rail = rail[:, outside]
        neutral = (
            (rail.max(axis=2) - rail.min(axis=2) <= 18)
            & (rail[:, :, 2] >= 85)
            & (rail[:, :, 2] <= 235)
        )
        if float(neutral.mean()) >= 0.98 and _classic_hotbar_slot_dividers_match(
            pixels, geometry, selected_slot=selected,
        ):
            return True
    return False


def _classic_hotbar_slot_dividers_match(
    pixels: Any,
    geometry: tuple[int, int],
    *,
    selected_slot: int = 0,
) -> bool:
    """A horizontal world stripe is not a HUD: require its vertical slot grid."""
    numpy = importlib.import_module("numpy")
    first_slot_x, rail_y = geometry
    # The selected slot replaces its neighboring dividers. Every unaffected
    # divider must retain the pinned four-pixel neutral inner rail.
    y0, y1 = rail_y + 12, rail_y + 54
    if y1 > pixels.shape[0]:
        return False
    indices = numpy.asarray([
        index for index in range(2, 9) if index not in {selected_slot, selected_slot + 1}
    ])
    columns = first_slot_x + indices * _CLASSIC_HOTBAR_SLOT_PITCH
    columns = (columns[:, None] + numpy.arange(4)).reshape(-1)
    dividers = pixels[y0:y1, columns, :3].astype(numpy.int16).reshape(42, len(indices), 4, 3)
    high = dividers.max(axis=3)
    low = dividers.min(axis=3)
    neutral = (high - low <= 22) & (dividers[:, :, :, 2] >= 70)
    neutral &= dividers[:, :, :, 2] <= 235
    return bool(numpy.all(neutral.mean(axis=(0, 2)) >= 0.96))


def _classic_hotbar_neutral_column(column: Any) -> bool:
    high = column.max(axis=1)
    low = column.min(axis=1)
    return bool(
        (((high - low) <= 18) & (column[:, 2] >= 85) & (column[:, 2] <= 235)).mean()
        >= 0.75
    )


def _classic_hotbar_geometry(pixels: Any) -> tuple[int, int] | None:
    """Locate the four-row neutral rail immediately above classic hotbar slots."""
    numpy = importlib.import_module("numpy")
    height, width = pixels.shape[:2]
    y_start, y_end = max(0, height - 160), max(0, height - 40)
    x_start, x_end = int(width * 0.30), int(width * 0.75)
    rows = pixels[y_start:y_end, x_start:x_end, :3].astype(numpy.int16)
    high = rows.max(axis=2)
    low = rows.min(axis=2)
    neutral = (high - low <= 18) & (rows[:, :, 2] >= 85) & (rows[:, :, 2] <= 235)
    transitions = numpy.diff(
        numpy.pad(neutral.astype(numpy.int8), ((0, 0), (1, 1))),
        axis=1,
    )
    start_rows, starts = numpy.nonzero(transitions == 1)
    end_rows, ends = numpy.nonzero(transitions == -1)
    if not numpy.array_equal(start_rows, end_rows):
        return None
    starts = starts + x_start
    ends = ends - 1 + x_start
    lengths = ends - starts + 1
    centers = (starts + ends) / 2
    selected = (
        (lengths >= _CLASSIC_HOTBAR_RAIL_MIN_LENGTH)
        & (lengths <= _CLASSIC_HOTBAR_RAIL_MAX_LENGTH)
        & (centers >= width * 0.45)
        & (centers <= width * 0.60)
    )
    candidates = tuple(
        (int(y_start + row), int(start), int(end))
        for row, start, end in zip(
            start_rows[selected],
            starts[selected],
            ends[selected],
            strict=True,
        )
    )
    for index in range(3, len(candidates)):
        group = candidates[index - 3 : index + 1]
        if any(
            current[0] != previous[0] + 1
            or abs(current[1] - previous[1]) > 1
            or abs(current[2] - previous[2]) > 1
            for previous, current in zip(group, group[1:], strict=False)
        ):
            continue
        first_y, first_start, _ = group[0]
        first_slot_x = first_start - _CLASSIC_HOTBAR_FIRST_SLOT_FROM_RAIL
        if (
            first_slot_x < 0
            or first_slot_x + 8 * _CLASSIC_HOTBAR_SLOT_PITCH + 78 > width
            or first_y + 58 > height
        ):
            return None
        geometry = first_slot_x, first_y
        if _classic_hotbar_geometry_matches(pixels, geometry):
            return geometry
    # A selected middle slot splits the neutral rail into two runs; selecting
    # the last slot leaves only the left run. Nominate the same centered grid
    # from those fragments, then require the exact selection frame and all
    # unaffected rail/divider pixels. A world stripe alone is insufficient.
    tried: set[tuple[int, int]] = set()
    for row, start, length in zip(start_rows, starts, lengths, strict=True):
        if length < 72:
            continue
        for offset in (6, *(94 + 80 * slot for slot in range(8))):
            first_slot_x = int(start) - offset
            if abs(first_slot_x + 366 - width / 2) > 2:
                continue
            geometry = first_slot_x, int(y_start + row)
            if geometry in tried:
                continue
            tried.add(geometry)
            if _classic_hotbar_selected_geometry_matches(pixels, geometry):
                return geometry
    return None


def _classic_hotbar_slot_kind(
    pixels: Any,
    *,
    slot_x: int,
    rail_y: int,
) -> Literal["log", "other", "ambiguous"]:
    numpy = importlib.import_module("numpy")
    patch = pixels[rail_y + 18 : rail_y + 38, slot_x + 20 : slot_x + 72, :3]
    if patch.shape != (20, 52, 3):
        return "ambiguous"
    # Captures are BGRA; the calibrated compact template is RGB.
    patch = patch[:, :, ::-1].astype(numpy.int32)
    grid = numpy.rint(
        patch.reshape(5, 4, 13, 4, 3).mean(axis=(1, 3))
    ).astype(numpy.int32)
    template, dirt_template, mask = _classic_hotbar_template_arrays()
    # Compare only the bright, warm diamond cells occupied by the rendered log
    # top, then independently constrain every remaining cell. The second gate
    # prevents an arbitrary icon/background that copies only the old mask from
    # being certified as a log.
    difference = numpy.abs(grid - template)
    mean_difference = float(difference[mask].mean())
    if (
        mean_difference <= 8.0
        and float(numpy.percentile(difference.max(axis=2)[mask], 95)) <= 20.0
        and float(difference[~mask].mean()) <= 18.0
        and float(numpy.percentile(difference.max(axis=2)[~mask], 95)) <= 45.0
    ):
        return "log"
    dirt_difference = numpy.abs(grid - dirt_template)
    if (
        float(dirt_difference.mean()) <= 4.0
        and float(numpy.percentile(dirt_difference.max(axis=2), 95)) <= 14.0
    ):
        return "other"
    if _classic_hotbar_slot_empty(pixels, slot_x=slot_x, rail_y=rail_y):
        return "other"
    return "ambiguous"


@lru_cache(maxsize=1)
def _classic_hotbar_template_arrays() -> tuple[Any, Any, Any]:
    """Compile immutable embedded templates once for the realtime fast path."""

    numpy = importlib.import_module("numpy")
    log = numpy.frombuffer(_CLASSIC_HOTBAR_LOG_RGB_5X13, dtype=numpy.uint8).reshape(
        5,
        13,
        3,
    ).astype(numpy.int32)
    dirt = numpy.frombuffer(_CLASSIC_HOTBAR_DIRT_RGB_5X13, dtype=numpy.uint8).reshape(
        5,
        13,
        3,
    ).astype(numpy.int32)
    mask = (log.mean(axis=2) > 75) & (log[:, :, 0] - log[:, :, 2] > 30)
    for value in (log, dirt, mask):
        value.flags.writeable = False
    return log, dirt, mask


def _classic_hotbar_slot_empty(
    pixels: Any,
    *,
    slot_x: int,
    rail_y: int,
) -> bool:
    """Certify a low-detail translucent slot, never an arbitrary non-log item."""
    numpy = importlib.import_module("numpy")
    bottom = min(pixels.shape[0], rail_y + 67)
    patch = pixels[rail_y + 11 : bottom, slot_x + 12 : slot_x + 72, :3].astype(
        numpy.float32
    )
    if patch.shape[0] < 40 or patch.shape[1:] != (60, 3):
        return False
    gray = patch.mean(axis=2)
    edge_strength = (
        float(numpy.abs(numpy.diff(gray, axis=0)).mean())
        + float(numpy.abs(numpy.diff(gray, axis=1)).mean())
    ) / 2
    center = patch[7:27, 8:60]
    border = numpy.concatenate(
        (
            patch[:8].reshape(-1, 3),
            patch[-8:].reshape(-1, 3),
            patch[:, :8].reshape(-1, 3),
            patch[:, -8:].reshape(-1, 3),
        ),
        axis=0,
    )
    return bool(
        float(gray.std()) <= 18.0
        and edge_strength <= 4.5
        and abs(float(center.mean() - border.mean())) <= 8.0
    )


def _classic_hotbar_stack_count(
    pixels: Any,
    *,
    slot_x: int,
    rail_y: int,
) -> int | None:
    """Decode only pinned 5x7 atlas glyphs through 16; no glyph means one."""
    numpy = importlib.import_module("numpy")
    height = pixels.shape[0]
    available_bottom = min(height, rail_y + 72)
    # Bedrock right-aligns the final 5-pixel glyph at +58 and advances a prior
    # glyph by six pixels. At the pinned 4-pixel UI scale this is an exact
    # 11x7 logical-cell region for all current survival counts (1..16).
    patch = pixels[rail_y + 44 : available_bottom, slot_x + 34 : slot_x + 78, :3]
    if patch.shape[1:] != (44, 3) or patch.shape[0] < 12:
        return None
    high = patch.max(axis=2).astype(numpy.int16)
    low = patch.min(axis=2).astype(numpy.int16)
    luma = patch[:, :, :3].mean(axis=2)
    white = (high - low <= 35) & (luma >= 210)
    if int(white.sum()) <= 2:
        return 1
    if patch.shape[0] != 28:
        return None
    fractions = white.reshape(7, 4, 11, 4).mean(axis=(1, 3))
    # Antialiasing or partial glyphs land in the indeterminate band.
    if bool(((fractions > 0.15) & (fractions < 0.65)).any()):
        return None
    observed = tuple(
        "".join("#" if fractions[row, column] >= 0.65 else "." for column in range(11))
        for row in range(7)
    )
    matches = tuple(
        value
        for value in range(2, _CLASSIC_HOTBAR_MAX_VERIFIED_COUNT + 1)
        if observed == _classic_hotbar_count_template(value)
    )
    return matches[0] if len(matches) == 1 else None


@lru_cache(maxsize=_CLASSIC_HOTBAR_MAX_VERIFIED_COUNT - 1)
def _classic_hotbar_count_template(value: int) -> tuple[str, ...]:
    digits = str(value)
    if len(digits) == 1:
        return tuple("......" + row for row in _CLASSIC_HOTBAR_DIGITS_5X7[value])
    if len(digits) == 2:
        left = _CLASSIC_HOTBAR_DIGITS_5X7[int(digits[0])]
        right = _CLASSIC_HOTBAR_DIGITS_5X7[int(digits[1])]
        return tuple(a + "." + b for a, b in zip(left, right, strict=True))
    raise ValueError("classic hotbar verifier supports at most two digits")


_CLASSIC_INVENTORY_CANONICAL_WIDTH = 1920
_CLASSIC_INVENTORY_CANONICAL_HEIGHT = 1054
_CLASSIC_INVENTORY_SLOT_SIZE = 68
_CLASSIC_INVENTORY_PLAYER_X = tuple(936 + 72 * column for column in range(9))
_CLASSIC_INVENTORY_PLAYER_Y = (540, 612, 684, 768)
# These are the non-player-grid slots that can contain the wood inputs/outputs
# in Bedrock survival. Armor and Bedrock's restricted offhand cannot hold them.
_CLASSIC_INVENTORY_AUXILIARY_SLOTS = (
    ("craft.input.0", 1292, 268),
    ("craft.input.1", 1364, 268),
    ("craft.input.2", 1292, 340),
    ("craft.input.3", 1364, 340),
    ("craft.output", 1516, 300),
)
# Manually verified vanilla dirt held in hotbar slot zero. The 36x34 source
# crop excludes the stack-count glyph; this 8x8 RGB mean grid was stable across
# independent count-5 and count-6 captures. Their full-frame SHA-256 values are
# 6c7602169f34ddb13ecba1567d18b7cd4f7bb20c5e0f65bad6208b89f7d937f8 and
# 85e3c9e97f01b0646c2c19ad91c1f61327b72c44d0f76b13d178031805b0b338.
# This is deliberately a one-item whitelist, not an open-ended item classifier.
# Unknown occupants abstain.
_CLASSIC_INVENTORY_DIRT_RGB_8X8 = bytes.fromhex(
    "898b8e8a8c8d8c8b8a86766a7f634d7d5b408e694b8268548c8b8a90847a8a"
    "726089654894684578573d856145815c3e8d746179573a8b664a7e5d4278553a"
    "835e41845d3f8c6647735034815a428963438e6c50886448805a3c916d4f856042"
    "58412e664a37724f327d583b815a3b9066468a634478583f60432c5c442d634935"
    "61452e70523a7a593e6d50394b33225f432e654b365c4029583f2b543a265b3e27"
    "422e1e453021634832684c36624732563c295d4430614732483629412e21"
)


def bedrock_inventory_slot_observation(
    frame: CapturedFrame,
) -> BedrockInventorySlotObservation | None:
    """Inspect the version-pinned classic survival inventory slot grid.

    The detector proves absence only. It recognizes uniform empty slot fill and
    one tightly calibrated non-wood dirt signature. Any other occupied slot,
    partial render, different GUI geometry, or unavailable NumPy path remains
    explicitly unresolved for the slower semantic fallback.
    """
    if not bedrock_inventory_overlay_present(frame):
        return None
    return _bedrock_inventory_slot_observation(frame)


def _bedrock_inventory_slot_observation(
    frame: CapturedFrame,
) -> BedrockInventorySlotObservation | None:
    if frame.width < 1280 or frame.height < 700:
        return None
    scale_x = frame.width / _CLASSIC_INVENTORY_CANONICAL_WIDTH
    scale_y = frame.height / _CLASSIC_INVENTORY_CANONICAL_HEIGHT
    if not 0.94 <= scale_x / scale_y <= 1.06:
        return None
    pixels = _numpy_bgra(frame)
    if pixels is None:
        return None

    slots = tuple(
        (
            f"inventory.{row * 9 + column}",
            _scaled_inventory_slot(frame, x=x, y=y),
        )
        for row, y in enumerate(_CLASSIC_INVENTORY_PLAYER_Y)
        for column, x in enumerate(_CLASSIC_INVENTORY_PLAYER_X)
    ) + tuple(
        (name, _scaled_inventory_slot(frame, x=x, y=y))
        for name, x, y in _CLASSIC_INVENTORY_AUXILIARY_SLOTS
    )
    if any(not _classic_inventory_slot_geometry(pixels, bounds) for _, bounds in slots):
        return None

    player_patches = tuple(
        _classic_inventory_slot_inner(pixels, bounds) for _, bounds in slots[:36]
    )
    background_candidates = tuple(
        median
        for patch in player_patches
        if (median := _uniform_empty_slot_median(patch)) is not None
    )
    # This is an absence fast path, not a general full-inventory reader. A
    # mostly occupied inventory remains the VLM's responsibility.
    if len(background_candidates) < 24:
        return None
    numpy = importlib.import_module("numpy")
    candidate_array = numpy.asarray(background_candidates, dtype=numpy.int16)
    background = numpy.rint(numpy.median(candidate_array, axis=0)).astype(numpy.int16)
    if int(numpy.max(numpy.abs(candidate_array - background))) > 6:
        return None

    occupied: list[str] = []
    known_non_wood: list[str] = []
    for name, bounds in slots:
        patch = _classic_inventory_slot_inner(pixels, bounds)
        classification = _classic_inventory_slot_occupancy(patch, background)
        if classification is None:
            return None
        if not classification:
            continue
        occupied.append(name)
        if _classic_inventory_dirt_matches(pixels, bounds):
            known_non_wood.append(name)
    return BedrockInventorySlotObservation(
        occupied_slots=tuple(occupied),
        known_non_wood_slots=tuple(known_non_wood),
    )


def _scaled_inventory_slot(
    frame: CapturedFrame,
    *,
    x: int,
    y: int,
) -> tuple[int, int, int, int]:
    scale_x = frame.width / _CLASSIC_INVENTORY_CANONICAL_WIDTH
    scale_y = frame.height / _CLASSIC_INVENTORY_CANONICAL_HEIGHT
    return (
        round(x * scale_x),
        round(y * scale_y),
        round((x + _CLASSIC_INVENTORY_SLOT_SIZE) * scale_x),
        round((y + _CLASSIC_INVENTORY_SLOT_SIZE) * scale_y),
    )


def _classic_inventory_slot_geometry(
    pixels: Any,
    bounds: tuple[int, int, int, int],
) -> bool:
    numpy = importlib.import_module("numpy")
    x0, y0, x1, y1 = bounds
    width, height = x1 - x0, y1 - y0
    border_x = max(1, round(width * 4 / _CLASSIC_INVENTORY_SLOT_SIZE))
    border_y = max(1, round(height * 4 / _CLASSIC_INVENTORY_SLOT_SIZE))
    top = pixels[y0 : y0 + border_y, x0:x1, :3].reshape(-1, 3).astype(numpy.int16)
    left = pixels[y0 + border_y : y1, x0 : x0 + border_x, :3].reshape(-1, 3).astype(
        numpy.int16
    )
    if not top.size or not left.size:
        return False
    border = numpy.concatenate((top, left), axis=0)
    chroma = border.max(axis=1) - border.min(axis=1)
    blue, green, red = border[:, 0], border[:, 1], border[:, 2]
    luma = (29 * blue + 150 * green + 77 * red) // 256
    signature = (chroma <= 8) & (luma >= 45) & (luma <= 70)
    return float(signature.mean()) >= 0.93


def _classic_inventory_slot_inner(
    pixels: Any,
    bounds: tuple[int, int, int, int],
) -> Any:
    x0, y0, x1, y1 = bounds
    width, height = x1 - x0, y1 - y0
    inset_x0 = round(width * 8 / _CLASSIC_INVENTORY_SLOT_SIZE)
    inset_x1 = round(width * 60 / _CLASSIC_INVENTORY_SLOT_SIZE)
    inset_y0 = round(height * 8 / _CLASSIC_INVENTORY_SLOT_SIZE)
    inset_y1 = round(height * 60 / _CLASSIC_INVENTORY_SLOT_SIZE)
    return pixels[
        y0 + inset_y0 : y0 + inset_y1,
        x0 + inset_x0 : x0 + inset_x1,
        :3,
    ].astype("int16")


def _uniform_empty_slot_median(patch: Any) -> Any | None:
    numpy = importlib.import_module("numpy")
    flat = patch.reshape(-1, 3)
    if not flat.size:
        return None
    median = numpy.rint(numpy.median(flat, axis=0)).astype(numpy.int16)
    luma = (29 * int(median[0]) + 150 * int(median[1]) + 77 * int(median[2])) // 256
    residual = numpy.max(numpy.abs(flat - median), axis=1)
    chroma = flat.max(axis=1) - flat.min(axis=1)
    uniform = (residual <= 4) & (chroma <= 8)
    if not 125 <= luma <= 150 or float(uniform.mean()) < 0.97:
        return None
    return median


def _classic_inventory_slot_occupancy(patch: Any, background: Any) -> bool | None:
    numpy = importlib.import_module("numpy")
    foreground = (numpy.max(numpy.abs(patch - background), axis=2) > 12) | (
        patch.max(axis=2) - patch.min(axis=2) > 12
    )
    fraction = float(foreground.mean())
    if fraction <= 0.03:
        return False
    if fraction >= 0.12:
        return True
    return None


def _classic_inventory_dirt_matches(
    pixels: Any,
    bounds: tuple[int, int, int, int],
) -> bool:
    numpy = importlib.import_module("numpy")
    x0, y0, x1, y1 = bounds
    width, height = x1 - x0, y1 - y0
    left = x0 + round(width * 10 / _CLASSIC_INVENTORY_SLOT_SIZE)
    right = x0 + round(width * 46 / _CLASSIC_INVENTORY_SLOT_SIZE)
    top = y0 + round(height * 8 / _CLASSIC_INVENTORY_SLOT_SIZE)
    bottom = y0 + round(height * 42 / _CLASSIC_INVENTORY_SLOT_SIZE)
    # Captured pixels are BGRA; the calibrated template is RGB.
    patch = pixels[top:bottom, left:right, :3][:, :, ::-1].astype(numpy.int32)
    if patch.shape[0] < 8 or patch.shape[1] < 8:
        return False
    y_edges = numpy.rint(numpy.linspace(0, patch.shape[0], 9)).astype(int)
    x_edges = numpy.rint(numpy.linspace(0, patch.shape[1], 9)).astype(int)
    grid = numpy.empty((8, 8, 3), dtype=numpy.int32)
    for row in range(8):
        for column in range(8):
            cell = patch[
                y_edges[row] : y_edges[row + 1],
                x_edges[column] : x_edges[column + 1],
            ]
            grid[row, column] = numpy.rint(cell.mean(axis=(0, 1))).astype(numpy.int32)
    template = numpy.frombuffer(_CLASSIC_INVENTORY_DIRT_RGB_8X8, dtype=numpy.uint8).reshape(
        8, 8, 3
    )
    difference = numpy.abs(grid - template.astype(numpy.int32))
    per_pixel = difference.max(axis=2)
    return bool(float(difference.mean()) <= 3.0 and float(numpy.percentile(per_pixel, 95)) <= 8)


def bedrock_inventory_overlay_present(frame: CapturedFrame) -> bool:
    """Detect Bedrock's survival or recipe-book inventory chrome.

    Survival inventory uses a wide split layout: its recipe pane is dominated
    by dark neutral slot fill while the player/crafting pane independently has
    mid and light neutral chrome. That asymmetric three-palette conjunction is
    much harder for a textured world frame to satisfy than a whole-screen gray
    ratio. The retained compact recipe-book layout is covered by the original
    upper/header conjunction below. This remains a negative-only safety signal:
    it does not assert which inventory, tab, recipe, or item is visible.
    """
    if not frame.bgra or frame.width < 320 or frame.height < 180:
        return False
    pixels = _numpy_bgra(frame)
    left_slot_ratio = _sampled_neutral_ratio(
        frame,
        x_start=0.15,
        x_end=0.46,
        y_start=0.18,
        y_end=0.75,
        luma_min=70,
        luma_max=110,
        pixels=pixels,
    )
    if left_slot_ratio >= 0.55:
        right_mid_ratio = _sampled_neutral_ratio(
            frame,
            x_start=0.47,
            x_end=0.84,
            y_start=0.18,
            y_end=0.75,
            luma_min=120,
            luma_max=160,
            pixels=pixels,
        )
        right_light_ratio = _sampled_neutral_ratio(
            frame,
            x_start=0.47,
            x_end=0.84,
            y_start=0.18,
            y_end=0.75,
            luma_min=175,
            luma_max=220,
            pixels=pixels,
        )
        if right_mid_ratio >= 0.25 and right_light_ratio >= 0.20:
            return True
    upper_light_ratio = _sampled_neutral_ratio(
        frame,
        x_start=0.10,
        x_end=0.90,
        y_start=0.12,
        y_end=0.24,
        luma_min=140,
        luma_max=235,
        pixels=pixels,
    )
    if upper_light_ratio < 0.40:
        return False
    header_neutral_ratio = _sampled_neutral_ratio(
        frame,
        x_start=0.10,
        x_end=0.90,
        y_start=0.20,
        y_end=0.34,
        luma_min=70,
        luma_max=235,
        pixels=pixels,
    )
    return header_neutral_ratio >= 0.65


def _sampled_neutral_ratio(
    frame: CapturedFrame,
    *,
    x_start: float,
    x_end: float,
    y_start: float,
    y_end: float,
    luma_min: int,
    luma_max: int,
    pixels: Any | None,
) -> float:
    """Measure neutral UI pixels on a bounded grid, with or without NumPy."""
    x0, x1 = int(frame.width * x_start), int(frame.width * x_end)
    y0, y1 = int(frame.height * y_start), int(frame.height * y_end)
    x_step = max(1, (x1 - x0) // 256)
    y_step = max(1, (y1 - y0) // 96)
    if pixels is not None:
        roi = pixels[y0:y1:y_step, x0:x1:x_step, :3].astype("int32")
        blue, green, red = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]
        high = roi.max(axis=2)
        low = roi.min(axis=2)
        luma = (29 * blue + 150 * green + 77 * red) // 256
        selected = (high - low <= 18) & (luma >= luma_min) & (luma <= luma_max)
        return float(selected.mean()) if selected.size else 0.0
    source = memoryview(frame.bgra)
    matched = 0
    sampled = 0
    for y in range(y0, y1, y_step):
        for x in range(x0, x1, x_step):
            offset = (y * frame.width + x) * 4
            blue, green, red = (int(value) for value in source[offset : offset + 3])
            luma = (29 * blue + 150 * green + 77 * red) // 256
            matched += int(
                max(blue, green, red) - min(blue, green, red) <= 18
                and luma_min <= luma <= luma_max
            )
            sampled += 1
    return matched / sampled if sampled else 0.0


def bedrock_survival_hud_present(frame: CapturedFrame) -> bool:
    """Verify an in-world survival HUD before supervisory camera calibration.

    This is a conservative actuator interlock, not semantic perception or a
    training label. Requiring both the red heart bank and neutral hotbar frame
    prevents calibration motion from being emitted over menus or loading UI.
    """
    return not bedrock_away_overlay_present(frame) and _bedrock_hud_present(
        frame, require_hearts=True,
    )


def bedrock_creative_hud_present(frame: CapturedFrame) -> bool:
    """Verify an in-world creative-mode HUD (hotbar present, no heart bank).

    Creative mode omits the heart bank entirely, so the survival interlock
    would wrongly reject it. The neutral hotbar frame plus a matching HUD band
    is sufficient to arm in-world creative play.
    """
    if not frame.bgra or frame.width < 320 or frame.height < 180:
        return False
    if bedrock_away_overlay_present(frame):
        return False
    pixels = _numpy_bgra(frame)
    hotbar_ratio = _hud_palette_ratio(
        frame,
        x_start=0.28,
        x_end=0.72,
        y_start=0.89,
        y_end=1.0,
        palette="hotbar",
        pixels=pixels,
    )
    # Info-lines and the crosshair are shown; the hotbar row must be present.
    return hotbar_ratio >= 0.03


def bedrock_in_world_hud_present(frame: CapturedFrame) -> bool:
    """Accept an in-world HUD in either survival or creative mode.

    Survival shows the heart bank; creative omits it. Either is playable.
    """
    return bedrock_survival_hud_present(frame) or bedrock_creative_hud_present(frame)


def bedrock_away_overlay_present(frame: CapturedFrame) -> bool:
    """Block the calibrated lower-center away panel, even with a visible HUD.

    Require its quiet gray inset, lighter bottom rim, and three separate
    light-on-gray text rows together. A hotbar, gray world, or one text row
    alone is insufficient. These bounded pixel checks are negative-only:
    they do not read the notice, identify a clickable point, or authorize a
    wake action. Exact text recognition remains owned by the menu navigator.
    Unknown UI scales/layouts are not certified by this detector.
    """
    if (
        frame.width < 320 or frame.height < 180
        or len(frame.bgra) != frame.width * frame.height * 4
    ):
        return False
    pixels = _numpy_bgra(frame)

    def neutral(bounds: tuple[float, float, float, float], low: int, high: int) -> float:
        left, top, right, bottom = bounds
        return _sampled_neutral_ratio(
            frame, x_start=left, x_end=right, y_start=top, y_end=bottom,
            luma_min=low, luma_max=high, pixels=pixels,
        )

    if neutral((0.42, 0.700, 0.69, 0.712), 75, 110) < 0.95:
        return False
    if neutral((0.694, 0.712, 0.703, 0.827), 75, 110) < 0.95:
        return False
    if neutral((0.42, 0.844, 0.69, 0.850), 125, 155) < 0.80:
        return False
    for bounds in (
        (0.43, 0.716, 0.68, 0.744),
        (0.43, 0.755, 0.69, 0.783),
        (0.43, 0.791, 0.56, 0.820),
    ):
        if neutral(bounds, 75, 110) < 0.50 or not 0.08 <= neutral(bounds, 210, 255) <= 0.40:
            return False
    return True


def _bedrock_hud_present(frame: CapturedFrame, *, require_hearts: bool) -> bool:
    if not frame.bgra or frame.width < 320 or frame.height < 180:
        return False
    pixels = _numpy_bgra(frame)
    hotbar_ratio = _hud_palette_ratio(
        frame,
        x_start=0.28,
        x_end=0.72,
        y_start=0.89,
        y_end=1.0,
        palette="hotbar",
        pixels=pixels,
    )
    if not require_hearts:
        return hotbar_ratio >= 0.03
    heart_ratio = _hud_palette_ratio(
        frame,
        x_start=0.29,
        x_end=0.49,
        y_start=0.82,
        y_end=0.91,
        palette="heart",
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
