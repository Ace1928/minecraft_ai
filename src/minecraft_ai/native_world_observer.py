"""Typed, fail-closed seam for game observations from the shared World owner.

Native CLIP comparisons are not resource detections, GUI measurements or chat
transcriptions. No current World capability is admitted for these outputs. The
request contract records exactly which pixels and deadline a future qualified
shared observer must consume; it does not fabricate a working visual head.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from minecraft_ai.native_world_model import _read_private_file
from minecraft_ai.platforms.bedrock_x11 import CapturedFrame

ObservationTask = Literal["resource", "inventory", "chat"]
ObservationStatus = Literal["unsupported", "unavailable", "expired"]
ObservationReason = Literal[
    "frame_deadline_expired", "world_readiness_unavailable", "world_readiness_invalid",
    "native_clip_has_no_qualified_game_outputs", "qualified_observer_transport_unavailable",
]
RUNTIME_ID = re.compile(r"[0-9a-f]{32}\Z")
TASK_OUTPUTS = {
    "resource": (
        "resource.tracks", "target.visible", "target.kind", "target.dx", "target.dy",
        "target.mineable", "target.near",
    ),
    "inventory": (
        "gui.mode", "gui.tracks", "inventory.logs", "inventory.planks", "inventory.crafting_table",
    ),
    "chat": ("chat.text", "chat.speaker", "chat.confidence"),
}


class NativeGameObservationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    contract: Literal["erais.minecraft-observation-request.v1"] = (
        "erais.minecraft-observation-request.v1"
    )
    request_id: str = Field(min_length=1, max_length=128)
    instance_id: str = Field(min_length=1, max_length=256)
    game_version: str = Field(pattern=r"^\d+\.\d+\.\d+\.\d+$")
    pack_catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task: ObservationTask
    frame_id: int = Field(ge=0)
    captured_ns: int = Field(gt=0)
    width: int = Field(gt=0, le=8192)
    height: int = Field(gt=0, le=8192)
    pixel_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    deadline_ns: int = Field(gt=0)

    @model_validator(mode="after")
    def frame_scope(self) -> NativeGameObservationRequest:
        if not self.instance_id.startswith(f"bedrock:{self.game_version}:"):
            raise ValueError("exact Bedrock instance/version binding required")
        if self.width * self.height * 4 > 32 * 1024 * 1024:
            raise ValueError("game observation pixels exceed bound")
        if not 0 < self.deadline_ns - self.captured_ns <= 5_000_000_000:
            raise ValueError("game observation must retain a bounded frame deadline")
        return self


class NativeGameObservationResult(BaseModel):
    """Negative readiness is explicit and grants no semantic/action authority."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    contract: Literal["erais.minecraft-observation-result.v1"] = (
        "erais.minecraft-observation-result.v1"
    )
    request_id: str
    frame_id: int
    captured_ns: int
    pixel_sha256: str
    task: ObservationTask
    status: ObservationStatus
    reason: ObservationReason
    unsupported_outputs: tuple[str, ...]
    runtime_id: str | None = None
    # No positive observation payload is admitted until an authenticated
    # profile, qualification receipt and World-owned transport exist together.


@dataclass(frozen=True)
class NativeWorldGameObserver:
    ready_file: str

    def inspect(
        self, request: NativeGameObservationRequest, frame: CapturedFrame,
    ) -> NativeGameObservationResult:
        """Use supplied game pixels; never start a capture or another model."""
        if (
            frame.frame_id != request.frame_id or frame.captured_ns != request.captured_ns
            or frame.width != request.width or frame.height != request.height
            or len(frame.bgra) != frame.width * frame.height * 4
            or hashlib.sha256(frame.bgra).hexdigest() != request.pixel_sha256
        ):
            raise ValueError("observer request does not bind the supplied game pixels")

        def result(
            status: ObservationStatus, reason: ObservationReason, runtime_id: str | None = None,
        ) -> NativeGameObservationResult:
            return NativeGameObservationResult(
                request_id=request.request_id, frame_id=request.frame_id,
                captured_ns=request.captured_ns, pixel_sha256=request.pixel_sha256,
                task=request.task, status=status, reason=reason,
                unsupported_outputs=TASK_OUTPUTS[request.task], runtime_id=runtime_id,
            )

        now = time.monotonic_ns()
        if not request.captured_ns <= now < request.deadline_ns:
            return result("expired", "frame_deadline_expired")
        try:
            ready = json.loads(_read_private_file(self.ready_file, limit=4_194_304))
        except (OSError, ValueError):
            return result("unavailable", "world_readiness_unavailable")
        if (
            type(ready) is not dict or ready.get("status") != "private_ready"
            or type(ready.get("runtime_id")) is not str
            or RUNTIME_ID.fullmatch(ready["runtime_id"]) is None
        ):
            return result("unavailable", "world_readiness_invalid")
        runtime_id = ready["runtime_id"]
        vision = ready.get("vision_backend")
        if (
            type(vision) is dict and vision.get("scope") == "native_clip_visual_matches"
            and vision.get("native_execution") is True
        ):
            return result("unsupported", "native_clip_has_no_qualified_game_outputs", runtime_id)
        # A flag or unfamiliar model name is insufficient to enable gameplay.
        # Implement the qualified protocol explicitly rather than trusting it.
        return result("unsupported", "qualified_observer_transport_unavailable", runtime_id)
