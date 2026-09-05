"""Observation-only first step toward a body-relative local world model.

A surface nomination is not a collision verdict. In particular, a clear centre
ray does not establish room for the nominal 0.6-wide, 1.8-high standing player.
RGB without reliable depth/pose cannot populate metric voxels. This module
retains a visible surface and its exact image; it publishes no motor targets,
block identities, free-space facts, or permissions.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import math
from dataclasses import dataclass
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .models import ModelResponse
from .perception import EvidenceRegion, PerceptionEvidence, ScreenRegion
from .platforms.bedrock_x11 import CapturedFrame


_Probability = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
SurfaceFeature = Literal["underside", "riser", "side_face"]


class _SurfaceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    feature: Literal["underside", "riser", "side_face", "unknown"] | None
    point: tuple[_Probability, _Probability] | None
    confidence: _Probability | None


@dataclass(frozen=True)
class VisibleClearanceSurface:
    """One point on a visible surface, not a 3D body collision or a mining target."""

    feature: SurfaceFeature
    point: tuple[float, float]
    confidence: float


@dataclass(frozen=True)
class BodyClearanceInspection:
    evidence: PerceptionEvidence
    candidate: VisibleClearanceSurface | None
    latency_ms: float
    raw_response: str
    model_input_sha256: str
    model_input_size: tuple[int, int]
    validation_error: str | None = None


class BodyClearanceValidationError(ValueError):
    """Rejected submitted reply with its evidence retained for honest evaluation."""

    def __init__(self, inspection: BodyClearanceInspection) -> None:
        super().__init__(inspection.validation_error)
        self.inspection = inspection


class _VisionModel(Protocol):
    def inspect(self, prompt: str, *, image_bytes: bytes, mime_type: str) -> ModelResponse: ...


def _survey_image(frame: CapturedFrame) -> tuple[bytes, PerceptionEvidence, tuple[int, int]]:
    if frame.width <= 0 or frame.height <= 0 or not frame.bgra:
        raise ValueError("clearance survey requires a nonempty captured frame")
    image_module = importlib.import_module("PIL.Image")
    source = image_module.frombytes(
        "RGBA", (frame.width, frame.height), frame.bgra, "raw", "BGRA"
    )
    # Same WORLD boundary as grounded perception. A deliberate lower view is
    # needed to see the feet; including the HUD cannot recover hidden pixels.
    crop_height = max(1, math.ceil(frame.height * 0.84))
    crop = source.crop((0, 0, frame.width, crop_height)).convert("RGB")
    evidence = PerceptionEvidence(
        evidence_id=f"frame-{frame.frame_id}:clearance-world",
        frame_id=frame.frame_id,
        captured_ns=frame.captured_ns,
        region_kind=EvidenceRegion.WORLD,
        region=ScreenRegion(x=0, y=0, width=1, height=crop_height / frame.height),
        pixel_sha256=hashlib.sha256(crop.tobytes()).hexdigest(),
        crop_width=frame.width,
        crop_height=crop_height,
    )
    # No labels, borders, letterboxing or duplicate views: model coordinates
    # refer to exactly this one crop, with the aspect ratio preserved.
    if crop.width > 512:
        crop = crop.resize(
            (512, max(1, round(crop.height * 512 / crop.width))),
            image_module.Resampling.LANCZOS,
        )
    buffer = io.BytesIO()
    crop.save(buffer, format="PNG")
    return buffer.getvalue(), evidence, (int(crop.width), int(crop.height))


def _surface_prompt(requested_feature: SurfaceFeature | None) -> str:
    selection = (
        "Locate ONE clearly visible nearby surface relevant to inspecting the immediate passage. "
        if requested_feature is None
        else f"Locate ONLY a clearly visible nearby {requested_feature}. "
        "Do not substitute another feature or the most visually prominent trunk face. "
        "If the requested feature is hidden, absent or uncertain, use unknown and a null point. "
    )
    return (
        "Minecraft first-person view. " + selection
        + "An underside is the bottom face of an "
        "overhang; a riser is the raised vertical front face of a step, not its top; "
        "a side_face is a vertical face bordering the passage. Pick a point INSIDE the "
        "visible face, not its edge. Do not select the ground supporting the player, "
        "distant terrain, animals, sky, or foliage hiding an uncertain face. If no such "
        "surface can be identified, use unknown and a null point. Return only JSON with "
        "feature,point,confidence. feature is underside/riser/side_face/unknown or null. "
        "point is [x,y] in this image, 0..1 left-to-right/top-to-bottom, or null. "
        "confidence is 0..1 or null. This is surface observation only: the player's "
        "nominal standing collision body is 0.6 blocks wide and 1.8 high, but distance, "
        "stance and exact body position are unknown. A surface is NOT proof of a head, "
        "foot or side collision, and an unknown answer does NOT mean an open passage."
    )


def _surface_grammar() -> str:
    names = " | ".join(json.dumps(json.dumps(value)) for value in (
        "underside", "riser", "side_face", "unknown",
    ))
    return "\n".join((
        'root ::= "{" ws "\\\"feature\\\"" ws ":" ws feature ws "," ws '
        '"\\\"point\\\"" ws ":" ws point ws "," ws "\\\"confidence\\\"" ws ":" ws '
        'confidence ws "}" ws',
        f'feature ::= "null" | {names}',
        'point ::= "null" | "[" ws probability ws "," ws probability ws "]"',
        'confidence ::= "null" | probability',
        'probability ::= "0" ("." [0-9]+)? | "1" ("." "0"+)?',
        "ws ::= [ \\t\\n]*",
    ))


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    fields: dict[str, object] = {}
    for key, value in pairs:
        if key in fields:
            raise ValueError(f"duplicate clearance response field: {key}")
        fields[key] = value
    return fields


@dataclass(frozen=True)
class BodyClearanceSurveyor:
    """One bounded image query; no repair loop, camera control or blackboard writes.

    Kept separate from general tracks: a high-confidence survey surface must
    never become an implicit target selected by the learned motor router.
    """

    model: _VisionModel

    def inspect(
        self, frame: CapturedFrame, *, requested_feature: SurfaceFeature | None = None,
    ) -> BodyClearanceInspection:
        if requested_feature is not None and requested_feature not in {
            "underside", "riser", "side_face",
        }:
            raise ValueError("unsupported requested clearance feature")
        image_bytes, evidence, image_size = _survey_image(frame)
        model_input_hash = hashlib.sha256(image_bytes).hexdigest()
        prompt = _surface_prompt(requested_feature)
        schema = _SurfaceResponse.model_json_schema()
        constrained = getattr(self.model, "inspect_constrained", None)
        structured = getattr(self.model, "inspect_structured", None)
        if callable(constrained):
            response = constrained(
                prompt, image_bytes=image_bytes, mime_type="image/png",
                name="minecraft_clearance_surface", schema=schema, grammar=_surface_grammar(),
            )
        elif callable(structured):
            response = structured(
                prompt, image_bytes=image_bytes, mime_type="image/png",
                name="minecraft_clearance_surface", schema=schema,
            )
        else:
            response = self.model.inspect(prompt, image_bytes=image_bytes, mime_type="image/png")
        # Do not let last-key-wins JSON hide a conflicting surface nomination.
        try:
            json.loads(response.text, object_pairs_hook=_unique_fields)
            parsed = _SurfaceResponse.model_validate_json(response.text)
        except ValueError as exc:
            raise BodyClearanceValidationError(BodyClearanceInspection(
                evidence=evidence, candidate=None, latency_ms=response.latency_ms,
                raw_response=response.text, model_input_sha256=model_input_hash,
                model_input_size=image_size,
                validation_error=f"{type(exc).__name__}: {exc}"[:2048],
            )) from exc
        candidate = None
        if (
            parsed.feature is not None and parsed.feature != "unknown"
            and parsed.point is not None and parsed.confidence is not None
            and parsed.confidence > 0
            and all(0 < coordinate < 1 for coordinate in parsed.point)
            and (requested_feature is None or parsed.feature == requested_feature)
        ):
            candidate = VisibleClearanceSurface(
                feature=parsed.feature,
                point=(
                    evidence.region.x + parsed.point[0] * evidence.region.width,
                    evidence.region.y + parsed.point[1] * evidence.region.height,
                ),
                confidence=parsed.confidence,
            )
        return BodyClearanceInspection(
            evidence=evidence, candidate=candidate,
            latency_ms=response.latency_ms, raw_response=response.text,
            model_input_sha256=model_input_hash, model_input_size=image_size,
        )
