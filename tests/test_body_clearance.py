from __future__ import annotations

import hashlib
import io
import json

import pytest
from PIL import Image

from minecraft_ai.body_clearance import BodyClearanceSurveyor
from minecraft_ai.models import ModelResponse
from minecraft_ai.perception import EvidenceRegion, ScreenRegion
from minecraft_ai.platforms.bedrock_x11 import CapturedFrame


def _frame(width: int = 320, height: int = 200) -> CapturedFrame:
    pixels = b"".join(
        bytes((x % 256, y % 256, (x + y) % 256, 255))
        for y in range(height) for x in range(width)
    )
    return CapturedFrame(7, 123_456, width, height, pixels)


class _PlainModel:
    model_id = "body-clearance-test"

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[tuple[str, str, bytes, dict[str, object]]] = []

    def _respond(
        self, route: str, prompt: str, image_bytes: bytes, kwargs: dict[str, object],
    ) -> ModelResponse:
        self.calls.append((route, prompt, image_bytes, kwargs))
        return ModelResponse(text=self.text, model=self.model_id, latency_ms=12.5)

    def inspect(self, prompt: str, *, image_bytes: bytes, mime_type: str) -> ModelResponse:
        assert mime_type == "image/png"
        return self._respond("plain", prompt, image_bytes, {})


class _StructuredModel(_PlainModel):
    def inspect_structured(
        self, prompt: str, *, image_bytes: bytes, mime_type: str,
        name: str, schema: dict[str, object],
    ) -> ModelResponse:
        assert mime_type == "image/png"
        return self._respond("structured", prompt, image_bytes, {"name": name, "schema": schema})


class _ConstrainedModel(_StructuredModel):
    def inspect_constrained(
        self, prompt: str, *, image_bytes: bytes, mime_type: str,
        name: str, schema: dict[str, object], grammar: str,
    ) -> ModelResponse:
        assert mime_type == "image/png"
        return self._respond("constrained", prompt, image_bytes, {
            "name": name, "schema": schema, "grammar": grammar,
        })


@pytest.mark.parametrize("feature", ("underside", "riser", "side_face"))
def test_body_clearance_reports_visible_candidate_with_exact_frame_provenance(feature: str) -> None:
    frame = _frame()
    raw = json.dumps({"feature": feature, "point": [0.5, 0.5], "confidence": 0.8})
    model = _PlainModel(raw)

    result = BodyClearanceSurveyor(model).inspect(frame)

    assert result.candidate is not None
    assert result.candidate.feature == feature
    assert result.candidate.point == pytest.approx((0.5, 0.42))
    assert result.candidate.confidence == 0.8
    assert result.raw_response == raw
    assert result.latency_ms == 12.5
    assert result.evidence.frame_id == frame.frame_id
    assert result.evidence.captured_ns == frame.captured_ns
    assert result.evidence.region_kind == EvidenceRegion.WORLD
    assert result.evidence.region == ScreenRegion(x=0, y=0, width=1, height=0.84)
    assert (result.evidence.crop_width, result.evidence.crop_height) == (320, 168)
    source = Image.frombytes("RGBA", (320, 200), frame.bgra, "raw", "BGRA").convert("RGB")
    expected = source.crop((0, 0, 320, 168))
    assert result.evidence.pixel_sha256 == hashlib.sha256(expected.tobytes()).hexdigest()
    assert len(model.calls) == 1
    image = Image.open(io.BytesIO(model.calls[0][2])).convert("RGB")
    assert image.size == expected.size
    assert image.tobytes() == expected.tobytes()  # No labels, header, or collage.
    assert not hasattr(result, "claims")
    assert not hasattr(result, "tracks")
    assert not hasattr(result.candidate, "action_permissions")


def test_body_clearance_downsizes_model_image_but_hashes_original_crop() -> None:
    frame = _frame(640, 400)
    model = _PlainModel('{"feature":"unknown","point":null,"confidence":null}')
    result = BodyClearanceSurveyor(model).inspect(frame)
    image = Image.open(io.BytesIO(model.calls[0][2]))
    assert image.width <= 512
    assert image.height == pytest.approx(image.width * 336 / 640, abs=1)
    assert (result.evidence.crop_width, result.evidence.crop_height) == (640, 336)
    source = Image.frombytes("RGBA", (640, 400), frame.bgra, "raw", "BGRA").convert("RGB")
    assert result.evidence.pixel_sha256 == hashlib.sha256(
        source.crop((0, 0, 640, 336)).tobytes()
    ).hexdigest()


def test_body_clearance_odd_height_matches_existing_world_crop_bounds() -> None:
    frame = _frame(320, 181)
    model = _PlainModel('{"feature":"riser","point":[0.5,0.5],"confidence":0.9}')
    result = BodyClearanceSurveyor(model).inspect(frame)
    # Existing WORLD uses ceil(0.84 * height), not truncation to an earlier row.
    assert result.evidence.crop_height == 153
    assert result.evidence.region.height == pytest.approx(153 / 181)
    assert result.candidate is not None
    assert result.candidate.point == pytest.approx((0.5, 0.5 * 153 / 181))
    source = Image.frombytes("RGBA", (320, 181), frame.bgra, "raw", "BGRA").convert("RGB")
    crop = source.crop((0, 0, 320, 153))
    assert result.evidence.pixel_sha256 == hashlib.sha256(crop.tobytes()).hexdigest()
    image = Image.open(io.BytesIO(model.calls[0][2])).convert("RGB")
    assert image.tobytes() == crop.tobytes()


@pytest.mark.parametrize("feature,point,confidence", (
    ("unknown", None, None), (None, None, None),
    ("unknown", [0.5, 0.5], 1.0), (None, [0.5, 0.5], 1.0),
    ("riser", None, 0.9), ("riser", [0.5, 0.5], None),
    ("riser", [0.5, 0.5], 0),
))
def test_body_clearance_abstentions_never_create_candidates(
    feature: str | None, point: list[float] | None, confidence: float | None,
) -> None:
    model = _PlainModel(json.dumps({"feature": feature, "point": point, "confidence": confidence}))
    assert BodyClearanceSurveyor(model).inspect(_frame()).candidate is None
    assert len(model.calls) == 1


@pytest.mark.parametrize("patch", (
    {"feature": "head"}, {"feature": True}, {"feature": 2},
    {"point": [True, 0.5]}, {"point": [0.5, False]},
    {"point": [float("nan"), 0.5]}, {"point": [float("inf"), 0.5]},
    {"point": [-0.01, 0.5]}, {"point": [0.5, 1.01]},
    {"point": [0.5]}, {"point": [0.5, 0.5, 0.5]}, {"point": "0.5,0.5"},
    {"confidence": True}, {"confidence": float("nan")},
    {"confidence": -0.1}, {"confidence": 1.1}, {"confidence": "0.9"},
    {"target.mineable": True}, {"action": "attack"},
))
def test_body_clearance_invalid_wire_fails_without_retry(patch: dict[str, object]) -> None:
    wire: dict[str, object] = {"feature": "riser", "point": [0.5, 0.5], "confidence": 0.9}
    wire.update(patch)
    model = _PlainModel(json.dumps(wire))
    with pytest.raises(ValueError):
        BodyClearanceSurveyor(model).inspect(_frame())
    assert len(model.calls) == 1


@pytest.mark.parametrize("raw", (
    "not JSON", "[]", "null", "{}", "true", "12", '"riser"',
    '{"feature":"riser","point":[0.5,0.5]}',
    '{"feature":"riser","confidence":0.9}',
    '{"point":[0.5,0.5],"confidence":0.9}',
))
def test_body_clearance_requires_exact_complete_object(raw: str) -> None:
    model = _PlainModel(raw)
    with pytest.raises(ValueError):
        BodyClearanceSurveyor(model).inspect(_frame())
    assert len(model.calls) == 1


@pytest.mark.parametrize("duplicate", (
    '"feature":"unknown",', '"point":null,', '"confidence":0,',
))
def test_body_clearance_duplicate_fields_are_not_last_value_wins(duplicate: str) -> None:
    raw = "{" + duplicate + '"feature":"riser","point":[0.5,0.5],"confidence":0.9}'
    model = _PlainModel(raw)
    with pytest.raises(ValueError):
        BodyClearanceSurveyor(model).inspect(_frame())
    assert len(model.calls) == 1


@pytest.mark.parametrize("model_type,route", (
    (_PlainModel, "plain"), (_StructuredModel, "structured"), (_ConstrainedModel, "constrained"),
))
def test_body_clearance_uses_best_available_model_contract_once(
    model_type: type[_PlainModel], route: str,
) -> None:
    model = model_type('{"feature":"side_face","point":[0,1],"confidence":1}')
    result = BodyClearanceSurveyor(model).inspect(_frame())
    assert len(model.calls) == 1
    assert model.calls[0][0] == route
    assert result.candidate is not None
    assert result.candidate.point == pytest.approx((0, 0.84))
    metadata = model.calls[0][3]
    if route != "plain":
        assert metadata["name"]
        schema = metadata["schema"]
        assert isinstance(schema, dict)
        assert schema.get("additionalProperties") is False
        assert set(schema["required"]) == {"feature", "point", "confidence"}
    if route == "constrained":
        assert isinstance(metadata["grammar"], str)
        assert "root" in metadata["grammar"]
