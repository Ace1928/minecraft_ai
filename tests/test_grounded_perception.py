from __future__ import annotations

import io
import hashlib
import json
import re
import time
from dataclasses import replace

import pytest
from PIL import Image
from pydantic import ValidationError

from minecraft_ai.grounded_perception import (
    ClaimStatus,
    GroundedClaim,
    GroundedPerceptionHarness,
    GroundedPerceptionReport,
    GroundedTrack,
    GroundedVLMResponse,
    RejectionCode,
    SegmentedFrameBuilder,
    resolve_grounded_output_keys,
    validate_grounded_response,
    _CROSSHAIR_PROBE_KEYS,
    _CompactCrosshairResponse,
    _build_crosshair_block_evidence,
    _build_crosshair_evidence,
    _crosshair_block_grammar,
    _crosshair_probe_requested,
    _crosshair_grammar,
    _expand_crosshair_block_response,
    _expand_crosshair_response,
    crosshair_block_crop_dimensions,
    crosshair_block_region,
    crosshair_block_rgb_grid,
    crosshair_block_rgb_grid_distance,
    crosshair_block_visually_equivalent,
)
from minecraft_ai.mining_control import is_hand_safe_soft_block
from minecraft_ai.models import ModelResponse
from minecraft_ai.perception import (
    EvidenceRegion,
    FrameState,
    PerceptionBlackboard,
    PerceptionEvidence,
    PerceptionFact,
    ScreenRegion,
)
from minecraft_ai.platforms.bedrock_x11 import CapturedFrame


def _frame(*, width: int = 320, height: int = 180) -> CapturedFrame:
    pixels = b"".join(
        bytes((x % 256, y % 256, (x + y) % 256, 255)) for y in range(height) for x in range(width)
    )
    return CapturedFrame(
        frame_id=7,
        captured_ns=123_456,
        width=width,
        height=height,
        bgra=pixels,
    )


def _evidence(kind: EvidenceRegion) -> PerceptionEvidence:
    regions = {
        EvidenceRegion.WORLD: ScreenRegion(x=0.0, y=0.0, width=1.0, height=0.84),
        EvidenceRegion.HUD: ScreenRegion(x=0.2, y=0.74, width=0.6, height=0.26),
        EvidenceRegion.HOTBAR: ScreenRegion(x=0.25, y=0.82, width=0.5, height=0.18),
        EvidenceRegion.CHAT: ScreenRegion(x=0.0, y=0.04, width=0.58, height=0.52),
        EvidenceRegion.GUI: ScreenRegion(x=0.12, y=0.04, width=0.76, height=0.9),
    }
    return PerceptionEvidence(
        evidence_id=f"frame-7:{kind.value}",
        frame_id=7,
        captured_ns=123_456,
        region_kind=kind,
        region=regions[kind],
        pixel_sha256=kind.value.encode().hex().ljust(64, "0")[:64],
        crop_width=160,
        crop_height=90,
    )


def _claim(
    key: str,
    value: bool | int | float | str,
    evidence_id: str,
    *,
    confidence: float = 0.9,
) -> GroundedClaim:
    return GroundedClaim(
        key=key,
        status=ClaimStatus.OBSERVED,
        value=value,
        confidence=confidence,
        evidence_ids=(evidence_id,),
    )


def _report(
    claims: tuple[GroundedClaim, ...],
    *,
    evidence: tuple[PerceptionEvidence, ...] | None = None,
    tracks: tuple[GroundedTrack, ...] = (),
    summary: str = "",
    requested_keys: tuple[str, ...] = (),
) -> GroundedPerceptionReport:
    manifest = tuple(EvidenceRegion)
    resolved_evidence = (
        tuple(_evidence(kind) for kind in manifest) if evidence is None else evidence
    )
    return validate_grounded_response(
        GroundedVLMResponse(
            uncertainty=0.2,
            prose_summary=summary,
            claims=claims,
            tracks=tracks,
        ),
        frame_id=7,
        evidence=resolved_evidence,
        requested_keys=requested_keys,
    )


def test_segmented_frame_builder_produces_content_addressed_explicit_regions() -> None:
    segmented = SegmentedFrameBuilder(panel_width=320, panel_height=180).build(
        _frame(),
        frame_id=7,
    )

    assert segmented.composite_png.startswith(b"\x89PNG")
    assert {item.region_kind for item in segmented.evidence} == set(EvidenceRegion)
    assert all(item.frame_id == 7 for item in segmented.evidence)
    assert all(len(item.pixel_sha256) == 64 for item in segmented.evidence)
    assert all(item.crop_width > 0 and item.crop_height > 0 for item in segmented.evidence)


def test_single_region_sheet_has_no_unused_second_panel() -> None:
    segmented = SegmentedFrameBuilder().build(
        _frame(),
        frame_id=7,
        region_kinds=(EvidenceRegion.WORLD,),
    )

    with Image.open(io.BytesIO(segmented.composite_png)) as image:
        assert image.size == (512, 288)


def _compact_answer(*, block: str = "mossy_cobblestone") -> dict[str, object]:
    return {
        "scene": "world",
        "playable": True,
        "danger": False,
        "visible": True,
        "block": block,
        "mineable": True,
        "dx": 0.0,
        "dy": 0.0,
        "box": [0.4, 0.4, 0.2, 0.2],
        "confidence": 0.85,
    }


def test_dedicated_crosshair_block_probe_is_one_exact_crop_and_two_fields() -> None:
    class _Model:
        model_id = "strict-crosshair-model"

        def inspect(self, prompt: str, *, image_bytes: bytes, mime_type: str) -> ModelResponse:
            raise AssertionError("native constrained path required")

        def inspect_constrained(
            self,
            prompt: str,
            *,
            image_bytes: bytes,
            mime_type: str,
            name: str,
            schema: dict[str, object],
            grammar: str,
        ) -> ModelResponse:
            self.prompt = prompt
            self.image = image_bytes
            self.name = name
            self.schema = schema
            self.grammar = grammar
            return ModelResponse(
                text='{"block":"dirt","confidence":0.85}',
                model=self.model_id,
                latency_ms=4,
            )

    model = _Model()
    frame = _frame(width=801, height=803)
    result = GroundedPerceptionHarness(model).inspect_crosshair_block_detailed(
        frame,
        frame_id=19,
    )
    assert result.report.observed_values() == {"recovery.crosshair.block": "dirt"}
    assert result.report.confidence_by_key() == {"recovery.crosshair.block": 0.85}
    assert result.report.tracks == ()
    assert len(result.report.evidence) == 1
    evidence = result.report.evidence[0]
    assert evidence.evidence_id == "frame-19:crosshair-block"
    assert evidence.frame_id == 19
    assert evidence.captured_ns == frame.captured_ns
    assert evidence.crop_width == evidence.crop_height == 511
    with Image.open(io.BytesIO(model.image)) as image:
        assert image.size == (511, 511)
    assert model.name == "minecraft_crosshair_block"
    assert set(model.schema["required"]) == {"block", "confidence"}
    assert "danger" not in model.prompt
    assert "mineable" not in model.prompt
    assert "box" not in model.prompt
    assert "dx" not in model.prompt
    assert '"\\\"block\\\""' in model.grammar


@pytest.mark.parametrize(
    "wire",
    (
        '{"block":"unknown","confidence":0.9}',
        '{"block":null,"confidence":0.9}',
        '{"block":"dirt","confidence":null}',
        '{"block":"dirt","confidence":0}',
    ),
)
def test_dedicated_crosshair_block_abstentions_emit_no_fact(wire: str) -> None:
    evidence = _build_crosshair_block_evidence(_frame(), frame_id=7).evidence[0]
    assert _expand_crosshair_block_response(wire, evidence).claims == ()


def test_dedicated_crosshair_block_hard_negative_remains_explicit() -> None:
    evidence = _build_crosshair_block_evidence(_frame(), frame_id=7).evidence[0]
    response = _expand_crosshair_block_response(
        '{"block":"mossy_cobblestone","confidence":0.91}', evidence
    )
    assert response.claims[0].value == "mossy_cobblestone"
    assert not is_hand_safe_soft_block(str(response.claims[0].value))


@pytest.mark.parametrize(
    "wire",
    (
        '{"block":"dirt","confidence":0.9,"danger":false}',
        '{"block":"dirt","block":"stone","confidence":0.9}',
        '{"block":"diamond_block","confidence":0.9}',
        '{"block":"dirt","confidence":"0.9"}',
    ),
)
def test_dedicated_crosshair_block_invalid_wire_fails_without_repair(wire: str) -> None:
    evidence = _build_crosshair_block_evidence(_frame(), frame_id=7).evidence[0]
    with pytest.raises((ValidationError, ValueError)):
        _expand_crosshair_block_response(wire, evidence)


def test_crosshair_rgb_signature_detects_equal_luma_center_color_swap() -> None:
    width = height = 512
    gray = bytearray(bytes((100, 100, 100, 255)) * width * height)
    cyan = bytearray(gray)
    for y in range(232, 281):
        for x in range(232, 281):
            offset = (y * width + x) * 4
            cyan[offset : offset + 4] = bytes((200, 132, 0, 255))
    gray_frame = CapturedFrame(
        frame_id=1, captured_ns=1, width=width, height=height, bgra=bytes(gray)
    )
    cyan_frame = replace(gray_frame, captured_ns=2, bgra=bytes(cyan))
    assert crosshair_block_rgb_grid_distance(
        crosshair_block_rgb_grid(gray_frame),
        crosshair_block_rgb_grid(cyan_frame),
    ) > 1
    grammar = _crosshair_block_grammar()
    assert 'probability ::= "0"' in grammar


@pytest.mark.parametrize(
    ("current_hash", "current_grid", "equivalent"),
    (
        ("0000000000000007", (bytes((1, 0)) * 384).hex(), True),
        ("000000000000000f", (bytes((1,)) * 768).hex(), True),
        ("0000000000000007", (bytes((2,)) * 768).hex(), False),
        ("000000000000001f", (bytes((0,)) * 768).hex(), False),
    ),
)
def test_crosshair_visual_equivalence_requires_small_hash_and_rgb_drift(
    current_hash: str,
    current_grid: str,
    equivalent: bool,
) -> None:
    assert (
        crosshair_block_visually_equivalent(
            "0" * 16,
            current_hash,
            (bytes((0,)) * 768).hex(),
            current_grid,
        )
        is equivalent
    )


@pytest.mark.parametrize(
    ("reference_hash", "reference_grid"),
    (
        ("-000000000000001", (bytes((0,)) * 768).hex()),
        ("0000000000000_00", (bytes((0,)) * 768).hex()),
        ("0" * 16, (bytes((0,)) * 768).hex() + " "),
    ),
)
def test_crosshair_visual_equivalence_rejects_noncanonical_hex(
    reference_hash: str,
    reference_grid: str,
) -> None:
    assert not crosshair_block_visually_equivalent(
        reference_hash,
        "0" * 16,
        reference_grid,
        (bytes((0,)) * 768).hex(),
    )


@pytest.mark.parametrize(("width", "height"), ((1280, 720), (65, 64), (64, 65)))
def test_crosshair_crop_is_exactly_centered_in_source_pixels(width: int, height: int) -> None:
    region = crosshair_block_region(width, height)
    crop_width, crop_height = crosshair_block_crop_dimensions(width, height)
    x0 = round(region.x * width)
    y0 = round(region.y * height)
    assert 2 * x0 + crop_width == width
    assert 2 * y0 + crop_height == height
    assert round(region.width * width) == crop_width
    assert round(region.height * height) == crop_height


def test_crosshair_crop_preserves_exact_original_pixel_provenance() -> None:
    frame = _frame(width=800, height=800)
    segmented = _build_crosshair_evidence(frame, frame_id=7)
    world, center = segmented.evidence
    assert world.region_kind == center.region_kind == EvidenceRegion.WORLD
    assert world.evidence_id == "frame-7:world"
    assert center.evidence_id == "frame-7:crosshair"
    assert center.region == ScreenRegion(x=0.18, y=0.18, width=0.64, height=0.64)
    assert center.crop_width == center.crop_height == 512
    source = Image.frombytes("RGBA", (800, 800), frame.bgra, "raw", "BGRA").convert("RGB")
    assert center.pixel_sha256 == hashlib.sha256(
        source.crop((144, 144, 656, 656)).tobytes()
    ).hexdigest()
    with Image.open(io.BytesIO(segmented.composite_png)) as image:
        assert image.size == (1024, 512)


def test_compact_crosshair_negative_remains_exact_unsafe_block() -> None:
    evidence = _build_crosshair_evidence(_frame(width=800, height=800), frame_id=7).evidence
    compact = _compact_answer()
    compact["dx"] = 0.1
    compact["dy"] = -0.1
    expanded = _expand_crosshair_response(json.dumps(compact), evidence)
    report = validate_grounded_response(
        expanded, frame_id=7, evidence=evidence, requested_keys=_CROSSHAIR_PROBE_KEYS
    )

    values = report.observed_values()
    assert values["target.kind"] == "mossy_cobblestone"
    assert is_hand_safe_soft_block(str(values["target.kind"])) is False
    assert values["target.dx"] == pytest.approx(0.064)
    assert values["target.dy"] == pytest.approx(-0.064)
    assert report.confidence_by_key()["target.kind"] == 0.85
    assert report.evidence_by_key()["target.kind"] == ("frame-7:crosshair",)
    assert report.evidence_by_key()["danger.immediate"] == ("frame-7:world",)
    assert report.tracks[0].label == "mossy_cobblestone"
    assert report.tracks[0].x == pytest.approx(0.436)
    assert report.tracks[0].y == pytest.approx(0.436)
    assert report.tracks[0].width == pytest.approx(0.128)
    assert report.tracks[0].height == pytest.approx(0.128)
    assert report.tracks[0].confidence == 0.85
    assert not report.rejections


def test_compact_crosshair_nulls_never_create_positive_or_safety_defaults() -> None:
    evidence = _build_crosshair_evidence(_frame(), frame_id=7).evidence
    wire = {key: None for key in _compact_answer()}
    raw = _expand_crosshair_response(json.dumps(wire), evidence)
    report = validate_grounded_response(
        raw, frame_id=7, evidence=evidence, requested_keys=_CROSSHAIR_PROBE_KEYS
    )
    assert report.observed_values() == {}
    assert report.tracks == ()
    assert report.uncertainty == 1.0


def test_compact_crosshair_explicit_dirt_keeps_all_eight_observations() -> None:
    evidence = _build_crosshair_evidence(_frame(), frame_id=7).evidence
    raw = _expand_crosshair_response(json.dumps(_compact_answer(block="dirt")), evidence)
    report = validate_grounded_response(
        raw, frame_id=7, evidence=evidence, requested_keys=_CROSSHAIR_PROBE_KEYS
    )
    values = report.observed_values()
    assert set(values) == set(_CROSSHAIR_PROBE_KEYS)
    assert values["target.kind"] == "dirt"
    assert is_hand_safe_soft_block(str(values["target.kind"])) is True
    assert values["target.dx"] == values["target.dy"] == 0
    assert all(value == 0.85 for value in report.confidence_by_key().values())
    assert report.tracks[0].label == "dirt"
    assert not report.rejections


@pytest.mark.parametrize("confidence", (None, 0.0))
def test_compact_crosshair_requires_explicit_positive_shared_confidence(
    confidence: float | None,
) -> None:
    evidence = _build_crosshair_evidence(_frame(), frame_id=7).evidence
    wire = _compact_answer(block="dirt")
    wire["confidence"] = confidence
    raw = _expand_crosshair_response(json.dumps(wire), evidence)
    assert raw.claims == raw.tracks == ()
    assert raw.uncertainty == 1


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("danger", 0),
        ("playable", 1),
        ("confidence", 1.5),
        ("dx", "0"),
    ),
)
def test_compact_crosshair_rejects_wrong_types_and_confidence(
    key: str, value: object
) -> None:
    wire = _compact_answer()
    wire[key] = value
    with pytest.raises(ValidationError):
        _CompactCrosshairResponse.model_validate_json(json.dumps(wire))


@pytest.mark.parametrize(
    "box", ([0.9, 0.4, 0.2, 0.2], [0.4, 0.4, 0, 0.2], [0.5, 0.5, 0.2, 0.2])
)
def test_compact_crosshair_does_not_clip_invalid_box_into_a_target(box: list[object]) -> None:
    evidence = _build_crosshair_evidence(_frame(), frame_id=7).evidence
    wire = _compact_answer(block="dirt")
    wire["box"] = box
    raw = _expand_crosshair_response(json.dumps(wire), evidence)
    report = validate_grounded_response(
        raw, frame_id=7, evidence=evidence, requested_keys=_CROSSHAIR_PROBE_KEYS
    )
    assert report.tracks == ()
    assert "target.visible" not in report.observed_values()
    assert "target.kind" not in report.observed_values()


def test_compact_crosshair_boundary_check_uses_exact_centered_crop_origin() -> None:
    evidence = _build_crosshair_evidence(_frame(width=801, height=803), frame_id=7).evidence
    center = evidence[1].region
    crosshair_x = (0.5 - center.x) / center.width
    assert crosshair_x == 0.5
    wire = _compact_answer(block="dirt")
    wire["box"] = [crosshair_x, 0.4, 0.2, 0.2]
    assert _expand_crosshair_response(json.dumps(wire), evidence).tracks == ()


def test_compact_crosshair_duplicate_safety_field_fails_closed() -> None:
    evidence = _build_crosshair_evidence(_frame(), frame_id=7).evidence
    wire = json.dumps(_compact_answer()).replace(
        '"danger": false', '"danger": true, "danger": false'
    )
    with pytest.raises(ValueError, match="duplicate compact crosshair field: danger"):
        _expand_crosshair_response(wire, evidence)


def test_exact_crosshair_harness_uses_compact_nullable_contract() -> None:
    class _Model:
        model_id = "test-compact"

        def inspect(self, *_args: object, **_kwargs: object) -> ModelResponse:
            raise AssertionError("compact probe should prefer enforced grammar")

        def inspect_constrained(self, prompt: str, **kwargs: object) -> ModelResponse:
            self.prompt = prompt
            self.schema = kwargs["schema"]
            self.grammar = kwargs["grammar"]
            self.image = kwargs["image_bytes"]
            return ModelResponse(
                text=json.dumps(_compact_answer()), model=self.model_id, latency_ms=1
            )

    model = _Model()
    inspection = GroundedPerceptionHarness(model, compact_crosshair_probe=True).inspect_detailed(
        _frame(), frame_id=7, question="Inspect only the crosshair block.",
        output_keys=_CROSSHAIR_PROBE_KEYS,
    )
    assert inspection.report.observed_values()["target.kind"] == "mossy_cobblestone"
    assert not inspection.schema_repaired
    assert "never choose adjacent easier dirt" in model.prompt
    assert "safe empty result" not in model.prompt
    assert isinstance(model.grammar, str) and '"\\\"mossy_cobblestone\\\""' in model.grammar
    assert isinstance(model.schema, dict)
    assert set(model.schema["required"]) == set(_compact_answer())
    assert all("default" not in field for field in model.schema["properties"].values())
    # Complete compact wire stays well below the old 1,141-char / 343-token answer.
    assert len(json.dumps(_compact_answer(), separators=(",", ":"))) < 250
    assert _crosshair_probe_requested(_CROSSHAIR_PROBE_KEYS, "Inspect crosshair") is True
    assert _crosshair_probe_requested(_CROSSHAIR_PROBE_KEYS, "Find a tree") is False


def test_crosshair_harness_defaults_to_production_grounded_contract() -> None:
    class _Model:
        model_id = "test-production-default"

        def inspect(self, *_args: object, **_kwargs: object) -> ModelResponse:
            raise AssertionError("grounded queries should prefer enforced grammar")

        def inspect_constrained(self, prompt: str, **kwargs: object) -> ModelResponse:
            self.schema = kwargs["schema"]
            return ModelResponse(
                text=GroundedVLMResponse(uncertainty=1.0).model_dump_json(),
                model=self.model_id,
                latency_ms=1,
            )

    model = _Model()
    inspection = GroundedPerceptionHarness(model).inspect_detailed(
        _frame(), frame_id=7, question="Inspect only the crosshair block.",
        output_keys=_CROSSHAIR_PROBE_KEYS,
    )
    assert isinstance(model.schema, dict)
    assert "claims" in model.schema["properties"]
    assert "block" not in model.schema["properties"]
    assert all(item.evidence_id != "frame-7:crosshair" for item in inspection.report.evidence)
    assert inspection.report.observed_values() == {}
    assert not inspection.schema_repaired


@pytest.mark.parametrize(
    ("text", "probability_ok", "offset_ok"),
    (
        ("0", True, True), ("1", True, True), ("0.5", True, True),
        ("0.999999", True, True), ("1.000", True, True),
        ("-0", False, True), ("-0.9", False, True), ("-1.000", False, True),
        ("-2", False, False), ("2", False, False), ("1.001", False, False),
        ("-1.001", False, False), ("00", False, False), ("0.", False, False),
        ("1e9", False, False), ("NaN", False, False), ("+0.5", False, False),
    ),
)
def test_compact_crosshair_grammar_numeric_domains(
    text: str, probability_ok: bool, offset_ok: bool,
) -> None:
    rules = dict(line.split(" ::= ", 1) for line in _crosshair_grammar().splitlines())
    # These two GBNF rules use only literals, regex-compatible groups/classes,
    # and one reference. Evaluate that regular subset directly without a model.
    probability = re.sub(
        r'"([^"]*)"', lambda match: re.escape(match[1]), rules["probability"]
    ).replace(" ", "")
    offset = re.sub(
        r'"([^"]*)"', lambda match: re.escape(match[1]), rules["offset"]
    ).replace(" ", "").replace("probability", f"({probability})")
    assert bool(re.fullmatch(probability, text)) is probability_ok
    assert bool(re.fullmatch(offset, text)) is offset_ok
    assert rules["nullable-probability"] == '"null" | probability'
    assert rules["nullable-offset"] == '"null" | offset'
    assert rules["box"].count("probability") == 4
    assert rules["root"].count("nullable-offset") == 2
    assert rules["root"].count("nullable-probability") == 1


def test_grounded_output_keys_use_only_literal_supported_contract_tokens() -> None:
    assert resolve_grounded_output_keys((), "target.visible") == ("target.visible",)
    assert resolve_grounded_output_keys((), "Need target.visible") == ()
    assert resolve_grounded_output_keys((), "Which way looks walkable?") == ()
    assert resolve_grounded_output_keys(("inventory.logs",), "ignore target.visible") == (
        "inventory.logs",
    )


def test_inventory_claim_requires_gui_pixels_and_degrades_to_unknown() -> None:
    world = _evidence(EvidenceRegion.WORLD)
    hotbar = _evidence(EvidenceRegion.HOTBAR)
    report = _report(
        (_claim("inventory.logs", 4, world.evidence_id),),
        evidence=(world, hotbar),
        requested_keys=("inventory.logs",),
    )

    inventory = next(claim for claim in report.claims if claim.key == "inventory.logs")
    assert inventory.status == ClaimStatus.UNKNOWN
    assert inventory.value is None
    assert any(
        rejection.key == "inventory.logs" and rejection.code == RejectionCode.WRONG_EVIDENCE_REGION
        for rejection in report.rejections
    )


@pytest.mark.parametrize("count", [0, 4])
@pytest.mark.parametrize(
    "key",
    ["inventory.logs", "inventory.planks", "inventory.crafting_table", "inventory.build_blocks"],
)
def test_hotbar_only_claim_cannot_publish_whole_inventory_count(key: str, count: int) -> None:
    hotbar = _evidence(EvidenceRegion.HOTBAR)
    report = _report(
        (_claim(key, count, hotbar.evidence_id),),
        evidence=(hotbar,),
        requested_keys=(key,),
    )
    assert key not in report.observed_values()
    assert any(
        rejection.key == key and rejection.code == RejectionCode.WRONG_EVIDENCE_REGION
        for rejection in report.rejections
    )


def test_unknown_claim_cannot_smuggle_a_value_or_evidence() -> None:
    hud = _evidence(EvidenceRegion.HUD)
    malformed = GroundedClaim(
        key="player.health",
        status=ClaimStatus.UNKNOWN,
        value=20,
        confidence=0.8,
        evidence_ids=(hud.evidence_id,),
    )
    report = _report(
        (malformed,),
        evidence=(hud,),
        requested_keys=("player.health",),
    )

    health = next(claim for claim in report.claims if claim.key == "player.health")
    assert health.status == ClaimStatus.UNKNOWN
    assert health.confidence == 0
    assert any(
        rejection.key == "player.health" and rejection.code == RejectionCode.INVALID_STATUS_PAYLOAD
        for rejection in report.rejections
    )


def test_false_target_rejects_inconsistent_target_details() -> None:
    world = _evidence(EvidenceRegion.WORLD)
    report = _report(
        (
            _claim("target.visible", False, world.evidence_id),
            _claim("target.kind", "oak_log", world.evidence_id),
            _claim("target.dx", 0.2, world.evidence_id),
            _claim("target.dy", -0.1, world.evidence_id),
        ),
        evidence=(world,),
    )

    values = report.observed_values()
    assert values["target.visible"] is False
    assert "target.kind" not in values
    assert "target.dx" not in values
    assert "target.dy" not in values
    assert (
        sum(rejection.code == RejectionCode.CROSS_FIELD_CONFLICT for rejection in report.rejections)
        == 3
    )


def test_visible_target_requires_localized_track_and_offsets() -> None:
    world = _evidence(EvidenceRegion.WORLD)
    claims = (
        _claim("target.visible", True, world.evidence_id),
        _claim("target.dx", 0.0, world.evidence_id),
        _claim("target.dy", 0.1, world.evidence_id),
    )
    without_track = _report(claims, evidence=(world,))
    with_track = _report(
        claims,
        evidence=(world,),
        tracks=(
            GroundedTrack(
                label="oak_log",
                confidence=0.9,
                evidence_id=world.evidence_id,
                x=0.4,
                y=0.2,
                width=0.2,
                height=0.4,
            ),
        ),
    )

    assert "target.visible" not in without_track.observed_values()
    assert with_track.observed_values()["target.visible"] is True


def test_conflicting_cited_prose_is_rejected_and_never_becomes_summary() -> None:
    world = _evidence(EvidenceRegion.WORLD)
    report = _report(
        (_claim("scene.playable", False, world.evidence_id),),
        evidence=(world,),
        summary="[scene.playable=true @frame-7:world]",
    )

    assert not report.summary_accepted
    assert report.model_summary != report.deterministic_summary
    assert "scene.playable=false" in report.deterministic_summary
    assert any(rejection.code == RejectionCode.PROSE_CONFLICT for rejection in report.rejections)


def test_visible_slot_evidence_cross_checks_aggregate_inventory_count() -> None:
    hotbar = _evidence(EvidenceRegion.HOTBAR)
    gui = _evidence(EvidenceRegion.GUI)
    report = _report(
        (
            _claim("hotbar.slot.0.item", "minecraft:oak_log", hotbar.evidence_id),
            _claim("hotbar.slot.0.count", 3, hotbar.evidence_id),
            _claim("inventory.logs", 2, gui.evidence_id),
        ),
        evidence=(hotbar, gui),
    )

    values = report.observed_values()
    assert values["hotbar.slot.0.count"] == 3
    assert "inventory.logs" not in values
    assert any(
        rejection.key == "inventory.logs" and rejection.code == RejectionCode.CROSS_FIELD_CONFLICT
        for rejection in report.rejections
    )


def test_full_inventory_count_can_exceed_visible_hotbar_subtotal() -> None:
    hotbar = _evidence(EvidenceRegion.HOTBAR)
    gui = _evidence(EvidenceRegion.GUI)
    report = _report(
        (
            _claim("hotbar.slot.0.item", "minecraft:oak_log", hotbar.evidence_id),
            _claim("hotbar.slot.0.count", 3, hotbar.evidence_id),
            _claim("inventory.logs", 9, gui.evidence_id),
        ),
        evidence=(hotbar, gui),
    )
    assert report.observed_values()["inventory.logs"] == 9


def test_blackboard_keeps_only_manifests_referenced_by_fresh_facts() -> None:
    board = PerceptionBlackboard()
    board.publish(
        FrameState(
            frame_id=7,
            captured_ns=123_456,
            instance_id="bedrock:test",
            width=320,
            height=180,
        )
    )
    hotbar = _evidence(EvidenceRegion.HOTBAR)
    fact = PerceptionFact(
        key="inventory.logs",
        value=3,
        confidence=0.95,
        observed_ns=time.monotonic_ns(),
        source="vlm:test",
        expires_after_ms=10_000,
        evidence_refs=(hotbar.evidence_id,),
    )

    assert board.merge_semantics(
        instance_id="bedrock:test",
        facts=(fact,),
        evidence=(hotbar,),
    )
    latest = board.latest()
    assert latest is not None
    assert latest.facts[0].evidence_refs == (hotbar.evidence_id,)
    assert latest.evidence == (hotbar,)
