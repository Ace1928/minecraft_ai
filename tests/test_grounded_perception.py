from __future__ import annotations

import io
import time

import pytest
from PIL import Image

from minecraft_ai.grounded_perception import (
    ClaimStatus,
    GroundedClaim,
    GroundedPerceptionReport,
    GroundedTrack,
    GroundedVLMResponse,
    RejectionCode,
    SegmentedFrameBuilder,
    resolve_grounded_output_keys,
    validate_grounded_response,
)
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
