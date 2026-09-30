from __future__ import annotations

import hashlib
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

import minecraft_ai.perception.service as service_module
from minecraft_ai.perception import PerceptionBlackboard
from minecraft_ai.perception_service import (
    BEDROCK_CLASSIC_HEALTH_SOURCE,
    BootstrapFastPerception,
    RealtimePerceptionService,
    bedrock_classic_health,
    bedrock_survival_hud_present,
)
from minecraft_ai.platforms.bedrock_x11 import CapturedFrame


pytest.importorskip("numpy")
FIXTURES = Path(__file__).parent / "fixtures" / "bedrock_health"
VERSION = "1.26.52.3"
CAPTURED_NS = 1_000_000_000


@lru_cache
def _image(name: str) -> Image.Image:
    return Image.open(FIXTURES / name).convert("RGBA")


def _frame(image: Image.Image, *, captured_ns: int = CAPTURED_NS) -> CapturedFrame:
    return CapturedFrame(
        frame_id=1, captured_ns=captured_ns, width=image.width, height=image.height,
        bgra=image.tobytes("raw", "BGRA"),
    )


def _read(image: Image.Image) -> int | None:
    return bedrock_classic_health(_frame(image), game_version=VERSION)


@pytest.mark.parametrize(
    ("name", "sha256", "expected"),
    [
        (
            "full_health_1920x1080.png",
            "8d04311268d7d1d132e98c10d8feb31a1e7a7241fe4827a0ba954a63172b5191",
            20,
        ),
        (
            "low_health_1920x1080.jpg",
            "2b38959564549be5832f1ff40128f0a40a2af87968affa3a009e5f94ce4afcc5",
            5,
        ),
        (
            "low_health_jitter_1920x1080.jpg",
            "05d5a80c3aa63c3acb0bd9e1c8827c979536613b4fc2a05d8d57c99d74141037",
            None,
        ),
    ],
)
def test_exact_retained_health_capture(name: str, sha256: str, expected: int | None) -> None:
    # File identity is independent of the decoder: do not regenerate these captures.
    assert hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest() == sha256
    image = _image(name)
    assert bedrock_survival_hud_present(_frame(image))
    assert _read(image) == expected


@pytest.mark.parametrize("version", [None, "1.26.45.1", "1.26.52.4", "1.26.52.3-modified"])
def test_unknown_or_different_build_has_no_health_authority(version: str | None) -> None:
    frame = _frame(_image("full_health_1920x1080.png"))
    assert bedrock_classic_health(frame, game_version=version) is None
    assert not any(
        fact.key.startswith("player.health")
        for fact in BootstrapFastPerception(game_version=version).infer(frame)
    )


@pytest.mark.parametrize("size", [(1280, 720), (960, 540), (1920, 1079), (1921, 1080)])
def test_scale_or_clipped_capture_abstains(size: tuple[int, int]) -> None:
    image = _image("full_health_1920x1080.png").resize(size)
    assert _read(image) is None


def test_incomplete_pixels_or_unavailable_numpy_abstains(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _frame(_image("full_health_1920x1080.png"))
    incomplete = replace(frame, bgra=frame.bgra[:-4])
    assert bedrock_classic_health(incomplete, game_version=VERSION) is None
    monkeypatch.setattr(service_module, "_numpy_bgra", lambda frame: None)
    assert bedrock_classic_health(frame, game_version=VERSION) is None
    assert not any(
        fact.key == "player.health"
        for fact in BootstrapFastPerception(game_version=VERSION).infer(frame)
    )


def test_real_death_and_translucent_server_menu_publish_no_health() -> None:
    menu = Image.open(
        FIXTURES.parent / "bedrock_menu" / "server_list_transfer_1920x1080.png"
    ).convert("RGBA")
    for image in (_image("death_1920x1080.png"), menu):
        frame = _frame(image)
        assert _read(image) is None
        assert not any(
            fact.key in ("player.health", "player.health_fraction", "player.critical_health")
            for fact in BootstrapFastPerception(game_version=VERSION).infer(frame)
        )


@pytest.mark.parametrize("broken", ["rail", "dividers", "bank", "world_stripe"])
def test_independent_hud_geometry_is_required(broken: str) -> None:
    image = _image("full_health_1920x1080.png").copy()
    draw = ImageDraw.Draw(image)
    if broken == "rail":
        draw.rectangle((595, 986, 1324, 995), fill=(25, 25, 25, 255))
    elif broken == "dividers":
        draw.rectangle((594, 1000, 1324, 1050), fill=(25, 25, 25, 255))
    elif broken == "bank":
        draw.rectangle((595, 916, 920, 952), fill=(255, 19, 19, 255))
    else:
        image = Image.new("RGBA", (1920, 1080), (20, 20, 20, 255))
        ImageDraw.Draw(image).rectangle((596, 916, 920, 952), fill=(255, 19, 19, 255))
    assert _read(image) is None


def _derived_bank(states: tuple[int, ...]) -> Image.Image:
    """Synthetic test control assembled from actual captured glyphs, never an observation."""
    assert len(states) == 10
    full = _image("full_health_1920x1080.png")
    low = _image("low_health_1920x1080.jpg")
    # These source rectangles are independently measured captured icons. No
    # decoder template or palette is used to manufacture expected test pixels.
    glyphs = {
        2: full.crop((596, 916, 632, 952)),
        1: low.crop((660, 916, 696, 952)),
        0: low.crop((724, 916, 760, 952)),
    }
    image = full.copy()
    for index, state in enumerate(states):
        image.paste(glyphs[state], (596 + 32 * index, 916))
    return image


@pytest.mark.parametrize("units", range(21))
def test_full_half_empty_units_using_derived_captured_glyphs(units: int) -> None:
    states = [2] * (units // 2) + ([1] if units % 2 else [])
    states += [0] * (10 - len(states))
    assert _read(_derived_bank(tuple(states))) == units


@pytest.mark.parametrize(
    "states",
    [(2, 0, 2, 0, 0, 0, 0, 0, 0, 0), (1, 1, 0, 0, 0, 0, 0, 0, 0, 0)],
)
def test_nonmonotone_or_multiple_half_hearts_abstain(states: tuple[int, ...]) -> None:
    assert _read(_derived_bank(states)) is None


@pytest.mark.parametrize(
    "mutation",
    ["poison", "wither", "frozen", "absorption", "flash", "missing_cell", "jitter"],
)
def test_effects_flash_or_animation_abstain(mutation: str) -> None:
    image = _image("full_health_1920x1080.png").copy()
    draw = ImageDraw.Draw(image)
    colors = {
        "poison": (140, 170, 25, 255),
        "wither": (55, 55, 55, 255),
        "frozen": (55, 185, 255, 255),
        "absorption": (255, 205, 25, 255),
        "flash": (255, 255, 255, 255),
        "missing_cell": (20, 70, 20, 255),
    }
    if mutation == "jitter":
        glyph = image.crop((596, 916, 632, 952))
        draw.rectangle((596, 916, 631, 951), fill=(20, 20, 20, 255))
        image.paste(glyph, (596, 912))
    elif mutation == "flash":
        # Black top outline cell becomes the white damage/healing outline.
        draw.rectangle((604, 916, 607, 919), fill=colors[mutation])
    else:
        draw.rectangle((608, 928, 611, 931), fill=colors[mutation])
    assert _read(image) is None


@pytest.mark.parametrize("color", [(255, 19, 19, 255), (255, 205, 25, 255)])
def test_extra_colored_health_row_is_unresolved(color: tuple[int, int, int, int]) -> None:
    image = _image("full_health_1920x1080.png").copy()
    ImageDraw.Draw(image).rectangle((608, 892, 619, 903), fill=color)
    assert _read(image) is None


class _Capture:
    def __init__(self, frame: CapturedFrame) -> None:
        self.frame = frame

    def capture(self) -> CapturedFrame:
        return self.frame

    def close(self) -> None:
        pass


@pytest.mark.parametrize(
    ("instance", "health"),
    [
        ("bedrock:1.26.52.3:retained-test", 20),
        ("bedrock:1.26.45.1:retained-test", None),
        ("java:1.26.52.3:retained-test", None),
        ("bedrock:1.26.52.3", None),
        ("bedrock:test", None),
    ],
)
def test_realtime_health_version_is_bound_to_authoritative_instance(
    instance: str, health: int | None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service_module.time, "monotonic_ns", lambda: CAPTURED_NS)
    frame = _frame(_image("full_health_1920x1080.png"))
    board = PerceptionBlackboard()
    # A caller-supplied version cannot override the service's actual instance.
    service = RealtimePerceptionService(
        capture_source=_Capture(frame), blackboard=board, instance_id=instance,
        fast_perception=BootstrapFastPerception(game_version=VERSION),
    )
    service.capture_once()
    fact = board.fact("player.health", min_confidence=0.8, now_ns=CAPTURED_NS)
    assert (fact.value if fact is not None else None) == health


def test_captured_damage_feedback_is_fresh_and_never_a_training_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [CAPTURED_NS]
    monkeypatch.setattr(service_module.time, "monotonic_ns", lambda: clock[0])
    board = PerceptionBlackboard()
    capture = _Capture(_frame(_image("full_health_1920x1080.png")))
    service = RealtimePerceptionService(
        capture_source=capture, blackboard=board, instance_id=f"bedrock:{VERSION}:retained-test",
    )
    service.capture_once()
    full = board.fact("player.health", min_confidence=0.8)
    assert full is not None and full.value == 20
    clock[0] += 50_000_000
    capture.frame = _frame(_image("low_health_1920x1080.jpg"), captured_ns=clock[0])
    service.capture_once()
    low = board.fact("player.health", min_confidence=0.8)
    assert low is not None and low.value == 5
    # This is the same 0..20, >=.8-confidence read contract consumed by
    # ERAIS AdaptiveInteractionPolicy. It is captured damage, not a model label.
    assert low.value < full.value
    assert low.source == BEDROCK_CLASSIC_HEALTH_SOURCE
    assert low.source.endswith(":not-training-label")
    assert low.observed_ns == capture.frame.captured_ns
    assert low.expires_after_ms == 250
    fraction = board.fact("player.health_fraction")
    critical = board.fact("player.critical_health")
    assert fraction is not None and fraction.value == 0.25
    assert critical is not None and critical.value is True
    assert board.fact("player.health", now_ns=low.observed_ns + 251_000_000) is None
    assert isinstance(service.fast_perception, BootstrapFastPerception)
    assert service.fast_perception.training_label_eligible is False
