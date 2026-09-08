from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

import minecraft_ai.perception.service as service
from minecraft_ai.platforms.bedrock_x11 import CapturedFrame


@pytest.fixture(params=[False, True], ids=["numpy", "fallback"])
def pixel_path(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.param:
        monkeypatch.setattr(service, "_numpy_bgra", lambda _frame: None)
    else:
        pytest.importorskip("numpy")


def _inventory_frame(
    *,
    width: int = 640,
    height: int = 360,
    omitted: frozenset[str] = frozenset(),
    dark_slots: bool = False,
) -> CapturedFrame:
    image = Image.new("RGB", (width, height), (35, 55, 30))
    draw = ImageDraw.Draw(image)

    def fill(
        region: tuple[float, float, float, float],
        color: tuple[int, int, int],
    ) -> None:
        x0, y0, x1, y1 = region
        draw.rectangle(
            (int(width * x0), int(height * y0), int(width * x1) - 1, int(height * y1) - 1),
            fill=color,
        )

    # Populated recipe tiles are colorful, not neutral dark slot fill.
    fill((0.15, 0.18, 0.46, 0.75), (93, 93, 93) if dark_slots else (178, 50, 44))
    if "right_mid" not in omitted:
        fill((0.47, 0.18, 0.84, 0.52), (139, 139, 139))
    if "right_light" not in omitted:
        fill((0.47, 0.52, 0.84, 0.75), (198, 198, 198))
    for name, region, color in (
        ("header", (0.177, 0.208, 0.448, 0.256), (198, 198, 198)),
        ("search", (0.184, 0.277, 0.38, 0.312), (93, 93, 93)),
        ("rail", (0.167, 0.345, 0.172, 0.75), (198, 198, 198)),
    ):
        if name not in omitted:
            fill(region, color)
    return _frame(image)


def _frame(image: Image.Image) -> CapturedFrame:
    return CapturedFrame(
        frame_id=1,
        captured_ns=1,
        width=image.width,
        height=image.height,
        bgra=image.convert("RGBA").tobytes("raw", "BGRA"),
    )


@pytest.mark.parametrize("size", [(320, 180), (640, 360), (1920, 1080)])
def test_colored_recipes_preserve_fixed_chrome_recognition(
    pixel_path: None, size: tuple[int, int],
) -> None:
    frame = _inventory_frame(width=size[0], height=size[1])

    assert service._sampled_neutral_ratio(
        frame,
        x_start=0.15, x_end=0.46, y_start=0.18, y_end=0.75,
        luma_min=70, luma_max=110, pixels=service._numpy_bgra(frame),
    ) < 0.55
    assert service.bedrock_inventory_overlay_present(frame)


@pytest.mark.parametrize(
    "omitted",
    [
        frozenset({"header"}),
        frozenset({"search"}),
        frozenset({"rail"}),
        frozenset({"right_mid"}),
        frozenset({"right_light"}),
        frozenset({"right_mid", "right_light"}),
        frozenset({"header", "search", "rail"}),
        frozenset({"header", "search", "rail", "right_mid", "right_light"}),
    ],
    ids=[
        "no-header", "no-search", "no-rail", "no-right-mid", "no-right-light",
        "left-only", "right-and-recipes-only", "colored-recipes-only",
    ],
)
def test_colored_recipe_path_requires_every_independent_chrome_region(
    pixel_path: None, omitted: frozenset[str],
) -> None:
    assert not service.bedrock_inventory_overlay_present(_inventory_frame(omitted=omitted))


@pytest.mark.parametrize("color", [(35, 55, 30), (93, 93, 93), (139, 139, 139)])
def test_world_or_uniform_gray_does_not_match_inventory(
    pixel_path: None, color: tuple[int, int, int],
) -> None:
    frame = _frame(Image.new("RGB", (640, 360), color))
    assert not service.bedrock_inventory_overlay_present(frame)


def test_original_dark_slot_path_does_not_require_new_chrome(pixel_path: None) -> None:
    frame = _inventory_frame(dark_slots=True, omitted=frozenset({"header", "search", "rail"}))

    assert service.bedrock_inventory_overlay_present(frame)


def test_original_compact_inventory_path_is_unchanged(pixel_path: None) -> None:
    image = Image.new("RGB", (640, 360), (35, 55, 30))
    draw = ImageDraw.Draw(image)
    draw.rectangle((64, 43, 511, 74), fill=(190, 190, 190))
    draw.rectangle((64, 72, 511, 121), fill=(110, 110, 110))

    assert service.bedrock_inventory_overlay_present(_frame(image))


def test_fixed_chrome_remains_only_existing_gui_safety_evidence(pixel_path: None) -> None:
    frame = _inventory_frame()

    facts = {fact.key: fact for fact in service.BootstrapFastPerception().infer(frame)}

    for key, value in (
        ("scene.inventory_overlay", True), ("scene.ui_overlay", True), ("scene.playable", False),
    ):
        fact = facts[key]
        assert fact.value is value
        assert fact.confidence == 0.995
        assert fact.expires_after_ms == 250
        assert fact.source == service.BEDROCK_HUD_SAFETY_SOURCE
    assert "scene.mode" not in facts
    assert not any(key.startswith(("inventory.", "target.", "obstacle.")) for key in facts)
    assert service.live_control_arm_reason(frame) == "inventory"


@pytest.mark.parametrize("size", [(256, 144), (319, 180), (320, 179)])
def test_subminimum_capture_still_abstains(pixel_path: None, size: tuple[int, int]) -> None:
    frame = _inventory_frame(width=size[0], height=size[1])
    assert not service.bedrock_inventory_overlay_present(frame)
