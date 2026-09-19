"""Keep the paired death-control contract without scanning impossible secondary controls."""

import pytest
import numpy as np

import minecraft_ai.perception.service as service
from minecraft_ai.platforms.bedrock_x11 import CapturedFrame


@pytest.mark.parametrize("primary_ratio", (0.0, 0.699, 0.70, 0.701, 1.0))
@pytest.mark.parametrize("secondary_ratio", (0.0, 0.749, 0.75, 1.0))
def test_death_pair_short_circuit_preserves_exact_thresholds(
    monkeypatch, primary_ratio, secondary_ratio,
):
    calls = []

    def palette(frame, *, palette, **kwargs):
        calls.append(palette)
        return primary_ratio if palette == "primary" else secondary_ratio

    monkeypatch.setattr(service, "_region_palette_ratio", palette)
    result = service._death_control_pair_present(
        CapturedFrame(1, 1, 1920, 1080, b""), *service._DEATH_CONTROL_PAIRS[0], pixels=None,
    )
    assert result == (primary_ratio >= 0.70 and secondary_ratio >= 0.75)
    assert calls == (["primary"] if primary_ratio < 0.70 else ["primary", "secondary"])


@pytest.mark.parametrize("dtype", ("uint8", "int32"))
@pytest.mark.parametrize("stride", (1, 2, 5))
def test_channel_range_is_exact_for_strided_pixel_crops(dtype, stride):
    pixels = np.random.default_rng(42).integers(0, 256, (37, 43, 4), dtype=np.uint8)
    pixels = pixels[::stride, ::stride, :3].astype(dtype)
    expected = pixels.max(axis=2).astype("int16") - pixels.min(axis=2).astype("int16")
    np.testing.assert_array_equal(service._channel_spread(pixels), expected)
    assert service._channel_spread(pixels[:0]).shape == (0, pixels.shape[1])


@pytest.mark.parametrize("palette", ("primary", "secondary", "heart", "hotbar", "neutral"))
def test_optimized_palette_ratios_match_the_scalar_pixel_contract(palette):
    rng = np.random.default_rng(42)
    pixels = rng.integers(0, 256, (48, 64, 4), dtype=np.uint8)
    # Include exact chroma/luma boundaries, not only obviously non-neutral pixels.
    for row, value in enumerate((0, 18, 19, 85, 100, 175, 230, 235, 255)):
        pixels[row, :, :3] = value
    frame = CapturedFrame(1, 1, 64, 48, pixels.tobytes())
    bounds = dict(x_start=0.125, x_end=0.89, y_start=0.0, y_end=0.9)
    if palette == "neutral":
        measure = service._sampled_neutral_ratio
        bounds.update(luma_min=100, luma_max=230)
    elif palette in {"heart", "hotbar"}:
        measure = service._hud_palette_ratio
        bounds["palette"] = palette
    else:
        measure = service._region_palette_ratio
        bounds["palette"] = palette
    assert measure(frame, pixels=pixels, **bounds) == measure(frame, pixels=None, **bounds)
