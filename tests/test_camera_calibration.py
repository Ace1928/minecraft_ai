from __future__ import annotations

import json
from pathlib import Path

import pytest

from minecraft_ai.camera_calibration import (
    load_camera_calibration,
    read_bedrock_mouse_sensitivity,
)


def _profile() -> dict[str, object]:
    return {
        "schema_version": 2,
        "captured_at": "2026-09-03T01:48:28+10:00",
        "game": "Minecraft Bedrock",
        "game_version": "1.26.45.1",
        "launcher": "bedrock-on-linux",
        "display": ":2",
        "capture_window_id": 6291460,
        "input_window_id": 12582913,
        "input_backend": "XTestFakeRelativeMotionEvent",
        "mouse_sensitivity_option": 0.03,
        "options_sha256": "a" * 64,
        "method": "measured full-yaw image wrap",
        "full_yaw_counts": 17267,
        "yaw_counts_per_degree": 47.9638888889,
        "full_pitch_counts": 11880,
        "pitch_counts_per_degree": 66.0,
        "horizon_from_upper_pole_counts": 5940,
        "configured_yaw_counts_per_degree": 47.96,
        "configured_pitch_counts_per_degree": 66.0,
        "restore_mean_absolute_pixel_error": 0.007,
        "pitch_method": "paced upper-pole homing and half-range return",
        "notes": "machine-local measurement",
    }


def test_exact_version_camera_profile_is_loaded_and_compatibility_checked(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "calibrations"
    directory.mkdir()
    (directory / "bedrock-camera-1.26.45.1.json").write_text(
        json.dumps(_profile()),
        encoding="utf-8",
    )

    profile = load_camera_calibration(tmp_path, game_version="1.26.45.1")
    profile.require_compatible(
        game_version="1.26.45.1",
        mouse_sensitivity=0.03,
        configured_yaw_counts_per_degree=47.96,
        configured_pitch_counts_per_degree=66.0,
    )

    assert profile.profile_id
    assert profile.yaw_counts_per_degree == pytest.approx(47.9638888889)


def test_camera_profile_rejects_changed_mouse_sensitivity(tmp_path: Path) -> None:
    directory = tmp_path / "calibrations"
    directory.mkdir()
    (directory / "bedrock-camera-1.26.45.1.json").write_text(
        json.dumps(_profile()),
        encoding="utf-8",
    )
    profile = load_camera_calibration(tmp_path, game_version="1.26.45.1")

    with pytest.raises(ValueError, match="sensitivity changed"):
        profile.require_compatible(
            game_version="1.26.45.1",
            mouse_sensitivity=0.05,
            configured_yaw_counts_per_degree=47.96,
            configured_pitch_counts_per_degree=66.0,
        )


def test_camera_profile_rejects_unmeasured_pitch_adapter(tmp_path: Path) -> None:
    directory = tmp_path / "calibrations"
    directory.mkdir()
    (directory / "bedrock-camera-1.26.45.1.json").write_text(
        json.dumps(_profile()),
        encoding="utf-8",
    )
    profile = load_camera_calibration(tmp_path, game_version="1.26.45.1")

    with pytest.raises(ValueError, match="pitch scale"):
        profile.require_compatible(
            game_version="1.26.45.1",
            mouse_sensitivity=0.03,
            configured_yaw_counts_per_degree=47.96,
            configured_pitch_counts_per_degree=47.96,
        )


def test_active_bedrock_mouse_sensitivity_is_read_from_wine_prefix(
    tmp_path: Path,
) -> None:
    options = (
        tmp_path
        / "drive_c/users/steamuser/AppData/Roaming/Minecraft Bedrock/Users/1/games/"
        "com.mojang/minecraftpe/options.txt"
    )
    options.parent.mkdir(parents=True)
    options.write_text(
        "gfx_fullscreen:0\nctrl_sensitivity2_mouse:0.03\n",
        encoding="utf-8",
    )

    assert read_bedrock_mouse_sensitivity(tmp_path) == pytest.approx(0.03)
