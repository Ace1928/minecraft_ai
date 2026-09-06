from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class CameraCalibrationProfile(BaseModel):
    """Measured Bedrock relative-mouse calibration for one exact game build."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(ge=2, le=2)
    captured_at: str
    game: str
    game_version: str
    launcher: str
    display: str
    capture_window_id: int
    input_window_id: int
    input_backend: str
    mouse_sensitivity_option: float = Field(gt=0.0, le=1.0)
    options_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    method: str
    full_yaw_counts: int = Field(gt=0)
    yaw_counts_per_degree: float = Field(gt=0.0, le=100.0)
    full_pitch_counts: int = Field(gt=0)
    pitch_counts_per_degree: float = Field(gt=0.0, le=100.0)
    horizon_from_upper_pole_counts: int = Field(gt=0)
    configured_yaw_counts_per_degree: float = Field(gt=0.0, le=100.0)
    configured_pitch_counts_per_degree: float = Field(gt=0.0, le=100.0)
    restore_mean_absolute_pixel_error: float = Field(ge=0.0)
    pitch_method: str
    notes: str = ""
    profile_id: str = Field(pattern=r"^[a-f0-9]{64}$")

    def require_compatible(
        self,
        *,
        game_version: str,
        mouse_sensitivity: float,
        configured_yaw_counts_per_degree: float | None,
        configured_pitch_counts_per_degree: float | None,
    ) -> None:
        if self.game_version != game_version:
            raise ValueError(
                f"camera calibration targets Bedrock {self.game_version}, not {game_version}"
            )
        if abs(self.mouse_sensitivity_option - mouse_sensitivity) > 1e-6:
            raise ValueError(
                "Bedrock mouse sensitivity changed after camera calibration "
                f"({self.mouse_sensitivity_option} -> {mouse_sensitivity})"
            )
        if configured_yaw_counts_per_degree is not None:
            relative_error = abs(
                configured_yaw_counts_per_degree - self.yaw_counts_per_degree
            ) / self.yaw_counts_per_degree
            if relative_error > 0.02:
                raise ValueError(
                    "policy yaw scale does not match the measured Bedrock calibration "
                    f"({configured_yaw_counts_per_degree} vs "
                    f"{self.yaw_counts_per_degree})"
                )
        if configured_pitch_counts_per_degree is not None:
            relative_error = abs(
                configured_pitch_counts_per_degree - self.pitch_counts_per_degree
            ) / self.pitch_counts_per_degree
            if relative_error > 0.02:
                raise ValueError(
                    "policy pitch scale does not match the measured Bedrock calibration "
                    f"({configured_pitch_counts_per_degree} vs "
                    f"{self.pitch_counts_per_degree})"
                )


def load_camera_calibration(
    data_dir: Path,
    *,
    game_version: str,
) -> CameraCalibrationProfile:
    safe_version = "".join(
        character if character.isalnum() or character in ".-_" else "_"
        for character in game_version
    )
    path = data_dir / "calibrations" / f"bedrock-camera-{safe_version}.json"
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError(f"invalid camera calibration profile: {path}")
    payload["profile_id"] = hashlib.sha256(raw).hexdigest()
    return CameraCalibrationProfile.model_validate(payload)


def read_bedrock_mouse_sensitivity(wine_prefix: Path) -> float:
    pattern = (
        "drive_c/users/*/AppData/Roaming/Minecraft Bedrock/Users/*/games/"
        "com.mojang/minecraftpe/options.txt"
    )
    candidates = sorted(
        wine_prefix.glob(pattern),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("Bedrock options.txt was not found in the active Wine prefix")
    for line in candidates[0].read_text(encoding="utf-8", errors="replace").splitlines():
        key, separator, value = line.partition(":")
        if separator and key == "ctrl_sensitivity2_mouse":
            return float(value)
    raise ValueError("Bedrock options.txt does not expose ctrl_sensitivity2_mouse")
