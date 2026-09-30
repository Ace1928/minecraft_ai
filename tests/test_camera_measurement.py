from __future__ import annotations

import math

import numpy as np
import pytest

from minecraft_ai.perception.camera_measurement import camera_rotation_delta_degrees


def _rotation_homography(
    width: int,
    height: int,
    field_of_view: float,
    angle_degrees: float,
    axis: str,
) -> np.ndarray:
    focal = height / (2.0 * math.tan(math.radians(field_of_view) / 2.0))
    intrinsics = np.array(
        [[focal, 0.0, (width - 1) / 2.0],
         [0.0, focal, (height - 1) / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    angle = math.radians(angle_degrees)
    if axis == "yaw":
        rotation = np.array(
            [[math.cos(angle), 0.0, math.sin(angle)],
             [0.0, 1.0, 0.0],
             [-math.sin(angle), 0.0, math.cos(angle)]],
            dtype=np.float64,
        )
    else:
        rotation = np.array(
            [[1.0, 0.0, 0.0],
             [0.0, math.cos(angle), -math.sin(angle)],
             [0.0, math.sin(angle), math.cos(angle)]],
            dtype=np.float64,
        )
    return intrinsics @ rotation @ np.linalg.inv(intrinsics)


@pytest.mark.parametrize("axis", ["yaw", "pitch"])
def test_camera_rotation_estimate_recovers_small_projective_turn(axis: str) -> None:
    cv2 = pytest.importorskip("cv2")
    width, height, fov, angle = 640, 360, 70.0, 5.0
    random = np.random.default_rng(7)
    texture = np.zeros((height, width), dtype=np.uint8)
    for _ in range(500):
        x = int(random.integers(0, width))
        y = int(random.integers(0, height))
        size = int(random.integers(4, 20))
        color = int(random.integers(25, 255))
        cv2.rectangle(
            texture,
            (x, y),
            (min(width - 1, x + size), min(height - 1, y + size)),
            color,
            thickness=-1,
        )
    before = np.repeat(texture[:, :, None], 4, axis=2)
    before[:, :, 3] = 255
    after = cv2.warpPerspective(
        before,
        _rotation_homography(width, height, fov, angle, axis),
        (width, height),
    )

    measured, inliers = camera_rotation_delta_degrees(
        before.tobytes(),
        after.tobytes(),
        width=width,
        height=height,
        vertical_fov_degrees=fov,
        axis=axis,  # type: ignore[arg-type]
    )

    assert measured == pytest.approx(angle, abs=0.35)
    assert inliers >= 20
