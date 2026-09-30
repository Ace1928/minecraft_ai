from __future__ import annotations

import math
from typing import Literal

import numpy as np


def camera_rotation_delta_degrees(
    before_bgra: bytes,
    after_bgra: bytes,
    *,
    width: int,
    height: int,
    vertical_fov_degrees: float,
    axis: Literal["yaw", "pitch"],
) -> tuple[float, int]:
    """Estimate a small camera-only rotation from a pair of Bedrock frames.

    ORB feature matches are fit to a RANSAC homography first. The surviving
    rays are then compared through the calibrated pinhole projection, which
    turns the screen-space rotation into degrees without assuming a mouse gain.
    The capture region excludes the avatar, hotbar, tooltip and crosshair.
    """
    if axis not in {"yaw", "pitch"}:
        raise ValueError("camera measurement axis must be yaw or pitch")
    if not 30.0 <= vertical_fov_degrees <= 150.0:
        raise ValueError("vertical field of view must be in [30, 150] degrees")
    expected_bytes = width * height * 4
    if width < 320 or height < 180 or len(before_bgra) != expected_bytes:
        raise ValueError("camera measurement received an invalid first frame")
    if len(after_bgra) != expected_bytes:
        raise ValueError("camera measurement received an invalid second frame")

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - covered by optional-extra packaging
        raise RuntimeError(
            "install the camera-calibration extra to measure Bedrock camera motion"
        ) from exc

    before = np.frombuffer(before_bgra, dtype=np.uint8).reshape(height, width, 4)
    after = np.frombuffer(after_bgra, dtype=np.uint8).reshape(height, width, 4)
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGRA2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGRA2GRAY)
    left, right = round(width * 0.08), round(width * 0.92)
    top, bottom = round(height * 0.10), round(height * 0.68)
    before_gray = before_gray[top:bottom, left:right]
    after_gray = after_gray[top:bottom, left:right]

    detector = cv2.ORB_create(nfeatures=1800, fastThreshold=8)
    keypoints_before, descriptors_before = detector.detectAndCompute(before_gray, None)
    keypoints_after, descriptors_after = detector.detectAndCompute(after_gray, None)
    if descriptors_before is None or descriptors_after is None:
        raise ValueError("camera frame has too few stable visual features")
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(
        descriptors_before,
        descriptors_after,
        k=2,
    )
    reliable_matches = [
        first
        for pair in matches
        if len(pair) == 2
        for first, second in (pair,)
        if first.distance < 0.78 * second.distance
    ]
    if len(reliable_matches) < 24:
        raise ValueError("camera probe retained too few matched visual features")
    initial = np.asarray(
        [keypoints_before[match.queryIdx].pt for match in reliable_matches],
        dtype=np.float32,
    )
    current = np.asarray(
        [keypoints_after[match.trainIdx].pt for match in reliable_matches],
        dtype=np.float32,
    )
    _, inlier_mask = cv2.findHomography(initial, current, cv2.RANSAC, 2.5)
    if inlier_mask is None:
        raise ValueError("camera rotation did not form a stable image homography")
    inliers = inlier_mask.reshape(-1).astype(bool)
    initial = initial[inliers]
    current = current[inliers]
    if len(initial) < 20:
        raise ValueError("camera rotation has fewer than 20 homography inliers")

    focal = height / (2.0 * math.tan(math.radians(vertical_fov_degrees) / 2.0))
    center_x = (width - 1) / 2.0 - left
    center_y = (height - 1) / 2.0 - top
    if axis == "yaw":
        ray_before = np.arctan2(initial[:, 0] - center_x, focal)
        ray_after = np.arctan2(current[:, 0] - center_x, focal)
    else:
        ray_before = np.arctan2(center_y - initial[:, 1], focal)
        ray_after = np.arctan2(center_y - current[:, 1], focal)
    deltas = np.degrees(ray_before - ray_after)
    median_delta = float(np.median(deltas))
    consistent = deltas[np.sign(deltas) == np.sign(median_delta)]
    if len(consistent) < 16:
        raise ValueError("camera rotation direction is not visually consistent")
    magnitude = abs(float(np.median(consistent)))
    if not 0.25 <= magnitude <= 30.0:
        raise ValueError(f"camera probe angle {magnitude:.3f} degrees is out of range")
    return magnitude, int(len(initial))


def camera_view_mean_absolute_error(
    first_bgra: bytes,
    second_bgra: bytes,
    *,
    width: int,
    height: int,
) -> float:
    """Return a normalized, HUD-masked pixel error for a reversible probe."""
    expected_bytes = width * height * 4
    if len(first_bgra) != expected_bytes or len(second_bgra) != expected_bytes:
        raise ValueError("camera restoration comparison received invalid frame bytes")
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - covered by optional-extra packaging
        raise RuntimeError(
            "install the camera-calibration extra to compare Bedrock camera frames"
        ) from exc
    first = np.frombuffer(first_bgra, dtype=np.uint8).reshape(height, width, 4)
    second = np.frombuffer(second_bgra, dtype=np.uint8).reshape(height, width, 4)
    first_gray = cv2.cvtColor(first, cv2.COLOR_BGRA2GRAY)
    second_gray = cv2.cvtColor(second, cv2.COLOR_BGRA2GRAY)
    left, right = round(width * 0.08), round(width * 0.92)
    top, bottom = round(height * 0.10), round(height * 0.68)
    first_gray = first_gray[top:bottom, left:right]
    second_gray = second_gray[top:bottom, left:right]
    size = (480, 320)
    first_small = cv2.resize(first_gray, size, interpolation=cv2.INTER_AREA)
    second_small = cv2.resize(second_gray, size, interpolation=cv2.INTER_AREA)
    first_small = cv2.GaussianBlur(first_small, (3, 3), 0)
    second_small = cv2.GaussianBlur(second_small, (3, 3), 0)
    return float(np.mean(cv2.absdiff(first_small, second_small)) / 255.0)
