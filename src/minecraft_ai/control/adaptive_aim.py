"""Acquisition-local camera system identification, independent of attack authority.

Only observed displacement after an emitted camera command calibrates a response.
These corrections remain synthetic actions, not native-policy demonstrations.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from statistics import median


@dataclass
class _AxisResponse:
    samples: deque[float] = field(default_factory=lambda: deque(maxlen=5))

    def observe(self, previous: float, current: float, command: int) -> None:
        displacement = previous - current
        if not command or abs(displacement) < 0.0001:
            return
        response = displacement / command
        if response <= 0:
            # A moving/relocalized target or an inverted response is not evidence
            # for increasing the gain of this positive-feedback convention.
            self.samples.clear()
            return
        self.samples.append(response)

    def command(self, error: float, *, previous: int, max_step: int) -> int:
        if error == 0:
            return 0
        if not self.samples:
            # Small identification probe, also the legacy cold-start behaviour.
            requested = error * 40.0
            limit = min(12, max_step)
        else:
            requested = 0.75 * error / median(self.samples)
            # Bounded growth makes one noisy localization insufficient to jump
            # straight from a small probe to the absolute camera-step ceiling.
            limit = min(max_step, max(12, abs(previous) * 4))
        magnitude = min(limit, max(1, round(abs(requested))))
        return magnitude if error > 0 else -magnitude


@dataclass
class AdaptiveMiningAim:
    """Estimate separate yaw/pitch responses during one mining acquisition.

    Reset with the acquisition; do not carry calibration between worlds, target
    identities, camera settings or input-release boundaries. This class cannot
    grant input authority or report a successful block break.
    """

    max_step: int = 256
    _axes: tuple[_AxisResponse, _AxisResponse] = field(
        default_factory=lambda: (_AxisResponse(), _AxisResponse()),
    )
    _target_id: str | None = None
    _center: tuple[float, float] | None = None
    _command: tuple[int, int] = (0, 0)
    _observed_ns: int = -1
    _issued_ns: int = -1
    _feedback_allowed: bool = False

    def __post_init__(self) -> None:
        if type(self.max_step) is not int or not 1 <= self.max_step <= 4096:
            raise ValueError("max_step must be an integer in [1, 4096]")

    def reset(self) -> None:
        self._axes = (_AxisResponse(), _AxisResponse())
        self._target_id = None
        self._center = None
        self._command = (0, 0)
        self._observed_ns = self._issued_ns = -1
        self._feedback_allowed = False

    def step(
        self,
        *,
        target_id: str,
        center: tuple[float, float],
        aligned: tuple[bool, bool],
        observed_ns: int,
        now_ns: int,
        stationary: bool = True,
    ) -> tuple[int, int]:
        if (not target_id or any(not math.isfinite(v) or not 0 <= v <= 1 for v in center)
                or observed_ns < 0 or observed_ns > now_ns):
            return (0, 0)
        if observed_ns <= max(self._observed_ns, self._issued_ns):
            return (0, 0)
        if self._target_id != target_id:
            self.reset()
        if self._center is not None and self._feedback_allowed and stationary:
            for axis, before, after, command in zip(
                self._axes, self._center, center, self._command, strict=True,
            ):
                axis.observe(before, after, command)
        commands = tuple(
            axis.command(0.0 if covered else position - 0.5,
                         previous=previous, max_step=self.max_step)
            for axis, position, covered, previous in zip(
                self._axes, center, aligned, self._command, strict=True,
            )
        )
        self._target_id = target_id
        self._center = center
        self._command = (commands[0], commands[1])
        self._observed_ns, self._issued_ns = observed_ns, now_ns
        self._feedback_allowed = stationary
        return self._command
