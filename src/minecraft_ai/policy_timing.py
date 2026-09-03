from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class InferenceRateHold:
    """Estimate a bounded zero-order-hold renewal horizon from model latency.

    This component never chooses an action.  It only determines how long the
    most recent learned key/button state remains fresh while its successor is
    being inferred.  The caller owns all safety, scene, option, target, and
    request-deadline invalidation.
    """

    minimum_ms: int = 50
    maximum_ms: int = 250
    latency_margin: float = 1.20
    ema_alpha: float = 0.25
    latency_ema_ms: float | None = None
    observations: int = 0
    renewals: int = 0
    invalidations: int = 0
    deadline_expirations: int = 0
    last_invalidation_reason: str | None = None

    def __post_init__(self) -> None:
        if not 50 <= self.minimum_ms <= 250:
            raise ValueError("minimum hold must be in [50, 250] ms")
        if not self.minimum_ms <= self.maximum_ms <= 250:
            raise ValueError("maximum hold must be in [minimum, 250] ms")
        if not 1.0 <= self.latency_margin <= 3.0:
            raise ValueError("hold latency margin must be in [1, 3]")
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("hold EMA alpha must be in (0, 1]")

    def observe(self, inference_ns: int) -> None:
        if inference_ns < 0:
            raise ValueError("inference latency cannot be negative")
        latency_ms = inference_ns / 1_000_000.0
        if self.latency_ema_ms is None:
            self.latency_ema_ms = latency_ms
        else:
            alpha = self.ema_alpha
            self.latency_ema_ms = alpha * latency_ms + (1.0 - alpha) * self.latency_ema_ms
        self.observations += 1

    @property
    def horizon_ms(self) -> int:
        if self.latency_ema_ms is None:
            return self.minimum_ms
        estimated = math.ceil(self.latency_ema_ms * self.latency_margin)
        return max(self.minimum_ms, min(self.maximum_ms, estimated))

    def renew_until_ns(
        self,
        *,
        now_ns: int,
        prediction_pending: bool,
        request_deadline_ns: int | None = None,
        state_valid: bool = True,
    ) -> int | None:
        """Return a bounded renewal only for a valid in-flight successor."""

        if not state_valid:
            self.invalidate("learned-state-invalid")
            return None
        if not prediction_pending:
            return None
        if request_deadline_ns is not None and now_ns >= request_deadline_ns:
            self.deadline_expirations += 1
            self.invalidate("request-deadline-expired")
            return None
        self.renewals += 1
        renewed_until_ns = now_ns + self.horizon_ms * 1_000_000
        if request_deadline_ns is not None:
            renewed_until_ns = min(renewed_until_ns, request_deadline_ns)
        return renewed_until_ns

    def invalidate(self, reason: str) -> None:
        """Record why the caller released rather than retaining learned state."""

        normalized = reason.strip()
        if not normalized:
            raise ValueError("hold invalidation reason is required")
        self.invalidations += 1
        self.last_invalidation_reason = normalized

    def status(self) -> dict[str, int | float | str | None]:
        return {
            "minimum_ms": self.minimum_ms,
            "maximum_ms": self.maximum_ms,
            "latency_margin": self.latency_margin,
            "ema_alpha": self.ema_alpha,
            "latency_ema_ms": (
                None if self.latency_ema_ms is None else round(self.latency_ema_ms, 3)
            ),
            "horizon_ms": self.horizon_ms,
            "observations": self.observations,
            "renewals": self.renewals,
            "invalidations": self.invalidations,
            "deadline_expirations": self.deadline_expirations,
            "last_invalidation_reason": self.last_invalidation_reason,
        }
