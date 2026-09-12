"""Cheap temporal luma evidence, never a block-break or reward classifier."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from minecraft_ai.perception import PerceptionBlackboard


@dataclass
class MiningVisualProgress:
    """Track local texture changes missed by brightness-invariant dHash.

    Only newer, current-frame observations count. Removing the common brightness
    delta rejects uniform lighting flicker. This may extend a bounded attempt;
    it cannot confirm success, grant input permission, or mint training reward.
    """

    reference: bytes | None = None
    last_observed_ns: int = -1
    changes: int = 0

    def observe(
        self,
        blackboard: PerceptionBlackboard,
        *,
        now_ns: int,
        not_before_ns: int = 0,
        min_mean_change: float = 6.0,
        pixel_delta: int = 12,
        min_changed_fraction: float = 0.12,
    ) -> bool:
        fact = blackboard.fact("frame.crosshair_luma_grid", min_confidence=1.0, now_ns=now_ns)
        latest = blackboard.raw_latest()
        if (
            fact is None
            or not isinstance(fact.value, str)
            or len(fact.value) != 128
            or not max(not_before_ns, self.last_observed_ns + 1) <= fact.observed_ns <= now_ns
            or (latest is not None and fact.observed_ns < latest.captured_ns)
        ):
            return False
        try:
            current = bytes.fromhex(fact.value)
        except ValueError:
            return False
        if len(current) != 64:
            return False
        self.last_observed_ns = fact.observed_ns
        if self.reference is None:
            self.reference = current
            return False
        signed = [a - b for a, b in zip(current, self.reference, strict=True)]
        common = median(signed)
        deltas = [abs(delta - common) for delta in signed]
        changed = (
            sum(deltas) / len(deltas) >= min_mean_change
            and sum(delta >= pixel_delta for delta in deltas) / len(deltas) >= min_changed_fraction
        )
        if changed:
            self.reference = current
            self.changes += 1
        return changed
