from __future__ import annotations

import time
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from .blackboard import PerceptFact, PerceptKind, PerceptSource


class PerceptionPriority(StrEnum):
    BACKGROUND = "background"
    NORMAL = "normal"
    TACTICAL = "tactical"
    EMERGENCY = "emergency"


class ActivePerceptionRequest(BaseModel):
    """A targeted semantic question for an asynchronous VLM/perception worker."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str = Field(min_length=1, max_length=256)
    question: str = Field(min_length=1, max_length=2048)
    output_key: str = Field(min_length=1, max_length=256)
    output_kind: PerceptKind
    priority: PerceptionPriority = PerceptionPriority.NORMAL
    created_monotonic_ns: int = Field(default_factory=time.monotonic_ns, gt=0)
    deadline_ms: int = Field(default=1000, ge=10, le=30_000)
    minimum_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    context_keys: tuple[str, ...] = ()

    def expired(self, now_ns: int | None = None) -> bool:
        now = time.monotonic_ns() if now_ns is None else now_ns
        return now - self.created_monotonic_ns >= self.deadline_ms * 1_000_000


class ActivePerceptionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str = Field(min_length=1, max_length=256)
    value: object
    confidence: float = Field(ge=0.0, le=1.0)
    sequence: int = Field(ge=0)
    observed_monotonic_ns: int = Field(default_factory=time.monotonic_ns, gt=0)
    provenance: str | None = Field(default=None, max_length=1024)


def result_to_fact(
    request: ActivePerceptionRequest,
    result: ActivePerceptionResult,
    *,
    ttl_ms: int = 1500,
) -> PerceptFact | None:
    """Accept only timely, matching, sufficiently confident semantic results."""
    if result.request_id != request.request_id:
        return None
    if request.expired(result.observed_monotonic_ns):
        return None
    if result.confidence < request.minimum_confidence:
        return None
    return PerceptFact(
        key=request.output_key,
        kind=request.output_kind,
        source=PerceptSource.VLM,
        value=result.value,
        confidence=result.confidence,
        sequence=result.sequence,
        observed_monotonic_ns=result.observed_monotonic_ns,
        ttl_ms=ttl_ms,
        provenance=result.provenance,
    )
