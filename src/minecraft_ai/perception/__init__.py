"""Perception types, blackboard, active queries, and visual services.

`PerceptionBlackboard` is the live capture/fact store used by the runtime.
The typed latest-fact store in `.blackboard` is a complementary contract,
available as `minecraft_ai.perception.blackboard.PerceptionBlackboard`.
"""
from __future__ import annotations

import time as time

from .active import (
    ActivePerceptionRequest,
    ActivePerceptionResult,
    PerceptionPriority,
    result_to_fact,
)
from .blackboard import BlackboardSnapshot, PerceptFact, PerceptKind, PerceptSource
from .types import (
    ActivePerceptionQuery,
    ChatLine,
    CognitionBlackboardSnapshot,
    CognitionReadView,
    EvidenceRegion,
    FrameState,
    PerceptionBlackboard,
    PerceptionEvidence,
    PerceptionFact,
    PerceptionQueryMode,
    PlayerEstimate,
    RingBuffer,
    ScreenRegion,
    Track,
)
from .types import _snapshot_json as _snapshot_json

__all__ = [
    "ActivePerceptionQuery",
    "ActivePerceptionRequest",
    "ActivePerceptionResult",
    "BlackboardSnapshot",
    "ChatLine",
    "CognitionBlackboardSnapshot",
    "CognitionReadView",
    "EvidenceRegion",
    "FrameState",
    "PerceptFact",
    "PerceptKind",
    "PerceptSource",
    "PerceptionBlackboard",
    "PerceptionEvidence",
    "PerceptionFact",
    "PerceptionPriority",
    "PerceptionQueryMode",
    "PlayerEstimate",
    "RingBuffer",
    "ScreenRegion",
    "Track",
    "_snapshot_json",
    "result_to_fact",
    "time",
]
