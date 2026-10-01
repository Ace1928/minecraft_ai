"""Fresh captured-world admission for the existing leased game-chat transport."""

from __future__ import annotations

import time

from .cognition.types import CognitionDecision, DecisionChatAuthority
from .pack_scope import recipe_identity_matches
from .perception import CognitionReadView, PerceptionBlackboard
from .perception_service import (
    BEDROCK_HUD_SAFETY_SOURCE,
    bedrock_classic_health,
    bedrock_death_screen_present,
    bedrock_in_world_hud_present,
    bedrock_survival_hud_present,
    bedrock_ui_chrome_present,
)
from .platforms.bedrock_x11 import CapturedFrame


def bind_game_chat_authority(
    decision: CognitionDecision, view: CognitionReadView,
) -> CognitionDecision:
    """Bind host-observed authority, never an LLM-proposed permission field."""
    if decision.game_chat is None:
        return decision
    latest = view.raw_latest()
    if latest is None:
        return decision.model_copy(update={"game_chat": None})
    for key in ("social.player_message", "operator.game_chat_authorized"):
        fact = view.fact(key, min_confidence=0.7)
        if fact is not None and bool(fact.value) and fact.fresh():
            bound = decision.model_copy()
            bound._chat_authority = DecisionChatAuthority(
                instance_id=latest.instance_id, key=key, value=fact.value,
                observed_ns=fact.observed_ns, source=fact.source,
                evidence_refs=fact.evidence_refs,
            )
            return bound
    return decision.model_copy(update={"game_chat": None})


def game_chat_authority_matches(
    decision: CognitionDecision, view: PerceptionBlackboard,
) -> bool:
    """A different/re-observed message cannot authorize this older answer."""
    authority = decision.chat_authority
    if authority is None:
        return False
    latest = view.raw_latest()
    fact = view.fact(authority.key, min_confidence=0.7)
    return (
        recipe_identity_matches(decision.pack_recipe_identity, view)
        and
        latest is not None and latest.instance_id == authority.instance_id
        and fact is not None and bool(fact.value) and fact.fresh()
        and fact.value == authority.value and type(fact.value) is type(authority.value)
        and fact.observed_ns == authority.observed_ns and fact.source == authority.source
        and fact.evidence_refs == authority.evidence_refs
    )


def game_chat_delivery_admitted(
    text: str,
    blackboard: PerceptionBlackboard,
    frame: CapturedFrame | None,
    *,
    now_ns: int | None = None,
) -> bool:
    """Chat authority alone cannot take focus from an unsafe or stale scene.

    This is an additional admission check, never a player-message observer,
    model, game input or motor lease. The supervisor independently verifies its
    current lease/isolation/stop interlocks at every typed character.
    """
    now = time.monotonic_ns() if now_ns is None else now_ns
    latest = blackboard.raw_latest()
    if (
        not text or len(text) > 256 or text.lstrip().startswith("/")
        or any(not 32 <= ord(char) <= 126 for char in text)
        or frame is None or latest is None
        or not latest.instance_id.startswith("bedrock:")
        or latest.captured_ns != frame.captured_ns
        or (latest.width, latest.height) != (frame.width, frame.height)
        or not 0 <= now - frame.captured_ns < 250_000_000
        or len(frame.bgra) != frame.width * frame.height * 4
        or bedrock_ui_chrome_present(frame) or bedrock_death_screen_present(frame)
        or not bedrock_in_world_hud_present(frame)
    ):
        return False
    if bedrock_survival_hud_present(frame):
        parts = latest.instance_id.split(":", 2)
        health = bedrock_classic_health(
            frame, game_version=parts[1] if len(parts) == 3 else None,
        )
        if health is None or health <= 6:
            return False
    required = (
        ("scene.playable", True), ("scene.ui_overlay", False),
        ("scene.death", False), ("danger.drowning", False),
        ("environment.underwater", False),
    )
    for key, expected in required:
        fact = blackboard.fact(key, min_confidence=0.99, now_ns=now)
        if (
            fact is None or fact.value is not expected
            or fact.source != BEDROCK_HUD_SAFETY_SOURCE
            or not frame.captured_ns <= fact.observed_ns <= now
        ):
            return False
    for key in ("danger.immediate", "player.critical_health"):
        fact = blackboard.fact(key, min_confidence=0.7, now_ns=now)
        if fact is not None and fact.value is not False:
            return False
    return True
