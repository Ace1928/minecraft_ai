from __future__ import annotations

from enum import StrEnum


class ActionLevel(StrEnum):
    """Explicit controller abstraction requested by an executable option.

    This is a routing contract, not a prediction made from a skill name or a
    mode string.  Keeping it shared by runtime intents and trajectory records
    makes expert selection observable and replayable.
    """

    RAW = "raw"
    MOTION = "motion"
    LATENT = "latent"
    GROUNDED = "grounded"
    GUI = "gui"
    SKILL = "skill"
