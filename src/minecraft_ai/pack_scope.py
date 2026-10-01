"""Fresh host-observed recipe scope, separate from configured snapshot intent.

This is an observation contract, not an input lease or a live server adapter.
The desktop observer must establish actual session/stack membership before
publishing it. Configuration files and model output cannot establish that fact.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import time

from .perception import CognitionReadView

ACTIVE_RECIPE_KEY = "world.active_recipe_identity"
ACTIVE_RECIPE_SOURCE = "game_forge:active-recipe-identity-v1"
RECIPE_SCOPE_UNAVAILABLE = (
    "I can't verify the active world and pack session, so I can't give its recipe yet."
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class ActiveRecipeIdentity:
    world: str
    bds_version: str
    scope_sha256: str
    catalog_sha256: str
    server_session: str
    instance_id: str


def active_recipe_identity(view: CognitionReadView) -> ActiveRecipeIdentity | None:
    """Accept a bounded fresh fact, including when view is an immutable snapshot.

    Heartbeat timestamps are deliberately outside identity: refreshing the same
    session keeps an answer valid; changing any identity field invalidates it.
    """
    latest = view.raw_latest()
    fact = view.fact(ACTIVE_RECIPE_KEY, min_confidence=0.99)
    now = time.monotonic_ns()
    if (
        latest is None or fact is None or fact.source != ACTIVE_RECIPE_SOURCE
        or type(fact.value) is not str or len(fact.value) > 2048
        or not 0 < fact.expires_after_ms <= 5000 or not fact.fresh(now)
    ):
        return None
    try:
        value = json.loads(fact.value)
    except (ValueError, TypeError, RecursionError):
        return None
    fields = ActiveRecipeIdentity.__dataclass_fields__
    if (
        type(value) is not dict
        or set(value) != {*fields, "schema_version", "state"}
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1 or value["state"] != "observed_active"
        or any(type(value[key]) is not str or not 0 < len(value[key]) <= 256
               for key in fields)
        or any(_SHA256.fullmatch(value[key]) is None
               for key in ("scope_sha256", "catalog_sha256"))
        or value["instance_id"] != latest.instance_id
        or not latest.instance_id.startswith("bedrock:" + value["bds_version"] + ":")
    ):
        return None
    return ActiveRecipeIdentity(**{key: value[key] for key in fields})


def recipe_identity_matches(
    identity: ActiveRecipeIdentity | None, view: CognitionReadView,
) -> bool:
    """Recheck a host-bound decision at consumption/delivery time."""
    return identity is None or active_recipe_identity(view) == identity
