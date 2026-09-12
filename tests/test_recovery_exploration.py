from __future__ import annotations

import math
import time

import pytest

from minecraft_ai.skills import SkillCondition, SkillStats
from minecraft_ai.skills.recovery import select_learned_recovery
from test_adaptive_gameplay import _library
from test_mining_control import _mining_board


def _select(library, **kwargs):
    return select_learned_recovery(
        library, ("backoff", "jump", "look"), _mining_board(now_ns=time.monotonic_ns()),
        context_key="wall", **kwargs,
    )


def test_uncertain_alternative_gets_a_trial_instead_of_permanent_greedy_lock_in() -> None:
    library = _library()
    library.stats[("backoff", "wall")] = SkillStats(successes=100)
    assert _select(library, exploration_strength=0).skill_id == "backoff"
    assert _select(library).skill_id == "jump"
    # One successful exploratory trial teaches the selector without having to
    # manufacture repeated failures for the previously preferred option.
    library.stats[("jump", "wall")] = SkillStats(successes=1)
    assert _select(library).skill_id == "jump"


def test_censored_failures_and_cancellations_do_not_consume_exploration_budget() -> None:
    library = _library()
    library.stats[("backoff", "wall")] = SkillStats(successes=100)
    library.stats[("jump", "wall")] = SkillStats(
        failures=1000, censored_failures=1000, cancellations=1000,
    )
    assert _select(library).skill_id == "jump"


def test_exploration_is_context_local() -> None:
    library = _library()
    library.stats[("backoff", "wall")] = SkillStats(successes=100)
    library.stats[("jump", "water")] = SkillStats(successes=1000)
    assert _select(library).skill_id == "jump"


def test_large_exploration_never_reactivates_exhausted_or_infeasible_choices() -> None:
    library = _library()
    library.stats[("backoff", "wall")] = SkillStats(failures=2, consecutive_failures=2)
    library.specs["jump"] = library.get("jump").model_copy(update={
        "preconditions": (SkillCondition(key="unknown", operator="truthy"),),
    })
    assert _select(library, exploration_strength=100).skill_id == "look"
    library.stats[("look", "wall")] = SkillStats(failures=2, consecutive_failures=2)
    assert _select(library, exploration_strength=100) is None


@pytest.mark.parametrize("value", [-1, math.nan, math.inf])
def test_invalid_exploration_strength_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="exploration_strength"):
        _select(_library(), exploration_strength=value)


@pytest.mark.parametrize("value", [0, True, 1.5])
def test_failure_cap_is_a_positive_integer(value) -> None:
    with pytest.raises(ValueError, match="max_context_failures"):
        _select(_library(), max_context_failures=value)
