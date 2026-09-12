"""Model-independent physical action permissions; never a movement policy."""
from __future__ import annotations

from collections.abc import Mapping

from minecraft_ai.safety import MotorAction

_KEYS = {
    "allow_movement": frozenset({"w", "a", "s", "d", "ctrl", "shift", "space"}),
    "allow_jump": frozenset({"space"}),
    "allow_drop": frozenset({"q"}),
    "allow_inventory": frozenset({"e"}),
    "allow_hotbar": frozenset("123456789"),
}
_BUTTONS = {"allow_attack": frozenset({"left"}), "allow_use": frozenset({"right"})}


def constrain_action(
    action: MotorAction, permissions: Mapping[str, object], *,
    held_keys: tuple[str, ...] = (), held_buttons: tuple[str, ...] = (),
) -> MotorAction:
    """Remove forbidden presses and explicitly release possible previous holds.

    Camera and duration are unchanged. The executor supplies its acknowledged
    hold estimate instead of trusting a third-party policy to release inputs.
    A masked proposal must be recorded as synthetic by its caller.
    """
    keys: set[str] = set().union(*(
        names for key, names in _KEYS.items() if permissions.get(key) is False
    ))
    buttons: set[str] = set().union(*(
        names for key, names in _BUTTONS.items() if permissions.get(key) is False
    ))
    if not keys and not buttons:
        return action
    released_keys = (set(held_keys) | set(action.keys_down)) & keys
    released_buttons = (set(held_buttons) | set(action.buttons_down)) & buttons
    return action.model_copy(update={
        "keys_down": tuple(key for key in action.keys_down if key not in keys),
        "keys_up": tuple(sorted(set(action.keys_up) | released_keys)),
        "buttons_down": tuple(key for key in action.buttons_down if key not in buttons),
        "buttons_up": tuple(sorted(set(action.buttons_up) | released_buttons)),
    })
