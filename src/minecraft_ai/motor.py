from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from .perception import PerceptionBlackboard
from .safety import MotorAction


class MotorIntent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    skill_id: str
    mode: str
    target_label: str | None = None
    parameters: dict[str, str | int | float | bool] = Field(default_factory=dict)


class MotorPolicy(Protocol):
    policy_id: str

    def act(
        self,
        blackboard: PerceptionBlackboard,
        intent: MotorIntent,
        *,
        sequence: int,
    ) -> MotorAction: ...

    def reset(self) -> MotorAction: ...


def _number(blackboard: PerceptionBlackboard, key: str) -> float | None:
    fact = blackboard.fact(key, min_confidence=0.35)
    if fact is None or not isinstance(fact.value, (int, float)):
        return None
    return float(fact.value)


@dataclass
class HeuristicMotorPolicy:
    """Deterministic bootstrap policy used until a learned GOA policy wins evals.

    This is intentionally small and bounded. It is a safe baseline and data
    collector, not the intended production intelligence.
    """

    policy_id: str = "heuristic-v1"
    mouse_gain: float = 22.0
    max_mouse_step: int = 90
    _held_keys: set[str] = field(default_factory=set)
    _held_buttons: set[str] = field(default_factory=set)
    _last_sequence: int = -1

    def act(
        self,
        blackboard: PerceptionBlackboard,
        intent: MotorIntent,
        *,
        sequence: int,
    ) -> MotorAction:
        if sequence <= self._last_sequence:
            raise ValueError("motor policy sequence must increase monotonically")
        self._last_sequence = sequence
        desired_keys: set[str] = set()
        desired_buttons: set[str] = set()
        mouse_dx = 0
        mouse_dy = 0

        dx = _number(blackboard, "target.dx")
        dy = _number(blackboard, "target.dy")
        if dx is not None:
            mouse_dx = _bounded_round(dx * self.mouse_gain, self.max_mouse_step)
        if dy is not None:
            mouse_dy = _bounded_round(dy * self.mouse_gain, self.max_mouse_step)

        mode = intent.mode.lower()
        if mode in {"approach", "navigate", "explore", "mine", "attack", "use"}:
            desired_keys.add("w")
        elif mode in {"retreat", "backoff"}:
            desired_keys.add("s")
        if bool(intent.parameters.get("sprint", False)):
            desired_keys.add("ctrl")
        if bool(intent.parameters.get("sneak", False)):
            desired_keys.add("shift")
        if bool(intent.parameters.get("jump", False)):
            desired_keys.add("space")
        if mode in {"mine", "attack"}:
            desired_buttons.add("left")
        elif mode in {"use", "place", "eat"}:
            desired_buttons.add("right")

        action = MotorAction(
            sequence=sequence,
            keys_down=tuple(sorted(desired_keys - self._held_keys)),
            keys_up=tuple(sorted(self._held_keys - desired_keys)),
            buttons_down=tuple(sorted(desired_buttons - self._held_buttons)),
            buttons_up=tuple(sorted(self._held_buttons - desired_buttons)),
            mouse_dx=mouse_dx,
            mouse_dy=mouse_dy,
            duration_ms=50,
        )
        self._held_keys = desired_keys
        self._held_buttons = desired_buttons
        return action

    def reset(self) -> MotorAction:
        sequence = self._last_sequence + 1
        action = MotorAction(
            sequence=sequence,
            keys_up=tuple(sorted(self._held_keys)),
            buttons_up=tuple(sorted(self._held_buttons)),
        )
        self._held_keys.clear()
        self._held_buttons.clear()
        self._last_sequence = sequence
        return action


def _bounded_round(value: float, limit: int) -> int:
    return max(-limit, min(limit, int(round(value))))
