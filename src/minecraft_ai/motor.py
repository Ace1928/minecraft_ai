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
class BootstrapMotorPolicy:
    """Deterministic fallback used for smoke tests and demonstration collection.

    This policy is deliberately hand-authored. It is a baseline and safety fallback,
    never a learned behavior prior or a benchmark-backed preferred controller.
    """

    policy_id: str = "bootstrap-motor-v1"
    mouse_gain: float = 35.0
    max_mouse_step: int = 120
    sweep_amplitude: float = 24.0
    jump_interval_ticks: int = 9
    _held_keys: set[str] = field(default_factory=set)
    _held_buttons: set[str] = field(default_factory=set)
    _last_sequence: int = -1
    _tick_count: int = 0
    _stuck_counter: int = 0

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
        self._tick_count += 1

        desired_keys: set[str] = set()
        desired_buttons: set[str] = set()
        mouse_dx = 0
        mouse_dy = 0

        dx = _number(blackboard, "target.dx")
        dy = _number(blackboard, "target.dy")
        obstacle_ahead = blackboard.fact("obstacle.ahead")
        danger = blackboard.fact("danger.immediate")
        mode = intent.mode.lower()

        pitch_up = blackboard.fact("pitch.looking_up")

        # 1. Camera orientation and target tracking
        if dx is not None:
            mouse_dx = _bounded_round(dx * self.mouse_gain, self.max_mouse_step)
        elif mode in {"explore", "reacquire", "navigate"}:
            # Controlled 40-tick scan with five turning ticks in each direction.
            cycle = self._tick_count % 40
            if cycle < 5:
                mouse_dx = 16
            elif 20 <= cycle < 25:
                mouse_dx = -16

        if dy is not None:
            mouse_dy = _bounded_round(dy * self.mouse_gain, self.max_mouse_step)
        elif pitch_up and bool(pitch_up.value):
            # Pitch correction: if looking up at sky, pitch down toward horizon
            mouse_dy = 35

        underwater = blackboard.fact("environment.underwater")

        # 2. Movement, swimming & obstacle auto-climbing
        if underwater and bool(underwater.value):
            # Swim to surface: hold space and look upward
            desired_keys.add("space")
            desired_keys.add("w")
            mouse_dy = -25
        elif mode in {"approach", "navigate", "explore", "mine", "attack", "use"}:
            desired_keys.add("w")
            obstacle_detected = bool(obstacle_ahead and obstacle_ahead.value)
            periodic_jump = self._tick_count % self.jump_interval_ticks == 0
            if obstacle_detected or periodic_jump:
                desired_keys.add("space")
                if obstacle_ahead and bool(obstacle_ahead.value) and (self._tick_count % 6 == 0):
                    # Lateral strafe to navigate around tree trunks / rock pillars
                    desired_keys.add("d" if self._tick_count % 12 < 6 else "a")
        elif mode in {"retreat", "backoff"}:
            desired_keys.add("s")
            if danger and bool(danger.value):
                desired_keys.add("space")

        # 3. Sprinting & Sneaking parameters
        exploring_on_land = mode == "explore" and not bool(underwater and underwater.value)
        if bool(intent.parameters.get("sprint", False)) or exploring_on_land:
            desired_keys.add("ctrl")
        if bool(intent.parameters.get("sneak", False)):
            desired_keys.add("shift")
        if bool(intent.parameters.get("jump", False)):
            desired_keys.add("space")

        # 4. Action button rhythms (mine, attack, place, use)
        if mode == "mine":
            desired_buttons.add("left")
        elif mode == "attack":
            # Tactical hit rhythm on left click
            if self._tick_count % 3 != 0:
                desired_buttons.add("left")
            else:
                desired_keys.add("a" if self._tick_count % 6 < 3 else "d")
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
