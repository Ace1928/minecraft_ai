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
    instruction: str | None = Field(default=None, min_length=1, max_length=1024)
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


def _number(
    blackboard: PerceptionBlackboard,
    key: str,
    *,
    min_confidence: float = 0.65,
) -> float | None:
    fact = blackboard.fact(key, min_confidence=min_confidence)
    if fact is None or not isinstance(fact.value, (int, float)):
        return None
    return float(fact.value)


def _reliable_truth(blackboard: PerceptionBlackboard, key: str) -> bool:
    fact = blackboard.fact(key, min_confidence=0.65)
    return bool(fact is not None and not fact.source.startswith("bootstrap:") and bool(fact.value))


@dataclass
class BootstrapMotorPolicy:
    """Deterministic fallback used for smoke tests and demonstration collection.

    This policy is deliberately hand-authored. It is a baseline and safety fallback,
    never a learned behavior prior or a benchmark-backed preferred controller.
    """

    policy_id: str = "bootstrap-motor-v2-smooth"
    mouse_gain: float = 20.0
    mouse_deadzone: float = 0.08
    mouse_smoothing: float = 0.35
    max_mouse_step: int = 28
    max_mouse_acceleration: int = 4
    _held_keys: set[str] = field(default_factory=set)
    _held_buttons: set[str] = field(default_factory=set)
    _last_sequence: int = -1
    _tick_count: int = 0
    _stuck_counter: int = 0
    _last_mouse_dx: int = 0
    _last_mouse_dy: int = 0

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

        target_visible = _reliable_truth(blackboard, "target.visible")
        dx = _number(blackboard, "target.dx") if target_visible else None
        dy = _number(blackboard, "target.dy") if target_visible else None
        obstacle_ahead = blackboard.fact("obstacle.ahead")
        danger = blackboard.fact("danger.immediate")
        mode = intent.mode.lower()

        # Camera motion is evidence-driven. Exploratory scanning must be an explicit
        # skill/action, never periodic jitter hidden inside locomotion.
        if dx is not None:
            mouse_dx = self._smooth_mouse(dx, self._last_mouse_dx)

        if dy is not None:
            mouse_dy = self._smooth_mouse(dy, self._last_mouse_dy)

        self._last_mouse_dx = mouse_dx
        self._last_mouse_dy = mouse_dy

        underwater = _reliable_truth(blackboard, "environment.underwater")
        scene_playable = blackboard.fact("scene.playable", min_confidence=0.65)
        playable = not (scene_playable is not None and not bool(scene_playable.value))

        # 2. Movement, swimming & obstacle auto-climbing
        if not playable:
            desired_keys.clear()
        elif underwater:
            desired_keys.add("space")
            desired_keys.add("w")
        elif mode in {"approach", "navigate", "explore", "mine", "attack", "use"}:
            desired_keys.add("w")
            obstacle_detected = bool(
                obstacle_ahead
                and obstacle_ahead.confidence >= 0.65
                and not obstacle_ahead.source.startswith("bootstrap:")
                and obstacle_ahead.value
            )
            if obstacle_detected:
                desired_keys.add("space")
        elif mode in {"retreat", "backoff"}:
            desired_keys.add("s")
            if danger and bool(danger.value):
                desired_keys.add("space")

        # 3. Sprinting & Sneaking parameters
        if bool(intent.parameters.get("sprint", False)):
            desired_keys.add("ctrl")
        if bool(intent.parameters.get("sneak", False)):
            desired_keys.add("shift")
        if bool(intent.parameters.get("jump", False)):
            desired_keys.add("space")

        # 4. Action button rhythms (mine, attack, place, use)
        if mode == "mine" and target_visible:
            desired_buttons.add("left")
        elif mode == "attack" and target_visible:
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

    def _smooth_mouse(self, error: float, previous: int) -> int:
        target = (
            0
            if abs(error) <= self.mouse_deadzone
            else _bounded_round(error * self.mouse_gain, self.max_mouse_step)
        )
        blended = int(round(previous + self.mouse_smoothing * (target - previous)))
        delta = max(
            -self.max_mouse_acceleration,
            min(self.max_mouse_acceleration, blended - previous),
        )
        return _bounded_round(previous + delta, self.max_mouse_step)

    def reset(self) -> MotorAction:
        sequence = self._last_sequence + 1
        action = MotorAction(
            sequence=sequence,
            keys_up=tuple(sorted(self._held_keys)),
            buttons_up=tuple(sorted(self._held_buttons)),
        )
        self._held_keys.clear()
        self._held_buttons.clear()
        self._last_mouse_dx = 0
        self._last_mouse_dy = 0
        self._last_sequence = sequence
        return action


def _bounded_round(value: float, limit: int) -> int:
    return max(-limit, min(limit, int(round(value))))
