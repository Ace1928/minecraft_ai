from __future__ import annotations

import time
from dataclasses import dataclass, field

from .crafting_control import (
    BoundedPlankCraftController,
    INVENTORY_TOGGLE_DURATION_MS,
    PlankCraftPhase,
)
from .mining_control import MiningLeaseGuard, is_rocket_source, track_contains_crosshair
from .motor import MotorIntent, MotorPolicy
from .outcome_verifier import (
    OutcomeKind,
    OutcomeSignal,
    OutcomeStatus,
    OutcomeVerification,
    TemporalOutcomeVerifier,
)
from .perception import PerceptionBlackboard
from .safety import MotorAction
from .skills import (
    SkillActionPermissions,
    SkillCondition,
    SkillFailureCode,
    SkillOutcome,
    SkillRun,
    SkillSpec,
)
from .trajectory import ActionOrigin


@dataclass(frozen=True)
class ExecutionTick:
    run: SkillRun
    action: MotorAction | None
    recovery_skills: tuple[str, ...] = ()
    motor_intent: MotorIntent | None = None
    policy_status: dict[str, object] = field(default_factory=dict)
    action_origin: ActionOrigin = ActionOrigin.POLICY
    outcome_verification: OutcomeVerification | None = None


@dataclass(frozen=True)
class _PendingMiningVerification:
    failure_code: SkillFailureCode
    deadline_ns: int


_MINING_POST_RELEASE_VERIFY_MS = 5_000
_MINING_SUCCESS_OVERRIDABLE_FAILURES = frozenset(
    {
        SkillFailureCode.MINING_TARGET_CHANGED,
        SkillFailureCode.MINING_VISUAL_STAGNATION,
        SkillFailureCode.MINING_LEASE_EXPIRED,
    }
)
_TRAVERSAL_SKILL_IDS = frozenset(
    {"explore_forward", "traverse_level_ground", "traverse_visible_obstacle"}
)
_LOCOMOTION_RELEASE_KEYS = ("a", "d", "s", "space", "w")
_REACQUIRE_MIN_CONFIDENCE = 0.65


class SkillExecutor:
    """Evaluate semantic skill contracts against live perception every tick."""

    def __init__(self, policy: MotorPolicy) -> None:
        self.policy = policy
        self._spec: SkillSpec | None = None
        self._run: SkillRun | None = None
        self._parameters: dict[str, str | int | float | bool] = {}
        self._instruction_override: str | None = None
        self._initiated = False
        self._last_intent: MotorIntent | None = None
        self._mining_guard = MiningLeaseGuard()
        self._plank_crafter = BoundedPlankCraftController()
        self._outcome_verifier = TemporalOutcomeVerifier()
        self._pending_mining_verification: _PendingMiningVerification | None = None
        self._inventory_open_sent = False
        self._inventory_close_sent = False

    @property
    def run(self) -> SkillRun | None:
        return self._run

    @property
    def instruction(self) -> str | None:
        if self._spec is None or self._run is None:
            return None
        if self._instruction_override:
            return self._instruction_override
        return _skill_instruction(self._spec, self._parameters)

    @property
    def parameters(self) -> dict[str, str | int | float | bool]:
        """Return the active option bindings without exposing mutable executor state."""
        return dict(self._parameters)

    @property
    def policy_parameters(self) -> dict[str, str | int | float | bool]:
        """Return option bindings intersected with the skill's action contract."""
        if self._spec is None or self._run is None:
            return {}
        return _policy_parameters(self._spec.action_permissions, self._parameters)

    def close(self) -> None:
        close = getattr(self.policy, "close", None)
        if callable(close):
            close()

    def plank_crafting_semantics_ready(
        self,
        blackboard: PerceptionBlackboard,
        *,
        now_ns: int,
    ) -> bool:
        return self._plank_crafter.semantic_request_ready(
            blackboard,
            now_ns=now_ns,
        )

    @property
    def plank_crafting_phase(self) -> PlankCraftPhase | None:
        return self._plank_crafter.phase

    def note_plank_crafting_semantic_completion(self, phase: PlankCraftPhase) -> None:
        self._plank_crafter.note_semantic_completion(phase)

    def plank_crafting_semantic_time_remaining_ms(self, *, now_ns: int) -> int | None:
        return self._plank_crafter.semantic_time_remaining_ms(now_ns)

    def start(
        self,
        spec: SkillSpec,
        *,
        run_id: str,
        context_key: str = "default",
        parameters: dict[str, str | int | float | bool] | None = None,
        now_ns: int | None = None,
        instruction: str | None = None,
    ) -> SkillRun:
        if self._run is not None and self._run.outcome == SkillOutcome.RUNNING:
            raise RuntimeError("a skill is already running")
        started = time.monotonic_ns() if now_ns is None else now_ns
        self._spec = spec
        self._parameters = dict(parameters or {})
        self._instruction_override = instruction
        self._initiated = False
        self._last_intent = None
        self._mining_guard.reset()
        self._plank_crafter.reset()
        self._outcome_verifier.reset()
        self._pending_mining_verification = None
        self._inventory_open_sent = False
        self._inventory_close_sent = False
        self._run = SkillRun(
            run_id=run_id,
            skill_id=spec.skill_id,
            started_ns=started,
            context_key=context_key,
            parameters=self._parameters,
        )
        return self._run

    def tick(
        self,
        blackboard: PerceptionBlackboard,
        *,
        sequence: int,
        now_ns: int | None = None,
    ) -> ExecutionTick:
        if self._spec is None or self._run is None:
            raise RuntimeError("no skill is running")
        if self._run.outcome != SkillOutcome.RUNNING:
            return ExecutionTick(run=self._run, action=None)
        now = time.monotonic_ns() if now_ns is None else now_ns

        if self._pending_mining_verification is not None:
            return self._verify_released_mining_outcome(blackboard, now_ns=now)
        if now - self._run.started_ns >= self._spec.max_duration_ms * 1_000_000:
            if self._spec.skill_id == "collect_recent_drop":
                return self._finish(
                    SkillOutcome.FAILED,
                    now,
                    SkillFailureCode.RESOURCE_PICKUP_UNVERIFIED.value,
                    failure_code=SkillFailureCode.RESOURCE_PICKUP_UNVERIFIED,
                    force_release_keys=_LOCOMOTION_RELEASE_KEYS,
                )
            return self._finish(
                SkillOutcome.TIMED_OUT,
                now,
                "skill-timeout",
                recover=self._spec.skill_id == "craft_wood_planks",
            )
        failed = _first_matching(self._spec.failure_conditions, blackboard, now_ns=now)
        if failed is not None:
            return self._finish(
                SkillOutcome.FAILED,
                now,
                f"failure-condition:{failed.key}",
                recover=True,
            )
        if self._spec.skill_id == "reacquire_target":
            if _reacquisition_satisfied(
                blackboard,
                run_started_ns=self._run.started_ns,
                target_label=_target_label(self._parameters),
                now_ns=now,
            ):
                return self._finish(SkillOutcome.SUCCEEDED, now, None)
        elif (
            self._spec.skill_id not in {"mine_visible_block", "craft_wood_planks"}
            and self._spec.success_conditions
            and conditions_satisfied(
                self._spec.success_conditions,
                blackboard,
                now_ns=now,
            )
        ):
            return self._finish(SkillOutcome.SUCCEEDED, now, None)
        if not self._initiated:
            if not initiation_satisfied(self._spec, blackboard, now_ns=now):
                return self._finish(
                    SkillOutcome.FAILED,
                    now,
                    "initiation-precondition-unsatisfied",
                    recover=True,
                )
            self._initiated = True
        if self._spec.invariants and not conditions_satisfied(
            self._spec.invariants, blackboard, now_ns=now
        ):
            return self._finish(
                SkillOutcome.FAILED,
                now,
                "invariant-lost",
                recover=True,
            )

        if self._spec.skill_id == "craft_wood_planks":
            return self._tick_plank_crafting(
                blackboard,
                sequence=sequence,
                now_ns=now,
            )
        if self._spec.skill_id == "open_inventory":
            return self._tick_inventory_open(sequence=sequence)
        if self._spec.skill_id == "close_open_inventory":
            return self._tick_inventory_close(sequence=sequence)

        intent = MotorIntent(
            skill_id=self._spec.skill_id,
            mode=self._spec.policy_ref or self._spec.skill_id,
            episode_id=self._run.run_id,
            action_level=self._spec.action_level,
            instruction=(
                self._instruction_override
                or _policy_instruction(self._spec)
            ),
            condition_scale=self._spec.policy_condition_scale,
            target_label=_target_label(self._parameters),
            parameters=self.policy_parameters,
        )
        action = self.policy.act(blackboard, intent, sequence=sequence)
        self._last_intent = intent
        mining = self._mining_guard.inspect(action, blackboard, intent, now_ns=now)
        verification = self._observe_mining_outcome(
            blackboard,
            action=mining.action,
            now_ns=now,
        )
        traversal_verification = self._observe_traversal_outcome(
            blackboard,
            action=mining.action,
            now_ns=now,
        )
        if mining.failure_code is not None:
            if (
                verification is not None
                and verification.status == OutcomeStatus.SUCCEEDED
                and mining.failure_code in _MINING_SUCCESS_OVERRIDABLE_FAILURES
            ):
                return self._finish(
                    SkillOutcome.SUCCEEDED,
                    now,
                    None,
                    outcome_verification=verification,
                )
            if (
                mining.failure_code in _MINING_SUCCESS_OVERRIDABLE_FAILURES
                and verification is not None
            ):
                if (
                    verification.status == OutcomeStatus.PROGRESS
                    and verification.signal == OutcomeSignal.BLOCK_DAMAGE_PROGRESS
                ):
                    return self._begin_released_mining_verification(
                        blackboard,
                        now_ns=now,
                        failure_code=mining.failure_code,
                        force_release_left=mining.force_release_left,
                        force_release_keys=mining.force_release_keys,
                        force_release_buttons=mining.force_release_buttons,
                    )
            return self._finish(
                SkillOutcome.FAILED,
                now,
                mining.failure_code.value,
                recover=True,
                failure_code=mining.failure_code,
                force_release_left=mining.force_release_left,
                force_release_keys=mining.force_release_keys,
                force_release_buttons=mining.force_release_buttons,
            )
        if verification is not None and verification.status == OutcomeStatus.SUCCEEDED:
            return self._finish(
                SkillOutcome.SUCCEEDED,
                now,
                None,
                outcome_verification=verification,
            )
        if verification is not None and verification.status == OutcomeStatus.STALLED:
            return self._finish(
                SkillOutcome.FAILED,
                now,
                SkillFailureCode.MINING_VISUAL_STAGNATION.value,
                recover=True,
                failure_code=SkillFailureCode.MINING_VISUAL_STAGNATION,
                force_release_left=True,
                outcome_verification=verification,
            )
        if (
            traversal_verification is not None
            and traversal_verification.status == OutcomeStatus.STALLED
        ):
            return self._finish(
                SkillOutcome.FAILED,
                now,
                SkillFailureCode.LOCOMOTION_STALLED.value,
                recover=True,
                failure_code=SkillFailureCode.LOCOMOTION_STALLED,
                force_release_keys=_LOCOMOTION_RELEASE_KEYS,
                outcome_verification=traversal_verification,
            )
        return ExecutionTick(
            run=self._run,
            action=mining.action,
            motor_intent=intent,
            policy_status=_policy_status_snapshot(self.policy),
            action_origin=(ActionOrigin.SYNTHETIC if mining.synthetic else ActionOrigin.POLICY),
            outcome_verification=(
                traversal_verification
                if traversal_verification is not None
                and traversal_verification.status == OutcomeStatus.PROGRESS
                else None
            ),
        )

    def _tick_plank_crafting(
        self,
        blackboard: PerceptionBlackboard,
        *,
        sequence: int,
        now_ns: int,
    ) -> ExecutionTick:
        if self._spec is None or self._run is None:
            raise RuntimeError("no skill is running")
        step = self._plank_crafter.step(
            blackboard,
            run_id=self._run.run_id,
            sequence=sequence,
            now_ns=now_ns,
        )
        intent = MotorIntent(
            skill_id=self._spec.skill_id,
            mode=step.mode,
            episode_id=self._run.run_id,
            action_level=self._spec.action_level,
            instruction=step.instruction,
            parameters=self.policy_parameters,
        )
        self._last_intent = intent
        if step.failure_reason is not None:
            return self._finish(
                SkillOutcome.FAILED,
                now_ns,
                step.failure_reason,
                recover=True,
            )
        if step.verification is not None:
            return self._finish(
                SkillOutcome.SUCCEEDED,
                now_ns,
                None,
                outcome_verification=step.verification,
            )
        return ExecutionTick(
            run=self._run,
            action=step.action,
            motor_intent=intent,
            policy_status=_policy_status_snapshot(self.policy),
            action_origin=ActionOrigin.SYNTHETIC,
        )

    def _tick_inventory_open(self, *, sequence: int) -> ExecutionTick:
        """Toggle the inventory once, then wait for calibrated overlay proof."""
        if self._spec is None or self._run is None:
            raise RuntimeError("no skill is running")
        intent = self._inventory_toggle_intent()
        if self._inventory_open_sent:
            return ExecutionTick(
                run=self._run,
                action=None,
                motor_intent=intent,
                policy_status=_policy_status_snapshot(self.policy),
                action_origin=ActionOrigin.SYNTHETIC,
            )
        self._inventory_open_sent = True
        return ExecutionTick(
            run=self._run,
            action=MotorAction(
                sequence=sequence,
                keys_down=("e",),
                keys_up=("e",),
                duration_ms=INVENTORY_TOGGLE_DURATION_MS,
            ),
            motor_intent=intent,
            policy_status=_policy_status_snapshot(self.policy),
            action_origin=ActionOrigin.SYNTHETIC,
        )

    def _inventory_toggle_intent(self) -> MotorIntent:
        if self._spec is None or self._run is None:
            raise RuntimeError("no skill is running")
        intent = MotorIntent(
            skill_id=self._spec.skill_id,
            mode=self._spec.policy_ref or self._spec.skill_id,
            episode_id=self._run.run_id,
            action_level=self._spec.action_level,
            instruction=self._instruction_override or _policy_instruction(self._spec),
            parameters=self.policy_parameters,
        )
        self._last_intent = intent
        return intent

    def _tick_inventory_close(self, *, sequence: int) -> ExecutionTick:
        """Toggle an open inventory once, then wait for world-frame proof.

        Closing an already-detected inventory is a fixed safety recovery, not a
        visuomotor search problem. One atomic, empirically reliable E toggle
        avoids repeated policy guesses while the normal success contract above
        still requires fresh perception to report a playable world.
        """
        if self._spec is None or self._run is None:
            raise RuntimeError("no skill is running")
        intent = self._inventory_toggle_intent()
        if self._inventory_close_sent:
            return ExecutionTick(
                run=self._run,
                action=None,
                motor_intent=intent,
                policy_status=_policy_status_snapshot(self.policy),
                action_origin=ActionOrigin.SYNTHETIC,
            )
        self._inventory_close_sent = True
        return ExecutionTick(
            run=self._run,
            action=MotorAction(
                sequence=sequence,
                keys_down=("e",),
                keys_up=("e",),
                duration_ms=INVENTORY_TOGGLE_DURATION_MS,
            ),
            motor_intent=intent,
            policy_status=_policy_status_snapshot(self.policy),
            action_origin=ActionOrigin.SYNTHETIC,
        )

    def _observe_mining_outcome(
        self,
        blackboard: PerceptionBlackboard,
        *,
        action: MotorAction | None,
        now_ns: int,
    ) -> OutcomeVerification | None:
        if self._spec is None or self._run is None:
            return None
        if self._spec.skill_id != "mine_visible_block":
            return None
        if self._outcome_verifier.active_run_id != self._run.run_id:
            observer_source = getattr(self.policy, "outcome_observer_source", None)
            trusted_transition_source = (
                observer_source() if callable(observer_source) else None
            )
            self._outcome_verifier.begin(
                self._run.run_id,
                OutcomeKind.MINING,
                blackboard,
                now_ns=now_ns,
                trusted_transition_source=(
                    trusted_transition_source
                    if isinstance(trusted_transition_source, str)
                    and trusted_transition_source
                    else None
                ),
            )
        return self._outcome_verifier.observe(
            blackboard,
            action=action,
            now_ns=now_ns,
        )

    def _observe_traversal_outcome(
        self,
        blackboard: PerceptionBlackboard,
        *,
        action: MotorAction | None,
        now_ns: int,
    ) -> OutcomeVerification | None:
        if self._spec is None or self._run is None:
            return None
        if self._spec.skill_id not in _TRAVERSAL_SKILL_IDS:
            return None
        if self._outcome_verifier.active_run_id != self._run.run_id:
            self._outcome_verifier.begin(
                self._run.run_id,
                OutcomeKind.TRAVERSAL,
                blackboard,
                now_ns=now_ns,
            )
        return self._outcome_verifier.observe(
            blackboard,
            action=action,
            now_ns=now_ns,
        )

    def _begin_released_mining_verification(
        self,
        blackboard: PerceptionBlackboard,
        *,
        now_ns: int,
        failure_code: SkillFailureCode,
        force_release_left: bool,
        force_release_keys: tuple[str, ...],
        force_release_buttons: tuple[str, ...],
    ) -> ExecutionTick:
        if self._run is None:
            raise RuntimeError("no skill is running")
        release = self._release_motor(
            force_release_left=force_release_left,
            force_release_keys=force_release_keys,
            force_release_buttons=force_release_buttons,
            preserve_outcome_perception=True,
        )
        self._outcome_verifier.observe(
            blackboard,
            action=release,
            now_ns=now_ns,
        )
        self._pending_mining_verification = _PendingMiningVerification(
            failure_code=failure_code,
            deadline_ns=now_ns + _MINING_POST_RELEASE_VERIFY_MS * 1_000_000,
        )
        return ExecutionTick(
            run=self._run,
            action=release,
            motor_intent=self._last_intent,
            policy_status=_policy_status_snapshot(self.policy),
            action_origin=ActionOrigin.RESET,
        )

    def _release_motor(
        self,
        *,
        force_release_left: bool = False,
        force_release_keys: tuple[str, ...] = (),
        force_release_buttons: tuple[str, ...] = (),
        preserve_outcome_perception: bool = False,
    ) -> MotorAction:
        guard_keys = self._mining_guard.held_keys
        guard_buttons = self._mining_guard.held_buttons
        crafting_sequence = self._plank_crafter.last_sequence
        mining_held = self._mining_guard.reset()
        self._plank_crafter.reset()
        release_for_observation = getattr(self.policy, "release_for_observation", None)
        release = (
            release_for_observation()
            if preserve_outcome_perception and callable(release_for_observation)
            else self.policy.reset()
        )
        if crafting_sequence >= 0 and release.sequence <= crafting_sequence:
            release = release.model_copy(update={"sequence": crafting_sequence + 1})
        if mining_held or force_release_left:
            release = _force_button_release(release, "left")
        for button in tuple(sorted({*guard_buttons, *force_release_buttons})):
            release = _force_button_release(release, button)
        all_release_keys = tuple(sorted({*guard_keys, *force_release_keys}))
        if all_release_keys:
            release = _force_key_release(release, all_release_keys)
        return release

    def _verify_released_mining_outcome(
        self,
        blackboard: PerceptionBlackboard,
        *,
        now_ns: int,
    ) -> ExecutionTick:
        if self._run is None or self._pending_mining_verification is None:
            raise RuntimeError("no mining outcome is awaiting verification")
        pending = self._pending_mining_verification
        unsafe = _urgent_mining_scene_reason(blackboard, now_ns=now_ns)
        if unsafe is not None:
            return self._finish(
                SkillOutcome.FAILED,
                now_ns,
                unsafe,
                recover=True,
                failure_code=SkillFailureCode.MINING_UNSAFE_SCENE,
                force_release_left=True,
            )
        assert self._spec is not None
        failed = _first_matching(self._spec.failure_conditions, blackboard, now_ns=now_ns)
        if failed is not None:
            return self._finish(
                SkillOutcome.FAILED,
                now_ns,
                f"failure-condition:{failed.key}",
                recover=True,
                force_release_left=True,
            )
        if self._spec.invariants and not conditions_satisfied(
            self._spec.invariants,
            blackboard,
            now_ns=now_ns,
        ):
            return self._finish(
                SkillOutcome.FAILED,
                now_ns,
                "invariant-lost",
                recover=True,
                force_release_left=True,
            )
        if now_ns - self._run.started_ns >= self._spec.max_duration_ms * 1_000_000:
            return self._finish(
                SkillOutcome.TIMED_OUT,
                now_ns,
                "skill-timeout",
                force_release_left=True,
            )
        if now_ns >= pending.deadline_ns:
            return self._finish(
                SkillOutcome.FAILED,
                now_ns,
                pending.failure_code.value,
                recover=True,
                failure_code=pending.failure_code,
                force_release_left=True,
            )
        poll_perception = getattr(self.policy, "poll_perception", None)
        if callable(poll_perception) and self._last_intent is not None:
            poll_perception(blackboard, self._last_intent)
        verification = self._outcome_verifier.observe(blackboard, now_ns=now_ns)
        if verification.status == OutcomeStatus.SUCCEEDED:
            return self._finish(
                SkillOutcome.SUCCEEDED,
                now_ns,
                None,
                outcome_verification=verification,
            )
        if verification.status == OutcomeStatus.STALLED:
            return self._finish(
                SkillOutcome.FAILED,
                now_ns,
                SkillFailureCode.MINING_VISUAL_STAGNATION.value,
                recover=True,
                failure_code=SkillFailureCode.MINING_VISUAL_STAGNATION,
                force_release_left=True,
                outcome_verification=verification,
            )
        return ExecutionTick(
            run=self._run,
            action=None,
            motor_intent=self._last_intent,
            policy_status=_policy_status_snapshot(self.policy),
            action_origin=ActionOrigin.RESET,
        )

    def cancel(self, *, now_ns: int | None = None) -> ExecutionTick:
        if self._run is None or self._spec is None:
            raise RuntimeError("no skill is running")
        now = time.monotonic_ns() if now_ns is None else now_ns
        return self._finish(
            SkillOutcome.CANCELLED,
            now,
            "cancelled",
            recover=self._spec.skill_id == "craft_wood_planks",
        )

    def _finish(
        self,
        outcome: SkillOutcome,
        ended_ns: int,
        reason: str | None,
        *,
        recover: bool = False,
        failure_code: SkillFailureCode | None = None,
        force_release_left: bool = False,
        force_release_keys: tuple[str, ...] = (),
        force_release_buttons: tuple[str, ...] = (),
        outcome_verification: OutcomeVerification | None = None,
    ) -> ExecutionTick:
        if self._run is None or self._spec is None:
            raise RuntimeError("no skill is running")
        current = self._run
        self._run = current.model_copy(
            update={
                "ended_ns": ended_ns,
                "outcome": outcome,
                "failure_reason": reason,
                "failure_code": failure_code,
            }
        )
        motor_intent = self._last_intent
        release = self._release_motor(
            force_release_left=force_release_left,
            force_release_keys=force_release_keys,
            force_release_buttons=force_release_buttons,
        )
        policy_status = _policy_status_snapshot(self.policy)
        self._last_intent = None
        self._outcome_verifier.reset()
        self._pending_mining_verification = None
        recovery = self._spec.recovery_skills if recover else ()
        return ExecutionTick(
            run=self._run,
            action=release,
            recovery_skills=recovery,
            motor_intent=motor_intent,
            policy_status=policy_status,
            action_origin=ActionOrigin.RESET,
            outcome_verification=outcome_verification,
        )


def _force_button_release(action: MotorAction, button: str) -> MotorAction:
    return action.model_copy(
        update={
            "buttons_down": tuple(item for item in action.buttons_down if item != button),
            "buttons_up": tuple(sorted({*action.buttons_up, button})),
        }
    )


def _force_key_release(action: MotorAction, keys: tuple[str, ...]) -> MotorAction:
    requested = set(keys)
    return action.model_copy(
        update={
            "keys_down": tuple(item for item in action.keys_down if item not in requested),
            "keys_up": tuple(sorted({*action.keys_up, *requested})),
        }
    )


def _urgent_mining_scene_reason(
    blackboard: PerceptionBlackboard,
    *,
    now_ns: int,
) -> str | None:
    danger = blackboard.fact("danger.immediate", min_confidence=0.65, now_ns=now_ns)
    if danger is not None and danger.value is True:
        return "mining-unsafe:danger.immediate"
    overlay = blackboard.fact("scene.ui_overlay", min_confidence=0.65, now_ns=now_ns)
    if overlay is not None and overlay.value is True:
        return "mining-unsafe:scene.ui_overlay"
    playable = blackboard.fact("scene.playable", min_confidence=0.65, now_ns=now_ns)
    if playable is not None and playable.value is False:
        return "mining-unsafe:scene.playable"
    for key in ("scene.mode", "gui.mode"):
        mode = blackboard.fact(key, min_confidence=0.65, now_ns=now_ns)
        if mode is not None and isinstance(mode.value, str) and mode.value not in {
            "world",
            "unknown",
        }:
            return f"mining-unsafe:{key}:{mode.value}"
    return None


def _policy_status_snapshot(policy: MotorPolicy) -> dict[str, object]:
    status = getattr(policy, "status", None)
    if callable(status):
        reported = status()
        if isinstance(reported, dict):
            return reported
    return {"policy_id": policy.policy_id}


def _skill_instruction(
    spec: SkillSpec,
    parameters: dict[str, str | int | float | bool],
) -> str:
    """Render the semantic option contract for a goal-conditioned motor policy."""
    instruction = spec.description.strip() or spec.name.strip()
    if not parameters:
        return instruction
    rendered = ", ".join(f"{key}={parameters[key]}" for key in sorted(parameters))
    return f"{instruction}. Parameters: {rendered}"


def _policy_instruction(spec: SkillSpec) -> str:
    """Return the concise command used to condition a learned motor policy.

    Planner-facing descriptions deliberately retain the complete option contract.
    Visuomotor policies are conditioned with the short command distribution used
    by their published training/evaluation interface instead of receiving prose
    intended for an LLM or an operator.
    """
    return spec.policy_instruction or spec.description.strip() or spec.name.strip()


def _target_label(parameters: dict[str, str | int | float | bool]) -> str | None:
    target = parameters.get("target")
    if not isinstance(target, str) or not target.strip():
        return None
    return target.strip()


def _reacquisition_satisfied(
    blackboard: PerceptionBlackboard,
    *,
    run_started_ns: int,
    target_label: str | None,
    now_ns: int,
) -> bool:
    """Require one coherent post-start ROCKET box under the crosshair."""

    visible = blackboard.fact(
        "target.visible",
        min_confidence=_REACQUIRE_MIN_CONFIDENCE,
        now_ns=now_ns,
    )
    tracking = blackboard.fact(
        "target.tracking_confidence",
        min_confidence=_REACQUIRE_MIN_CONFIDENCE,
        now_ns=now_ns,
    )
    probability = blackboard.fact(
        "target.exists_probability",
        min_confidence=_REACQUIRE_MIN_CONFIDENCE,
        now_ns=now_ns,
    )
    kind = blackboard.fact(
        "target.kind",
        min_confidence=_REACQUIRE_MIN_CONFIDENCE,
        now_ns=now_ns,
    )
    facts = (visible, tracking, probability, kind)
    if any(fact is None for fact in facts):
        return False
    assert visible is not None and tracking is not None
    assert probability is not None and kind is not None
    coherent_facts = (visible, tracking, probability, kind)
    observed_ns = visible.observed_ns
    source = visible.source
    if (
        visible.value is not True
        or observed_ns <= run_started_ns
        or not is_rocket_source(source)
        or any(
            fact.source != source or fact.observed_ns != observed_ns
            for fact in coherent_facts
        )
        or not _number_at_least(tracking.value, _REACQUIRE_MIN_CONFIDENCE)
        or not _number_at_least(probability.value, _REACQUIRE_MIN_CONFIDENCE)
        or not isinstance(kind.value, str)
    ):
        return False
    latest = blackboard.latest()
    if latest is None:
        return False
    for track in latest.tracks:
        track_probability = track.attributes.get("target_exists_probability")
        if (
            track.last_seen_ns == observed_ns
            and track.confidence >= _REACQUIRE_MIN_CONFIDENCE
            and track.attributes.get("tracking_source") == source
            and _number_at_least(track_probability, _REACQUIRE_MIN_CONFIDENCE)
            and track.label.casefold() == kind.value.casefold()
            and (target_label is None or track.label.casefold() == target_label.casefold())
            and track_contains_crosshair(track)
        ):
            return True
    return False


def _number_at_least(value: object, minimum: float) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and float(value) >= minimum
    )


def _policy_parameters(
    permissions: SkillActionPermissions,
    parameters: dict[str, str | int | float | bool],
) -> dict[str, str | int | float | bool]:
    """Intersect planner/operator bindings with an option's learned-action envelope."""
    merged = dict(parameters)
    for name in (
        "allow_attack",
        "allow_use",
        "allow_jump",
        "allow_drop",
        "allow_inventory",
        "allow_hotbar",
    ):
        skill_allows = bool(getattr(permissions, name))
        runtime_allows = parameters.get(name) is not False
        merged[name] = skill_allows and runtime_allows
    return merged


def conditions_satisfied(
    conditions: tuple[SkillCondition, ...],
    blackboard: PerceptionBlackboard,
    now_ns: int | None = None,
) -> bool:
    """Evaluate a complete semantic option condition set against fresh observations."""
    return all(_matches(condition, blackboard, now_ns=now_ns) for condition in conditions)


def initiation_satisfied(
    spec: SkillSpec,
    blackboard: PerceptionBlackboard,
    now_ns: int | None = None,
) -> bool:
    """Evaluate OR-of-AND initiation groups for a learned option contract."""
    groups = tuple(group for group in (spec.preconditions, *spec.initiation_alternatives) if group)
    if not groups:
        return True
    return any(conditions_satisfied(group, blackboard, now_ns=now_ns) for group in groups)


def _first_matching(
    conditions: tuple[SkillCondition, ...],
    blackboard: PerceptionBlackboard,
    now_ns: int | None = None,
) -> SkillCondition | None:
    for condition in conditions:
        if _matches(condition, blackboard, now_ns=now_ns):
            return condition
    return None


def _matches(
    condition: SkillCondition,
    blackboard: PerceptionBlackboard,
    now_ns: int | None = None,
) -> bool:
    try:
        fact = blackboard.fact(
            condition.key,
            min_confidence=condition.min_confidence,
            now_ns=now_ns,
        )
    except TypeError:
        fact = blackboard.fact(condition.key, min_confidence=condition.min_confidence)
    if condition.operator == "exists":
        return fact is not None
    if fact is None:
        return False
    value = fact.value
    expected = condition.value
    if condition.operator == "truthy":
        return bool(value)
    if condition.operator == "falsy":
        return not bool(value)
    if condition.operator == "eq":
        return value == expected
    if condition.operator == "neq":
        return value != expected
    if not isinstance(value, (int, float)) or not isinstance(expected, (int, float)):
        return False
    if condition.operator == "gte":
        return float(value) >= float(expected)
    if condition.operator == "lte":
        return float(value) <= float(expected)
    if condition.operator == "gt":
        return float(value) > float(expected)
    if condition.operator == "lt":
        return float(value) < float(expected)
    return False
