from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from .perception import PerceptionBlackboard, PerceptionFact, Track
from .safety import MotorAction


class OutcomeKind(StrEnum):
    """Temporal contracts implemented by the first outcome verifier."""

    MINING = "mining"
    TRAVERSAL = "traversal"
    INVENTORY_OPEN = "inventory_open"
    INVENTORY_CLOSE = "inventory_close"
    CRAFTING = "crafting"


class OutcomeStatus(StrEnum):
    PENDING = "pending"
    PROGRESS = "progress"
    SUCCEEDED = "succeeded"
    STALLED = "stalled"


class OutcomeSignal(StrEnum):
    NONE = "none"
    BLOCK_DAMAGE_PROGRESS = "block_damage_progress"
    BLOCK_BROKEN = "block_broken"
    MINING_STALLED = "mining_stalled"
    LOCOMOTION_PROGRESS = "locomotion_progress"
    LOCOMOTION_STALLED = "locomotion_stalled"
    GUI_TRANSITION_PROGRESS = "gui_transition_progress"
    INVENTORY_OPENED = "inventory_opened"
    INVENTORY_CLOSED = "inventory_closed"
    PLANKS_CRAFTED = "planks_crafted"
    GUI_TRANSITION_STALLED = "gui_transition_stalled"


class OutcomeVerification(BaseModel):
    """One typed observation about the active option."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1, max_length=256)
    kind: OutcomeKind
    status: OutcomeStatus
    signal: OutcomeSignal = OutcomeSignal.NONE
    observed_ns: int = Field(ge=0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=512)
    evidence_keys: tuple[str, ...] = ()

    @property
    def terminal(self) -> bool:
        return self.status in {OutcomeStatus.SUCCEEDED, OutcomeStatus.STALLED}


@dataclass(frozen=True)
class OutcomeVerifierConfig:
    min_fact_confidence: float = 0.70
    mining_absent_probability: float = 0.20
    static_hash_distance: int = 2
    mining_change_distance: int = 8
    mining_min_attack_ms: int = 400
    mining_post_release_settle_ms: int = 300
    mining_stable_samples: int = 3
    mining_target_loss_samples: int = 2
    mining_damage_phase_samples: int = 5
    mining_peak_damage_distance: int = 24
    mining_stall_ms: int = 1_500
    traversal_change_distance: int = 6
    traversal_crosshair_change_distance: int = 4
    traversal_progress_samples: int = 2
    traversal_stall_ms: int = 2_000
    camera_quiet_ms: int = 200
    inventory_change_distance: int = 6
    inventory_stable_samples: int = 2
    inventory_stall_ms: int = 2_500

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_fact_confidence <= 1.0:
            raise ValueError("min_fact_confidence must be in 0..1")
        if not 0.0 <= self.mining_absent_probability <= 1.0:
            raise ValueError("mining_absent_probability must be in 0..1")
        thresholds = (
            self.static_hash_distance,
            self.mining_change_distance,
            self.mining_min_attack_ms,
            self.mining_post_release_settle_ms,
            self.mining_stable_samples,
            self.mining_target_loss_samples,
            self.mining_damage_phase_samples,
            self.mining_peak_damage_distance,
            self.mining_stall_ms,
            self.traversal_change_distance,
            self.traversal_crosshair_change_distance,
            self.traversal_progress_samples,
            self.traversal_stall_ms,
            self.camera_quiet_ms,
            self.inventory_change_distance,
            self.inventory_stable_samples,
            self.inventory_stall_ms,
        )
        if any(value < 1 for value in thresholds):
            raise ValueError("outcome verifier thresholds must be positive")


@dataclass
class _MiningState:
    baseline_hash: str | None
    target_was_visible: bool
    target_source: str | None
    target_track_id: str | None
    target_kind: str | None
    trusted_transition_source: str | None
    last_hash: str | None
    last_hash_ns: int = -1
    last_change_ns: int = 0
    samples: int = 0
    candidate_hash: str | None = None
    candidate_samples: int = 0
    damage_phase_hashes: list[str] = field(default_factory=list)
    peak_damage_distance: int = 0
    target_fact_ns: int = -1
    target_loss_samples: int = 0
    target_loss_source: str | None = None
    attack_started_ns: int | None = None
    attack_released_ns: int | None = None
    invalidated: bool = False


@dataclass
class _TraversalState:
    frame_hash: str | None
    crosshair_hash: str | None
    last_sample_ns: int = -1
    last_change_ns: int = 0
    samples: int = 0
    progress_samples: int = 0
    movement_started_ns: int | None = None
    last_camera_ns: int = -1


@dataclass
class _InventoryState:
    baseline: tuple[str | None, str | None]
    interaction_started_ns: int | None = None
    candidate: tuple[str | None, str | None] | None = None
    candidate_samples: int = 0
    last_sample_ns: int = -1


@dataclass(frozen=True)
class _InputDelta:
    movement_active: bool
    movement_started: bool
    attack_active: bool
    attack_started: bool
    attack_released: bool
    camera_changed: bool
    interaction: bool


_MOVEMENT_KEYS = frozenset({"w", "a", "s", "d", "space"})
_INVENTORY_KEYS = frozenset({"e", "escape", "esc"})
@dataclass
class TemporalOutcomeVerifier:
    """Verify narrow outcomes from action-bound temporal visual evidence.

    One hash change proves only that pixels changed. Mining success additionally
    requires an explicit break fact, repeated exact-bound target loss, or a
    complete action-bound damage cycle: several distinct attack-time phases and
    stable replacement pixels after release. Consequently a crack animation
    that clears, one transient occlusion, and unrelated inventory changes can
    report no more than progress.
    """

    config: OutcomeVerifierConfig = field(default_factory=OutcomeVerifierConfig)
    _run_id: str | None = field(default=None, init=False)
    _kind: OutcomeKind | None = field(default=None, init=False)
    _started_ns: int = field(default=0, init=False)
    _state: _MiningState | _TraversalState | _InventoryState | None = field(
        default=None, init=False
    )
    _held_keys: set[str] = field(default_factory=set, init=False)
    _held_buttons: set[str] = field(default_factory=set, init=False)

    @property
    def active_run_id(self) -> str | None:
        return self._run_id

    def begin(
        self,
        run_id: str,
        kind: OutcomeKind,
        blackboard: PerceptionBlackboard,
        *,
        now_ns: int | None = None,
        trusted_transition_source: str | None = None,
    ) -> None:
        if not run_id:
            raise ValueError("run_id must not be empty")
        now = time.monotonic_ns() if now_ns is None else now_ns
        self.reset()
        self._run_id = run_id
        self._kind = kind
        self._started_ns = now
        if kind == OutcomeKind.MINING:
            target = _semantic_fact(
                blackboard,
                "target.visible",
                now_ns=now,
                min_confidence=self.config.min_fact_confidence,
            )
            crosshair = _hash_observation(blackboard, "frame.crosshair_dhash", now)
            target_binding = _target_binding(
                blackboard,
                target,
            )
            self._state = _MiningState(
                baseline_hash=None if crosshair is None else crosshair[0],
                target_was_visible=target_binding is not None,
                target_source=None if target_binding is None else target_binding[0],
                target_track_id=None if target_binding is None else target_binding[1],
                target_kind=None if target_binding is None else target_binding[2],
                trusted_transition_source=trusted_transition_source,
                last_hash=None if crosshair is None else crosshair[0],
                last_hash_ns=-1 if crosshair is None else crosshair[1],
                last_change_ns=now,
            )
        elif kind == OutcomeKind.TRAVERSAL:
            self._state = _TraversalState(
                frame_hash=_hash_value(blackboard, "frame.dhash", now),
                crosshair_hash=_hash_value(blackboard, "frame.crosshair_dhash", now),
                last_change_ns=now,
            )
        else:
            self._state = _InventoryState(baseline=_scene_signature(blackboard, now))

    def reset(self) -> None:
        self._run_id = None
        self._kind = None
        self._started_ns = 0
        self._state = None
        self._held_keys.clear()
        self._held_buttons.clear()

    def observe(
        self,
        blackboard: PerceptionBlackboard,
        *,
        action: MotorAction | None = None,
        now_ns: int | None = None,
    ) -> OutcomeVerification:
        if self._run_id is None or self._kind is None or self._state is None:
            raise RuntimeError("outcome verifier has no active run")
        now = time.monotonic_ns() if now_ns is None else now_ns
        delta = self._apply_action(action)
        if isinstance(self._state, _MiningState):
            return self._observe_mining(blackboard, self._state, delta, now)
        if isinstance(self._state, _TraversalState):
            return self._observe_traversal(blackboard, self._state, delta, now)
        return self._observe_inventory(blackboard, self._state, delta, now)

    def _apply_action(self, action: MotorAction | None) -> _InputDelta:
        previous_keys = set(self._held_keys)
        previous_buttons = set(self._held_buttons)
        if action is not None:
            self._held_keys.update(action.keys_down)
            self._held_keys.difference_update(action.keys_up)
            self._held_buttons.update(action.buttons_down)
            self._held_buttons.difference_update(action.buttons_up)
        was_moving = bool(previous_keys & _MOVEMENT_KEYS)
        moving = bool(self._held_keys & _MOVEMENT_KEYS)
        was_attacking = "left" in previous_buttons
        attacking = "left" in self._held_buttons
        return _InputDelta(
            movement_active=moving,
            movement_started=moving and not was_moving,
            attack_active=attacking,
            attack_started=attacking and not was_attacking,
            attack_released=was_attacking and not attacking,
            camera_changed=bool(action is not None and (action.mouse_dx or action.mouse_dy)),
            interaction=bool(
                action is not None
                and (
                    set(action.keys_down) & _INVENTORY_KEYS
                    or set(action.buttons_down)
                )
            ),
        )

    def _observe_mining(
        self,
        blackboard: PerceptionBlackboard,
        state: _MiningState,
        delta: _InputDelta,
        now_ns: int,
    ) -> OutcomeVerification:
        if state.attack_started_ns is None and not delta.attack_started:
            _refresh_mining_baseline(
                state,
                blackboard,
                now_ns=now_ns,
                min_confidence=self.config.min_fact_confidence,
            )
            return self._pending(now_ns, "waiting for an attack press")
        if delta.attack_started:
            _refresh_mining_baseline(
                state,
                blackboard,
                now_ns=now_ns,
                min_confidence=self.config.min_fact_confidence,
            )
            state.attack_started_ns = now_ns
            state.attack_released_ns = None
            state.last_change_ns = now_ns
            state.invalidated = delta.camera_changed or delta.movement_active
        elif delta.camera_changed or delta.movement_active:
            state.invalidated = True
        if delta.attack_released:
            state.attack_released_ns = now_ns

        changed = _update_mining_hash(state, blackboard, self.config, now_ns)
        _update_target_loss(state, blackboard, self.config, now_ns)
        assert state.attack_started_ns is not None
        attack_end = state.attack_released_ns or now_ns
        attack_ms = max(0.0, (attack_end - state.attack_started_ns) / 1_000_000)
        stable_change = state.candidate_samples >= self.config.mining_stable_samples
        release_settled = bool(
            state.attack_released_ns is not None
            and now_ns - state.attack_released_ns
            >= self.config.mining_post_release_settle_ms * 1_000_000
        )
        strong, strong_keys = _strong_break_evidence(
            state,
            blackboard,
            now_ns=now_ns,
            min_confidence=self.config.min_fact_confidence,
        )
        settled_target_loss = bool(
            release_settled
            and state.target_was_visible
            and state.target_loss_samples >= self.config.mining_target_loss_samples
        )
        replacement_target = _bound_target_still_present_after_release(
            state,
            blackboard,
            now_ns=now_ns,
            min_probability=self.config.min_fact_confidence,
        )
        settled_damage_cycle = bool(
            release_settled
            and state.target_was_visible
            and len(state.damage_phase_hashes) >= self.config.mining_damage_phase_samples
            and state.peak_damage_distance >= self.config.mining_peak_damage_distance
            and state.candidate_hash is not None
            and all(
                _hash_distance(state.candidate_hash, phase_hash)
                > self.config.static_hash_distance
                for phase_hash in state.damage_phase_hashes
            )
            and replacement_target is not None
        )
        if (
            not state.invalidated
            and attack_ms >= self.config.mining_min_attack_ms
            and stable_change
            and (strong or settled_target_loss or settled_damage_cycle)
        ):
            evidence = (
                "frame.crosshair_dhash",
                *strong_keys,
                *(
                    ("target.exists_probability",)
                    if settled_target_loss
                    else ()
                ),
                *(("target.track_id", "target.tracking_source") if settled_target_loss else ()),
                *(
                    (
                        f"target.track_id={state.target_track_id}",
                        f"target.tracking_source={state.target_loss_source}",
                    )
                    if settled_target_loss
                    else ()
                ),
                *(
                    (
                        "target.track_id",
                        f"target.track_id={state.target_track_id}",
                        f"target.binding_source={state.target_source}",
                        "target.exists_probability",
                        f"target.tracking_source={replacement_target.source}",
                    )
                    if settled_damage_cycle and replacement_target is not None
                    else ()
                ),
            )
            return self._result(
                OutcomeStatus.SUCCEEDED,
                OutcomeSignal.BLOCK_BROKEN,
                now_ns,
                0.98 if strong else (0.92 if settled_target_loss else 0.88),
                (
                    "stationary sustained attack produced an explicit bound break observation"
                    if strong
                    else (
                        "stationary sustained attack produced stable replacement pixels and "
                        "two consecutive exact-bound low target-existence observations"
                        if settled_target_loss
                        else "stationary sustained attack traversed multiple damage phases, "
                        "settled on stable replacement pixels, and the exact-bound observer "
                        "found another matching target behind them"
                    )
                ),
                evidence,
            )
        if (
            delta.attack_active
            and state.samples >= self.config.mining_stable_samples
            and now_ns - state.last_change_ns >= self.config.mining_stall_ms * 1_000_000
        ):
            return self._result(
                OutcomeStatus.STALLED,
                OutcomeSignal.MINING_STALLED,
                now_ns,
                0.92,
                "attack remained active without meaningful crosshair-region change",
                ("frame.crosshair_dhash",),
            )
        if changed and attack_ms >= self.config.mining_min_attack_ms:
            return self._result(
                OutcomeStatus.PROGRESS,
                OutcomeSignal.BLOCK_DAMAGE_PROGRESS,
                now_ns,
                0.55,
                "crosshair pixels changed during attack but break evidence is incomplete",
                ("frame.crosshair_dhash",),
            )
        reason = (
            "mining evidence invalidated by camera or locomotion"
            if state.invalidated
            else "waiting for sustained, corroborated block-change evidence"
        )
        return self._pending(now_ns, reason)

    def _observe_traversal(
        self,
        blackboard: PerceptionBlackboard,
        state: _TraversalState,
        delta: _InputDelta,
        now_ns: int,
    ) -> OutcomeVerification:
        if delta.camera_changed:
            state.last_camera_ns = now_ns
        if delta.movement_started or (
            delta.movement_active and state.movement_started_ns is None
        ):
            state.movement_started_ns = now_ns
            state.last_change_ns = now_ns
            _refresh_traversal_hashes(state, blackboard, now_ns)
        if not delta.movement_active:
            state.movement_started_ns = None
            return self._pending(now_ns, "waiting for locomotion input")

        frame = _hash_observation(blackboard, "frame.dhash", now_ns)
        crosshair = _hash_observation(blackboard, "frame.crosshair_dhash", now_ns)
        sample_ns = max(
            -1 if frame is None else frame[1],
            -1 if crosshair is None else crosshair[1],
        )
        if sample_ns <= state.last_sample_ns:
            return self._pending(now_ns, "waiting for a fresh traversal frame")
        state.last_sample_ns = sample_ns
        state.samples += 1
        camera_quiet = bool(
            state.last_camera_ns < 0
            or now_ns - state.last_camera_ns >= self.config.camera_quiet_ms * 1_000_000
        )
        if not camera_quiet:
            _refresh_traversal_hashes(state, blackboard, now_ns)
            state.last_change_ns = now_ns
            return self._pending(now_ns, "camera motion currently confounds translation")

        frame_changed = bool(
            frame is not None
            and state.frame_hash is not None
            and _hash_distance(state.frame_hash, frame[0])
            >= self.config.traversal_change_distance
        )
        crosshair_changed = bool(
            crosshair is not None
            and state.crosshair_hash is not None
            and _hash_distance(state.crosshair_hash, crosshair[0])
            >= self.config.traversal_crosshair_change_distance
        )
        if frame_changed and crosshair_changed:
            state.progress_samples += 1
            state.last_change_ns = now_ns
            _refresh_traversal_hashes(state, blackboard, now_ns)
            if state.progress_samples >= self.config.traversal_progress_samples:
                return self._result(
                    OutcomeStatus.PROGRESS,
                    OutcomeSignal.LOCOMOTION_PROGRESS,
                    now_ns,
                    0.78,
                    "repeated world changes followed commanded quiet-camera locomotion",
                    ("frame.dhash", "frame.crosshair_dhash"),
                )
        if (
            state.samples >= self.config.traversal_progress_samples
            and now_ns - state.last_change_ns >= self.config.traversal_stall_ms * 1_000_000
        ):
            return self._result(
                OutcomeStatus.STALLED,
                OutcomeSignal.LOCOMOTION_STALLED,
                now_ns,
                0.92,
                "locomotion remained commanded while world pixels stayed static",
                ("frame.dhash", "frame.crosshair_dhash"),
            )
        return self._pending(now_ns, "collecting quiet-camera locomotion evidence")

    def _observe_inventory(
        self,
        blackboard: PerceptionBlackboard,
        state: _InventoryState,
        delta: _InputDelta,
        now_ns: int,
    ) -> OutcomeVerification:
        if delta.interaction and state.interaction_started_ns is None:
            state.interaction_started_ns = now_ns
        signature = _scene_signature(blackboard, now_ns)
        sample_ns = max(
            _hash_observed_ns(blackboard, "frame.ui_dhash", now_ns),
            _hash_observed_ns(blackboard, "frame.dhash", now_ns),
        )
        changed = _signature_changed(
            state.baseline,
            signature,
            threshold=self.config.inventory_change_distance,
        )
        if sample_ns > state.last_sample_ns:
            state.last_sample_ns = sample_ns
            if changed:
                if state.candidate is not None and not _signature_changed(
                    state.candidate,
                    signature,
                    threshold=self.config.static_hash_distance + 1,
                ):
                    state.candidate_samples += 1
                else:
                    state.candidate = signature
                    state.candidate_samples = 1
            else:
                state.candidate = None
                state.candidate_samples = 0

        expected, expected_keys = self._expected_inventory_state(blackboard, now_ns)
        if (
            state.interaction_started_ns is not None
            and state.candidate_samples >= self.config.inventory_stable_samples
            and expected
        ):
            opened = self._kind == OutcomeKind.INVENTORY_OPEN
            return self._result(
                OutcomeStatus.SUCCEEDED,
                OutcomeSignal.INVENTORY_OPENED if opened else OutcomeSignal.INVENTORY_CLOSED,
                now_ns,
                0.97,
                "input-triggered stable screen transition matches the observed inventory state",
                ("frame.ui_dhash", "frame.dhash", *expected_keys),
            )
        if (
            state.interaction_started_ns is not None
            and now_ns - state.interaction_started_ns
            >= self.config.inventory_stall_ms * 1_000_000
        ):
            return self._result(
                OutcomeStatus.STALLED,
                OutcomeSignal.GUI_TRANSITION_STALLED,
                now_ns,
                0.90,
                "inventory interaction did not produce a fully verified transition",
                ("frame.ui_dhash", "frame.dhash"),
            )
        if state.interaction_started_ns is not None and (changed or expected):
            return self._result(
                OutcomeStatus.PROGRESS,
                OutcomeSignal.GUI_TRANSITION_PROGRESS,
                now_ns,
                0.65,
                "screen or scene state changed, but transition evidence is incomplete",
                ("frame.ui_dhash", "frame.dhash", *expected_keys),
            )
        return self._pending(now_ns, "waiting for an inventory interaction and transition")

    def _expected_inventory_state(
        self,
        blackboard: PerceptionBlackboard,
        now_ns: int,
    ) -> tuple[bool, tuple[str, ...]]:
        modes = tuple(
            (
                key,
                _semantic_fact(
                    blackboard,
                    key,
                    now_ns=now_ns,
                    min_confidence=self.config.min_fact_confidence,
                ),
            )
            for key in ("scene.mode", "gui.mode")
        )
        fresh_modes = tuple(
            (key, fact)
            for key, fact in modes
            if fact is not None and fact.observed_ns >= self._started_ns
        )
        wanted = "inventory" if self._kind == OutcomeKind.INVENTORY_OPEN else "world"
        mode_key = next((key for key, fact in fresh_modes if fact.value == wanted), None)
        if mode_key is not None:
            return True, (mode_key,)
        if self._kind != OutcomeKind.INVENTORY_CLOSE:
            return False, ()
        playable = _semantic_fact(
            blackboard,
            "scene.playable",
            now_ns=now_ns,
            min_confidence=self.config.min_fact_confidence,
        )
        verified = bool(
            playable is not None
            and playable.value is True
            and playable.observed_ns >= self._started_ns
        )
        return verified, (() if not verified else ("scene.playable",))

    def _pending(self, now_ns: int, reason: str) -> OutcomeVerification:
        return self._result(
            OutcomeStatus.PENDING,
            OutcomeSignal.NONE,
            now_ns,
            0.0,
            reason,
        )

    def _result(
        self,
        status: OutcomeStatus,
        signal: OutcomeSignal,
        now_ns: int,
        confidence: float,
        reason: str,
        evidence_keys: tuple[str, ...] = (),
    ) -> OutcomeVerification:
        assert self._run_id is not None and self._kind is not None
        return OutcomeVerification(
            run_id=self._run_id,
            kind=self._kind,
            status=status,
            signal=signal,
            observed_ns=now_ns,
            confidence=confidence,
            reason=reason,
            evidence_keys=tuple(dict.fromkeys(evidence_keys)),
        )


def _refresh_mining_baseline(
    state: _MiningState,
    blackboard: PerceptionBlackboard,
    *,
    now_ns: int,
    min_confidence: float,
) -> None:
    crosshair = _hash_observation(blackboard, "frame.crosshair_dhash", now_ns)
    if crosshair is not None:
        state.baseline_hash = crosshair[0]
        state.last_hash = crosshair[0]
        state.last_hash_ns = crosshair[1]
    target = _semantic_fact(
        blackboard,
        "target.visible",
        now_ns=now_ns,
        min_confidence=min_confidence,
    )
    target_binding = _target_binding(
        blackboard,
        target,
    )
    state.target_was_visible = target_binding is not None
    state.target_source = None if target_binding is None else target_binding[0]
    state.target_track_id = None if target_binding is None else target_binding[1]
    state.target_kind = None if target_binding is None else target_binding[2]


def _update_mining_hash(
    state: _MiningState,
    blackboard: PerceptionBlackboard,
    config: OutcomeVerifierConfig,
    now_ns: int,
) -> bool:
    observation = _hash_observation(blackboard, "frame.crosshair_dhash", now_ns)
    if observation is None or observation[1] <= state.last_hash_ns:
        return False
    current, observed_ns = observation
    state.last_hash_ns = observed_ns
    state.samples += 1
    if state.last_hash is None or (
        _hash_distance(state.last_hash, current) > config.static_hash_distance
    ):
        state.last_change_ns = now_ns
    state.last_hash = current
    changed = bool(
        state.baseline_hash is not None
        and _hash_distance(state.baseline_hash, current) >= config.mining_change_distance
    )
    if (
        changed
        and state.attack_started_ns is not None
        and state.attack_released_ns is None
        and now_ns - state.attack_started_ns >= config.mining_min_attack_ms * 1_000_000
    ):
        assert state.baseline_hash is not None
        baseline_distance = _hash_distance(state.baseline_hash, current)
        state.peak_damage_distance = max(state.peak_damage_distance, baseline_distance)
        if not state.damage_phase_hashes or (
            _hash_distance(state.damage_phase_hashes[-1], current)
            > config.static_hash_distance
        ):
            state.damage_phase_hashes.append(current)
    if not changed:
        state.candidate_hash = None
        state.candidate_samples = 0
    elif (
        state.candidate_hash is not None
        and _hash_distance(state.candidate_hash, current) <= config.static_hash_distance
    ):
        state.candidate_samples += 1
    else:
        state.candidate_hash = current
        state.candidate_samples = 1
    return changed


def _update_target_loss(
    state: _MiningState,
    blackboard: PerceptionBlackboard,
    config: OutcomeVerifierConfig,
    now_ns: int,
) -> None:
    probability = _semantic_fact(
        blackboard,
        "target.exists_probability",
        now_ns=now_ns,
        min_confidence=1.0,
    )
    if (
        probability is None
        or state.attack_started_ns is None
        or probability.observed_ns <= state.target_fact_ns
        or probability.observed_ns <= state.attack_started_ns
        or not isinstance(probability.value, (int, float))
        or isinstance(probability.value, bool)
        or not _matches_target_binding(
            state,
            blackboard,
            probability,
        )
    ):
        return
    state.target_fact_ns = probability.observed_ns
    if float(probability.value) <= config.mining_absent_probability:
        state.target_loss_samples += 1
        state.target_loss_source = probability.source
    else:
        state.target_loss_samples = 0
        state.target_loss_source = None


def _strong_break_evidence(
    state: _MiningState,
    blackboard: PerceptionBlackboard,
    *,
    now_ns: int,
    min_confidence: float,
) -> tuple[bool, tuple[str, ...]]:
    if state.attack_started_ns is None:
        return False, ()
    broken = _semantic_fact(
        blackboard,
        "target.broken",
        now_ns=now_ns,
        min_confidence=min_confidence,
    )
    verified = bool(
        broken is not None
        and broken.value is True
        and broken.observed_ns > state.attack_started_ns
        and _matches_target_binding(
            state,
            blackboard,
            broken,
        )
    )
    if not verified:
        return False, ()
    return True, ("target.broken", "target.track_id")


def _bound_target_still_present_after_release(
    state: _MiningState,
    blackboard: PerceptionBlackboard,
    *,
    now_ns: int,
    min_probability: float,
) -> PerceptionFact | None:
    """Return a fresh exact-bound positive after the attacked pixels settled.

    This is deliberately not success evidence by itself. Combined with the
    complete damage cycle it distinguishes an identical block behind the one
    attacked from a classifier that simply stayed positive throughout mining.
    """
    if state.attack_released_ns is None:
        return None
    probability = _semantic_fact(
        blackboard,
        "target.exists_probability",
        now_ns=now_ns,
        min_confidence=1.0,
    )
    if (
        probability is None
        or probability.observed_ns <= state.attack_released_ns
        or not isinstance(probability.value, (int, float))
        or isinstance(probability.value, bool)
        or float(probability.value) < min_probability
        or not _matches_target_binding(state, blackboard, probability)
    ):
        return None
    return probability


def _refresh_traversal_hashes(
    state: _TraversalState,
    blackboard: PerceptionBlackboard,
    now_ns: int,
) -> None:
    state.frame_hash = _hash_value(blackboard, "frame.dhash", now_ns)
    state.crosshair_hash = _hash_value(blackboard, "frame.crosshair_dhash", now_ns)


def _hash_observation(
    blackboard: PerceptionBlackboard,
    key: str,
    now_ns: int,
) -> tuple[str, int] | None:
    fact = blackboard.fact(key, min_confidence=1.0, now_ns=now_ns)
    if fact is None or not isinstance(fact.value, str) or len(fact.value) != 16:
        return None
    try:
        int(fact.value, 16)
    except ValueError:
        return None
    return fact.value, fact.observed_ns


def _hash_value(blackboard: PerceptionBlackboard, key: str, now_ns: int) -> str | None:
    observation = _hash_observation(blackboard, key, now_ns)
    return None if observation is None else observation[0]


def _hash_observed_ns(blackboard: PerceptionBlackboard, key: str, now_ns: int) -> int:
    observation = _hash_observation(blackboard, key, now_ns)
    return -1 if observation is None else observation[1]


def _semantic_fact(
    blackboard: PerceptionBlackboard,
    key: str,
    *,
    now_ns: int,
    min_confidence: float,
) -> PerceptionFact | None:
    fact = blackboard.fact(key, min_confidence=min_confidence, now_ns=now_ns)
    if fact is None or fact.source.startswith("bootstrap:"):
        return None
    return fact


def _target_binding(
    blackboard: PerceptionBlackboard,
    target: PerceptionFact | None,
) -> tuple[str, str, str] | None:
    if target is None or target.value is not True:
        return None
    latest = blackboard.latest()
    if latest is None:
        return None

    operator_prefix = "operator:explicit-grounding:"
    if target.source.startswith(operator_prefix):
        track_id = target.source.removeprefix(operator_prefix)
        track = next((item for item in latest.tracks if item.track_id == track_id), None)
        if (
            track is None
            or track.attributes.get("source") != "operator"
            or not _track_contains_crosshair(track)
        ):
            return None
        normalized_kind = _normalize_target_kind(track.label)
        return target.source, track.track_id, normalized_kind

    candidates = tuple(
        track
        for track in latest.tracks
        if _track_contains_crosshair(track)
        and _track_matches_source(track.track_id, track.attributes, target.source)
    )
    if not candidates:
        return None
    track = max(candidates, key=lambda item: (item.last_seen_ns, item.confidence))
    return target.source, track.track_id, _normalize_target_kind(track.label)


def _matches_target_binding(
    state: _MiningState,
    blackboard: PerceptionBlackboard,
    fact: PerceptionFact,
) -> bool:
    if (
        state.target_source is None
        or state.target_track_id is None
        or state.target_kind is None
        or state.attack_started_ns is None
        or fact.observed_ns <= state.attack_started_ns
    ):
        return False
    latest = blackboard.latest()
    if latest is None:
        return False
    track = next(
        (item for item in latest.tracks if item.track_id == state.target_track_id),
        None,
    )
    if (
        track is None
        or _normalize_target_kind(track.label) != state.target_kind
        or track.last_seen_ns != fact.observed_ns
    ):
        return False

    operator_source = f"operator:explicit-grounding:{state.target_track_id}"
    if state.target_source == operator_source:
        return bool(
            state.trusted_transition_source is not None
            and fact.source == state.trusted_transition_source
            and track.attributes.get("source") == "operator"
            and track.attributes.get("tracking_source") == state.trusted_transition_source
        )
    return bool(
        fact.source == state.target_source
        and _track_matches_source(track.track_id, track.attributes, state.target_source)
    )


def _track_matches_source(
    track_id: str,
    attributes: dict[str, str | int | float | bool],
    source: str,
) -> bool:
    if source == f"operator:explicit-grounding:{track_id}":
        return True
    if attributes.get("tracking_source") == source:
        return True
    if not source.startswith("vlm:"):
        return False
    query_id = source.rsplit(":", 1)[-1]
    return track_id.startswith(f"vlm:{query_id}:")


def _contains_crosshair(x: float, y: float, width: float, height: float) -> bool:
    return x <= 0.5 <= x + width and y <= 0.5 <= y + height


def _track_contains_crosshair(track: Track) -> bool:
    region = track.region
    return _contains_crosshair(region.x, region.y, region.width, region.height)


def _normalize_target_kind(value: str) -> str:
    return value.strip().casefold().replace(" ", "_")


def _scene_signature(
    blackboard: PerceptionBlackboard,
    now_ns: int,
) -> tuple[str | None, str | None]:
    return (
        _hash_value(blackboard, "frame.ui_dhash", now_ns),
        _hash_value(blackboard, "frame.dhash", now_ns),
    )


def _signature_changed(
    baseline: tuple[str | None, str | None],
    current: tuple[str | None, str | None],
    *,
    threshold: int,
) -> bool:
    distances = tuple(
        _hash_distance(before, after)
        for before, after in zip(baseline, current, strict=True)
        if before is not None and after is not None
    )
    return bool(distances and max(distances) >= threshold)


def _hash_distance(first: str, second: str) -> int:
    if len(first) != 16 or len(second) != 16:
        return 64
    try:
        return (int(first, 16) ^ int(second, 16)).bit_count()
    except ValueError:
        return 64
