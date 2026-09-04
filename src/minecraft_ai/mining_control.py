from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from .motor import MotorIntent
from .perception import PerceptionBlackboard, PerceptionFact, Track
from .safety import MotorAction
from .skills import SkillFailureCode


_MINING_MODES = frozenset({"mine", "gather", "gather_wood", "break"})
_LOCOMOTION_KEYS = frozenset({"w", "a", "s", "d", "space"})
_HOTBAR_KEYS = frozenset("123456789")
_OPERATOR_AIM_GAIN = 40.0
_OPERATOR_AIM_MAX_STEP = 12
_OPERATOR_AIM_INITIAL_FRAME_GRACE_MS = 250
_EMPTY_ITEMS = frozenset({"air", "empty", "empty_slot", "hand", "none"})
_UNVERIFIED_ITEM = "unverified_item"
_WOOD_SPECIES = frozenset(
    {
        "acacia",
        "birch",
        "cherry",
        "crimson",
        "jungle",
        "mangrove",
        "oak",
        "spruce",
        "warped",
    }
)
_LOG_DESCRIPTOR_WORDS = frozenset(
    {
        "block",
        "central",
        "dark",
        "log",
        "nearby",
        "pale",
        "stem",
        "stripped",
        "target",
        "tree",
        "trunk",
        "vertical",
        "visible",
    }
)
_UNBREAKABLE = frozenset(
    {
        "barrier",
        "bedrock",
        "chain_command_block",
        "command_block",
        "end_gateway",
        "end_portal",
        "end_portal_frame",
        "jigsaw",
        "light",
        "moving_block",
        "repeating_command_block",
        "structure_block",
    }
)
_SOFT_BLOCKS = frozenset(
    {
        "clay",
        "coarse_dirt",
        "dirt",
        "grass_block",
        "gravel",
        "mud",
        "muddy_mangrove_roots",
        "mycelium",
        "podzol",
        "red_sand",
        "sand",
        "snow_block",
        "soul_sand",
        "soul_soil",
    }
)
_STONE_BLOCKS = frozenset(
    {
        "andesite",
        "basalt",
        "blackstone",
        "calcite",
        "cobbled_deepslate",
        "cobblestone",
        "deepslate",
        "diorite",
        "dripstone_block",
        "end_stone",
        "granite",
        "netherrack",
        "prismarine",
        "sandstone",
        "stone",
        "tuff",
    }
)
_TOOL_TIERS = {
    "wooden": 1,
    "gold": 1,
    "golden": 1,
    "stone": 2,
    "copper": 2,
    "iron": 3,
    "diamond": 4,
    "netherite": 5,
}


class _BlockFamily(StrEnum):
    LOG = "log"
    WOOD = "wood"
    SOFT = "soft"
    PICKAXE = "pickaxe"
    DEEPSLATE = "deepslate"
    OBSIDIAN = "obsidian"
    ANCIENT_DEBRIS = "ancient_debris"
    UNBREAKABLE = "unbreakable"


_HAND_SAFE_FAMILIES = frozenset(
    {
        _BlockFamily.LOG,
        _BlockFamily.WOOD,
        _BlockFamily.SOFT,
    }
)


@dataclass(frozen=True)
class _BlockRule:
    family: _BlockFamily
    minimum_pick_tier: int | None = None


@dataclass(frozen=True)
class _VerifiedTarget:
    track_id: str
    kind: str
    rule: _BlockRule
    selected_item: str
    lease_ms: int


@dataclass
class _MiningLease:
    episode_id: str
    target: _VerifiedTarget
    started_ns: int
    deadline_ns: int
    visual_hash: str
    visual_observed_ns: int
    visual_seen_ns: int
    visual_changed_ns: int
    visual_samples: int = 1
    override_active: bool = False


@dataclass
class _PendingMiningAcquisition:
    episode_id: str
    started_ns: int
    deadline_ns: int
    motion_seen: bool = False
    settling: bool = False
    settle_after_ns: int = 0
    last_aim_observation_ns: int = 0


@dataclass(frozen=True)
class MiningGuardDecision:
    action: MotorAction
    failure_code: SkillFailureCode | None = None
    synthetic: bool = False
    force_release_left: bool = False
    force_release_keys: tuple[str, ...] = ()
    force_release_buttons: tuple[str, ...] = ()


@dataclass
class MiningLeaseGuard:
    """Bound one learned attack press to one visually verified block attempt."""

    min_confidence: float = 0.70
    max_track_age_ms: int = 15_000
    visual_signal_grace_ms: int = 750
    stagnation_ms: int = 1_250
    minimum_visual_samples: int = 4
    static_hash_distance: int = 2
    absolute_max_ms: int = 12_000
    acquisition_timeout_ms: int = 6_000
    acquisition_motion_grace_ms: int = 750
    operator_grounding_grace_ms: int = 1_500
    _lease: _MiningLease | None = None
    _pending: _PendingMiningAcquisition | None = None
    _held_keys: set[str] = field(default_factory=set)
    _held_buttons: set[str] = field(default_factory=set)
    _last_targeting_change_ns: int = 0

    @property
    def held_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._held_keys))

    @property
    def held_buttons(self) -> tuple[str, ...]:
        return tuple(sorted(self._held_buttons))

    def reset(self) -> bool:
        held = self._lease is not None or "left" in self._held_buttons
        self._lease = None
        self._pending = None
        self._held_keys.clear()
        self._held_buttons.clear()
        self._last_targeting_change_ns = 0
        return held

    def inspect(
        self,
        action: MotorAction,
        blackboard: PerceptionBlackboard,
        intent: MotorIntent,
        *,
        now_ns: int,
    ) -> MiningGuardDecision:
        held_keys = set(self._held_keys)
        held_buttons = set(self._held_buttons)
        # Input backends apply positive events before matching releases, so a
        # token present in both sets is an atomic tap and must end released.
        next_keys = (held_keys | set(action.keys_down)) - set(action.keys_up)
        next_buttons = (held_buttons | set(action.buttons_down)) - set(
            action.buttons_up
        )
        release_keys = tuple(
            sorted(held_keys | set(action.keys_down) | set(action.keys_up))
        )
        release_buttons = tuple(
            sorted(held_buttons | set(action.buttons_down) | set(action.buttons_up))
        )
        locomotion_requested = bool(
            _LOCOMOTION_KEYS.intersection(action.keys_down)
            or _LOCOMOTION_KEYS.intersection(next_keys)
        )
        tool_change_requested = bool(
            _HOTBAR_KEYS.intersection(action.keys_down)
            or _HOTBAR_KEYS.intersection(next_keys)
        )
        targeting_changed = bool(
            action.mouse_dx
            or action.mouse_dy
            or action.keys_down
            or action.keys_up
            or (set(action.buttons_down) - {"left"})
            or (set(action.buttons_up) - {"left"})
        )
        # Do not let an atomic tap evade mining's interaction interlock merely
        # because its final held-state is empty.
        conflicting_input = bool(
            (next_buttons | set(action.buttons_down)) - {"left"}
            or (next_keys | set(action.keys_down)) - _LOCOMOTION_KEYS - _HOTBAR_KEYS
        )
        lease = self._lease
        if lease is None:
            return self._inspect_without_lease(
                action,
                blackboard,
                intent,
                now_ns=now_ns,
                held_keys=held_keys,
                held_buttons=held_buttons,
                next_keys=next_keys,
                next_buttons=next_buttons,
                release_keys=release_keys,
                release_buttons=release_buttons,
                locomotion_requested=locomotion_requested,
                tool_change_requested=tool_change_requested,
                targeting_changed=targeting_changed,
                conflicting_input=conflicting_input,
            )

        continuation_failure = self._continuation_interlock(
            action,
            blackboard,
            intent,
            lease=lease,
            now_ns=now_ns,
            tool_change_requested=tool_change_requested,
            conflicting_input=conflicting_input,
        )
        if continuation_failure is not None:
            return MiningGuardDecision(
                action,
                failure_code=continuation_failure,
                force_release_left=True,
                force_release_keys=release_keys,
                force_release_buttons=release_buttons,
            )

        policy_action = action
        action = _quiesce_active_lease_action(action, held_keys=held_keys)
        next_keys = (held_keys | set(action.keys_down)) - set(action.keys_up)
        buttons_up = tuple(button for button in action.buttons_up if button != "left")
        buttons_down = action.buttons_down
        if lease.override_active:
            buttons_down = tuple(button for button in buttons_down if button != "left")
        suppressed_release = len(buttons_up) != len(action.buttons_up)
        lease.override_active = lease.override_active or suppressed_release
        synthetic = lease.override_active or action != policy_action
        self._held_keys = next_keys
        self._held_buttons = (held_buttons | set(buttons_down)) - set(buttons_up)
        if buttons_up == action.buttons_up and buttons_down == action.buttons_down:
            return MiningGuardDecision(action, synthetic=synthetic)
        return MiningGuardDecision(
            action.model_copy(
                update={
                    "buttons_down": buttons_down,
                    "buttons_up": buttons_up,
                }
            ),
            synthetic=synthetic,
        )

    def _inspect_without_lease(
        self,
        action: MotorAction,
        blackboard: PerceptionBlackboard,
        intent: MotorIntent,
        *,
        now_ns: int,
        held_keys: set[str],
        held_buttons: set[str],
        next_keys: set[str],
        next_buttons: set[str],
        release_keys: tuple[str, ...],
        release_buttons: tuple[str, ...],
        locomotion_requested: bool,
        tool_change_requested: bool,
        targeting_changed: bool,
        conflicting_input: bool,
    ) -> MiningGuardDecision:
        mode = intent.mode.casefold()
        pending = self._pending
        operator_authorized = _explicit_operator_mining_authorized(
            blackboard,
            intent,
            now_ns=now_ns,
            min_confidence=self.min_confidence,
        )
        if pending is None and (
            mode not in _MINING_MODES
            or (
                "left" not in action.buttons_down
                and not operator_authorized
            )
        ):
            if targeting_changed:
                self._last_targeting_change_ns = now_ns
            self._held_keys = next_keys
            self._held_buttons = next_buttons
            return MiningGuardDecision(action)

        if pending is not None and (
            mode not in _MINING_MODES or (intent.episode_id or "") != pending.episode_id
        ):
            return MiningGuardDecision(
                action,
                failure_code=SkillFailureCode.MINING_EPISODE_CHANGED,
                force_release_left=True,
                force_release_keys=release_keys,
                force_release_buttons=release_buttons,
            )

        hard_failure = self._prelease_hard_failure(
            blackboard,
            action,
            now_ns=now_ns,
            tool_change_requested=tool_change_requested,
            conflicting_input=conflicting_input,
        )
        if hard_failure is not None:
            return MiningGuardDecision(
                action,
                failure_code=hard_failure,
                force_release_left=True,
                force_release_keys=release_keys,
                force_release_buttons=release_buttons,
            )
        semantic_failure = _prelease_semantic_failure(
            blackboard,
            intent,
            now_ns=now_ns,
            min_confidence=self.min_confidence,
        )
        if semantic_failure is not None:
            return MiningGuardDecision(
                action,
                failure_code=semantic_failure,
                force_release_left=True,
                force_release_keys=release_keys,
                force_release_buttons=release_buttons,
            )

        if pending is None:
            pending = _PendingMiningAcquisition(
                episode_id=intent.episode_id or "",
                started_ns=now_ns,
                deadline_ns=now_ns + self.acquisition_timeout_ms * 1_000_000,
            )
            self._pending = pending
        elif now_ns >= pending.deadline_ns:
            return MiningGuardDecision(
                action,
                failure_code=SkillFailureCode.MINING_ACQUISITION_TIMEOUT,
                force_release_left=True,
                force_release_keys=release_keys,
                force_release_buttons=release_buttons,
            )

        # STEVE/VPT represents attack as a short pulse. Once a safe mining
        # acquisition has latched that pulse, its ordinary zero-order release
        # must not erase the request while asynchronous grounding catches up.
        # The bounded timeout and all hard interlocks above remain authoritative.
        policy_action = action
        if "left" in action.buttons_up:
            action = action.model_copy(
                update={
                    "buttons_up": tuple(
                        button for button in action.buttons_up if button != "left"
                    )
                }
            )

        target = _verified_target(
            blackboard,
            intent,
            now_ns=now_ns,
            min_confidence=self.min_confidence,
            max_track_age_ms=self.max_track_age_ms,
            require_scene_match=True,
            evidence_after_ns=(
                pending.settle_after_ns
                if pending.settling
                else self._last_targeting_change_ns
            ),
        )
        if target in {
            SkillFailureCode.MINING_WRONG_TOOL,
            SkillFailureCode.MINING_TARGET_MISMATCH,
        }:
            return MiningGuardDecision(
                action,
                failure_code=target,
                force_release_left=True,
                force_release_keys=release_keys,
                force_release_buttons=release_buttons,
            )

        # The semantic body does not consume ROCKET's target coordinates and
        # ROCKET's own body action is intentionally discarded. During an exact
        # operator mining acquisition, use each new localization once to nudge
        # only the camera. Locomotion and attack remain suppressed, and a later
        # newer, centered observation must still pass the normal target/tool
        # gates before attack can start.
        aim_track = _operator_aim_track(
            blackboard,
            intent,
            pending=pending,
            now_ns=now_ns,
            min_confidence=self.min_confidence,
            max_track_age_ms=self.max_track_age_ms,
        )
        if aim_track is not None and not track_contains_crosshair(aim_track):
            aimed = _aim_at_operator_track(
                action,
                aim_track,
                held_keys=held_keys,
            )
            pending.motion_seen = True
            pending.settling = True
            pending.settle_after_ns = now_ns
            pending.last_aim_observation_ns = aim_track.last_seen_ns
            self._last_targeting_change_ns = now_ns
            self._remember_emitted(aimed)
            return MiningGuardDecision(aimed, synthetic=aimed != policy_action)

        if pending.settling:
            quiesced = _quiesce_pending_action(action, held_keys=held_keys)
            visual = _visual_hash(blackboard, now_ns=now_ns)
            fresh_visual = bool(
                visual is not None and visual[1] > pending.settle_after_ns
            )
            if isinstance(target, _VerifiedTarget) and fresh_visual:
                assert visual is not None
                started = _press_left(quiesced)
                self._pending = None
                self._start_lease(
                    intent,
                    target,
                    visual_hash=visual[0],
                    visual_observed_ns=visual[1],
                    now_ns=now_ns,
                )
                self._remember_emitted(started)
                return MiningGuardDecision(started, synthetic=True)
            self._remember_emitted(quiesced)
            return MiningGuardDecision(quiesced, synthetic=quiesced != policy_action)

        has_current_motion = bool(
            action.mouse_dx
            or action.mouse_dy
            or _LOCOMOTION_KEYS.intersection(action.keys_down)
            or _LOCOMOTION_KEYS.intersection(action.keys_up)
        )
        held_locomotion = bool(_LOCOMOTION_KEYS.intersection(held_keys))
        pending.motion_seen = pending.motion_seen or has_current_motion or held_locomotion
        visual = _visual_hash(blackboard, now_ns=now_ns)
        fresh_visual = bool(
            visual is not None and visual[1] > self._last_targeting_change_ns
        )
        if (
            isinstance(target, _VerifiedTarget)
            and fresh_visual
            and not locomotion_requested
            and not has_current_motion
        ):
            assert visual is not None
            started = _press_left(_suppress_left_press(action))
            self._pending = None
            self._start_lease(
                intent,
                target,
                visual_hash=visual[0],
                visual_observed_ns=visual[1],
                now_ns=now_ns,
            )
            self._remember_emitted(started)
            return MiningGuardDecision(started, synthetic=started != policy_action)

        # The first combined attack/approach or attack/aim action may proceed,
        # but only with attack removed. Once grounding is current, quiesce any
        # still-held locomotion and wait for one post-release frame before press.
        if isinstance(target, _VerifiedTarget) and not has_current_motion:
            pending.settling = True
            pending.settle_after_ns = now_ns
            quiesced = _quiesce_pending_action(action, held_keys=held_keys)
            self._last_targeting_change_ns = now_ns
            self._remember_emitted(quiesced)
            return MiningGuardDecision(quiesced, synthetic=quiesced != policy_action)
        if isinstance(target, _VerifiedTarget) and held_locomotion:
            pending.settling = True
            pending.settle_after_ns = now_ns
            quiesced = _quiesce_pending_action(action, held_keys=held_keys)
            self._last_targeting_change_ns = now_ns
            self._remember_emitted(quiesced)
            return MiningGuardDecision(quiesced, synthetic=quiesced != policy_action)
        if (
            pending.motion_seen
            and now_ns - pending.started_ns
            >= self.acquisition_motion_grace_ms * 1_000_000
        ):
            pending.settling = True
            pending.settle_after_ns = now_ns
            quiesced = _quiesce_pending_action(action, held_keys=held_keys)
            self._last_targeting_change_ns = now_ns
            self._remember_emitted(quiesced)
            return MiningGuardDecision(quiesced, synthetic=quiesced != policy_action)

        waiting = _suppress_left_press(action)
        if targeting_changed:
            self._last_targeting_change_ns = now_ns
        self._remember_emitted(waiting)
        return MiningGuardDecision(waiting, synthetic=waiting != policy_action)

    def _prelease_hard_failure(
        self,
        blackboard: PerceptionBlackboard,
        action: MotorAction,
        *,
        now_ns: int,
        tool_change_requested: bool,
        conflicting_input: bool,
    ) -> SkillFailureCode | None:
        if _unsafe_scene(blackboard, now_ns=now_ns):
            return SkillFailureCode.MINING_UNSAFE_SCENE
        if action.camera_semantics != "world":
            return SkillFailureCode.MINING_CAMERA_CHANGED
        if tool_change_requested:
            return SkillFailureCode.MINING_TOOL_CHANGED
        if conflicting_input:
            return SkillFailureCode.MINING_CONFLICTING_INPUT
        return None

    def _start_lease(
        self,
        intent: MotorIntent,
        target: _VerifiedTarget,
        *,
        visual_hash: str,
        visual_observed_ns: int,
        now_ns: int,
    ) -> None:
        lease_ms = min(target.lease_ms, self.absolute_max_ms)
        self._lease = _MiningLease(
            episode_id=intent.episode_id or "",
            target=target,
            started_ns=now_ns,
            deadline_ns=now_ns + lease_ms * 1_000_000,
            visual_hash=visual_hash,
            visual_observed_ns=visual_observed_ns,
            visual_seen_ns=now_ns,
            visual_changed_ns=now_ns,
        )

    def _remember_emitted(self, action: MotorAction) -> None:
        self._held_keys.update(action.keys_down)
        self._held_keys.difference_update(action.keys_up)
        self._held_buttons.update(action.buttons_down)
        self._held_buttons.difference_update(action.buttons_up)

    def _continuation_interlock(
        self,
        action: MotorAction,
        blackboard: PerceptionBlackboard,
        intent: MotorIntent,
        *,
        lease: _MiningLease,
        now_ns: int,
        tool_change_requested: bool,
        conflicting_input: bool,
    ) -> SkillFailureCode | None:
        if intent.mode.casefold() not in _MINING_MODES or (intent.episode_id or "") != (
            lease.episode_id
        ):
            return SkillFailureCode.MINING_EPISODE_CHANGED
        if _unsafe_scene(blackboard, now_ns=now_ns):
            return SkillFailureCode.MINING_UNSAFE_SCENE
        if action.camera_semantics != "world":
            return SkillFailureCode.MINING_CAMERA_CHANGED
        if tool_change_requested:
            return SkillFailureCode.MINING_TOOL_CHANGED
        if conflicting_input:
            return SkillFailureCode.MINING_CONFLICTING_INPUT
        target_failure = _continuation_target_failure(
            blackboard,
            lease.target,
            now_ns=now_ns,
            min_confidence=self.min_confidence,
            max_track_age_ms=self.max_track_age_ms,
            lease_started_ns=lease.started_ns,
            operator_grounding_grace_ms=self.operator_grounding_grace_ms,
        )
        if target_failure is not None:
            return target_failure

        visual = _visual_hash(blackboard, now_ns=now_ns)
        if visual is None:
            if now_ns - lease.visual_seen_ns >= self.visual_signal_grace_ms * 1_000_000:
                return SkillFailureCode.MINING_VISUAL_SIGNAL_LOST
        elif visual[1] > lease.visual_observed_ns:
            lease.visual_samples += 1
            lease.visual_observed_ns = visual[1]
            lease.visual_seen_ns = now_ns
            if _hash_distance(lease.visual_hash, visual[0]) > self.static_hash_distance:
                lease.visual_hash = visual[0]
                lease.visual_changed_ns = now_ns

        if (
            lease.visual_samples >= self.minimum_visual_samples
            and now_ns - lease.visual_changed_ns >= self.stagnation_ms * 1_000_000
        ):
            return SkillFailureCode.MINING_VISUAL_STAGNATION
        if now_ns >= lease.deadline_ns:
            return SkillFailureCode.MINING_LEASE_EXPIRED
        return None


def _quiesce_active_lease_action(
    action: MotorAction,
    *,
    held_keys: set[str],
) -> MotorAction:
    """Keep a bound block attempt stationary despite ordinary policy drift."""
    locomotion = _LOCOMOTION_KEYS.intersection(
        held_keys | set(action.keys_down) | set(action.keys_up)
    )
    return action.model_copy(
        update={
            "keys_down": tuple(
                key for key in action.keys_down if key not in _LOCOMOTION_KEYS
            ),
            "keys_up": tuple(sorted(set(action.keys_up) | locomotion)),
            "mouse_dx": 0,
            "mouse_dy": 0,
        }
    )


def _suppress_left_press(action: MotorAction) -> MotorAction:
    buttons_down = tuple(button for button in action.buttons_down if button != "left")
    if buttons_down == action.buttons_down:
        return action
    return action.model_copy(update={"buttons_down": buttons_down})


def _quiesce_pending_action(
    action: MotorAction,
    *,
    held_keys: set[str],
) -> MotorAction:
    """Stop body/camera drift without ever forwarding the remembered attack."""
    locomotion = _LOCOMOTION_KEYS.intersection(
        held_keys | set(action.keys_down) | set(action.keys_up)
    )
    return action.model_copy(
        update={
            "keys_down": tuple(
                key for key in action.keys_down if key not in _LOCOMOTION_KEYS
            ),
            "keys_up": tuple(sorted(set(action.keys_up) | locomotion)),
            "buttons_down": tuple(
                button for button in action.buttons_down if button != "left"
            ),
            "mouse_dx": 0,
            "mouse_dy": 0,
        }
    )


def _aim_at_operator_track(
    action: MotorAction,
    track: Track,
    *,
    held_keys: set[str],
) -> MotorAction:
    """Emit one bounded camera-only correction toward an exact operator track."""
    quiesced = _quiesce_pending_action(action, held_keys=held_keys)
    region = track.region
    center_x = region.x + region.width / 2.0
    center_y = region.y + region.height / 2.0
    horizontal_error = (
        0.0 if region.x <= 0.5 <= region.x + region.width else center_x - 0.5
    )
    vertical_error = (
        0.0 if region.y <= 0.5 <= region.y + region.height else center_y - 0.5
    )
    return quiesced.model_copy(
        update={
            "mouse_dx": _bounded_operator_aim_step(horizontal_error),
            "mouse_dy": _bounded_operator_aim_step(vertical_error),
        }
    )


def _bounded_operator_aim_step(error: float) -> int:
    if error == 0.0:
        return 0
    step = round(error * _OPERATOR_AIM_GAIN)
    if step == 0:
        step = 1 if error > 0.0 else -1
    return max(-_OPERATOR_AIM_MAX_STEP, min(_OPERATOR_AIM_MAX_STEP, step))


def _press_left(action: MotorAction) -> MotorAction:
    return action.model_copy(
        update={
            "buttons_down": tuple(dict.fromkeys((*action.buttons_down, "left"))),
            "buttons_up": tuple(button for button in action.buttons_up if button != "left"),
        }
    )


def _prelease_semantic_failure(
    blackboard: PerceptionBlackboard,
    intent: MotorIntent,
    *,
    now_ns: int,
    min_confidence: float,
) -> SkillFailureCode | None:
    """Reject fresh negative evidence while allowing absent async evidence to arrive."""
    mineable = blackboard.fact("target.mineable", min_confidence=min_confidence, now_ns=now_ns)
    visible = blackboard.fact("target.visible", min_confidence=min_confidence, now_ns=now_ns)
    kind = blackboard.fact("target.kind", min_confidence=min_confidence, now_ns=now_ns)
    mineable_is_bound = bool(
        mineable is not None
        and visible is not None
        and mineable.source == visible.source
        and (kind is None or kind.source == visible.source)
    )
    if (
        mineable_is_bound
        and mineable is not None
        and mineable.value is False
        and not mineable.source.startswith("bootstrap:")
    ):
        return SkillFailureCode.MINING_TARGET_UNVERIFIED
    if intent.mode.casefold() != "gather_wood":
        return None
    if (
        kind is None
        or visible is None
        or visible.value is not True
        or not isinstance(kind.value, str)
        or kind.source != visible.source
        or not is_rocket_source(kind.source)
    ):
        return None
    rule = _block_rule(_normalize_name(kind.value))
    if rule is None:
        return SkillFailureCode.MINING_TARGET_UNVERIFIED
    if rule.family != _BlockFamily.LOG:
        return SkillFailureCode.MINING_TARGET_MISMATCH
    return None


def _explicit_operator_mining_authorized(
    blackboard: PerceptionBlackboard,
    intent: MotorIntent,
    *,
    now_ns: int,
    min_confidence: float,
) -> bool:
    """Authorize only the exact operator-marked target for this mining intent."""
    return (
        _explicit_operator_mining_track(
            blackboard,
            intent,
            now_ns=now_ns,
            min_confidence=min_confidence,
        )
        is not None
    )


def _explicit_operator_mining_track(
    blackboard: PerceptionBlackboard,
    intent: MotorIntent,
    *,
    now_ns: int,
    min_confidence: float,
) -> Track | None:
    if (
        intent.skill_id != "mine_visible_block"
        or intent.mode.casefold() != "mine"
        or intent.target_label is None
    ):
        return None
    latest = blackboard.latest()
    if latest is None:
        return None

    track_id: str | None = None
    reference = blackboard.fact(
        "target.reference_available",
        min_confidence=min_confidence,
        now_ns=now_ns,
    )
    reference_prefix = "operator:cross-view-reference:"
    if (
        reference is not None
        and reference.value is True
        and reference.source.startswith(reference_prefix)
        and 0 <= now_ns - reference.observed_ns <= 500_000_000
    ):
        candidate = reference.source.removeprefix(reference_prefix)
        if candidate:
            track_id = candidate

    visible = blackboard.fact("target.visible", min_confidence=min_confidence, now_ns=now_ns)
    kind = blackboard.fact("target.kind", min_confidence=min_confidence, now_ns=now_ns)
    explicit_prefix = "operator:explicit-grounding:"
    if (
        track_id is None
        and visible is not None
        and kind is not None
        and visible.value is True
        and isinstance(kind.value, str)
        and visible.source == kind.source
        and visible.source.startswith(explicit_prefix)
        and 0 <= now_ns - visible.observed_ns <= 500_000_000
        and 0 <= now_ns - kind.observed_ns <= 500_000_000
    ):
        candidate = visible.source.removeprefix(explicit_prefix)
        if candidate:
            track_id = candidate
    if track_id is None:
        return None

    track = next((item for item in latest.tracks if item.track_id == track_id), None)
    if (
        track is None
        or track.attributes.get("source") != "operator"
        or track.confidence < min_confidence
        or (intent.target_track_id is not None and intent.target_track_id != track_id)
    ):
        return None
    kind_name = _normalize_name(track.label)
    rule = _block_rule(kind_name)
    if (
        rule is None
        or rule.family == _BlockFamily.UNBREAKABLE
        or not _requested_target_matches(intent.target_label, kind_name, rule)
    ):
        return None
    return track


def _operator_aim_track(
    blackboard: PerceptionBlackboard,
    intent: MotorIntent,
    *,
    pending: _PendingMiningAcquisition,
    now_ns: int,
    min_confidence: float,
    max_track_age_ms: int,
) -> Track | None:
    """Return one new ROCKET localization of the exact operator mining target."""
    track = _explicit_operator_mining_track(
        blackboard,
        intent,
        now_ns=now_ns,
        min_confidence=min_confidence,
    )
    if track is None or not _track_fresh(
        track,
        now_ns=now_ns,
        max_track_age_ms=max_track_age_ms,
    ):
        return None
    tracking_source = track.attributes.get("tracking_source")
    probability = track.attributes.get("target_exists_probability")
    if (
        not isinstance(tracking_source, str)
        or not is_rocket_source(tracking_source)
        or not isinstance(probability, (int, float))
        or isinstance(probability, bool)
        or float(probability) < min_confidence
        or track.last_seen_ns <= pending.last_aim_observation_ns
    ):
        return None

    # Never replay an observation captured before the most recent synthetic
    # nudge or body-settle boundary. The first observation may precede skill
    # admission only by one capture tick, matching the live request path.
    if pending.settling:
        evidence_after_ns = pending.settle_after_ns
    else:
        evidence_after_ns = (
            pending.started_ns - _OPERATOR_AIM_INITIAL_FRAME_GRACE_MS * 1_000_000
        )
    if track.last_seen_ns <= evidence_after_ns:
        return None
    return track


def _verified_target(
    blackboard: PerceptionBlackboard,
    intent: MotorIntent,
    *,
    now_ns: int,
    min_confidence: float,
    max_track_age_ms: int,
    require_scene_match: bool,
    evidence_after_ns: int,
) -> _VerifiedTarget | SkillFailureCode:
    visible = blackboard.fact("target.visible", min_confidence=min_confidence, now_ns=now_ns)
    mineable = blackboard.fact("target.mineable", min_confidence=min_confidence, now_ns=now_ns)
    kind_fact = blackboard.fact("target.kind", min_confidence=min_confidence, now_ns=now_ns)
    if visible is None or visible.value is not True or visible.source.startswith("bootstrap:"):
        return SkillFailureCode.MINING_TARGET_UNVERIFIED

    kind = (
        _normalize_name(kind_fact.value)
        if kind_fact is not None
        and isinstance(kind_fact.value, str)
        and not kind_fact.source.startswith("bootstrap:")
        else None
    )
    if kind is None:
        if intent.mode.casefold() != "gather_wood":
            return SkillFailureCode.MINING_TARGET_UNVERIFIED
        track = _fresh_rocket_log_track(
            blackboard,
            visible=visible,
            now_ns=now_ns,
            min_confidence=min_confidence,
            max_track_age_ms=max_track_age_ms,
        )
        if track is None:
            return SkillFailureCode.MINING_TARGET_UNVERIFIED
        kind = _normalize_name(track.label)
    else:
        initial_rule = _block_rule(kind)
        if initial_rule is None:
            return SkillFailureCode.MINING_TARGET_UNVERIFIED
        track = _crosshair_track(
            blackboard,
            kind=kind,
            rule=initial_rule,
            visible=visible,
            kind_fact=kind_fact,
            now_ns=now_ns,
            min_confidence=min_confidence,
            max_track_age_ms=max_track_age_ms,
        )
        if track is None:
            return SkillFailureCode.MINING_TARGET_UNVERIFIED

    rocket_current = _fresh_rocket_target(
        track,
        visible=visible,
        now_ns=now_ns,
        max_track_age_ms=max_track_age_ms,
    )
    operator_current = _fresh_operator_target(
        track,
        visible=visible,
        kind_fact=kind_fact,
        now_ns=now_ns,
    )
    current_grounding = rocket_current or operator_current
    mineable_is_bound = bool(
        mineable is not None
        and mineable.source == visible.source
        and (kind_fact is None or kind_fact.source == visible.source)
    )
    latest_frame = blackboard.raw_latest()
    if evidence_after_ns and (
        latest_frame is None
        or latest_frame.captured_ns <= evidence_after_ns
        or visible.observed_ns <= evidence_after_ns
        or (rocket_current and track.last_seen_ns <= evidence_after_ns)
        or (
            operator_current
            and (kind_fact is None or kind_fact.observed_ns <= evidence_after_ns)
        )
    ):
        return SkillFailureCode.MINING_TARGET_UNVERIFIED
    if current_grounding:
        # ROCKET localizes the current pixels. Prefer its bound track label over
        # a potentially older global VLM fact when establishing this lease. The
        # same rule applies to an exact operator region whose reference hash is
        # still matched by the runtime.
        kind = _normalize_name(track.label)
    rule = _block_rule(kind)
    if rule is None:
        return SkillFailureCode.MINING_TARGET_UNVERIFIED
    if rule.family == _BlockFamily.UNBREAKABLE:
        return SkillFailureCode.MINING_WRONG_TOOL
    if intent.mode.casefold() == "gather_wood" and rule.family != _BlockFamily.LOG:
        return SkillFailureCode.MINING_TARGET_MISMATCH
    if intent.target_track_id is not None and intent.target_track_id != track.track_id:
        return SkillFailureCode.MINING_TARGET_MISMATCH
    if intent.target_label is not None and not _requested_target_matches(
        intent.target_label,
        kind,
        rule,
    ):
        return SkillFailureCode.MINING_TARGET_MISMATCH
    if mineable_is_bound and mineable is not None and mineable.value is not True:
        return SkillFailureCode.MINING_TARGET_UNVERIFIED
    inferred_hand_safe = current_grounding and rule.family in _HAND_SAFE_FAMILIES
    if not inferred_hand_safe and (
        mineable is None
        or not mineable_is_bound
        or mineable.value is not True
        or mineable.source.startswith("bootstrap:")
    ):
        return SkillFailureCode.MINING_TARGET_UNVERIFIED
    if not current_grounding:
        if (
            kind_fact is None
            or mineable is None
            or visible.source != kind_fact.source
            or visible.source != mineable.source
        ):
            return SkillFailureCode.MINING_TARGET_UNVERIFIED
        if require_scene_match and not _scene_hash_matches(
            blackboard,
            now_ns=now_ns,
            expected_source=visible.source,
            evidence_after_ns=evidence_after_ns,
        ):
            return SkillFailureCode.MINING_TARGET_UNVERIFIED

    selected_item = _selected_item(
        blackboard,
        now_ns=now_ns,
        min_confidence=min_confidence,
        evidence_after_ns=evidence_after_ns,
    )
    if selected_item is None:
        if rule.family not in _HAND_SAFE_FAMILIES:
            return SkillFailureCode.MINING_TOOL_UNVERIFIED
        selected_item = _UNVERIFIED_ITEM
    tool_tier = _tool_tier(selected_item, suffix="pickaxe")
    if rule.minimum_pick_tier is not None and (
        tool_tier is None or tool_tier < rule.minimum_pick_tier
    ):
        return SkillFailureCode.MINING_WRONG_TOOL
    return _VerifiedTarget(
        track_id=track.track_id,
        kind=kind,
        rule=rule,
        selected_item=selected_item,
        lease_ms=_lease_duration_ms(rule, selected_item),
    )


def _continuation_target_failure(
    blackboard: PerceptionBlackboard,
    expected: _VerifiedTarget,
    *,
    now_ns: int,
    min_confidence: float,
    max_track_age_ms: int,
    lease_started_ns: int,
    operator_grounding_grace_ms: int,
) -> SkillFailureCode | None:
    latest = blackboard.latest()
    if latest is None:
        return SkillFailureCode.MINING_TARGET_UNVERIFIED
    track = next((item for item in latest.tracks if item.track_id == expected.track_id), None)
    visible = blackboard.fact("target.visible", min_confidence=min_confidence, now_ns=now_ns)
    kind_fact = blackboard.fact("target.kind", min_confidence=min_confidence, now_ns=now_ns)
    operator_current = bool(
        track is not None
        and visible is not None
        and _fresh_operator_target(
            track,
            visible=visible,
            kind_fact=kind_fact,
            now_ns=now_ns,
        )
    )
    operator_grace = bool(
        track is not None
        and track.attributes.get("source") == "operator"
        and now_ns - lease_started_ns
        <= operator_grounding_grace_ms * 1_000_000
    )
    if (
        track is None
        or track.confidence < min_confidence
        or not (
            _track_fresh(track, now_ns=now_ns, max_track_age_ms=max_track_age_ms)
            or operator_current
            or operator_grace
        )
        or not track_contains_crosshair(track)
    ):
        return SkillFailureCode.MINING_TARGET_CHANGED
    track_kind = _normalize_name(track.label)
    track_rule = _block_rule(track_kind)
    same_kind = track_kind == expected.kind or (
        expected.rule.family == _BlockFamily.LOG
        and track_rule is not None
        and track_rule.family == _BlockFamily.LOG
    )
    if not same_kind:
        return SkillFailureCode.MINING_TARGET_CHANGED

    if visible is not None and visible.value is not True:
        # Target disappearance alone is not strong enough evidence that a block
        # broke. End the attempt as a typed failure and let later inventory or
        # block-change evidence establish success.
        return SkillFailureCode.MINING_TARGET_CHANGED
    mineable = blackboard.fact("target.mineable", min_confidence=min_confidence, now_ns=now_ns)
    mineable_is_bound = bool(
        mineable is not None
        and visible is not None
        and mineable.source == visible.source
        and (kind_fact is None or kind_fact.source == visible.source)
    )
    if mineable_is_bound and mineable is not None and mineable.value is not True:
        return SkillFailureCode.MINING_TARGET_UNVERIFIED
    if (
        kind_fact is not None
        and isinstance(kind_fact.value, str)
        and not kind_fact.source.startswith("bootstrap:")
        and visible is not None
        and kind_fact.source == visible.source
    ):
        observed_kind = _normalize_name(kind_fact.value)
        observed_rule = _block_rule(observed_kind)
        same_observation = observed_kind == expected.kind or (
            expected.rule.family == _BlockFamily.LOG
            and observed_rule is not None
            and observed_rule.family == _BlockFamily.LOG
        )
        if not same_observation:
            return SkillFailureCode.MINING_TARGET_CHANGED

    selected_item = _selected_item(
        blackboard,
        now_ns=now_ns,
        min_confidence=min_confidence,
    )
    if selected_item is None and expected.rule.family not in _HAND_SAFE_FAMILIES:
        return SkillFailureCode.MINING_TOOL_UNVERIFIED
    if (
        expected.selected_item != _UNVERIFIED_ITEM
        and selected_item is not None
        and selected_item != expected.selected_item
    ):
        return SkillFailureCode.MINING_TOOL_CHANGED
    return None


def _selected_item(
    blackboard: PerceptionBlackboard,
    *,
    now_ns: int,
    min_confidence: float,
    evidence_after_ns: int = 0,
) -> str | None:
    slot_fact = blackboard.fact(
        "player.selected_slot",
        min_confidence=min_confidence,
        now_ns=now_ns,
    )
    if (
        slot_fact is None
        or not isinstance(slot_fact.value, int)
        or isinstance(slot_fact.value, bool)
        or slot_fact.source.startswith("bootstrap:")
        or slot_fact.observed_ns <= evidence_after_ns
    ):
        return None
    item_fact = blackboard.fact(
        f"hotbar.slot.{slot_fact.value}.item",
        min_confidence=min_confidence,
        now_ns=now_ns,
    )
    if (
        item_fact is None
        or not isinstance(item_fact.value, str)
        or item_fact.source.startswith("bootstrap:")
        or item_fact.source != slot_fact.source
        or item_fact.observed_ns <= evidence_after_ns
    ):
        return None
    return _normalize_name(item_fact.value)


def _crosshair_track(
    blackboard: PerceptionBlackboard,
    *,
    kind: str,
    rule: _BlockRule,
    visible: PerceptionFact,
    kind_fact: PerceptionFact | None,
    now_ns: int,
    min_confidence: float,
    max_track_age_ms: int,
) -> Track | None:
    latest = blackboard.latest()
    if latest is None:
        return None
    candidates: list[Track] = []
    for track in latest.tracks:
        track_rule = _block_rule(_normalize_name(track.label))
        same_target = _normalize_name(track.label) == kind or (
            rule.family == _BlockFamily.LOG
            and track_rule is not None
            and track_rule.family == _BlockFamily.LOG
        )
        if (
            same_target
            and track_contains_crosshair(track)
            and track.confidence >= min_confidence
            and (
                _track_fresh(
                    track,
                    now_ns=now_ns,
                    max_track_age_ms=max_track_age_ms,
                )
                or _fresh_operator_target(
                    track,
                    visible=visible,
                    kind_fact=kind_fact,
                    now_ns=now_ns,
                )
            )
        ):
            candidates.append(track)
    return max(candidates, key=lambda item: (item.last_seen_ns, item.confidence), default=None)


def _fresh_rocket_log_track(
    blackboard: PerceptionBlackboard,
    *,
    visible: PerceptionFact,
    now_ns: int,
    min_confidence: float,
    max_track_age_ms: int,
) -> Track | None:
    latest = blackboard.latest()
    if latest is None:
        return None
    candidates = (
        track
        for track in latest.tracks
        if track.confidence >= min_confidence
        and track_contains_crosshair(track)
        and _track_fresh(
            track,
            now_ns=now_ns,
            max_track_age_ms=max_track_age_ms,
        )
        and _is_log_label(track.label)
        and _fresh_rocket_target(
            track,
            visible=visible,
            now_ns=now_ns,
            max_track_age_ms=max_track_age_ms,
        )
    )
    return max(candidates, key=lambda item: (item.last_seen_ns, item.confidence), default=None)


def _fresh_rocket_target(
    track: Track,
    *,
    visible: PerceptionFact,
    now_ns: int,
    max_track_age_ms: int,
) -> bool:
    """Return whether ROCKET currently grounds this track in the live frame.

    The global VLM can describe a target long after the view has moved. ROCKET's
    auxiliary localization refreshes both the track and `target.visible`; those
    two agreeing, bounded-age signals are sufficient to omit the slower VLM's
    scene hash for a block whose exact ontology is safe to mine by hand.
    """
    tracking_source = track.attributes.get("tracking_source")
    return bool(
        visible.value is True
        and isinstance(tracking_source, str)
        and visible.source == tracking_source
        and is_rocket_source(visible.source)
        and 0 <= now_ns - visible.observed_ns <= max_track_age_ms * 1_000_000
        and _track_fresh(
            track,
            now_ns=now_ns,
            max_track_age_ms=max_track_age_ms,
        )
    )


def _fresh_operator_target(
    track: Track,
    *,
    visible: PerceptionFact,
    kind_fact: PerceptionFact | None,
    now_ns: int,
) -> bool:
    """Verify an operator region that still matches its captured frame."""
    source = f"operator:explicit-grounding:{track.track_id}"
    return bool(
        track.attributes.get("source") == "operator"
        and visible.value is True
        and visible.source == source
        and 0 <= now_ns - visible.observed_ns <= 500_000_000
        and kind_fact is not None
        and isinstance(kind_fact.value, str)
        and kind_fact.source == source
        and 0 <= now_ns - kind_fact.observed_ns <= 500_000_000
        and _normalize_name(kind_fact.value) == _normalize_name(track.label)
    )


def is_rocket_source(source: str) -> bool:
    """Return whether a fact came from the learned ROCKET localization head."""

    normalized = source.casefold()
    return normalized.startswith("learned:") and "rocket" in normalized


def track_contains_crosshair(track: Track) -> bool:
    """Return whether a normalized target box contains the world crosshair."""

    region = track.region
    return (
        region.x <= 0.5 <= region.x + region.width
        and region.y <= 0.5 <= region.y + region.height
    )


def normalize_block_kind(value: str) -> str:
    """Return the canonical block label used by mining's safety rules."""

    return _normalize_name(value)


def is_hand_safe_soft_block(value: str) -> bool:
    """Return whether mining already classifies this exact block as hand-safe soft terrain."""

    rule = _block_rule(_normalize_name(value))
    return rule is not None and rule.family == _BlockFamily.SOFT


def _track_fresh(track: Track, *, now_ns: int, max_track_age_ms: int) -> bool:
    age_ns = now_ns - track.last_seen_ns
    return 0 <= age_ns <= max_track_age_ms * 1_000_000


def _is_log_label(label: str) -> bool:
    rule = _block_rule(_normalize_name(label))
    return rule is not None and rule.family == _BlockFamily.LOG


def _unsafe_scene(blackboard: PerceptionBlackboard, *, now_ns: int) -> bool:
    danger = blackboard.fact("danger.immediate", min_confidence=0.65, now_ns=now_ns)
    overlay = blackboard.fact("scene.ui_overlay", min_confidence=0.65, now_ns=now_ns)
    playable = blackboard.fact("scene.playable", min_confidence=0.65, now_ns=now_ns)
    return bool(
        (danger is not None and danger.value is True)
        or (overlay is not None and overlay.value is True)
        or (playable is not None and playable.value is False)
    )


def _scene_hash_matches(
    blackboard: PerceptionBlackboard,
    *,
    now_ns: int,
    expected_source: str,
    evidence_after_ns: int,
) -> bool:
    observed = blackboard.fact("scene.observation_dhash", min_confidence=1.0, now_ns=now_ns)
    current = blackboard.fact("frame.dhash", min_confidence=1.0, now_ns=now_ns)
    if (
        observed is None
        or current is None
        or not isinstance(observed.value, str)
        or not isinstance(current.value, str)
        or observed.source != expected_source
        or observed.observed_ns <= evidence_after_ns
    ):
        return False
    try:
        return _hash_distance(observed.value, current.value) <= 6
    except ValueError:
        return False


def _visual_hash(
    blackboard: PerceptionBlackboard,
    *,
    now_ns: int,
) -> tuple[str, int] | None:
    fact = blackboard.fact("frame.crosshair_dhash", min_confidence=1.0, now_ns=now_ns)
    if fact is None or not isinstance(fact.value, str):
        return None
    try:
        _hash_distance(fact.value, fact.value)
    except ValueError:
        return None
    return fact.value, fact.observed_ns


def _block_rule(kind: str) -> _BlockRule | None:
    if kind in _UNBREAKABLE:
        return _BlockRule(_BlockFamily.UNBREAKABLE)
    if _describes_log(kind):
        return _BlockRule(_BlockFamily.LOG)
    if kind.endswith(("_wood", "_hyphae", "_planks")) or kind in {
        "bookshelf",
        "crafting_table",
    }:
        return _BlockRule(_BlockFamily.WOOD)
    if kind in _SOFT_BLOCKS:
        return _BlockRule(_BlockFamily.SOFT)
    if kind in {"obsidian", "crying_obsidian"}:
        return _BlockRule(_BlockFamily.OBSIDIAN, minimum_pick_tier=4)
    if kind in {"ancient_debris", "netherite_block"}:
        return _BlockRule(_BlockFamily.ANCIENT_DEBRIS, minimum_pick_tier=4)

    minimum_tier = 1
    if any(token in kind for token in ("iron_ore", "copper_ore", "lapis_ore")):
        minimum_tier = 2
    elif any(token in kind for token in ("gold_ore", "diamond_ore", "emerald_ore", "redstone_ore")):
        minimum_tier = 3
    elif kind.endswith("_ore") and "coal_ore" not in kind and "quartz_ore" not in kind:
        return None
    is_pickaxe_block = (
        kind in _STONE_BLOCKS
        or kind == "nether_quartz_ore"
        or "coal_ore" in kind
        or any(
            token in kind
            for token in (
                "iron_ore",
                "copper_ore",
                "lapis_ore",
                "gold_ore",
                "diamond_ore",
                "emerald_ore",
                "redstone_ore",
            )
        )
    )
    if not is_pickaxe_block:
        return None
    family = _BlockFamily.DEEPSLATE if "deepslate" in kind else _BlockFamily.PICKAXE
    return _BlockRule(family, minimum_pick_tier=minimum_tier)


def _lease_duration_ms(rule: _BlockRule, selected_item: str) -> int:
    if rule.family in {_BlockFamily.LOG, _BlockFamily.WOOD}:
        material = _tool_material(selected_item, suffix="axe")
        if material is None:
            return 3_600
        return {
            "wooden": 1_800,
            "gold": 550,
            "golden": 550,
            "stone": 1_100,
            "copper": 1_000,
            "iron": 800,
            "diamond": 700,
            "netherite": 650,
        }.get(material, 3_600)
    if rule.family == _BlockFamily.SOFT:
        return 2_500
    if rule.family == _BlockFamily.OBSIDIAN:
        return 9_200 if _tool_tier(selected_item, suffix="pickaxe") == 5 else 10_200
    if rule.family == _BlockFamily.ANCIENT_DEBRIS:
        return 6_500
    material = _tool_material(selected_item, suffix="pickaxe") or "wooden"
    duration = {
        "wooden": 1_500,
        "gold": 550,
        "golden": 550,
        "stone": 900,
        "copper": 850,
        "iron": 700,
        "diamond": 650,
        "netherite": 600,
    }.get(material, 1_500)
    return int(duration * 1.8) if rule.family == _BlockFamily.DEEPSLATE else duration


def _tool_tier(item: str, *, suffix: str) -> int | None:
    material = _tool_material(item, suffix=suffix)
    if material is None:
        return None
    return _TOOL_TIERS.get(material)


def _tool_material(item: str, *, suffix: str) -> str | None:
    if item in _EMPTY_ITEMS or not item.endswith(f"_{suffix}"):
        return None
    return item[: -len(suffix) - 1]


def _normalize_name(value: str) -> str:
    normalized = value.strip().casefold().removeprefix("minecraft:")
    return re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")


def _describes_log(kind: str) -> bool:
    tokens = frozenset(kind.split("_"))
    return bool(
        tokens.intersection({"log", "stem", "trunk"})
        and ("tree" in tokens or tokens.intersection(_WOOD_SPECIES))
        and tokens.issubset(_WOOD_SPECIES | _LOG_DESCRIPTOR_WORDS)
    )


def _requested_target_matches(
    requested_label: str,
    observed_kind: str,
    observed_rule: _BlockRule,
) -> bool:
    requested = _normalize_name(requested_label)
    if requested == observed_kind:
        return True
    if requested in {"log", "tree", "tree_trunk", "trunk"}:
        return observed_rule.family == _BlockFamily.LOG
    requested_rule = _block_rule(requested)
    return bool(
        requested_rule is not None
        and requested_rule.family == _BlockFamily.LOG
        and observed_rule.family == _BlockFamily.LOG
    )


def _hash_distance(first: str, second: str) -> int:
    if len(first) != 16 or len(second) != 16:
        raise ValueError("perceptual hashes must be 64-bit hexadecimal strings")
    try:
        return (int(first, 16) ^ int(second, 16)).bit_count()
    except ValueError as exc:
        raise ValueError("perceptual hashes must be hexadecimal") from exc
