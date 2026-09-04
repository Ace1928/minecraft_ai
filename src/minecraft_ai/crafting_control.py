from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .outcome_verifier import (
    OutcomeKind,
    OutcomeSignal,
    OutcomeStatus,
    OutcomeVerification,
)
from .perception import EvidenceRegion, PerceptionBlackboard, PerceptionFact, Track
from .safety import MotorAction


INVENTORY_TOGGLE_DURATION_MS = 150


class PlankCraftPhase(StrEnum):
    OPEN_INVENTORY = "open_inventory"
    LOCATE_RECIPE = "locate_recipe"
    VERIFY_OUTPUT = "verify_output"
    CLOSE_INVENTORY = "close_inventory"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True)
class PlankCraftStep:
    action: MotorAction | None = None
    mode: str = "craft_planks"
    instruction: str = "craft one set of wood planks"
    verification: OutcomeVerification | None = None
    failure_reason: str | None = None


@dataclass
class BoundedPlankCraftController:
    """Small visual Bedrock inventory controller for one log-to-planks craft.

    Bedrock's recipe book supports right-clicking a visible recipe to craft one
    set directly into inventory.  The controller uses only a current GUI track
    for that click and accepts completion only after a coherent, pixel-grounded
    inventory observation reports both a consumed log and at least four planks.
    """

    open_timeout_ms: int = 10_000
    locate_timeout_ms: int = 80_000
    outcome_timeout_ms: int = 80_000
    close_timeout_ms: int = 10_000
    retry_interval_ms: int = 2_000
    max_toggle_attempts: int = 1
    max_recipe_attempts: int = 1
    minimum_confidence: float = 0.70
    _run_id: str | None = field(default=None, init=False)
    _phase: PlankCraftPhase | None = field(default=None, init=False)
    _phase_started_ns: int = field(default=0, init=False)
    _baseline_logs: int | None = field(default=None, init=False)
    _inventory_observed_ns: int = field(default=0, init=False)
    _inventory_toggle_ns: int = field(default=0, init=False)
    _interaction_ns: int = field(default=0, init=False)
    _last_attempt_ns: int = field(default=0, init=False)
    _toggle_attempts: int = field(default=0, init=False)
    _recipe_attempts: int = field(default=0, init=False)
    _last_sequence: int = field(default=-1, init=False)
    _verified_counts: tuple[PerceptionFact, PerceptionFact] | None = field(
        default=None,
        init=False,
    )

    @property
    def phase(self) -> PlankCraftPhase | None:
        return self._phase

    @property
    def last_sequence(self) -> int:
        return self._last_sequence

    def semantic_request_ready(
        self,
        blackboard: PerceptionBlackboard,
        *,
        now_ns: int,
    ) -> bool:
        """Admit slow GUI perception only after this transaction owns a GUI."""
        if self._phase == PlankCraftPhase.VERIFY_OUTPUT:
            return True
        if self._phase == PlankCraftPhase.LOCATE_RECIPE:
            observed = _inventory_gui_observation(
                blackboard,
                now_ns=now_ns,
                min_confidence=self.minimum_confidence,
                evidence_captured_after_ns=self._inventory_toggle_ns,
            )
            if observed is None:
                return True
            observed_ns = max(item.observed_ns for item in observed)
            if observed_ns < self._phase_started_ns:
                return True
            baseline = _inventory_count(
                blackboard,
                "inventory.logs",
                now_ns=now_ns,
                min_confidence=self.minimum_confidence,
                observed_at_or_after_ns=observed_ns,
                evidence_region=EvidenceRegion.GUI,
                evidence_captured_after_ns=self._inventory_toggle_ns,
            )
            if baseline is not None and int(baseline.value) < 1:
                # The executor will fail closed on this already-grounded result;
                # another pre-failure VLM request cannot add useful evidence.
                return False
            track = _fresh_planks_recipe_track(
                blackboard,
                observed_after_ns=observed_ns,
                evidence_captured_after_ns=self._inventory_toggle_ns,
                min_confidence=self.minimum_confidence,
            )
            if baseline is not None and track is not None and self._attempt_ready(now_ns):
                # Runtime asks for semantics before the controller ticks.  Do
                # not occupy the sole VLM worker with another pre-click frame
                # when this frame already makes the recipe click-ready; the
                # following VERIFY_OUTPUT tick must get the first post-click
                # capture instead.
                return False
            return True
        return bool(
            self._phase == PlankCraftPhase.OPEN_INVENTORY
            and self._last_attempt_ns > 0
            and _fresh_ui_overlay_observation(
                blackboard,
                now_ns=now_ns,
                observed_after_ns=self._inventory_toggle_ns,
                min_confidence=self.minimum_confidence,
            )
            is not None
        )

    def reset(self) -> None:
        self._run_id = None
        self._phase = None
        self._phase_started_ns = 0
        self._baseline_logs = None
        self._inventory_observed_ns = 0
        self._inventory_toggle_ns = 0
        self._interaction_ns = 0
        self._last_attempt_ns = 0
        self._toggle_attempts = 0
        self._recipe_attempts = 0
        self._last_sequence = -1
        self._verified_counts = None

    def step(
        self,
        blackboard: PerceptionBlackboard,
        *,
        run_id: str,
        sequence: int,
        now_ns: int,
    ) -> PlankCraftStep:
        if self._run_id != run_id:
            self.reset()
            self._run_id = run_id
            self._phase = PlankCraftPhase.OPEN_INVENTORY
            self._phase_started_ns = now_ns

        assert self._phase is not None
        if self._phase == PlankCraftPhase.OPEN_INVENTORY:
            return self._open_inventory(blackboard, sequence=sequence, now_ns=now_ns)
        if self._phase == PlankCraftPhase.LOCATE_RECIPE:
            return self._locate_recipe(blackboard, sequence=sequence, now_ns=now_ns)
        if self._phase == PlankCraftPhase.VERIFY_OUTPUT:
            return self._verify_output(blackboard, sequence=sequence, now_ns=now_ns)
        if self._phase == PlankCraftPhase.CLOSE_INVENTORY:
            return self._close_inventory(blackboard, sequence=sequence, now_ns=now_ns)
        if self._phase == PlankCraftPhase.FAILED:
            return PlankCraftStep(failure_reason="crafting-controller-already-failed")
        return PlankCraftStep()

    def _open_inventory(
        self,
        blackboard: PerceptionBlackboard,
        *,
        sequence: int,
        now_ns: int,
    ) -> PlankCraftStep:
        overlay = _fresh_ui_overlay_observation(
            blackboard,
            now_ns=now_ns,
            observed_after_ns=self._inventory_toggle_ns,
            min_confidence=self.minimum_confidence,
        )
        if self._last_attempt_ns > 0 and overlay is not None:
            self._phase = PlankCraftPhase.LOCATE_RECIPE
            self._phase_started_ns = now_ns
            return PlankCraftStep(
                mode="craft_planks",
                instruction="wait for grounded inventory GUI perception",
            )
        if self._elapsed_ms(now_ns) >= self.open_timeout_ms:
            return self._fail("crafting-inventory-open-timeout")
        if self._attempt_ready(now_ns) and self._toggle_attempts < self.max_toggle_attempts:
            self._toggle_attempts += 1
            self._last_attempt_ns = now_ns
            self._inventory_toggle_ns = now_ns
            return PlankCraftStep(
                action=self._tap_key(sequence, "e"),
                mode="open_inventory",
                instruction="open inventory",
            )
        return PlankCraftStep(mode="open_inventory", instruction="open inventory")

    def _locate_recipe(
        self,
        blackboard: PerceptionBlackboard,
        *,
        sequence: int,
        now_ns: int,
    ) -> PlankCraftStep:
        observed = _inventory_gui_observation(
            blackboard,
            now_ns=now_ns,
            min_confidence=self.minimum_confidence,
            evidence_captured_after_ns=self._inventory_toggle_ns,
        )
        if observed is None:
            if self._elapsed_ms(now_ns) >= self.locate_timeout_ms:
                return self._fail("crafting-inventory-gui-not-verified")
            return PlankCraftStep(
                mode="craft_planks",
                instruction="wait for grounded inventory GUI perception",
            )
        observed_ns = max(item.observed_ns for item in observed)
        if observed_ns < self._phase_started_ns:
            if self._elapsed_ms(now_ns) >= self.locate_timeout_ms:
                return self._fail("crafting-inventory-gui-not-verified")
            return PlankCraftStep(
                mode="craft_planks",
                instruction="wait for fresh grounded inventory GUI perception",
            )
        self._inventory_observed_ns = observed_ns
        if self._baseline_logs is None:
            baseline = _inventory_count(
                blackboard,
                "inventory.logs",
                now_ns=now_ns,
                min_confidence=self.minimum_confidence,
                observed_at_or_after_ns=self._inventory_observed_ns,
                evidence_region=EvidenceRegion.GUI,
                evidence_captured_after_ns=self._inventory_toggle_ns,
            )
            if baseline is not None:
                if int(baseline.value) < 1:
                    return self._fail("crafting-no-logs-observed-in-inventory")
                self._baseline_logs = int(baseline.value)
        track = _fresh_planks_recipe_track(
            blackboard,
            observed_after_ns=self._inventory_observed_ns,
            evidence_captured_after_ns=self._inventory_toggle_ns,
            min_confidence=self.minimum_confidence,
        )
        if (
            self._baseline_logs is not None
            and track is not None
            and self._attempt_ready(now_ns)
        ):
            self._recipe_attempts += 1
            self._last_attempt_ns = now_ns
            self._interaction_ns = now_ns
            self._phase = PlankCraftPhase.VERIFY_OUTPUT
            self._phase_started_ns = now_ns
            center_x = track.region.x + track.region.width / 2.0
            center_y = track.region.y + track.region.height / 2.0
            return PlankCraftStep(
                action=self._click(sequence, center_x, center_y, button="right"),
                mode="craft_planks",
                instruction="right-click the grounded craftable wood planks recipe once",
            )
        if self._elapsed_ms(now_ns) >= self.locate_timeout_ms:
            reason = (
                "crafting-baseline-log-count-unavailable"
                if self._baseline_logs is None
                else "crafting-planks-recipe-not-grounded"
            )
            return self._fail(reason)
        return PlankCraftStep(
            mode="craft_planks",
            instruction=(
                "read the visible log count and locate the craftable wood planks recipe"
            ),
        )

    def _verify_output(
        self,
        blackboard: PerceptionBlackboard,
        *,
        sequence: int,
        now_ns: int,
    ) -> PlankCraftStep:
        counts = self._verified_inventory_delta(blackboard, now_ns=now_ns)
        if counts is not None:
            self._verified_counts = counts
            self._phase = PlankCraftPhase.CLOSE_INVENTORY
            self._phase_started_ns = now_ns
            self._last_attempt_ns = 0
            self._toggle_attempts = 0
            return self._close_inventory(blackboard, sequence=sequence, now_ns=now_ns)
        if self._elapsed_ms(now_ns) < self.outcome_timeout_ms:
            return PlankCraftStep(
                mode="craft_planks",
                instruction="wait for a fresh inventory count after the recipe click",
            )
        if self._recipe_attempts < self.max_recipe_attempts and _inventory_gui_is_current(
            blackboard,
            now_ns=now_ns,
            min_confidence=self.minimum_confidence,
        ):
            self._phase = PlankCraftPhase.LOCATE_RECIPE
            self._phase_started_ns = now_ns
            self._inventory_observed_ns = now_ns
            self._last_attempt_ns = 0
            return PlankCraftStep(
                mode="craft_planks",
                instruction="re-ground the craftable wood planks recipe for one final attempt",
            )
        return self._fail("crafting-output-delta-not-verified")

    def _close_inventory(
        self,
        blackboard: PerceptionBlackboard,
        *,
        sequence: int,
        now_ns: int,
    ) -> PlankCraftStep:
        world = _playable_world_observation(
            blackboard,
            now_ns=now_ns,
            observed_after_ns=self._phase_started_ns,
            min_confidence=self.minimum_confidence,
        )
        if world is not None:
            assert self._run_id is not None
            assert self._baseline_logs is not None
            assert self._verified_counts is not None
            logs, planks = self._verified_counts
            self._phase = PlankCraftPhase.COMPLETE
            confidence = min(
                logs.confidence,
                planks.confidence,
                *(item.confidence for item in world),
            )
            return PlankCraftStep(
                verification=OutcomeVerification(
                    run_id=self._run_id,
                    kind=OutcomeKind.CRAFTING,
                    status=OutcomeStatus.SUCCEEDED,
                    signal=OutcomeSignal.PLANKS_CRAFTED,
                    observed_ns=now_ns,
                    confidence=confidence,
                    reason=(
                        "fresh grounded inventory delta consumed a log "
                        f"({self._baseline_logs}->{int(logs.value)}) and observed "
                        f"{int(planks.value)} planks before returning to the world"
                    ),
                    evidence_keys=(
                        "inventory.logs",
                        "inventory.planks",
                        "scene.mode",
                        "scene.playable",
                    ),
                ),
                mode="close_inventory",
                instruction="close inventory",
            )
        if self._elapsed_ms(now_ns) >= self.close_timeout_ms:
            return self._fail("crafting-inventory-close-timeout")
        if self._attempt_ready(now_ns) and self._toggle_attempts < self.max_toggle_attempts:
            self._toggle_attempts += 1
            self._last_attempt_ns = now_ns
            return PlankCraftStep(
                action=self._tap_key(sequence, "e"),
                mode="close_inventory",
                instruction="close inventory",
            )
        return PlankCraftStep(mode="close_inventory", instruction="close inventory")

    def _verified_inventory_delta(
        self,
        blackboard: PerceptionBlackboard,
        *,
        now_ns: int,
    ) -> tuple[PerceptionFact, PerceptionFact] | None:
        assert self._baseline_logs is not None
        logs = _inventory_count(
            blackboard,
            "inventory.logs",
            now_ns=now_ns,
            min_confidence=self.minimum_confidence,
            observed_at_or_after_ns=self._interaction_ns + 1,
            evidence_region=EvidenceRegion.GUI,
            evidence_captured_after_ns=self._interaction_ns,
        )
        planks = _inventory_count(
            blackboard,
            "inventory.planks",
            now_ns=now_ns,
            min_confidence=self.minimum_confidence,
            observed_at_or_after_ns=self._interaction_ns + 1,
            evidence_region=EvidenceRegion.GUI,
            evidence_captured_after_ns=self._interaction_ns,
        )
        if logs is None or planks is None:
            return None
        if logs.observed_ns != planks.observed_ns:
            return None
        if int(logs.value) > self._baseline_logs - 1 or int(planks.value) < 4:
            return None
        return logs, planks

    def _attempt_ready(self, now_ns: int) -> bool:
        return self._last_attempt_ns == 0 or (
            now_ns - self._last_attempt_ns >= self.retry_interval_ms * 1_000_000
        )

    def _elapsed_ms(self, now_ns: int) -> int:
        return max(0, (now_ns - self._phase_started_ns) // 1_000_000)

    def _tap_key(self, sequence: int, key: str) -> MotorAction:
        self._last_sequence = sequence
        return MotorAction(
            sequence=sequence,
            keys_down=(key,),
            keys_up=(key,),
            # Bedrock/Proton samples gameplay keys less reliably than its
            # render cadence suggests. Retained live trajectories prove that
            # 112 ms and 321 ms inventory holds register while 50 ms can be
            # missed. Keep this one atomic, bounded transaction so a stalled
            # next frame can never leave E held.
            duration_ms=INVENTORY_TOGGLE_DURATION_MS,
        )

    def _click(
        self,
        sequence: int,
        cursor_x: float,
        cursor_y: float,
        *,
        button: str,
    ) -> MotorAction:
        self._last_sequence = sequence
        return MotorAction(
            sequence=sequence,
            buttons_down=(button,),
            buttons_up=(button,),
            cursor_x=cursor_x,
            cursor_y=cursor_y,
            camera_semantics="cursor",
            duration_ms=50,
        )

    def _fail(self, reason: str) -> PlankCraftStep:
        self._phase = PlankCraftPhase.FAILED
        return PlankCraftStep(failure_reason=reason)


def _inventory_count(
    blackboard: PerceptionBlackboard,
    key: str,
    *,
    now_ns: int,
    min_confidence: float,
    observed_at_or_after_ns: int = 0,
    evidence_region: EvidenceRegion | None = None,
    evidence_captured_after_ns: int = 0,
) -> PerceptionFact | None:
    fact = blackboard.fact(key, min_confidence=min_confidence, now_ns=now_ns)
    if fact is None or isinstance(fact.value, bool) or not isinstance(fact.value, int):
        return None
    if fact.observed_ns < observed_at_or_after_ns:
        return None
    if evidence_region is not None:
        latest = blackboard.latest()
        if latest is None:
            return None
        supported = {
            item.evidence_id
            for item in latest.evidence
            if item.region_kind == evidence_region
            and item.captured_ns > evidence_captured_after_ns
        }
        if not supported.intersection(fact.evidence_refs):
            return None
    return fact


def _inventory_gui_observation(
    blackboard: PerceptionBlackboard,
    *,
    now_ns: int,
    min_confidence: float,
    evidence_captured_after_ns: int = 0,
) -> tuple[PerceptionFact, PerceptionFact] | None:
    scene = blackboard.fact("scene.mode", min_confidence=min_confidence, now_ns=now_ns)
    gui = blackboard.fact("gui.mode", min_confidence=min_confidence, now_ns=now_ns)
    if scene is None or gui is None or scene.value != "gui" or gui.value != "inventory":
        return None
    if evidence_captured_after_ns > 0:
        latest = blackboard.latest()
        if latest is None:
            return None
        fresh_evidence = {
            item.evidence_id
            for item in latest.evidence
            if item.captured_ns > evidence_captured_after_ns
        }
        if not fresh_evidence.intersection(scene.evidence_refs):
            return None
        if not fresh_evidence.intersection(gui.evidence_refs):
            return None
    return scene, gui


def _inventory_gui_is_current(
    blackboard: PerceptionBlackboard,
    *,
    now_ns: int,
    min_confidence: float,
) -> bool:
    return (
        _inventory_gui_observation(
            blackboard,
            now_ns=now_ns,
            min_confidence=min_confidence,
        )
        is not None
    )


def _fresh_ui_overlay_observation(
    blackboard: PerceptionBlackboard,
    *,
    now_ns: int,
    observed_after_ns: int,
    min_confidence: float,
) -> tuple[PerceptionFact, PerceptionFact] | None:
    overlay = blackboard.fact(
        "scene.ui_overlay",
        min_confidence=min_confidence,
        now_ns=now_ns,
    )
    playable = blackboard.fact(
        "scene.playable",
        min_confidence=min_confidence,
        now_ns=now_ns,
    )
    if overlay is None or playable is None:
        return None
    if overlay.observed_ns <= observed_after_ns or playable.observed_ns <= observed_after_ns:
        return None
    if overlay.value is not True or playable.value is not False:
        return None
    if not overlay.source.startswith(("bootstrap:", "safety:")):
        return None
    if not playable.source.startswith(("bootstrap:", "safety:")):
        return None
    return overlay, playable


def _playable_world_observation(
    blackboard: PerceptionBlackboard,
    *,
    now_ns: int,
    observed_after_ns: int,
    min_confidence: float,
) -> tuple[PerceptionFact, PerceptionFact] | None:
    scene = blackboard.fact("scene.mode", min_confidence=min_confidence, now_ns=now_ns)
    playable = blackboard.fact("scene.playable", min_confidence=min_confidence, now_ns=now_ns)
    if scene is None or playable is None:
        return None
    if scene.observed_ns < observed_after_ns or playable.observed_ns < observed_after_ns:
        return None
    if scene.value != "world" or playable.value is not True:
        return None
    return scene, playable


def _fresh_planks_recipe_track(
    blackboard: PerceptionBlackboard,
    *,
    observed_after_ns: int,
    evidence_captured_after_ns: int,
    min_confidence: float,
) -> Track | None:
    latest = blackboard.latest()
    if latest is None:
        return None
    gui_evidence = {
        item.evidence_id
        for item in latest.evidence
        if item.region_kind == EvidenceRegion.GUI
        and item.captured_ns > evidence_captured_after_ns
    }
    candidates: list[Track] = []
    for track in latest.tracks:
        label = "_".join(track.label.casefold().replace("-", " ").split())
        if label != "craftable_planks_recipe":
            continue
        if track.confidence < min_confidence or track.last_seen_ns < observed_after_ns:
            continue
        if not gui_evidence.intersection(track.evidence_refs):
            continue
        candidates.append(track)
    return max(candidates, key=lambda item: (item.confidence, item.last_seen_ns), default=None)
