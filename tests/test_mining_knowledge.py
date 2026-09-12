from dataclasses import replace
import time

import pytest

from minecraft_ai.control.mining import MiningLeaseGuard, _verified_target
from minecraft_ai.control.mining_knowledge import (
    MiningKey,
    MiningKnowledge,
    MiningRule,
    MiningRuleSnapshot,
    MiningTrial,
)
from minecraft_ai.execution import SkillExecutor
from minecraft_ai.memory import MemoryStore
from minecraft_ai.motor import MotorIntent
from minecraft_ai.outcome_verifier import (
    OutcomeKind,
    OutcomeSignal,
    OutcomeStatus,
    OutcomeVerification,
)
from minecraft_ai.runtime import _HeadroomRecovery
from minecraft_ai.runtime_support.helpers import _headroom_retry_advances_plan
from minecraft_ai.safety import MotorAction
from minecraft_ai.skills import SkillActionPermissions, SkillFailureCode, SkillOutcome, SkillSpec
from minecraft_ai.storage import StateDatabase
from test_headroom_recovery import _mining_success, _retry_progress, _runtime_for_probe
from test_mining_control import _ScriptedPolicy, _mining_board


def _book(**kwargs):
    return MiningKnowledge(MemoryStore(), "pack-v1-survival", **kwargs)


def _trial(book, name="one", **kwargs):
    return MiningTrial(
        attempt_id=name,
        key=book.key("pack:rock", "pack:drill"),
        elapsed_ms=4000.0,
        evidence="fixture:resolved-outcome",
        **kwargs,
    )


@pytest.mark.parametrize("name", ["", " ", "minecraft:", " MINECRAFT: "])
def test_empty_canonical_identifier_refused(name):
    for cls in (MiningKey, MiningRule):
        args = dict(block=name, tool="hand")
        if cls is MiningKey:
            args["ruleset"] = "v1"
        with pytest.raises(ValueError):
            cls(**args)


def test_namespace_identity_and_immutable_rule_content_scope(tmp_path):
    a = MiningRuleSnapshot(
        ruleset_id="pack",
        provenance="local resolved game adapter",
        rules=(
            MiningRule(block="pack:rock", tool="pack:drill", can_break=True, expected_ms=7000.0),
        ),
    )
    p = tmp_path / "rules.json"
    p.write_text(a.model_dump_json())
    assert MiningRuleSnapshot.load(p) == a
    b = a.model_copy(update={"rules": (a.rules[0].model_copy(update={"can_break": False}),)})
    assert a.scope != b.scope
    book = _book(snapshot=a)
    assert book.rule("other:rock", "pack:drill") is None
    assert book.rule("pack:rock", "hand") is None
    assert book.budget_ms("pack:rock", "pack:drill", 1000, cap_ms=30000) == 8750
    assert book.key("minecraft:stone", "hand") == book.key("stone", "hand")


@pytest.mark.parametrize(
    "data",
    [
        {"can_break": "yes"},
        {"expected_ms": float("nan")},
        {"expected_ms": -1.0},
        {"can_harvest": 1},
        {"unknown": True},
    ],
)
def test_rule_contract_rejects_unresolved_or_invalid_values(data):
    with pytest.raises(ValueError):
        MiningRule(block="pack:rock", tool="pack:drill", **data)


def test_duplicate_rule_names_after_canonicalization_refused():
    with pytest.raises(ValueError):
        MiningRuleSnapshot(
            ruleset_id="v1",
            provenance="adapter",
            rules=(
                MiningRule(block="minecraft:stone", tool="hand"),
                MiningRule(block="stone", tool="hand"),
            ),
        )


def test_missing_pickup_never_teaches_wrong_tool_or_timeout_duration():
    book = _book()
    book.record(_trial(book, broke=True, picked_up=False))
    for i in range(20):
        book.record(_trial(book, str(i), broke=None))
    belief = book.belief("pack:rock", "pack:drill")
    assert (belief.breaks, belief.censored, belief.nonharvests, belief.uncollected) == (1, 20, 0, 1)
    assert book.budget_ms("pack:rock", "pack:drill", 5000, cap_ms=30000) == 5000
    assert book.rule("pack:rock", "pack:drill") is None
    belief.breaks = 99
    assert book.belief("pack:rock", "pack:drill").breaks == 1


def test_learned_durations_only_from_completed_same_pair_and_bounded():
    book = _book()
    for i in range(3):
        trial = _trial(book, str(i), broke=True).model_copy(update={"elapsed_ms": 12000.0})
        book.record(trial)
    assert book.budget_ms("pack:rock", "pack:drill", 5000, cap_ms=30000) == 15000
    assert book.budget_ms("pack:rock", "hand", 5000, cap_ms=30000) == 5000
    assert book.budget_ms("pack:rock", "pack:drill", 5000, cap_ms=10000) == 10000


@pytest.mark.parametrize("fallback,cap", [(True, 100), (1, 0), (float("nan"), 100), (1, 1.2)])
def test_invalid_budget_refused(fallback, cap):
    with pytest.raises(ValueError):
        _book().budget_ms("rock", "hand", fallback, cap_ms=cap)


def test_collection_has_exact_parent_and_no_double_break_count(tmp_path):
    book = _book()
    parent = _trial(book, broke=True)
    child = _trial(
        book, "pickup", parent_attempt_id=parent.attempt_id, harvested=True, picked_up=True
    )
    with pytest.raises(ValueError):
        book.record(child)
    book.record(parent)
    book.record(child)
    assert book.record(parent) is None and book.record(child) is None
    with pytest.raises(ValueError):
        book.record(parent.model_copy(update={"elapsed_ms": 10.0}))
    with StateDatabase(tmp_path / "state.sqlite3") as db:
        for memory in book.memories.records.values():
            db.save_memory(memory)
        reloaded = MiningKnowledge(db.load_memories(), book.scope)
        belief = reloaded.belief("pack:rock", "pack:drill")
        assert (belief.breaks, belief.harvests, belief.pickups, belief.censored) == (1, 1, 1, 0)
        assert reloaded.record(child) is None
        other = MiningKnowledge(db.load_memories(), "pack-v2")
        assert other.belief("pack:rock", "pack:drill").breaks == 0
    book.memories.remove(parent.memory_id)
    orphan = MiningKnowledge(book.memories, book.scope)
    assert orphan.belief("pack:rock", "pack:drill").harvests == 0


@pytest.mark.parametrize(
    "fields",
    [
        {"picked_up": True},
        {"harvested": True},
        {"broke": False, "harvested": True},
        {"parent_attempt_id": "parent", "broke": True},
    ],
)
def test_inconsistent_outcomes_refused(fields):
    with pytest.raises(ValueError):
        _trial(_book(), **fields)


def _target(book, *, block="pack:rock", tool="pack:drill", parameters=None, **board_kwargs):
    now = time.monotonic_ns()
    return _verified_target(
        _mining_board(now_ns=now, kind=block, item=tool, **board_kwargs),
        MotorIntent(skill_id="mine", mode="mine", parameters=parameters or {}),
        now_ns=now,
        min_confidence=0.7,
        max_track_age_ms=15000,
        require_scene_match=True,
        evidence_after_ns=0,
        knowledge=book,
    )


def test_resolved_mod_rule_and_empirical_breaks_reach_actual_guard():
    book = _book()
    assert _target(book) == SkillFailureCode.MINING_TARGET_UNVERIFIED
    for i in range(3):
        book.record(_trial(book, str(i), broke=True))
    assert _target(book).kind == "pack:rock"
    assert _target(book, tool=None) == SkillFailureCode.MINING_TARGET_UNVERIFIED
    assert _target(book, scene_hash="ffffffffffffffff") == SkillFailureCode.MINING_TARGET_UNVERIFIED
    snap = MiningRuleSnapshot(
        ruleset_id="pack",
        provenance="adapter",
        rules=(
            MiningRule(block="pack:rock", tool="pack:drill", can_break=True, can_harvest=False),
        ),
    )
    book = _book(snapshot=snap)
    assert _target(book) == SkillFailureCode.MINING_WRONG_TOOL
    assert _target(book, parameters={"harvest_required": False}).kind == "pack:rock"
    denied = snap.model_copy(
        update={"rules": (snap.rules[0].model_copy(update={"can_break": False}),)}
    )
    assert (
        _target(_book(snapshot=denied), parameters={"harvest_required": False})
        == SkillFailureCode.MINING_WRONG_TOOL
    )


def test_unknown_probe_does_not_bypass_safety_or_equipped_tool_evidence():
    assert _target(_book(), parameters={"allow_unknown_block_probe": True}).kind == "pack:rock"
    assert (
        _target(_book(), tool=None, parameters={"allow_unknown_block_probe": True})
        == SkillFailureCode.MINING_TARGET_UNVERIFIED
    )
    assert (
        _target(_book(), parameters={"allow_unknown_block_probe": True}, include_visible=False)
        == SkillFailureCode.MINING_TARGET_UNVERIFIED
    )


def test_probe_permission_belongs_to_skill_not_model_bindings():
    executor = SkillExecutor(_ScriptedPolicy())
    spec = SkillSpec(skill_id="custom", name="Custom", policy_ref="mine")
    executor.start(spec, run_id="r", now_ns=1, parameters={"allow_unknown_block_probe": True})
    assert executor.policy_parameters.get("allow_unknown_block_probe") is not True
    executor.cancel()
    executor.start(
        spec.model_copy(update={"allow_unknown_block_probe": True}), run_id="r2", now_ns=1
    )
    assert executor.policy_parameters["allow_unknown_block_probe"] is True


@pytest.mark.parametrize("prohibited", [False, True])
def test_empty_or_prohibited_policy_does_not_reset_inactivity_watchdog(prohibited):
    proposal = MotorAction(sequence=0, keys_down=("w",)) if prohibited else MotorAction(sequence=0)
    executor = SkillExecutor(_ScriptedPolicy(proposal, proposal))
    executor.start(
        SkillSpec(
            skill_id="quiet",
            name="Quiet",
            policy_ref="look",
            inactivity_timeout_ms=1000,
            max_duration_ms=10000,
            action_permissions=SkillActionPermissions(allow_movement=False),
        ),
        run_id="idle",
        now_ns=1,
    )
    executor.tick(_mining_board(now_ns=1), sequence=1, now_ns=1)
    tick = executor.tick(_mining_board(now_ns=1_000_000_001), sequence=2, now_ns=1_000_000_001)
    assert tick.run.failure_code == SkillFailureCode.CONTROLLER_STARVATION
    assert tick.action.keys_down == tick.action.buttons_down == ()


def test_actual_camera_activity_renews_inactivity_not_success():
    executor = SkillExecutor(
        _ScriptedPolicy(MotorAction(sequence=0), MotorAction(sequence=0, mouse_dx=5))
    )
    executor.start(
        SkillSpec(
            skill_id="look",
            name="Look",
            policy_ref="look",
            inactivity_timeout_ms=1000,
            max_duration_ms=10000,
        ),
        run_id="r",
        now_ns=1,
    )
    executor.tick(_mining_board(now_ns=1), sequence=1, now_ns=1)
    tick = executor.tick(_mining_board(now_ns=1_000_000_001), sequence=2, now_ns=1_000_000_001)
    assert tick.run.outcome == SkillOutcome.RUNNING
    assert tick.action.mouse_dx == 5


@pytest.mark.parametrize("verified", [True, False])
def test_executor_emits_causal_trial_after_release_not_imagined_success(verified):
    now = time.monotonic_ns()
    executor = SkillExecutor(_ScriptedPolicy(MotorAction(sequence=1, buttons_down=("left",))))
    book = _book()
    executor.configure_mining_knowledge(book)
    executor.start(
        SkillSpec(skill_id="mine_visible_block", name="Mine", policy_ref="mine"),
        run_id="r",
        now_ns=now,
    )
    executor.tick(_mining_board(now_ns=now), sequence=1, now_ns=now)
    assert executor._mining_guard.attempt is not None
    end = now + 2_000_000_000
    verification = (
        OutcomeVerification(
            run_id="r",
            kind=OutcomeKind.MINING,
            status=OutcomeStatus.SUCCEEDED,
            signal=OutcomeSignal.BLOCK_BROKEN,
            observed_ns=end,
            confidence=1.0,
            reason="fixture joined break",
            target_kind="oak_log",
        )
        if verified
        else None
    )
    result = executor._finish(
        SkillOutcome.SUCCEEDED if verified else SkillOutcome.TIMED_OUT,
        end,
        "fixture",
        recover=False,
        outcome_verification=verification,
    )
    assert result.action.buttons_up == ("left",)
    trial = executor.take_mining_trial("r")
    assert trial.broke is (True if verified else None)
    assert trial.picked_up is None and trial.harvested is None
    assert executor.take_mining_trial("r") is None
    assert not book.memories.records  # executor does not do IO or preempt durable runtime ownership


@pytest.mark.parametrize("skill", ["traverse_level_ground", "mod_walker"])
def test_obstruction_clear_retries_original_traversal_and_completes_only_its_plan(skill):
    runtime, _, _ = _runtime_for_probe()
    if skill == "mod_walker":
        runtime.skills.register(
            SkillSpec(
                skill_id=skill, name="Mod Walker", outcome_kind="traversal", policy_ref="navigate"
            )
        )
    recovery = _HeadroomRecovery(
        context_key="goal",
        traversal_parameters={},
        deadline_ns=time.monotonic_ns() + 60_000_000_000,
        phase="mining",
        origin_skill_id=skill,
        origin_run_id="stalled",
        mining_run_id="clear",
    )
    runtime._headroom_recovery = recovery
    runtime._route_headroom_terminal(_mining_success("clear"))
    assert runtime.executor.run.skill_id == skill
    proof = _retry_progress(recovery.retry_run_id, context_key="goal")
    proof = replace(proof, run=proof.run.model_copy(update={"skill_id": skill}))
    assert _headroom_retry_advances_plan(
        proof, recovery, plan_steps=(skill,), plan_index=0, plan_goal_id="goal"
    )
    recovery.origin_skill_id = "gather_nearby_wood"
    assert not _headroom_retry_advances_plan(
        proof, recovery, plan_steps=(skill,), plan_index=0, plan_goal_id="goal"
    )


def test_extended_runtime_mining_bounds_are_configurable_not_unbounded():
    executor = SkillExecutor(_ScriptedPolicy())
    executor.configure_mining_timing(max_hold_ms=30000, acquisition_ms=10000)
    assert executor._mining_guard.absolute_max_ms == 30000
    assert executor._mining_guard.acquisition_timeout_ms == 10000
    with pytest.raises(ValueError):
        executor.configure_mining_timing(max_hold_ms=300000, acquisition_ms=10000)
    assert MiningLeaseGuard().absolute_max_ms == 12000  # standalone compatibility


def test_runtime_records_trial_once_and_exposes_tool_evidence_to_planning(tmp_path):
    from test_agent_core import _runtime_for_learning

    now = time.monotonic_ns()
    with StateDatabase(tmp_path / "state.sqlite3") as db:
        runtime = _runtime_for_learning(db)
        runtime.executor = SkillExecutor(_ScriptedPolicy())
        book = MiningKnowledge(runtime.memories, "versioned-survival")
        runtime.executor.configure_mining_knowledge(book)
        trial = MiningTrial(
            attempt_id="r",
            key=book.key("oak_log", "stick"),
            broke=True,
            elapsed_ms=3000.0,
            evidence="fixture:break",
        )
        runtime.executor._mining_trials["r"] = trial
        run = _mining_success("r").run
        runtime._record_terminal_run(run)
        runtime._record_terminal_run(run)
        runtime._flush_pending_learning_records(force=True)
        restored = MiningKnowledge(db.load_memories(), book.scope)
        assert restored.belief("oak_log", "stick").breaks == 1
        runtime.blackboard = _mining_board(now_ns=now)
        evidence = runtime._mining_planning_evidence(book)
        assert evidence[0]["observed_breaks"] == 1
        assert evidence[0]["equipped"] is True
        assert evidence[0]["observed_nonharvests"] == 0
        assert evidence[0]["game_rule"] is None
