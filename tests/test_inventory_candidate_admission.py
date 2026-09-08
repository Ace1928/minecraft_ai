"""Explicit GUI inspection may be considered without reopening the crafting loop."""

from __future__ import annotations

import json

import pytest

from minecraft_ai.builtin_skills import build_bootstrap_skill_library
from minecraft_ai.cognition import (
    CognitionContext, CognitionDecision, HighLevelController, planks_retry_requires_wood,
)
from minecraft_ai.cognition.prompts import (
    _explicit_action_constraints, _operator_requested_skill_ids,
)
from minecraft_ai.execution import SkillExecutor
from minecraft_ai.models import ModelResponse
from minecraft_ai.motor import BootstrapMotorPolicy
from minecraft_ai.perception import PerceptionBlackboard
from minecraft_ai.roles import get_role
from minecraft_ai.social import OperatorMessage, OperatorMessageKind, OperatorMessageStatus
from minecraft_ai.storage import StateDatabase
from test_agent_core import _active_operator_messages, _runtime_with_completed_decision


REQUEST = (
    "Open the inventory once, verify it is visible, then close it. "
    "Do not move or attack during the check."
)


def _context(text=REQUEST, *, kind=OperatorMessageKind.INSTRUCTION,
             status=OperatorMessageStatus.DELIVERED):
    return CognitionContext(
        role=get_role("generalist"), goals=(), memories=(), promises=(), wiki=(),
        planks_retry_requires_wood=True,
        operator_messages=(OperatorMessage(
            message_id="inventory-check", created_ns=1, text=text, kind=kind, status=status,
        ),),
    )


@pytest.mark.parametrize("text", (
    REQUEST,
    (
        "One-time command-following check: open the inventory once, verify the inventory "
        "screen is visible, then close it and return to autonomous survival. Do not move or "
        "attack during this inventory check. Do not repeat this check after returning to the world."
    ),
    "Please open inventory. Do not attack.",
    "Inspect the inventory for logs without moving or attacking.",
    "Check the inventory contents; do not jump.",
    "Audit the inventory. Do not attack.",
    "View the inventory. Don't move.",
    "Do not attack. Open inventory once.",
    "Inventory task: open the inventory once. Do not move or attack.",
    "One-time command-following check: open inventory once. Do not attack.",
    "Open inventory once. Don't do that again.",
    "Open inventory once. Do not do it again.",
))
def test_unrelated_prohibition_allows_only_inventory_candidate(text):
    context = _context(text)
    assert _operator_requested_skill_ids(text) == ()
    assert not planks_retry_requires_wood(context, skill_id="open_inventory")
    assert planks_retry_requires_wood(context, skill_id="craft_wood_planks")
    assert planks_retry_requires_wood(context)
    assert context.planks_retry_requires_wood  # No durable prerequisite repair.


@pytest.mark.parametrize("text", (
    "Do not open inventory.",
    "Never inspect inventory.",
    "Avoid opening the inventory.",
    "Do not move and open inventory.",
    "Open inventory. Do not open inventory after all.",
    "Open inventory, but don't open it after all.",
    "Open inventory, but do not open my inventory.",
    "Open inventory. Don't do that.",
    "Open inventory. Do not do it.",
    "Open inventory. Don't do this.",
    "Open inventory. Actually, don't.",
    "Open inventory. Actually, do not.",
    "Could you open inventory?",
    "How to open inventory: explain the procedure.",
    "Tell me this command: open inventory.",
    "Example: open inventory.",
    "The player said: open inventory.",
    "The sign reads: open inventory.",
    "I wrote: open inventory.",
    "The banner: open inventory.",
    "When safe: open inventory.",
    "Open inventory only when it is safe.",
    '"Open inventory."',
    "'Open inventory.'",
    "`Open inventory.`",
    "“Open inventory.”",
    "If safe: open inventory.",
    "Imagine a check: open inventory.",
    "Find the inventory button.",
))
def test_no_exception_for_negated_quoted_conditional_or_query_only_requests(text):
    # Exercise the new semantic exception, not broaden the legacy literal mapper.
    context = _context(text + " Do not attack.")
    assert _operator_requested_skill_ids(context.operator_messages[0].text) == ()
    assert planks_retry_requires_wood(context, skill_id="open_inventory")


@pytest.mark.parametrize("text", (
    "Open inventory. Don't do that.",
    "Open inventory. Do not do it.",
    "Open inventory. Actually, don't.",
    "Open inventory. Don’t do that.",
    "Open inventory. Actually, don’t.",
))
def test_cancellation_without_extra_prohibition_blocks_both_admission_paths(text):
    context = _context(text)
    assert _operator_requested_skill_ids(text) == ()
    assert planks_retry_requires_wood(context, skill_id="open_inventory")
    assert planks_retry_requires_wood(context, skill_id="craft_wood_planks")
    assert planks_retry_requires_wood(context)
    controller = HighLevelController(_Model(CognitionDecision()), build_bootstrap_skill_library())
    assert controller._operator_fast_path_decision(PerceptionBlackboard(), context) is None


def test_smart_apostrophe_keeps_literal_actuator_prohibitions():
    text = "Open inventory. Don’t attack, interact or jump."
    assert _operator_requested_skill_ids(text) == ()
    assert _explicit_action_constraints(text) == {
        "allow_attack": False, "allow_use": False, "allow_jump": False,
    }
    assert not planks_retry_requires_wood(_context(text), skill_id="open_inventory")
    assert planks_retry_requires_wood(_context(text), skill_id="craft_wood_planks")


@pytest.mark.parametrize("kind", tuple(OperatorMessageKind))
@pytest.mark.parametrize("status", tuple(OperatorMessageStatus))
def test_inventory_exception_follows_instruction_and_correction_authority(kind, status):
    context = _context(kind=kind, status=status)
    qualifies = (
        kind in {OperatorMessageKind.INSTRUCTION, OperatorMessageKind.CORRECTION}
        and status in {OperatorMessageStatus.QUEUED, OperatorMessageStatus.DELIVERED}
    ) or (
        kind == OperatorMessageKind.INSTRUCTION and status == OperatorMessageStatus.ACKNOWLEDGED
    )
    assert planks_retry_requires_wood(context, skill_id="open_inventory") is not qualifies
    assert planks_retry_requires_wood(context, skill_id="craft_wood_planks")
    assert planks_retry_requires_wood(context)


@pytest.mark.parametrize("kind", (OperatorMessageKind.QUESTION, OperatorMessageKind.FEEDBACK))
def test_new_exception_cannot_borrow_older_instruction_behind_active_question_or_feedback(kind):
    context = _context()
    context.operator_messages = (OperatorMessage(
        message_id="newer", created_ns=2, text="What are you doing?", kind=kind,
        status=OperatorMessageStatus.DELIVERED,
    ), *context.operator_messages)
    assert planks_retry_requires_wood(context, skill_id="open_inventory")


@pytest.mark.parametrize("kind,status,text,allowed", (
    (OperatorMessageKind.INSTRUCTION, OperatorMessageStatus.ACKNOWLEDGED, REQUEST, True),
    (OperatorMessageKind.INSTRUCTION, OperatorMessageStatus.ACKNOWLEDGED, "Explore.", False),
    (OperatorMessageKind.CORRECTION, OperatorMessageStatus.ACKNOWLEDGED, REQUEST, False),
    (OperatorMessageKind.INSTRUCTION, OperatorMessageStatus.ARCHIVED, REQUEST, False),
    (OperatorMessageKind.QUESTION, OperatorMessageStatus.QUEUED, "What are you doing?", False),
))
def test_resolved_authority_never_borrows_archived_or_tombstoned_inventory_request(
    kind, status, text, allowed,
):
    context = _context(status=OperatorMessageStatus.ACKNOWLEDGED)
    prior = context.operator_messages[0]
    # A previous correction already retired the older inventory instruction.
    tombstone = prior.model_copy(update={
        "message_id": "prior-correction", "created_ns": 2,
        "kind": OperatorMessageKind.CORRECTION,
    })
    latest = prior.model_copy(update={
        "message_id": "latest", "created_ns": 3, "kind": kind, "status": status, "text": text,
    })
    context.operator_messages = _active_operator_messages((latest, tombstone, prior))
    assert planks_retry_requires_wood(context, skill_id="open_inventory") is not allowed
    assert planks_retry_requires_wood(context, skill_id="craft_wood_planks")


class _Model:
    model_id = "inventory-candidate-test"

    def __init__(self, decision):
        self.decision = decision
        self.calls = []

    def complete_constrained(self, messages, *, name, schema, grammar):
        self.calls.append((messages, name, schema, grammar))
        return ModelResponse(
            text=self.decision.model_dump_json(), model=self.model_id, latency_ms=1,
        )


@pytest.mark.parametrize("skill_id,questions,replan", (
    ("open_inventory", (), False),
    (None, (), False),
    (None, ("scene.playable",), True),
))
def test_candidate_does_not_force_skill_or_remove_observation_and_replan(
    skill_id, questions, replan,
):
    context = _context()
    model = _Model(CognitionDecision(
        skill_id=skill_id, ask_perception=questions, request_replan=replan,
        instruction=REQUEST, skill_parameters={"allow_attack": False},
    ))
    controller = HighLevelController(model, build_bootstrap_skill_library())
    board = PerceptionBlackboard()
    assert controller._operator_fast_path_decision(board, context) is None
    candidates = controller._feasible_skill_payloads(board, query_text=REQUEST, context=context)
    assert "open_inventory" in {item["skill_id"] for item in candidates}
    assert "craft_wood_planks" not in {item["skill_id"] for item in candidates}
    bounds = controller._decision_repair_bounds(board, context)
    assert bounds.requested_skill_ids == ()
    assert "open_inventory" in dict(bounds.allowed_skills)
    assert "craft_wood_planks" not in dict(bounds.allowed_skills)

    decision = controller.decide(board, context)

    assert decision.skill_id == skill_id
    assert decision.ask_perception == questions
    assert decision.request_replan == replan
    assert decision.chosen_goal_id == "operator:inventory-check"
    assert decision.skill_parameters["allow_attack"] is False
    assert len(model.calls) == 1
    payload = json.loads(model.calls[0][0][1].content)
    assert payload["planks_retry_requires_wood"] is True
    skill_rule = next(line for line in model.calls[0][3].splitlines()
                      if line.startswith("skill ::="))
    assert "open_inventory" in skill_rule and '"null"' in skill_rule
    assert "craft_wood_planks" not in skill_rule


@pytest.mark.parametrize("repair", ("infeasible", "repeated"))
def test_repair_candidate_lists_keep_inventory_only_exception(repair):
    from minecraft_ai.skills import SkillOutcome, SkillRun

    model = _Model(CognitionDecision(skill_id="open_inventory", instruction=REQUEST))
    controller = HighLevelController(model, build_bootstrap_skill_library())
    decision = CognitionDecision(skill_id="craft_wood_planks")
    if repair == "infeasible":
        result = controller._repair_infeasible(
            decision, PerceptionBlackboard(), _context(), reason="missing wood",
        )
    else:
        result = controller._repair_repeated_failure(
            decision, PerceptionBlackboard(), _context(),
            SkillRun(run_id="old", skill_id="craft_wood_planks", started_ns=1,
                     ended_ns=2, outcome=SkillOutcome.FAILED),
        )
    assert result.skill_id == "open_inventory"
    assert len(model.calls) == 1
    skill_rule = next(line for line in model.calls[0][3].splitlines()
                      if line.startswith("skill ::="))
    assert "open_inventory" in skill_rule and "craft_wood_planks" not in skill_rule


@pytest.mark.parametrize("skill_id,status,admitted", (
    ("open_inventory", OperatorMessageStatus.DELIVERED, True),
    ("open_inventory", OperatorMessageStatus.ACKNOWLEDGED, True),
    ("craft_wood_planks", OperatorMessageStatus.DELIVERED, False),
))
def test_runtime_rechecks_selected_skill_and_active_directive(tmp_path, skill_id, status, admitted):
    context = _context(status=status)
    with StateDatabase(tmp_path / "state.sqlite3") as database:
        database.save_operator_message(context.operator_messages[0])
        pending = ("inventory-check",) if status == OperatorMessageStatus.DELIVERED else ()
        runtime = _runtime_with_completed_decision(
            CognitionDecision(skill_id=skill_id, chosen_goal_id="operator:inventory-check",
                              instruction=REQUEST, skill_parameters={"allow_attack": False}),
            database=database, pending_message_ids=pending,
        )
        runtime.skills = build_bootstrap_skill_library()
        runtime.executor = SkillExecutor(BootstrapMotorPolicy())
        runtime._planks_retry_requires_wood = lambda: True
        runtime._cognition_context = lambda: context

        runtime._consume_cognition()

        assert (runtime.executor.run is not None) is admitted
        if admitted:
            assert runtime.executor.run.skill_id == "open_inventory"
            assert runtime.executor.parameters["allow_attack"] is False
        else:
            assert runtime._last_decision is None
        assert database.load_operator_messages(limit=1)[0].status == (
            OperatorMessageStatus.ACKNOWLEDGED if admitted else status
        )
