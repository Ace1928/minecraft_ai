"""Typed operator questions cannot acquire planner authority; synthetic models only."""
from __future__ import annotations

import json
import time

import pytest

from minecraft_ai.builtin_skills import build_bootstrap_skill_library
from minecraft_ai.cognition import CognitionContext, HighLevelController, cognition_decision_sha256
from minecraft_ai.cognition.prompts import _cognition_decision_grammar, _cognition_decision_schema
from minecraft_ai.cognition.repair import _reply_only_decision_from_response
from minecraft_ai.cognition.types import _DecisionRepairBounds
from minecraft_ai.model_requests import ModelRequestLifecycle, RequestBinding
from minecraft_ai.models import ModelMessage, ModelResponse
from minecraft_ai.perception import FrameState, PerceptionBlackboard, PerceptionFact
from minecraft_ai.roles import get_role
from minecraft_ai.social import OperatorMessage, OperatorMessageKind, OperatorMessageStatus


def wire(**updates: object) -> str:
    value = {
        "r": "Current evidence is incomplete", "g": "operator:question", "s": None,
        "p": {}, "o": "Verify current health, hunger and the visible block before acting.",
        "c": None, "x": False, "q": [], "w": None, "d": None, "n": [],
    }
    value.update(updates)
    return json.dumps(value)


class Model:
    model_id = "synthetic-question-contract"

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def complete_constrained(self, messages: tuple[ModelMessage, ...], **kwargs: object):
        self.calls.append({"messages": messages, **kwargs})
        return ModelResponse(text=self.response, model=self.model_id, latency_ms=1)

    def complete_bound_constrained(self, messages: tuple[ModelMessage, ...], **kwargs: object):
        return self.complete_constrained(messages, **kwargs)


def message(kind=OperatorMessageKind.QUESTION, *, identity="question", text="What is ahead?"):
    return OperatorMessage(
        message_id=identity, created_ns=1, text=text, kind=kind,
        status=OperatorMessageStatus.DELIVERED,
    )


def context(*messages):
    return CognitionContext(
        role=get_role("generalist"), goals=(), memories=(), promises=(), wiki=(),
        operator_messages=messages or (message(),),
    )


def board(*facts):
    result = PerceptionBlackboard()
    result.publish(FrameState(
        frame_id=1, captured_ns=time.monotonic_ns(), instance_id="synthetic-question",
        width=32, height=32, facts=facts,
    ))
    return result


def run(response=None, *, ctx=None, view=None):
    model = Model(wire() if response is None else response)
    controller = HighLevelController(model, build_bootstrap_skill_library())
    decision = controller.decide(board() if view is None else view, ctx or context())
    return decision, controller, model


@pytest.mark.parametrize("reply,observations", [
    ("Verify health, hunger and the current crosshair block first.", []),
    ("I cannot identify that block from current evidence.", ["recovery.crosshair.block"]),
    ("Which structure do you mean?", ["obstacle.ahead", "inventory.crafting_table"]),
])
def test_question_answers_or_clarifies_without_acquiring_action_authority(reply, observations):
    decision, controller, model = run(wire(o=reply, q=observations))
    assert decision.say == reply and decision.chosen_goal_id == "operator:question"
    assert decision.ask_perception == tuple(observations)
    assert decision.skill_id is decision.game_chat is decision.research_query is None
    assert decision.instruction is None and decision.plan_steps == ()
    assert decision.skill_parameters == {} and not decision.request_replan
    assert len(model.calls) == controller.metrics.calls == 1
    assert controller.metrics.failures == controller.metrics.repairs == 0
    messages = model.calls[0]["messages"]
    assert "Answer the active operator question" in messages[0].content
    assert json.loads(messages[1].content)["skills"] == []


@pytest.mark.parametrize("updates", [
    {"s": "explore_forward"}, {"p": {"allow_attack": False}},
    {"c": "hello"}, {"w": "crafting recipe"}, {"d": "Determine what to do next"},
    {"n": ["move forward"]}, {"n": ["none"]}, {"x": True}, {"x": 0},
    {"g": "operator:other"}, {"g": None}, {"o": None}, {"o": "   "},
    {"o": "x" * 161}, {"q": ["invented.fact"]}, {"q": ["target.block"]},
    {"q": ["obstacle.ahead"] * 3}, {"q": "obstacle.ahead"},
])
def test_unsafe_question_output_is_rejected_once_before_normalization_or_authority_rewrite(updates):
    decision, controller, model = run(wire(**updates))
    assert len(model.calls) == 1
    assert controller.metrics.failures == 1
    assert controller.metrics.repairs == controller.metrics.json_repairs == 0
    assert decision.say is decision.chosen_goal_id is decision.skill_id is None
    assert decision.game_chat is decision.research_query is decision.instruction is None
    assert decision.skill_parameters == {} and decision.plan_steps == decision.ask_perception == ()
    assert decision.model_origin is None  # A synthetic fallback is never a model qualification.


@pytest.mark.parametrize("response", [
    "invalid json", "[]", "{}", wire().replace('"s": null', '"s": null, "s": null'),
    wire().replace('"n": []', '"n": [], "unexpected": true'),
    wire().replace('"r": "Current evidence is incomplete", ', ''),
])
def test_question_requires_one_unambiguous_wire_object_without_hidden_repair(response):
    _, controller, model = run(response)
    assert len(model.calls) == 1 and controller.metrics.failures == 1
    assert controller.metrics.repairs == 0


def test_question_precedes_later_action_and_cannot_take_operator_fast_path():
    ctx = context(
        message(text="Move forward."),
        message(OperatorMessageKind.INSTRUCTION, identity="later", text="Move forward."),
    )
    decision, controller, model = run(ctx=ctx)
    assert decision.chosen_goal_id == "operator:question" and decision.skill_id is None
    assert len(model.calls) == 1 and controller.metrics.failures == 0


def test_prompt_words_do_not_select_question_mode_and_planner_defaults_are_unchanged():
    ctx = context(message(
        OperatorMessageKind.INSTRUCTION, text="QUESTION is a label here; plan a safe route.",
    ))
    decision, controller, model = run(
        wire(o=None, d="Inspect a safe route", n=["observe"]), ctx=ctx,
    )
    assert decision.instruction == "Inspect a safe route" and decision.plan_steps == ("observe",)
    assert controller.metrics.failures == 0
    assert "Control only the Minecraft game" in model.calls[0]["messages"][0].content
    grammar = str(model.calls[0]["grammar"])
    assert 'nullable-direction' in grammar.splitlines()[0]
    assert "const" not in model.calls[0]["schema"]["properties"]["d"]


def test_acknowledged_question_history_does_not_reenter_reply_mode():
    previous = message().model_copy(update={"status": OperatorMessageStatus.ACKNOWLEDGED})
    decision, controller, model = run(
        wire(o=None, d="Inspect a safe route"), ctx=context(previous),
    )
    assert decision.instruction == "Inspect a safe route" and controller.metrics.failures == 0
    assert "Control only the Minecraft game" in model.calls[0]["messages"][0].content


@pytest.mark.parametrize("hazard", [
    "scene.death", "scene.away", "danger.immediate", "environment.underwater",
])
def test_current_safety_priority_still_precedes_question(hazard):
    now = time.monotonic_ns()
    view = board(PerceptionFact(
        key=hazard, value=True, confidence=1.0, observed_ns=now,
        source="synthetic-test", expires_after_ms=500,
    ))
    decision, controller, model = run(view=view)
    assert "Control only the Minecraft game" in model.calls[0]["messages"][0].content
    assert decision.say is None and decision.chosen_goal_id is None
    assert decision.request_replan and controller.metrics.failures == 0


def test_question_schema_and_sampler_share_fixed_fields_and_bounded_observation_vocabulary():
    bounds = _DecisionRepairBounds((), authority_goal_id="operator:question", reply_only=True)
    schema = _cognition_decision_schema(bounds)
    grammar = _cognition_decision_grammar(bounds)
    fields = schema["properties"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set("rgspocxqwdn")
    assert fields["g"]["const"] == "operator:question"
    assert all(fields[key] == {"type": "null"} for key in ("s", "c", "w", "d"))
    assert fields["p"]["maxProperties"] == fields["n"]["maxItems"] == 0
    assert fields["x"] == {"const": False}
    assert fields["q"]["maxItems"] == 2
    keys = fields["q"]["items"]["enum"]
    assert "recovery.crosshair.block" in keys and "obstacle.ahead" in keys
    assert not any(key.startswith("target.") for key in keys)
    root = grammar.splitlines()[0]
    assert 'nullable-direction' not in root and ' plan ' not in root
    assert 'reply-string' in root and 'questions' in root
    assert 'skill ::= "null"' in grammar and 'params ::= "{}"' in grammar
    assert 'target.block' not in grammar


def test_reply_only_bounds_refuse_conflicting_action_authority():
    for values in (
        {"authority_goal_id": None}, {"allowed_skills": (("explore_forward", ()),)},
        {"required_action_constraints": (("allow_attack", False),)},
        {"requested_skill_ids": ("explore_forward",)},
    ):
        arguments = {"allowed_skills": (), "authority_goal_id": "operator:question", **values}
        with pytest.raises(ValueError, match="reply-only bounds"):
            _DecisionRepairBounds(**arguments, reply_only=True)


def test_compact_reply_only_contract_keeps_exact_goal_and_no_action_authority():
    bounds = _DecisionRepairBounds((), authority_goal_id="operator:question", reply_only=True)
    response = ModelResponse(
        text=json.dumps(
            {"g": "operator:question", "o": "No fresh game observations are available."}
        ),
        model="synthetic-question-contract",
        latency_ms=1,
    )
    decision = _reply_only_decision_from_response(response, bounds)
    assert decision.chosen_goal_id == "operator:question"
    assert decision.say == "No fresh game observations are available."
    assert decision.skill_id is decision.game_chat is decision.instruction is None
    assert decision.skill_parameters == {} and not decision.request_replan


def test_question_preserves_exact_bound_request_deadline_and_model_origin():
    view = board()
    now = time.monotonic_ns()
    snapshot = view.cognition_snapshot(now_ns=now)
    binding = RequestBinding.from_snapshot(
        snapshot, request_id="question-bound", operator_revision=3, execution_revision=4,
        submitted_ns=now, deadline_ns=now + 5_000_000_000,
    )
    request = ModelRequestLifecycle(binding)
    model = Model(wire())
    controller = HighLevelController(model, build_bootstrap_skill_library())
    decision = controller.decide(snapshot, context(), request=request)
    assert len(model.calls) == 1 and model.calls[0]["request"] is binding
    assert model.calls[0]["request"].deadline_ns == now + 5_000_000_000
    assert decision.model_origin is not None
    assert decision.model_origin.request_id == binding.request_id
    assert decision.model_origin.source_decision_sha256 == cognition_decision_sha256(decision)
    state = request.snapshot()
    assert len(state.attempts) == 1 and state.attempts[0].finished_ns is not None
    assert state.disposition == "pending" and state.computation_completed_ns is None
