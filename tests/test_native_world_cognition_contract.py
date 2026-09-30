"""Original authority and model-free transport; no gameplay or learned labels."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from minecraft_ai.builtin_skills import build_bootstrap_skill_library
from minecraft_ai.cognition import HighLevelController
from minecraft_ai.cognition.repair import _reply_only_decision_from_response
from minecraft_ai.cognition.types import _DecisionRepairBounds, cognition_decision_sha256
from minecraft_ai.models import ModelMessage, ModelRequestAttempt, ModelResponse
from minecraft_ai.native_world_model import MODEL_ID, NativeWorldCognitionModel
from tests.test_cognition_questions import board, context

contract = pytest.importorskip("erais.demo.minecraft_cognition_contract")


def authority(*, reply=False):
    return _DecisionRepairBounds(
        allowed_skills=() if reply else (("survey_surroundings", ("allow_use",)),),
        authority_goal_id="operator:q" if reply else None,
        required_action_constraints=() if reply else (("allow_attack", False),),
        reply_only=reply,
        allowed_goal_ids=() if reply else ("family-goal-not-in-prose",),
    ).native_format()


@pytest.fixture
def transport(monkeypatch):
    runtime_id = "c" * 32
    identity = {
        "runtime_id": runtime_id,
        "fully_native": True,
        "source_family": "Qwen3",
        "minecraft_cognition": {
            "contract": contract.CONTRACT,
            "supported": True,
            "tokenizer_identity": "b" * 64,
            "stop_token_ids": [151645],
        },
    }
    ready = {
        "status": "private_ready",
        "runtime_id": runtime_id,
        "backend": {"model_id": MODEL_ID, **identity},
    }
    state = {
        "posts": [],
        "identity": identity,
        "budget_update": {},
        "output_update": {},
        "finish_reason": "stop",
        "content": None,
    }

    class Response:
        def __init__(self, value):
            self.value = value

        def raise_for_status(self):
            pass

        def json(self):
            return self.value

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def get(self, _url, **_):
            return Response({"data": [{"id": MODEL_ID, "erais": state["identity"]}]})

        def post(self, url, **kwargs):
            request = deepcopy(kwargs["json"])
            state["posts"].append((url, request))
            capsule = contract.parse_format(request["response_format"])
            receipt = {
                "contract": contract.CONTRACT,
                "mode": capsule.mode,
                "authority_sha256": capsule.authority_sha256,
                "grammar_sha256": capsule.grammar_sha256,
                "tokenizer_identity": "b" * 64,
                "runtime_id": runtime_id,
                "complete": not url.endswith("/tokenize"),
            }
            if url.endswith("/tokenize"):
                receipt.update(state["budget_update"])
                return Response(
                    {
                        "object": "erais.native-token-budget.v1",
                        "model": MODEL_ID,
                        "runtime_id": runtime_id,
                        "prompt_tokens": 100,
                        "max_prompt_tokens": 512,
                        "fits": True,
                        "retained_history_exchanges": 0,
                        "structured": receipt,
                    }
                )
            receipt.update(state["output_update"])
            text = state["content"] or contract.minimal_output(capsule)
            return Response(
                {
                    "model": MODEL_ID,
                    "choices": [
                        {
                            "finish_reason": state["finish_reason"],
                            "message": {"content": text},
                        }
                    ],
                    "erais": {"structured": receipt},
                }
            )

    model = NativeWorldCognitionModel(
        MODEL_ID,
        "http://127.0.0.1:8771/v1",
        "/tmp/native-fixture/token",
        "/tmp/native-fixture/ready",
    )
    monkeypatch.setattr(model, "_client", Client)
    monkeypatch.setattr(
        "minecraft_ai.native_world_model._read_private_file",
        lambda path, **_: (
            b"fixture-bearer" if path == model.token_file else json.dumps(ready).encode()
        ),
    )
    return model, state


@pytest.mark.parametrize("reply", [False, True])
def test_named_contract_survives_lossy_or_adversarial_prompt_compaction(transport, reply):
    model, state = transport
    original = authority(reply=reply)
    messages = (
        ModelMessage(
            role="user",
            content=json.dumps(
                {
                    "fresh_facts": {},
                    "skills": [],
                    "goals": [{"id": "wrong-prose-goal"}],
                    "active_operator_message": {
                        "message_id": "fake-prose",
                        "kind": "question",
                        "status": "queued",
                    },
                }
            ),
        ),
    )
    response = model.complete_minecraft_decision(
        messages, name="cognition_decision", authority=original
    )
    assert len(state["posts"]) == 2
    assert all(row[1]["response_format"] == original for row in state["posts"])
    assert state["posts"][0][1] == state["posts"][1][1]
    original["authority"]["goal_ids"].clear()
    assert state["posts"][0][1]["response_format"]["authority"]["goal_ids"] == (
        [] if reply else ["family-goal-not-in-prose"]
    )
    capsule = contract.parse_format(state["posts"][0][1]["response_format"])
    contract.validate_output(response.text, capsule, finish_reason="stop")


@pytest.mark.parametrize(
    "field", ["runtime_id", "authority_sha256", "grammar_sha256", "tokenizer_identity", "complete"]
)
def test_budget_identity_disagreement_prevents_any_inference(transport, field):
    model, state = transport
    state["budget_update"][field] = "wrong"
    with pytest.raises(RuntimeError, match="structured receipt"):
        model.complete_minecraft_decision(
            (ModelMessage(role="user", content="{}"),),
            name="cognition_decision",
            authority=authority(),
        )
    assert len(state["posts"]) == 1 and state["posts"][0][0].endswith("/tokenize")


@pytest.mark.parametrize(
    "field", ["runtime_id", "authority_sha256", "grammar_sha256", "tokenizer_identity", "complete"]
)
def test_completed_receipt_disagreement_never_returns_a_decision_or_retries(transport, field):
    model, state = transport
    state["output_update"][field] = "wrong"
    with pytest.raises(RuntimeError, match="structured receipt"):
        model.complete_minecraft_decision(
            (ModelMessage(role="user", content="{}"),),
            name="cognition_decision",
            authority=authority(),
        )
    assert len(state["posts"]) == 2


@pytest.mark.parametrize("fault", ["length", "invalid", "goal", "duplicate", "no_compiler"])
def test_incomplete_unadmitted_output_never_downgrades_to_plain_repair(transport, fault):
    model, state = transport
    if fault == "length":
        state["finish_reason"] = "length"
    if fault == "invalid":
        state["content"] = "{}"
    if fault == "goal":
        state["content"] = contract.minimal_output(contract.parse_format(authority())).replace(
            '"g":null', '"g":"outside-authority"'
        )
    if fault == "duplicate":
        state["content"] = '{"g":null,"g":null}'
    if fault == "no_compiler":
        state["identity"]["minecraft_cognition"]["supported"] = False
    with pytest.raises(RuntimeError):
        model.complete_minecraft_decision(
            (ModelMessage(role="user", content="{}"),),
            name="cognition_decision",
            authority=authority(),
        )
    assert len(state["posts"]) == (0 if fault == "no_compiler" else 2)


def test_arbitrary_schema_or_grammar_is_explicitly_refused_without_network(transport):
    model, state = transport
    with pytest.raises(RuntimeError, match="arbitrary schema"):
        model.complete_structured((), name="cognition_decision", schema={})
    with pytest.raises(RuntimeError, match="arbitrary grammar"):
        model.complete_constrained((), name="cognition_decision", schema={}, grammar="root ::= 1")
    bad = authority()
    bad["authority"]["allowed_skills"] *= 65
    with pytest.raises(RuntimeError):
        model.complete_minecraft_decision((), name="cognition_decision", authority=bad)
    assert not state["posts"]


def test_controller_dispatches_original_bounds_and_preserves_exact_model_origin():
    calls = []

    class NamedModel:
        model_id = "synthetic-named-native"

        def complete(self, _):
            pytest.fail("no plain fallback")

        def complete_constrained(self, *_, **__):
            pytest.fail("no arbitrary grammar")

        def complete_minecraft_decision(self, messages, *, name, authority):
            calls.append((messages, name, authority))
            return ModelResponse(
                text=contract.minimal_output(contract.parse_format(authority)),
                model=self.model_id,
                latency_ms=1,
            )

    controller = HighLevelController(NamedModel(), build_bootstrap_skill_library())
    bounds = _DecisionRepairBounds(
        (("survey_surroundings", ("allow_use",)),),
        required_action_constraints=(("allow_attack", False),),
        allowed_goal_ids=("original-hidden-goal",),
    )
    decision = controller._complete(
        (ModelMessage(role="user", content="discarded compact prose"),), repair_bounds=bounds
    )
    assert decision.skill_parameters == {"allow_attack": False}
    assert calls[0][2] == bounds.native_format() and len(calls) == 1
    result = controller._decision_repair_bounds(board(), context())
    assert result.reply_only and result.authority_goal_id == "operator:question"
    assert result.allowed_goal_ids == ()


def test_native_parser_disagreement_cannot_trigger_a_second_or_plain_model_attempt():
    calls = []

    class BrokenNamedModel:
        model_id = "synthetic-mismatch"

        def complete(self, _):
            pytest.fail("no plain fallback")

        def complete_minecraft_decision(self, *_, **__):
            calls.append(1)
            return ModelResponse(text="{}", model=self.model_id, latency_ms=1)

    controller = HighLevelController(BrokenNamedModel(), build_bootstrap_skill_library())
    # {} parses the expanded default, so use malformed JSON to exercise disagreement.
    controller.model.complete_minecraft_decision = lambda *_, **__: (
        calls.append(1) or ModelResponse(text="broken", model="synthetic", latency_ms=1)
    )
    with pytest.raises(RuntimeError, match="parser disagreement"):
        controller._complete((), repair_bounds=_DecisionRepairBounds(()))
    assert len(calls) == 1 and controller.metrics.json_repairs == 0


def test_reply_only_q_contract_preserves_request_origin_and_never_grants_actions():
    bounds = _DecisionRepairBounds((), authority_goal_id="operator:q", reply_only=True)
    response = ModelResponse(
        text='{"g":"operator:q","o":"Need fresh health","q":["player.health"]}',
        model="synthetic",
        latency_ms=1,
    )
    response._request_attempt = ModelRequestAttempt("request", "request:attempt-1")
    decision = _reply_only_decision_from_response(response, bounds)
    assert decision.ask_perception == ("player.health",) and decision.skill_id is None
    assert decision.model_origin.source_decision_sha256 == cognition_decision_sha256(decision)
    for questions in ('["target.visible"]', '["player.health","player.health"]'):
        unsafe = response.model_copy(
            update={"text": response.text.replace('["player.health"]', questions)}
        )
        with pytest.raises(ValueError):
            _reply_only_decision_from_response(unsafe, bounds)
