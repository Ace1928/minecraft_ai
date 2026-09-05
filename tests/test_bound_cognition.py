"""Private request origins for generic cognition; no real model or control fixtures."""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from minecraft_ai.builtin_skills import build_bootstrap_skill_library
from minecraft_ai.cognition import (
    CognitionContext,
    CognitionDecision,
    HighLevelController,
    cognition_decision_sha256,
)
from minecraft_ai.model_requests import ModelRequestLifecycle, RequestBinding
from minecraft_ai.models import (
    ModelMessage,
    ModelResponse,
    local_model_inference_available,
    local_model_inference_lane,
)
from minecraft_ai.perception import CognitionBlackboardSnapshot, FrameState, PerceptionBlackboard
from minecraft_ai.roles import get_role
from minecraft_ai.skills import SkillOutcome, SkillRun
from minecraft_ai.social import OperatorMessage


def _wire(skill: str | None = "explore_forward", **updates: object) -> str:
    value: dict[str, object] = {
        "r": "Move to observe nearby terrain", "g": None, "s": skill, "p": {},
        "o": None, "c": None, "x": False, "q": [], "w": None, "d": None, "n": [],
    }
    value.update(updates)
    return json.dumps(value)


def _response(text: str) -> ModelResponse:
    return ModelResponse(text=text, model="synthetic-bound-test", latency_ms=1.0)


def _request(
    request_id: str = "request-one",
) -> tuple[CognitionBlackboardSnapshot, CognitionContext, ModelRequestLifecycle]:
    now = time.monotonic_ns()
    board = PerceptionBlackboard()
    board.publish(FrameState(
        frame_id=1, captured_ns=now, instance_id="synthetic-test", width=32, height=32,
    ))
    snapshot = board.cognition_snapshot(now_ns=now)
    context = CognitionContext(
        role=get_role("generalist"), goals=(), memories=(), promises=(), wiki=(),
    )
    binding = RequestBinding.from_snapshot(
        snapshot, request_id=request_id, operator_revision=3, execution_revision=4,
        submitted_ns=now, deadline_ns=now + 10_000_000_000,
    )
    return snapshot, context, ModelRequestLifecycle(binding)


class _BoundModel:
    model_id = "synthetic-bound-test"

    def __init__(self, *results: str | BaseException) -> None:
        self.results = list(results or (_wire(),))
        self.calls: list[dict[str, object]] = []
        self.legacy_calls = 0
        self.shared_response: ModelResponse | None = None

    def complete(self, messages: tuple[ModelMessage, ...]) -> ModelResponse:
        self.legacy_calls += 1
        return self.shared_response or _response(_wire())

    def complete_bound_constrained(
        self,
        messages: tuple[ModelMessage, ...],
        *,
        name: str,
        schema: dict[str, object],
        grammar: str,
        request: RequestBinding,
        attempt_id: str,
    ) -> ModelResponse:
        self.calls.append({
            "messages": messages, "name": name, "schema": schema, "grammar": grammar,
            "request": request, "attempt_id": attempt_id,
        })
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return self.shared_response or _response(result)


class _LegacyModel:
    model_id = "synthetic-legacy-test"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages: tuple[ModelMessage, ...]) -> ModelResponse:
        self.calls += 1
        return _response(_wire())


def test_bound_adapter_gets_exact_binding_and_unchanged_wire_schema() -> None:
    snapshot, context, request = _request()
    model = _BoundModel()
    controller = HighLevelController(model, build_bootstrap_skill_library())
    decision = controller.decide(snapshot, context, request=request)
    assert decision.skill_id == "explore_forward"
    assert model.legacy_calls == 0
    assert len(model.calls) == 1
    call = model.calls[0]
    assert call["request"] is request.binding
    assert call["attempt_id"] == "request-one:attempt-1"
    schema = call["schema"]
    assert isinstance(schema, dict)
    assert set(schema["properties"]) == set("rgspocxqwdn")
    assert isinstance(call["grammar"], str) and "root ::=" in call["grammar"]
    assert "request-one" not in str(call["messages"])
    assert decision.model_origin is not None
    assert decision.model_origin.request_id == request.binding.request_id
    assert decision.model_origin.attempt_id == call["attempt_id"]
    assert decision.model_origin.source_decision_sha256 == cognition_decision_sha256(decision)
    state = request.snapshot()
    assert len(state.attempts) == 1 and state.attempts[0].finished_ns is not None
    assert state.attempts[0].error_type is None
    # Only the future/runtime may mark computation completion or publication.
    assert state.computation_completed_ns is None and state.disposition == "pending"


def test_legacy_adapter_still_runs_and_accounts_its_bound_attempt() -> None:
    snapshot, context, request = _request()
    model = _LegacyModel()
    decision = HighLevelController(model, build_bootstrap_skill_library()).decide(
        snapshot, context, request=request,
    )
    assert model.calls == 1
    assert decision.model_origin is not None
    assert decision.model_origin.attempt_id == request.snapshot().attempts[0].attempt_id


def test_absent_request_uses_legacy_capability_without_inventing_origin() -> None:
    snapshot, context, _ = _request()
    model = _BoundModel()
    decision = HighLevelController(model, build_bootstrap_skill_library()).decide(snapshot, context)
    assert model.legacy_calls == 1 and not model.calls
    assert decision.model_origin is None


def test_json_repair_origin_selects_second_actual_attempt() -> None:
    snapshot, context, request = _request()
    model = _BoundModel('not JSON', _wire())
    decision = HighLevelController(model, build_bootstrap_skill_library()).decide(
        snapshot, context, request=request,
    )
    assert decision.skill_id == "explore_forward"
    assert decision.model_origin is not None
    assert decision.model_origin.attempt_id == "request-one:attempt-2"
    attempts = request.snapshot().attempts
    assert [item.name for item in attempts] == [
        "cognition_decision", "cognition_decision_json_repair",
    ]
    assert all(item.finished_ns is not None and item.error_type is None for item in attempts)


@pytest.mark.parametrize("first_response", ['not JSON', _wire("invented_skill")])
def test_first_answer_finishing_after_deadline_never_calls_repair_adapter(
    first_response: str,
) -> None:
    snapshot, context, original_request = _request()
    clock = [original_request.binding.submitted_ns]
    binding = replace(original_request.binding, deadline_ns=clock[0] + 10)
    request = ModelRequestLifecycle(binding, clock_ns=lambda: clock[0])

    class LateModel(_BoundModel):
        def complete_bound_constrained(
            self, messages: tuple[ModelMessage, ...], *, name: str,
            schema: dict[str, object], grammar: str, request: RequestBinding, attempt_id: str,
        ) -> ModelResponse:
            response = super().complete_bound_constrained(
                messages, name=name, schema=schema, grammar=grammar,
                request=request, attempt_id=attempt_id,
            )
            clock[0] = request.deadline_ns + 5
            return response

    model = LateModel(first_response, _wire())
    controller = HighLevelController(model, build_bootstrap_skill_library())
    decision = controller.decide(snapshot, context, request=request)
    assert [call["name"] for call in model.calls] == ["cognition_decision"]
    assert model.results == [_wire()] and model.legacy_calls == 0
    attempt, = request.snapshot().attempts
    assert attempt.name == "cognition_decision"
    assert attempt.started_ns == binding.submitted_ns
    assert attempt.finished_ns == binding.deadline_ns + 5 and attempt.error_type is None
    assert decision.skill_id is None and decision.request_replan and decision.model_origin is None
    assert controller.metrics.calls == 1 and controller.metrics.failures == 1
    assert controller.metrics.last_error is not None and "deadline" in controller.metrics.last_error
    assert controller._request_context.get() is None
    # The controller fallback does not claim runtime completion or rejection.
    assert request.snapshot().disposition == "pending"
    assert request.snapshot().computation_completed_ns is None
    request.mark_computation_complete()
    assert request.reject("deadline_before_publication")
    assert request.snapshot().attempts == (attempt,)
    assert request.take_discard_notice() == "deadline_before_publication"


def test_semantic_repair_origin_selects_repaired_attempt() -> None:
    snapshot, context, request = _request()
    model = _BoundModel(_wire("invented_skill"), _wire())
    decision = HighLevelController(model, build_bootstrap_skill_library()).decide(
        snapshot, context, request=request,
    )
    assert decision.skill_id == "explore_forward"
    assert len(request.snapshot().attempts) == 2
    assert decision.model_origin is not None
    assert decision.model_origin.attempt_id == "request-one:attempt-2"


@pytest.mark.parametrize("repaired_skill", ["craft_wood_planks", "invented_skill"])
def test_repeated_execution_repair_tracks_selection_or_clears_fallback(
    repaired_skill: str,
) -> None:
    snapshot, context, request = _request()
    library = build_bootstrap_skill_library()
    failed = SkillRun(
        run_id="failure-one", skill_id="explore_forward", started_ns=1, ended_ns=2,
        outcome=SkillOutcome.TIMED_OUT, failure_reason="synthetic-no-displacement",
    )
    library.record(failed)
    second = failed.model_copy(update={"run_id": "failure-two"})
    library.record(second)
    context.recent_skill_runs = (second,)
    controller = HighLevelController(_BoundModel(_wire(), _wire(repaired_skill)), library)
    decision = controller.decide(snapshot, context, request=request)
    assert controller.metrics.retry_repairs == 1
    assert len(request.snapshot().attempts) == 2
    if repaired_skill == "craft_wood_planks":
        assert decision.skill_id == repaired_skill and decision.model_origin is not None
        assert decision.model_origin.attempt_id == "request-one:attempt-2"
    else:
        assert decision.skill_id is None and decision.request_replan
        assert decision.model_origin is None


@pytest.mark.parametrize("results", [
    ("invalid", "still invalid"),
    (_wire("invented_skill"), _wire("also_invented")),
    (_wire("establish_basic_shelter"), _wire("establish_basic_shelter")),
])
def test_invalid_or_infeasible_repair_fallback_has_no_model_origin(
    results: tuple[str, str],
) -> None:
    snapshot, context, request = _request()
    decision = HighLevelController(
        _BoundModel(*results), build_bootstrap_skill_library(),
    ).decide(snapshot, context, request=request)
    assert decision.skill_id is None and decision.request_replan
    assert decision.model_origin is None
    assert len(request.snapshot().attempts) == 2
    assert all(item.finished_ns is not None for item in request.snapshot().attempts)


def test_model_exception_finishes_attempt_before_controller_fallback() -> None:
    snapshot, context, request = _request()
    controller = HighLevelController(_BoundModel(ValueError("fixture")),
                                     build_bootstrap_skill_library())
    decision = controller.decide(snapshot, context, request=request)
    assert decision.model_origin is None and decision.request_replan
    attempt, = request.snapshot().attempts
    assert attempt.error_type == "ValueError" and attempt.finished_ns is not None
    assert controller._request_context.get() is None


def test_rejected_request_does_not_start_another_model_call() -> None:
    snapshot, context, request = _request()
    request.reject("operator_changed")
    model = _BoundModel()
    decision = HighLevelController(model, build_bootstrap_skill_library()).decide(
        snapshot, context, request=request,
    )
    assert decision.model_origin is None and decision.request_replan
    assert request.snapshot().attempts == () and model.calls == []


def test_deterministic_operator_fast_path_has_no_paid_attempt_or_origin() -> None:
    snapshot, context, request = _request()
    context.operator_messages = (
        OperatorMessage(message_id="operator-one", created_ns=1, text="Craft planks once."),
    )
    model = _BoundModel()
    decision = HighLevelController(model, build_bootstrap_skill_library()).decide(
        snapshot, context, request=request,
    )
    assert decision.skill_id == "craft_wood_planks"
    assert decision.model_origin is None
    assert request.snapshot().attempts == () and not model.calls


def test_authority_rewrite_retains_original_digest_and_private_copy_origin() -> None:
    snapshot, context, request = _request()
    model = _BoundModel(_wire(o="Operator-only text without an operator"))
    decision = HighLevelController(model, build_bootstrap_skill_library()).decide(
        snapshot, context, request=request,
    )
    origin = decision.model_origin
    assert origin is not None and decision.say is None
    assert origin.source_decision_sha256 != cognition_decision_sha256(decision)
    cloned = decision.model_copy(update={"instruction": "A runtime-authorized rewrite"})
    assert cloned.model_origin == origin
    assert "model_origin" not in decision.model_dump()
    assert "source_decision_sha256" not in decision.model_dump_json()
    assert "model_origin" not in CognitionDecision.model_json_schema()["properties"]


def test_reused_adapter_response_is_not_mutated_and_legacy_origin_is_cleared() -> None:
    snapshot, context, request = _request()
    model = _BoundModel()
    model.shared_response = _response(_wire())
    controller = HighLevelController(model, build_bootstrap_skill_library())
    bound = controller.decide(snapshot, context, request=request)
    legacy = controller.decide(snapshot, context)
    assert bound.model_origin is not None and legacy.model_origin is None
    assert model.shared_response.request_attempt is None
    assert set(model.shared_response.model_dump()) == {"text", "model", "latency_ms"}


def test_worker_contexts_remain_distinct_and_reset_on_exit() -> None:
    snapshot, context, first = _request("one")
    second = ModelRequestLifecycle(replace(first.binding, request_id="two"))
    barrier = threading.Barrier(2)
    main_thread = threading.get_ident()

    class ConcurrentModel(_BoundModel):
        def complete_bound_constrained(
            self, messages: tuple[ModelMessage, ...], *, name: str,
            schema: dict[str, object], grammar: str, request: RequestBinding, attempt_id: str,
        ) -> ModelResponse:
            assert threading.get_ident() != main_thread
            assert attempt_id == f"{request.request_id}:attempt-1"
            barrier.wait(timeout=2.0)
            return _response(_wire())

    controller = HighLevelController(ConcurrentModel(), build_bootstrap_skill_library())

    def run(request: ModelRequestLifecycle) -> CognitionDecision:
        result = controller.decide(snapshot, context, request=request)
        assert controller._request_context.get() is None
        return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, request) for request in (first, second)]
        results = [future.result(timeout=3.0) for future in futures]
    assert [result.model_origin.request_id for result in results
            if result.model_origin is not None] == ["one", "two"]
    assert controller._request_context.get() is None


def test_worker_context_is_reset_even_for_non_exception_base_error() -> None:
    snapshot, context, request = _request()
    controller = HighLevelController(_BoundModel(KeyboardInterrupt()),
                                     build_bootstrap_skill_library())
    with pytest.raises(KeyboardInterrupt):
        controller.decide(snapshot, context, request=request)
    assert controller._request_context.get() is None
    assert request.snapshot().attempts[0].error_type == "KeyboardInterrupt"


def test_exported_local_lane_shares_lock_and_releases_on_error() -> None:
    assert local_model_inference_available()
    with pytest.raises(ValueError, match="fixture"):
        with local_model_inference_lane():
            assert not local_model_inference_available()
            raise ValueError("fixture")
    assert local_model_inference_available()
