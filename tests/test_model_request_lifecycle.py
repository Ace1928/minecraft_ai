from __future__ import annotations

import concurrent.futures
import hashlib
import threading
from dataclasses import FrozenInstanceError, asdict, replace
from types import SimpleNamespace

import pytest

from minecraft_ai.model_requests import ModelRequestLifecycle, RequestBinding


_DECISION_SHA = "d" * 64


class _Clock:
    def __init__(self, now: int = 100) -> None:
        self.now = now

    def __call__(self) -> int:
        return self.now


def _binding() -> RequestBinding:
    return RequestBinding(
        request_id="request-1", source_sha256="a" * 64, instance_id="world-1",
        frame_id=7, captured_ns=90, operator_revision=2, execution_revision=3,
        submitted_ns=100, deadline_ns=200,
    )


def _request() -> tuple[ModelRequestLifecycle, _Clock]:
    clock = _Clock()
    return ModelRequestLifecycle(_binding(), clock_ns=clock), clock


def _completed() -> tuple[ModelRequestLifecycle, str, _Clock]:
    request, clock = _request()
    attempt = request.start_attempt("cognition_decision")
    clock.now = 120
    request.finish_attempt(attempt)
    clock.now = 130
    request.mark_computation_complete()
    return request, attempt, clock


def test_binding_copies_semantic_snapshot_identity_without_reinterpreting_it() -> None:
    semantic_digest = hashlib.sha256(b'frame identity plus merged semantic facts').hexdigest()
    snapshot = SimpleNamespace(
        source_sha256=semantic_digest, instance_id="world-1", frame_id=7, captured_ns=90,
    )
    binding = RequestBinding.from_snapshot(
        snapshot, request_id="request-1", operator_revision=2, execution_revision=3,
        submitted_ns=100, deadline_ns=200,
    )
    assert binding.source_sha256 == semantic_digest
    assert binding.frame_id == snapshot.frame_id
    assert binding.captured_ns == snapshot.captured_ns
    snapshot.source_sha256 = "b" * 64
    assert binding.source_sha256 == semantic_digest
    with pytest.raises(FrozenInstanceError):
        binding.frame_id = 8


@pytest.mark.parametrize("change", (
    {"source_sha256": "pixel-label"}, {"source_sha256": "A" * 64},
    {"request_id": ""}, {"instance_id": ""}, {"frame_id": True},
    {"operator_revision": -1}, {"execution_revision": False},
    {"captured_ns": 101}, {"deadline_ns": 99}, {"submitted_ns": 1.5},
))
def test_binding_rejects_invalid_identity_revision_and_clock_fields(change) -> None:
    with pytest.raises(ValueError):
        replace(_binding(), **change)


def test_attempts_record_actual_failures_and_repairs_without_accepting_work() -> None:
    request, clock = _request()
    first = request.start_attempt("cognition_decision")
    before = request.snapshot()
    clock.now = 110
    request.finish_attempt(first, error_type="ValueError")
    clock.now = 115
    repair = request.start_attempt("cognition_decision_json_repair")
    clock.now = 125
    request.finish_attempt(repair)
    clock.now = 130
    request.mark_computation_complete()
    result = request.snapshot()

    assert first != repair
    assert before.attempts[0].finished_ns is None
    assert result.attempts[0].started_ns == 100
    assert result.attempts[0].finished_ns == 110
    assert result.attempts[0].error_type == "ValueError"
    assert result.attempts[1].started_ns == 115
    assert result.attempts[1].finished_ns == 125
    assert result.attempts[1].error_type is None
    assert result.computation_completed_ns == 130
    assert result.disposition == "pending"
    assert result.disposition_ns is None
    assert result.selected_attempt_id is None
    assert request.take_discard_notice() is None
    assert "token_count" not in asdict(result)
    with pytest.raises(FrozenInstanceError):
        result.attempts[0].finished_ns = 999


@pytest.mark.parametrize("finished_ns", [20, 25])
def test_late_finished_attempt_is_preserved_when_deadline_refuses_repair(
    finished_ns: int,
) -> None:
    clock = _Clock(10)
    request = ModelRequestLifecycle(
        replace(_binding(), captured_ns=10, submitted_ns=10, deadline_ns=20), clock_ns=clock,
    )
    first = request.start_attempt("cognition_decision")
    clock.now = finished_ns
    request.finish_attempt(first)
    before = request.snapshot()
    with pytest.raises(RuntimeError, match="deadline"):
        request.start_attempt("json-repair")
    assert request.snapshot() == before
    assert len(before.attempts) == 1
    assert before.attempts[0].started_ns == 10
    assert before.attempts[0].finished_ns == finished_ns
    assert before.attempts[0].error_type is None
    assert before.disposition == "pending" and before.computation_completed_ns is None
    assert request.take_discard_notice() is None
    request.mark_computation_complete()
    assert request.reject("deadline_before_publication")
    assert request.snapshot().attempts == before.attempts
    assert request.snapshot().computation_completed_ns == finished_ns
    assert request.take_discard_notice() == "deadline_before_publication"
    assert request.take_discard_notice() is None


@pytest.mark.parametrize("started_ns", [20, 25])
def test_expired_initial_attempt_refusal_does_not_invent_work_or_disposition(
    started_ns: int,
) -> None:
    clock = _Clock(started_ns)
    request = ModelRequestLifecycle(
        replace(_binding(), captured_ns=10, submitted_ns=10, deadline_ns=20), clock_ns=clock,
    )
    before = request.snapshot()
    with pytest.raises(RuntimeError, match="deadline"):
        request.start_attempt("cognition_decision")
    assert request.snapshot() == before
    assert before.attempts == () and before.disposition == "pending"
    assert before.computation_completed_ns is None


def test_rejection_during_work_does_not_fabricate_finish_or_erase_spent_work() -> None:
    request, clock = _request()
    attempt = request.start_attempt("cognition_decision")
    clock.now = 105
    assert request.reject("operator_revision_changed")
    rejected = request.snapshot()
    assert rejected.disposition_ns == 105
    assert rejected.computation_completed_ns is None
    assert rejected.attempts[0].finished_ns is None
    assert request.take_discard_notice() is None
    clock.now = 110
    assert not request.reject("replacement_reason")
    with pytest.raises(RuntimeError, match="cannot start"):
        request.start_attempt("unwanted_repair")

    clock.now = 120
    request.finish_attempt(attempt, "TimeoutError")
    clock.now = 125
    request.mark_computation_complete()
    result = request.snapshot()
    assert result.disposition == "rejected"
    assert result.reason == "operator_revision_changed"
    assert result.disposition_ns == 105
    assert result.attempts[0].finished_ns == 120
    assert result.attempts[0].error_type == "TimeoutError"
    assert result.computation_completed_ns == 125
    assert request.take_discard_notice() == "operator_revision_changed"
    assert request.take_discard_notice() is None


def test_work_completion_requires_finished_attempts_and_is_idempotent() -> None:
    request, clock = _request()
    attempt = request.start_attempt("cognition_decision")
    with pytest.raises(RuntimeError, match="still running"):
        request.mark_computation_complete()
    clock.now = 110
    request.finish_attempt(attempt)
    clock.now = 120
    request.mark_computation_complete()
    clock.now = 140
    request.mark_computation_complete()
    assert request.snapshot().computation_completed_ns == 120
    with pytest.raises(RuntimeError, match="cannot start"):
        request.start_attempt("late_attempt")


def test_attempt_finish_is_repeatable_but_cannot_rewrite_its_error() -> None:
    request, clock = _request()
    attempt = request.start_attempt("model")
    clock.now = 110
    request.finish_attempt(attempt, "TimeoutError")
    clock.now = 150
    request.finish_attempt(attempt, "TimeoutError")
    assert request.snapshot().attempts[0].finished_ns == 110
    with pytest.raises(ValueError, match="cannot be rewritten"):
        request.finish_attempt(attempt)
    with pytest.raises(ValueError, match="unknown"):
        request.finish_attempt("other-request:attempt-1")


def test_acceptance_requires_completed_work_and_a_successful_selected_attempt() -> None:
    request, clock = _request()
    attempt = request.start_attempt("model")
    published = []
    with pytest.raises(RuntimeError, match="not complete"):
        request.accept(attempt, _DECISION_SHA, lambda: published.append(True))
    clock.now = 120
    request.finish_attempt(attempt, "RuntimeError")
    request.mark_computation_complete()
    for selected in (attempt, "missing"):
        with pytest.raises(ValueError, match="completed successful"):
            request.accept(selected, _DECISION_SHA, lambda: published.append(True))
    assert not published
    assert request.snapshot().disposition == "pending"


def test_successful_acceptance_selects_identity_and_publishes_only_once() -> None:
    request, attempt, clock = _completed()
    published = []

    def publish() -> None:
        # Reentrant read is safe, but success has not been asserted yet.
        assert request.snapshot().disposition == "pending"
        published.append("published")
        clock.now = 150

    assert request.accept(attempt, _DECISION_SHA, publish)
    result = request.snapshot()
    assert result.disposition == "accepted"
    assert result.disposition_ns == 150
    assert result.selected_attempt_id == attempt
    assert result.final_decision_sha256 == _DECISION_SHA
    assert request.accept(attempt, _DECISION_SHA, publish)
    assert not request.accept(attempt, "e" * 64, publish)
    assert published == ["published"]
    assert not request.reject("late_rejection")
    assert request.take_discard_notice() is None


def test_rejected_completed_work_cannot_execute_publication_callback() -> None:
    request, attempt, _clock = _completed()
    request.reject("frame_changed")

    def forbidden_publication() -> None:
        raise AssertionError("discarded work cannot be published")

    assert not request.accept(attempt, _DECISION_SHA, forbidden_publication)
    assert request.take_discard_notice() == "frame_changed"


def test_publication_exception_is_terminal_and_exposes_only_its_type() -> None:
    request, attempt, clock = _completed()
    error = RuntimeError("private callback details must remain outside the generic receipt")
    calls = []

    def fail() -> None:
        calls.append(True)
        clock.now = 145
        raise error

    with pytest.raises(RuntimeError) as caught:
        request.accept(attempt, _DECISION_SHA, fail)
    assert caught.value is error
    snapshot = request.snapshot()
    assert snapshot.disposition == "publication_failed"
    assert snapshot.publication_error_type == "RuntimeError"
    assert snapshot.disposition_ns == 145
    assert snapshot.reason == "publication_failed"
    assert "private callback details" not in repr(snapshot)
    assert not request.accept(attempt, _DECISION_SHA, fail)
    assert calls == [True]
    assert request.take_discard_notice() == "publication_failed"
    assert request.take_discard_notice() is None


def test_unavailable_terminal_clock_cannot_reopen_already_published_work() -> None:
    request, attempt, clock = _completed()
    calls = []

    def publish() -> None:
        calls.append(True)
        clock.now = -1

    assert request.accept(attempt, _DECISION_SHA, publish)
    assert request.snapshot().disposition == "accepted"
    assert request.snapshot().disposition_ns is None
    assert request.accept(attempt, _DECISION_SHA, publish)
    assert calls == [True]


def test_two_discard_paths_can_claim_only_one_notice() -> None:
    request, _attempt, _clock = _completed()
    request.reject("execution_revision_changed")
    barrier = threading.Barrier(2)

    def claim() -> str | None:
        barrier.wait(timeout=1)
        return request.take_discard_notice()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        left, right = pool.submit(claim), pool.submit(claim)
        notices = [left.result(timeout=1), right.result(timeout=1)]
    assert notices.count("execution_revision_changed") == 1
    assert notices.count(None) == 1


def test_accept_callback_and_rejection_are_serialized() -> None:
    request, attempt, _clock = _completed()
    entered, release, rejecting = threading.Event(), threading.Event(), threading.Event()
    calls = []

    def publish() -> None:
        entered.set()
        assert release.wait(timeout=1)
        calls.append(True)

    def reject() -> bool:
        rejecting.set()
        return request.reject("competing_rejection")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        accepted = pool.submit(request.accept, attempt, _DECISION_SHA, publish)
        try:
            assert entered.wait(timeout=1)
            rejected = pool.submit(reject)
            assert rejecting.wait(timeout=1)
            assert not rejected.done()
        finally:
            release.set()
        assert accepted.result(timeout=1)
        assert not rejected.result(timeout=1)
    assert calls == [True]
    assert request.snapshot().disposition == "accepted"


def test_reentrant_callback_cannot_reject_or_publish_a_second_time() -> None:
    request, attempt, _clock = _completed()

    def publish() -> None:
        assert not request.reject("too_late_inside_publication")
        with pytest.raises(RuntimeError, match="reenter"):
            request.accept(attempt, _DECISION_SHA, lambda: None)

    assert request.accept(attempt, _DECISION_SHA, publish)


def test_failed_attempt_and_clock_validation_do_not_invent_accounting() -> None:
    request, clock = _request()
    attempt = request.start_attempt("model")
    clock.now = 99
    with pytest.raises(ValueError, match="precedes submission"):
        request.finish_attempt(attempt)
    assert request.snapshot().attempts[0].finished_ns is None
    assert request.snapshot().computation_completed_ns is None


def test_completion_callback_can_finish_and_discard_without_any_publish_hook() -> None:
    request, clock = _request()
    attempt = request.start_attempt("model")
    future: concurrent.futures.Future[str] = concurrent.futures.Future()
    discarded = []

    def done(completed: concurrent.futures.Future[str]) -> None:
        failure = completed.exception()
        request.finish_attempt(attempt, None if failure is None else type(failure).__name__)
        request.mark_computation_complete()
        notice = request.take_discard_notice()
        if notice is not None:
            discarded.append(notice)

    future.add_done_callback(done)
    request.reject("request_retired")
    clock.now = 150
    future.set_result("unused model candidate")
    assert discarded == ["request_retired"]
    assert request.snapshot().disposition == "rejected"
    assert request.snapshot().computation_completed_ns == 150
    assert request.snapshot().selected_attempt_id is None
    assert request.take_discard_notice() is None
