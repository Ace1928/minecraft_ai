"""Planned shutdown fences semantic output without touching a model process."""

from __future__ import annotations

import threading
import time

import pytest

import minecraft_ai.perception.service as service_module
from minecraft_ai.perception.service import (
    ActiveVLMWorker,
    RealtimePerceptionService,
    SemanticJob,
    SemanticObservation,
)
from minecraft_ai.perception.types import (
    ActivePerceptionQuery,
    FrameState,
    PerceptionBlackboard,
)
from minecraft_ai.platforms.bedrock_x11 import CapturedFrame


class _UnusedModel:
    model_id = "test:never-invoked"


def _worker_and_job() -> tuple[ActiveVLMWorker, SemanticJob]:
    now = time.monotonic_ns()
    board = PerceptionBlackboard()
    board.publish(FrameState(
        frame_id=1, captured_ns=now, instance_id="test", width=9, height=8,
    ))
    worker = ActiveVLMWorker(_UnusedModel(), board, "test")
    job = SemanticJob(
        query=ActivePerceptionQuery(
            query_id="q-retire", question="obstacle.ahead", frame_id=1,
            output_keys=("obstacle.ahead",),
        ),
        frame=CapturedFrame(
            frame_id=1, captured_ns=now, width=9, height=8, bgra=bytes(9 * 8 * 4),
        ),
        frame_dhash="0" * 16,
    )
    return worker, job


def _observation() -> SemanticObservation:
    return SemanticObservation(
        facts={"obstacle.ahead": True}, confidences={"obstacle.ahead": 0.95},
    )


def test_retired_inflight_semantics_cannot_publish_or_admit_more_work(monkeypatch) -> None:
    worker, job = _worker_and_job()
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def inspect(actual_job):
        calls.append(actual_job.query.query_id)
        entered.set()
        assert release.wait(2)
        return _observation(), 1.0

    monkeypatch.setattr(worker, "_inspect", inspect)
    worker.start()
    try:
        assert worker.submit(job)
        assert entered.wait(2)
        assert worker.stop(deadline_ns=time.monotonic_ns()) is False
        assert not worker.submit(job)
        assert not worker.available()
        release.set()
        assert worker.stop(deadline_ns=time.monotonic_ns() + 2_000_000_000)
        assert worker.blackboard.fact("obstacle.ahead") is None
        assert calls == ["q-retire"]
        assert worker.metrics.completed == 1  # Completion is not publication.
    finally:
        release.set()
        worker.stop()


def test_retirement_after_dequeue_does_not_start_inspection(monkeypatch) -> None:
    worker, job = _worker_and_job()
    assert worker.submit(job)

    def get(*, timeout):
        worker._stop.set()
        return job

    def forbidden(_job):
        raise AssertionError("retired queued job must not invoke a model")

    monkeypatch.setattr(worker._jobs, "get", get)
    monkeypatch.setattr(worker, "_inspect", forbidden)
    worker._run()
    assert not worker._busy.is_set()
    assert worker._work_admitted is False
    assert worker.metrics.completed == 0


def test_retirement_between_publication_preparation_and_merge_is_fenced(monkeypatch) -> None:
    worker, job = _worker_and_job()
    original_latest = worker.blackboard.raw_latest

    def retire_then_latest():
        assert worker.stop(deadline_ns=time.monotonic_ns())
        return original_latest()

    monkeypatch.setattr(worker.blackboard, "raw_latest", retire_then_latest)
    worker._publish(job, _observation())
    assert worker.blackboard.fact("obstacle.ahead") is None
    assert worker.blackboard.fact("scene.observation_dhash") is None


@pytest.mark.parametrize("remaining_ns", [0, -1, 125_000_000])
def test_stop_only_joins_for_remaining_absolute_budget(monkeypatch, remaining_ns) -> None:
    worker, _ = _worker_and_job()
    waits = []

    class Thread:
        def join(self, *, timeout):
            waits.append(timeout)

        def is_alive(self):
            return True

    worker._thread = Thread()
    monkeypatch.setattr(service_module.time, "monotonic_ns", lambda: 1_000_000_000)
    assert worker.stop(deadline_ns=1_000_000_000 + remaining_ns) is False
    assert waits == [max(0, remaining_ns) / 1e9]
    assert worker._stop.is_set()


def test_no_argument_stop_retains_default_two_second_wait(monkeypatch) -> None:
    worker, _ = _worker_and_job()
    waits = []

    class Thread:
        def join(self, *, timeout):
            waits.append(timeout)

        def is_alive(self):
            return False

    worker._thread = Thread()
    monkeypatch.setattr(service_module.time, "monotonic_ns", lambda: 1_000_000_000)
    assert worker.stop()
    assert waits == [2.0]


@pytest.mark.parametrize("deadline", [True, 0.5, float("nan"), "later"])
def test_invalid_deadline_cannot_begin_retirement(deadline) -> None:
    worker, _ = _worker_and_job()
    with pytest.raises(ValueError, match="deadline"):
        worker.stop(deadline_ns=deadline)
    assert not worker._stop.is_set()


@pytest.mark.parametrize("result", [True, False, RuntimeError("worker stop failed")])
def test_service_always_closes_capture_with_shared_deadline(result) -> None:
    calls = []

    class Worker:
        def stop(self, *, deadline_ns):
            calls.append(("worker", deadline_ns))
            if isinstance(result, Exception):
                raise result
            return result

    class Capture:
        def close(self):
            calls.append(("capture", None))

    service = RealtimePerceptionService(
        capture_source=Capture(), blackboard=PerceptionBlackboard(),
        instance_id="test", active_vlm=Worker(),
    )
    if isinstance(result, Exception):
        with pytest.raises(RuntimeError, match="worker stop failed"):
            service.close(deadline_ns=123)
    else:
        assert service.close(deadline_ns=123) is result
    assert calls == [("worker", 123), ("capture", None)]


def test_service_default_close_accepts_legacy_worker_stop_signature() -> None:
    calls = []

    class Worker:
        def stop(self):
            calls.append("worker")
            return True

    class Capture:
        def close(self):
            calls.append("capture")

    service = RealtimePerceptionService(
        capture_source=Capture(), blackboard=PerceptionBlackboard(),
        instance_id="test", active_vlm=Worker(),
    )
    assert service.close()
    assert calls == ["worker", "capture"]
