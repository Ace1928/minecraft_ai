from __future__ import annotations

import concurrent.futures
import threading
import time
from typing import Any

from minecraft_ai.models import ModelMessage, OpenAICompatibleLocalModel


class _Response:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {"model": "test", "choices": [{"message": {"content": "ok"}}]}


class _TrackingClient:
    def __init__(self, state: dict[str, Any], guard: threading.Lock) -> None:
        self.state = state
        self.guard = guard

    def __enter__(self) -> _TrackingClient:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def post(self, *_args: object, **_kwargs: object) -> _Response:
        self.state["json"] = _kwargs.get("json")
        with self.guard:
            self.state["active"] += 1
            self.state["peak"] = max(self.state["peak"], self.state["active"])
        time.sleep(0.02)
        with self.guard:
            self.state["active"] -= 1
        return _Response()


def test_local_model_calls_serialize_shared_gpu_work(monkeypatch: Any) -> None:
    state = {"active": 0, "peak": 0}
    guard = threading.Lock()

    def client(_model: OpenAICompatibleLocalModel) -> _TrackingClient:
        return _TrackingClient(state, guard)

    monkeypatch.setattr(OpenAICompatibleLocalModel, "_client", client)
    models = (
        OpenAICompatibleLocalModel(model_id="planner"),
        OpenAICompatibleLocalModel(model_id="vision", base_url="http://127.0.0.1:8081/v1"),
    )
    message = (ModelMessage(role="user", content="test"),)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(model.complete, message) for model in models]
        assert [future.result().text for future in futures] == ["ok", "ok"]

    assert state["peak"] == 1


def test_local_model_uses_bounded_configured_generation_budget(monkeypatch: Any) -> None:
    state: dict[str, Any] = {"active": 0, "peak": 0}
    guard = threading.Lock()

    def client(_model: OpenAICompatibleLocalModel) -> _TrackingClient:
        return _TrackingClient(state, guard)

    monkeypatch.setattr(OpenAICompatibleLocalModel, "_client", client)
    model = OpenAICompatibleLocalModel(
        model_id="planner",
        max_tokens=96,
        thinking_budget_tokens=0,
        reasoning_format="none",
    )
    model.complete((ModelMessage(role="user", content="decide"),))

    assert state["json"]["max_tokens"] == 96
    assert state["json"]["thinking_budget_tokens"] == 0
    assert state["json"]["reasoning_format"] == "none"


def test_local_model_omits_backend_specific_reasoning_controls_by_default(
    monkeypatch: Any,
) -> None:
    state: dict[str, Any] = {"active": 0, "peak": 0}
    guard = threading.Lock()

    def client(_model: OpenAICompatibleLocalModel) -> _TrackingClient:
        return _TrackingClient(state, guard)

    monkeypatch.setattr(OpenAICompatibleLocalModel, "_client", client)
    OpenAICompatibleLocalModel(model_id="portable").complete(
        (ModelMessage(role="user", content="decide"),)
    )

    assert "thinking_budget_tokens" not in state["json"]
    assert "reasoning_format" not in state["json"]


def test_local_model_uses_native_grammar_for_constrained_completion(
    monkeypatch: Any,
) -> None:
    state: dict[str, Any] = {"active": 0, "peak": 0}
    guard = threading.Lock()

    def client(_model: OpenAICompatibleLocalModel) -> _TrackingClient:
        return _TrackingClient(state, guard)

    monkeypatch.setattr(OpenAICompatibleLocalModel, "_client", client)
    model = OpenAICompatibleLocalModel(model_id="planner")
    model._grammar_supported = True

    model.complete_constrained(
        (ModelMessage(role="user", content="decide"),),
        name="decision",
        schema={"type": "object"},
        grammar='root ::= "{}"',
    )

    assert state["json"]["grammar"] == 'root ::= "{}"'
    assert "response_format" not in state["json"]
