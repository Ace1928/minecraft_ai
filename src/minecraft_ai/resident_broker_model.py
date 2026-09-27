"""Explicit shared-resident adapter with the runtime's existing authority hooks."""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import ModelConfig
from .model_requests import RequestBinding
from .models import (
    ModelMessage,
    ModelResponse,
    OpenAICompatibleLocalModel,
    local_model_inference_lane,
)
from .resident_broker_transport import UPSTREAM_MODEL, BrokerError, checked_completion, exchange


@dataclass
class _Bound:
    binding: RequestBinding
    canceled: threading.Event = field(default_factory=threading.Event)
    attempts: dict[str, str | None] = field(default_factory=dict)
    accepted: tuple[str, str] | None = None


@dataclass
class BrokeredLocalModel(OpenAICompatibleLocalModel):
    broker_socket: str = ""
    purpose: str = "cognition"
    _scope: threading.local = field(default_factory=threading.local, init=False, repr=False)
    _bindings: dict[str, _Bound] = field(default_factory=dict, init=False, repr=False)
    _bindings_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        path = Path(self.broker_socket)
        if (
            self.model_id != UPSTREAM_MODEL
            or not path.is_absolute()
            or ".." in path.parts
            or len(self.broker_socket.encode()) > 107
            or self.purpose not in {"cognition", "vision"}
        ):
            raise ValueError("explicit matching resident identity and private socket required")

    @contextmanager
    def request_scope(
        self, *, deadline_ns: int | None = None, cancel_requested: Callable[[], bool] | None = None
    ) -> Iterator[None]:
        """One absolute budget covers local wait, discovery, fallback and network."""
        previous = getattr(self._scope, "request", None)
        deadline = time.monotonic_ns() + int(self.timeout_s * 1e9)
        if deadline_ns is not None:
            deadline = min(deadline, deadline_ns)
        canceled = cancel_requested or (lambda: False)
        if previous is not None:
            deadline = min(deadline, previous[0])
            inner_cancel = canceled

            def nested_cancel() -> bool:
                return bool(previous[1]()) or inner_cancel()

            canceled = nested_cancel
        self._scope.request = (deadline, canceled)
        try:
            yield
        finally:
            if previous is None:
                del self._scope.request
            else:
                self._scope.request = previous

    def _exchange(self, operation: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        deadline, canceled = self._scope.request
        return exchange(
            Path(self.broker_socket),
            operation=operation,
            deadline_ns=deadline,
            cancel_requested=canceled,
            purpose=self.purpose,
            payload=payload,
        )

    def _request_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.request_scope():
            deadline, canceled = self._scope.request
            with local_model_inference_lane(deadline_ns=deadline, cancel_requested=canceled):
                result = self._exchange("infer", payload)
                return checked_completion(result.get("result"), max_tokens=self.max_tokens)

    def _llama_grammar_available(self) -> bool:
        if self._grammar_supported is None:
            with self.request_scope():
                result = self._exchange("capabilities")
            if type(result.get("grammar")) is not bool:
                raise BrokerError("invalid private grammar capability")
            self._grammar_supported = result["grammar"]
        return bool(self._grammar_supported)

    def complete_constrained(
        self,
        messages: tuple[ModelMessage, ...],
        *,
        name: str,
        schema: dict[str, object],
        grammar: str,
    ) -> ModelResponse:
        with self.request_scope():
            return super().complete_constrained(messages, name=name, schema=schema, grammar=grammar)

    def inspect_constrained(
        self,
        prompt: str,
        *,
        image_bytes: bytes,
        mime_type: str,
        name: str,
        schema: dict[str, object],
        grammar: str,
    ) -> ModelResponse:
        with self.request_scope():
            return super().inspect_constrained(
                prompt,
                image_bytes=image_bytes,
                mime_type=mime_type,
                name=name,
                schema=schema,
                grammar=grammar,
            )

    def _binding(self, request: RequestBinding) -> _Bound:
        if type(request) is not RequestBinding:
            raise TypeError("request binding required")
        now = time.monotonic_ns()
        for key, old in tuple(self._bindings.items()):
            if old.binding.deadline_ns <= now:
                old.canceled.set()
                del self._bindings[key]
        current = self._bindings.get(request.request_id)
        if current is None:
            if now >= request.deadline_ns or len(self._bindings) >= 4:
                raise TimeoutError("bound model request expired or capacity exhausted")
            current = self._bindings[request.request_id] = _Bound(request)
        if current.binding != request:
            raise ValueError("bound request identity changed")
        return current

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
        if type(attempt_id) is not str or not 0 < len(attempt_id) <= 256:
            raise ValueError("invalid bound attempt identity")
        with self._bindings_lock:
            state = self._binding(request)
            if (
                state.canceled.is_set()
                or state.accepted is not None
                or attempt_id in state.attempts
            ):
                raise RuntimeError("bound request cannot dispatch")
            if len(state.attempts) >= 8:
                raise RuntimeError("bound attempt budget exhausted")
            state.attempts[attempt_id] = None
        with self.request_scope(
            deadline_ns=request.deadline_ns, cancel_requested=state.canceled.is_set
        ):
            response = self.complete_constrained(
                messages, name=name, schema=schema, grammar=grammar
            )
        # Bind the existing parser's source digest, while preserving its ordinary
        # repair behavior for malformed output. No game publication occurs here.
        from .cognition.repair import _parse_decision
        from .cognition.types import cognition_decision_sha256

        try:
            digest = cognition_decision_sha256(_parse_decision(response.text))
        except (ValueError, RuntimeError):
            digest = None
        with self._bindings_lock:
            if state.canceled.is_set() or time.monotonic_ns() >= request.deadline_ns:
                raise TimeoutError("bound model request retired")
            state.attempts[attempt_id] = digest
        return response

    def admit_bound_decision(
        self,
        *,
        request: RequestBinding,
        attempt_id: str,
        source_decision_sha256: str,
        final_decision: dict[str, object],
        final_decision_sha256: str,
        rewritten: bool,
    ) -> None:
        """Short adapter identity check, inside the runtime's authority transaction."""
        from .cognition.types import CognitionDecision, cognition_decision_sha256

        with self._bindings_lock:
            state = self._binding(request)
            if (
                state.canceled.is_set()
                or state.attempts.get(attempt_id) is None
                or state.attempts[attempt_id] != source_decision_sha256
                or type(rewritten) is not bool
                or rewritten != (source_decision_sha256 != final_decision_sha256)
                or re.fullmatch(r"[0-9a-f]{64}", final_decision_sha256) is None
                or cognition_decision_sha256(CognitionDecision.model_validate(final_decision))
                != final_decision_sha256
                or state.accepted not in (None, (attempt_id, final_decision_sha256))
            ):
                raise RuntimeError("bound decision identity or authority changed")
            state.accepted = (attempt_id, final_decision_sha256)

    def discard_bound_request(self, *, request: RequestBinding, reason: str) -> None:
        # Tombstone before an executor has started also fences its later call.
        with self._bindings_lock:
            if time.monotonic_ns() < request.deadline_ns:
                self._binding(request).canceled.set()


def configured_model(config: ModelConfig, *, purpose: str) -> OpenAICompatibleLocalModel:
    options = {
        name: getattr(config, name)
        for name in (
            "model_id",
            "base_url",
            "api_key",
            "timeout_s",
            "max_tokens",
            "thinking_budget_tokens",
            "reasoning_format",
        )
    }
    if config.broker_socket is None:
        return OpenAICompatibleLocalModel(**options)
    return BrokeredLocalModel(**options, broker_socket=config.broker_socket, purpose=purpose)
