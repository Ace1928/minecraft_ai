from __future__ import annotations

import base64
import importlib
import threading
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


_LOCAL_MODEL_INFERENCE_LOCK = threading.Lock()


class ModelMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str = Field(pattern="^(system|user|assistant)$")
    content: str


class ModelResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    model: str
    latency_ms: float = Field(ge=0.0)


class LanguageModel(Protocol):
    model_id: str

    def complete(self, messages: tuple[ModelMessage, ...]) -> ModelResponse: ...


class VisionLanguageModel(Protocol):
    model_id: str

    def inspect(self, prompt: str, *, image_bytes: bytes, mime_type: str) -> ModelResponse: ...


@dataclass
class OpenAICompatibleLocalModel:
    """Adapter for local llama.cpp/vLLM/Ollama-compatible chat endpoints.

    The base URL is explicit and defaults to loopback. The adapter refuses
    non-loopback HTTP unless `allow_remote=True`; local-first operation must not
    accidentally turn an API-compatible configuration into a cloud dependency.
    """

    model_id: str
    base_url: str = "http://127.0.0.1:8080/v1"
    api_key: str = "local"
    timeout_s: float = 60.0
    allow_remote: bool = False

    def __post_init__(self) -> None:
        lowered = self.base_url.lower()
        if not self.allow_remote and not any(
            host in lowered for host in ("127.0.0.1", "localhost", "[::1]")
        ):
            raise ValueError("local model adapter refuses non-loopback base_url")

    def _client(self) -> Any:
        try:
            httpx = importlib.import_module("httpx")
        except ImportError as exc:
            raise RuntimeError("install minecraft-ai[knowledge] for HTTP model adapters") from exc
        return httpx.Client(timeout=self.timeout_s)

    def complete(self, messages: tuple[ModelMessage, ...]) -> ModelResponse:
        return self._complete(messages)

    def complete_structured(
        self,
        messages: tuple[ModelMessage, ...],
        *,
        name: str,
        schema: dict[str, object],
    ) -> ModelResponse:
        return self._complete(
            messages,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": name, "strict": True, "schema": schema},
            },
        )

    def _complete(
        self,
        messages: tuple[ModelMessage, ...],
        *,
        response_format: dict[str, object] | None = None,
    ) -> ModelResponse:
        import time

        started = time.perf_counter()
        payload = {
            "model": self.model_id,
            "messages": [message.model_dump(mode="json") for message in messages],
            "temperature": 0.2,
            "max_tokens": 512,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        # Multiple local llama.cpp servers may share one GPU. Concurrent VLM
        # prefill and strategic decoding caused both requests to take roughly
        # six times longer on the managed machine. Serialize local inference
        # at the process boundary while capture and motor loops remain async.
        with _LOCAL_MODEL_INFERENCE_LOCK:
            with self._client() as client:
                response = client.post(
                    self.base_url.rstrip("/") + "/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                )
                response.raise_for_status()
                raw = response.json()
        text = _extract_chat_text(raw)
        return ModelResponse(
            text=text,
            model=str(raw.get("model", self.model_id)) if isinstance(raw, dict) else self.model_id,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def inspect(
        self,
        prompt: str,
        *,
        image_bytes: bytes,
        mime_type: str = "image/png",
    ) -> ModelResponse:
        return self._inspect(
            prompt,
            image_bytes=image_bytes,
            mime_type=mime_type,
        )

    def inspect_structured(
        self,
        prompt: str,
        *,
        image_bytes: bytes,
        mime_type: str,
        name: str,
        schema: dict[str, object],
    ) -> ModelResponse:
        return self._inspect(
            prompt,
            image_bytes=image_bytes,
            mime_type=mime_type,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": name, "strict": True, "schema": schema},
            },
        )

    def _inspect(
        self,
        prompt: str,
        *,
        image_bytes: bytes,
        mime_type: str,
        response_format: dict[str, object] | None = None,
    ) -> ModelResponse:
        import time

        started = time.perf_counter()
        encoded = base64.b64encode(image_bytes).decode("ascii")
        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                        },
                    ],
                }
            ],
            "temperature": 0.1,
            "max_tokens": 512,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        with _LOCAL_MODEL_INFERENCE_LOCK:
            with self._client() as client:
                response = client.post(
                    self.base_url.rstrip("/") + "/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                )
                response.raise_for_status()
                raw = response.json()
        text = _extract_chat_text(raw)
        return ModelResponse(
            text=text,
            model=str(raw.get("model", self.model_id)) if isinstance(raw, dict) else self.model_id,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )


def _extract_chat_text(raw: Any) -> str:
    if not isinstance(raw, dict):
        raise RuntimeError("model returned non-object JSON")
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("model returned no choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise RuntimeError("model returned malformed choice")
    message = first.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("model returned malformed message")
    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError("model returned non-text content")
    return content
