from __future__ import annotations

import pytest

from minecraft_ai.config import ModelConfig, RuntimeConfig
from minecraft_ai.models import OpenAICompatibleLocalModel
from minecraft_ai.resident_broker_model import configured_model_services


def test_native_world_route_does_not_start_a_dense_model_service() -> None:
    config = RuntimeConfig(
        high_level=ModelConfig(
            enabled=True,
            provider="erais-native-world",
            model_id="erais-native-qwen3",
            base_url="http://127.0.0.1:8771/v1",
            native_world_token_file="/tmp/world/token",
            native_world_ready_file="/tmp/world/ready.json",
        )
    )

    assert configured_model_services(config) == ()


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8081/v1",
        "http://localhost:8081/v1/",
        "http://[::1]:8081/v1",
    ],
)
def test_enabled_dense_compat_route_requests_its_configured_local_service(url: str) -> None:
    config = RuntimeConfig(
        high_level=ModelConfig(enabled=True, model_id="gemma-local", base_url=url)
    )

    assert configured_model_services(config) == (
        "minecraft-ai-vlm-gemma4-vulkan-all.service",
    )


def test_disabled_or_brokered_dense_route_does_not_start_the_direct_model_service() -> None:
    disabled = RuntimeConfig(
        high_level=ModelConfig(enabled=False, model_id="gemma-local", base_url="http://127.0.0.1:8081/v1")
    )
    brokered = RuntimeConfig(
        high_level=ModelConfig(
            enabled=True,
            model_id="gemma-local",
            base_url="http://127.0.0.1:8081/v1",
            broker_socket="/run/user/1000/erais/model.sock",
        )
    )

    assert configured_model_services(disabled) == ()
    assert configured_model_services(brokered) == ()


def test_openai_compatible_readiness_requires_exact_registry_identity(monkeypatch) -> None:
    model = OpenAICompatibleLocalModel(
        model_id="configured-model",
        base_url="http://127.0.0.1:8081/v1",
        api_key="private-local-key",
    )
    seen = []

    class Response:
        def __init__(self, body):
            self.body = body

        def raise_for_status(self):
            return None

        def json(self):
            return self.body

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def get(self, url, **kwargs):
            seen.append((url, kwargs))
            return Response({"data": [{"id": "configured-model"}]})

    monkeypatch.setattr(model, "_client", lambda: Client())

    assert model.verify_ready() == "configured-model"
    assert seen[0][0].endswith("/models")
    assert seen[0][1]["headers"]["Authorization"] == "Bearer private-local-key"


def test_openai_compatible_readiness_rejects_a_different_model(monkeypatch) -> None:
    model = OpenAICompatibleLocalModel(
        model_id="configured-model", base_url="http://127.0.0.1:8081/v1"
    )

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"id": "other-model"}]}

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def get(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setattr(model, "_client", lambda: Client())

    with pytest.raises(RuntimeError, match="absent from the model registry"):
        model.verify_ready()
