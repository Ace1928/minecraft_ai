"""Private broker integration with a fake Unix peer; no provider/model calls."""

from __future__ import annotations

import base64
import json
import socketserver
import struct
import threading
import time

import pytest

from minecraft_ai.cognition.repair import _parse_decision
from minecraft_ai.cognition.types import cognition_decision_sha256
from minecraft_ai.config import ModelConfig
from minecraft_ai.model_requests import RequestBinding
from minecraft_ai.models import ModelMessage, OpenAICompatibleLocalModel, local_model_inference_lane
from minecraft_ai.resident_broker_model import BrokeredLocalModel, configured_model
from minecraft_ai.resident_broker_transport import CONTRACT, MODEL_ID, UPSTREAM_MODEL, BrokerError


MESSAGES = (ModelMessage(role="user", content="fixture command"),)
OUTPUT = '{"r":"fixture","p":{}}'


@pytest.fixture
def broker(tmp_path):
    directory = tmp_path / "private"
    directory.mkdir(mode=0o700)
    path = directory / "model.sock"
    state = {
        "requests": [],
        "blocked": False,
        "entered": threading.Event(),
        "disconnected": threading.Event(),
        "identity": MODEL_ID,
        "result_model": UPSTREAM_MODEL,
        "grammar": True,
        "status": 200,
        "usage": {"prompt_tokens": 4, "completion_tokens": 8, "total_tokens": 12},
    }

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            def read(n):
                data = b""
                while len(data) < n:
                    part = self.request.recv(n - len(data))
                    if not part:
                        raise EOFError()
                    data += part
                return data

            message = json.loads(read(struct.unpack("!I", read(4))[0]))
            state["requests"].append(message)
            reply = {
                "contract": CONTRACT,
                "request_id": message["request_id"],
                "model_id": state["identity"],
                "upstream_model": UPSTREAM_MODEL,
                "ok": True,
            }
            if message["operation"] == "capabilities":
                reply.update(grammar=state["grammar"], max_tokens=2048)
            else:
                state["entered"].set()
                if state["blocked"]:
                    self.request.settimeout(2)
                    try:
                        if self.request.recv(1) == b"":
                            state["disconnected"].set()
                    except OSError:
                        pass
                    return
                reply["result"] = {
                    "model": state["result_model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": OUTPUT},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": state["usage"],
                }
                if state["status"] != 200 and "grammar" in message["payload"]:
                    reply.update(ok=False, http_status=state["status"], error="owner_unavailable")
            raw = json.dumps(reply).encode()
            try:
                self.request.sendall(struct.pack("!I", len(raw)) + raw)
            except OSError:
                pass

    class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
        daemon_threads = True
        block_on_close = False

    server = Server(str(path), Handler)
    path.chmod(0o600)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    model = BrokeredLocalModel(
        model_id=UPSTREAM_MODEL,
        broker_socket=str(path),
        timeout_s=1,
        max_tokens=256,
        thinking_budget_tokens=0,
        reasoning_format="none",
    )
    try:
        yield model, state, path
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def constrained(model):
    return model.complete_constrained(
        MESSAGES, name="test", schema={"type": "object"}, grammar='root ::= "ok"'
    )


def binding():
    now = time.monotonic_ns()
    return RequestBinding(
        request_id="fixture-request",
        source_sha256="a" * 64,
        instance_id="fixture-world",
        frame_id=1,
        captured_ns=now,
        operator_revision=3,
        execution_revision=4,
        submitted_ns=now,
        deadline_ns=now + 2_000_000_000,
    )


def test_opt_in_preserves_existing_configuration_and_direct_adapter():
    config = ModelConfig(enabled=True, model_id="custom", max_tokens=96, timeout_s=42)
    model = configured_model(config, purpose="cognition")
    assert type(model) is OpenAICompatibleLocalModel and model.max_tokens == 96
    assert config.model_dump()["broker_socket"] is None
    with pytest.raises(ValueError):
        configured_model(
            config.model_copy(update={"broker_socket": "/private/model.sock"}), purpose="cognition"
        )


def test_grammar_tokens_and_reasoning_preserved_without_credentials_or_direct_http(
    broker, monkeypatch
):
    model, seen, _ = broker
    monkeypatch.setattr(model, "_client", lambda: pytest.fail("direct HTTP bypass"))
    assert constrained(model).text == OUTPUT
    assert len(seen["requests"]) == 2
    sent = seen["requests"][1]
    assert sent["purpose"] == "cognition" and "consumer_id" not in sent
    assert sent["payload"] == {
        "model": UPSTREAM_MODEL,
        "messages": [{"role": "user", "content": "fixture command"}],
        "temperature": 0.2,
        "max_tokens": 256,
        "thinking_budget_tokens": 0,
        "reasoning_format": "none",
        "grammar": 'root ::= "ok"',
        "cache_prompt": True,
    }
    assert "api_key" not in json.dumps(sent)
    assert seen["requests"][0]["deadline_ns"] == sent["deadline_ns"]


def test_existing_grammar_schema_fallback_uses_same_absolute_budget(broker):
    model, seen, _ = broker
    seen["status"] = 400
    assert constrained(model).text == OUTPUT
    assert len(seen["requests"]) == 3
    assert len({r["deadline_ns"] for r in seen["requests"]}) == 1
    assert "grammar" in seen["requests"][1]["payload"]
    assert seen["requests"][2]["payload"]["response_format"]["json_schema"]["strict"] is True


def test_readiness_uses_private_owner_capability_handshake_without_inference(broker):
    model, seen, _ = broker

    assert model.verify_ready() == UPSTREAM_MODEL
    assert len(seen["requests"]) == 1
    assert seen["requests"][0]["operation"] == "capabilities"


def test_image_bytes_schema_and_sampler_options_preserved(broker):
    model, seen, _ = broker
    model.purpose = "vision"
    model.inspect_structured(
        "inspect fixture",
        image_bytes=b"fixture png",
        mime_type="image/png",
        name="vision",
        schema={"type": "object"},
    )
    sent = seen["requests"][0]
    assert sent["purpose"] == "vision" and sent["payload"]["temperature"] == 0.1
    parts = sent["payload"]["messages"][0]["content"]
    assert (
        parts[1]["image_url"]["url"]
        == "data:image/png;base64," + base64.b64encode(b"fixture png").decode()
    )
    assert sent["payload"]["max_tokens"] == 256 and "grammar" not in sent["payload"]


@pytest.mark.parametrize("kind", ["identity", "result_model", "total", "usage"])
def test_wrong_identity_or_usage_fails_closed(broker, kind):
    model, seen, _ = broker
    if kind in {"identity", "result_model"}:
        seen[kind] = "wrong"
    elif kind == "total":
        seen["usage"]["total_tokens"] = 999
    else:
        seen["usage"]["completion_tokens"] = True
    with pytest.raises(BrokerError):
        model.complete(MESSAGES)
    assert len(seen["requests"]) == 1


def test_expired_local_lane_wait_never_sends_to_broker(broker):
    model, seen, _ = broker
    model.timeout_s = 0.05
    failures = []

    def run():
        try:
            model.complete(MESSAGES)
        except TimeoutError:
            failures.append(True)

    with local_model_inference_lane():
        thread = threading.Thread(target=run)
        thread.start()
        thread.join(1)
    assert failures == [True] and not thread.is_alive() and not seen["requests"]


def test_bound_discard_cancels_socket_and_cannot_publish(broker):
    model, seen, _ = broker
    seen["blocked"] = True
    request = binding()
    failures = []

    def run():
        try:
            model.complete_bound_constrained(
                MESSAGES,
                name="test",
                schema={},
                grammar='root ::= "x"',
                request=request,
                attempt_id="attempt-1",
            )
        except TimeoutError:
            failures.append(True)

    thread = threading.Thread(target=run)
    thread.start()
    assert seen["entered"].wait(1)
    model.discard_bound_request(request=request, reason="operator changed")
    thread.join(1)
    assert failures == [True] and not thread.is_alive() and seen["disconnected"].wait(1)
    with pytest.raises(RuntimeError):
        model.admit_bound_decision(
            request=request,
            attempt_id="attempt-1",
            source_decision_sha256="a" * 64,
            final_decision={},
            final_decision_sha256="a" * 64,
            rewritten=False,
        )


def test_pre_dispatch_discard_tombstone_prevents_future_worker(broker):
    model, seen, _ = broker
    request = binding()
    model.discard_bound_request(request=request, reason="operator changed")
    with pytest.raises(RuntimeError):
        model.complete_bound_constrained(
            MESSAGES,
            name="test",
            schema={},
            grammar='root ::= "x"',
            request=request,
            attempt_id="attempt-1",
        )
    assert not seen["requests"]


def test_bound_publication_validates_selected_source_and_final_digest(broker):
    model, seen, _ = broker
    request = binding()
    result = model.complete_bound_constrained(
        MESSAGES,
        name="test",
        schema={},
        grammar='root ::= "x"',
        request=request,
        attempt_id="attempt-1",
    )
    decision = _parse_decision(result.text)
    digest = cognition_decision_sha256(decision)
    fields = dict(
        request=request,
        attempt_id="attempt-1",
        source_decision_sha256=digest,
        final_decision=decision.model_dump(mode="json"),
        final_decision_sha256=digest,
        rewritten=False,
    )
    model.admit_bound_decision(**fields)
    model.admit_bound_decision(**fields)
    with pytest.raises(RuntimeError):
        model.admit_bound_decision(**{**fields, "source_decision_sha256": "b" * 64})
    assert len(seen["requests"]) == 2  # admission performs no inference or inputs


def test_insecure_socket_or_symlink_parent_refuses_without_send(broker, tmp_path):
    model, seen, path = broker
    path.chmod(0o666)
    with pytest.raises(BrokerError):
        model.complete(MESSAGES)
    path.chmod(0o600)
    alias = tmp_path / "alias"
    alias.symlink_to(path.parent, target_is_directory=True)
    model.broker_socket = str(alias / path.name)
    with pytest.raises(OSError):
        model.complete(MESSAGES)
    assert not seen["requests"]


def test_retirement_scope_cancels_vision_socket(broker):
    model, seen, _ = broker
    seen["blocked"] = True
    stopped = threading.Event()
    failures = []

    def run():
        try:
            with model.request_scope(cancel_requested=stopped.is_set):
                model.inspect("fixture", image_bytes=b"png", mime_type="image/png")
        except TimeoutError:
            failures.append(True)

    thread = threading.Thread(target=run)
    thread.start()
    assert seen["entered"].wait(1)
    stopped.set()
    thread.join(1)
    assert failures == [True] and seen["disconnected"].wait(1)
