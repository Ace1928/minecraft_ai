"""Real-pipe transport checks; no worker, model or actuator is launched."""

import json
import os
import select
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from test_policy_service import _policy_config

import minecraft_ai.policy_service as policy_service_module
from minecraft_ai.motor import MotorIntent
from minecraft_ai.perception import PerceptionBlackboard
from minecraft_ai.platforms.bedrock_x11 import CapturedFrame
from minecraft_ai.policy_service import TemporalPolicyClient

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="select requires POSIX pipes")


class _PipeProcess:
    def __init__(self):
        command_read, command_write = os.pipe()
        reply_read, reply_write = os.pipe()
        self.stdin = os.fdopen(command_write, "w", buffering=1)
        self.stdout = os.fdopen(reply_read, "r", buffering=1)
        self.command_reader = os.fdopen(command_read, "rb", buffering=0)
        self.reply_writer = os.fdopen(reply_write, "wb", buffering=0)
        self.returncode = None

    def poll(self):
        return self.returncode

    def send(self, payload):
        encoded = json.dumps(payload, separators=(",", ":")).encode() + b"\n"
        assert self.reply_writer.write(encoded) == len(encoded)

    def commands(self):
        records = b""
        while select.select([self.command_reader], [], [], 0)[0]:
            records += os.read(self.command_reader.fileno(), 65536)
        return [json.loads(line) for line in records.splitlines()]

    def close(self):
        for stream in (self.stdin, self.stdout, self.command_reader, self.reply_writer):
            stream.close()


@pytest.fixture
def pipe_client(tmp_path, monkeypatch):
    frames = [CapturedFrame(frame_id=1, captured_ns=1, width=2, height=2, bgra=b"a" * 16)]
    client = TemporalPolicyClient(
        config=_policy_config(tmp_path, deadline_ms=5000), frame_provider=lambda: frames[0]
    )
    process = _PipeProcess()
    memory = SimpleNamespace(buf=bytearray(16))
    client._process = process
    client._memory = memory
    client._memory_size = len(memory.buf)
    monkeypatch.setattr(client, "_ensure_started", lambda _size: None)
    closes = []
    monkeypatch.setattr(client, "close", lambda: closes.append(True))
    try:
        yield client, process, frames, memory, closes
    finally:
        process.close()


def test_response_reader_drains_two_queued_records_without_new_pipe_bytes(pipe_client):
    client, process, *_ = pipe_client
    payloads = [{"type": "ready", "marker": "first"}, {"type": "prediction", "marker": "second"}]
    encoded = b"".join(json.dumps(value).encode() + b"\n" for value in payloads)
    assert process.reply_writer.write(encoded) == len(encoded)

    assert client._read_response(0) == payloads[0]
    # The first read has consumed the kernel bytes, but the second full line
    # remains parent-owned and must not depend on future fd readiness.
    assert not select.select([process.stdout], [], [], 0)[0]
    assert client._read_response(0) == payloads[1]
    assert client._read_response(0) is None


@pytest.mark.parametrize("timeout_s", [0.0, 0.02])
def test_partial_response_returns_promptly_and_decodes_only_after_completion(
    pipe_client, timeout_s
):
    client, process, *_ = pipe_client
    process.reply_writer.write(b'{"type":"prediction","marker":"par')
    # A broken blocking readline implementation fails boundedly instead of
    # hanging the suite forever. The correct reader never needs this fallback.
    fallback = threading.Timer(1.0, lambda: process.reply_writer.write(b'tial"}\n'))
    fallback.start()
    try:
        started = time.monotonic()
        result = client._read_response(timeout_s)
        elapsed = time.monotonic() - started
        assert elapsed < 0.5
        assert result is None
    finally:
        fallback.cancel()
        fallback.join()
    process.reply_writer.write(b'tial"}\n')
    assert client._read_response(0) == {"type": "prediction", "marker": "partial"}


@pytest.mark.parametrize("partial", [b"", b'{"type":"prediction"', b'{"type":"prediction"}'])
def test_response_reader_rejects_eof_including_unterminated_record(pipe_client, partial):
    client, process, *_ = pipe_client
    process.reply_writer.write(partial)
    process.reply_writer.close()
    if partial:
        assert client._read_response(0) is None
    with pytest.raises(RuntimeError):
        client._read_response(0)


def test_zero_timeout_reads_at_most_one_chunk_then_next_poll_continues(pipe_client):
    client, process, *_ = pipe_client
    payload = {"marker": "x" * 6000}
    encoded = json.dumps(payload).encode() + b"\n"
    # This payload fits a Linux pipe but requires two parent 4096-byte reads.
    assert process.reply_writer.write(encoded) == len(encoded)

    assert client._read_response(0) is None
    assert bytes(client._response_bytes) == encoded[:4096]
    assert select.select([process.stdout], [], [], 0)[0]
    assert client._read_response(0) == payload
    assert client._read_response(0) is None


def test_always_readable_fragments_do_not_extend_positive_call_deadline(pipe_client, monkeypatch):
    client, process, *_ = pipe_client
    clock = [100.0]
    selections, reads = [], []
    client._pending_request_id = "owned-request"
    client._pending_deadline_ns = 12345

    def readable(sources, _write, _error, timeout):
        selections.append(timeout)
        return sources, [], []

    def slow_fragment(fd, size):
        assert fd == process.stdout.fileno() and size == 4096
        reads.append(size)
        clock[0] += 0.006
        return b" " * size

    monkeypatch.setattr(policy_service_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(policy_service_module.select, "select", readable)
    monkeypatch.setattr(policy_service_module.os, "read", slow_fragment)

    assert client._read_response(0.010) is None
    assert reads == [4096, 4096]
    assert selections == pytest.approx([0.010, 0.004])
    assert client._response_bytes == bytearray(b" " * 8192)
    assert client._pending_request_id == "owned-request"
    assert client._pending_deadline_ns == 12345


def test_positive_deadline_rechecked_after_select_before_read(pipe_client, monkeypatch):
    client, process, *_ = pipe_client
    clock = [100.0]

    def readiness_at_deadline(sources, _write, _error, timeout):
        assert timeout == pytest.approx(0.01)
        clock[0] += 0.011
        return sources, [], []

    def forbidden_read(*_args):
        pytest.fail("read started after this call's absolute deadline")

    monkeypatch.setattr(policy_service_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(policy_service_module.select, "select", readiness_at_deadline)
    monkeypatch.setattr(policy_service_module.os, "read", forbidden_read)

    assert client._read_response(0.01) is None
    assert client._response_bytes == bytearray()
    assert client._process is process


@pytest.mark.parametrize("timeout_s", [0.0, 0.01])
def test_complete_buffered_reply_has_priority_over_elapsed_call_budget(
    pipe_client,
    monkeypatch,
    timeout_s,
):
    client, _process, *_ = pipe_client
    client._response_bytes.extend(b'{"type":"prediction"}\npartial')
    clock = iter((100.0, 101.0))
    monkeypatch.setattr(policy_service_module.time, "monotonic", lambda: next(clock))

    def forbidden_select(*_args):
        pytest.fail("buffered complete reply must not wait for new pipe readiness")

    monkeypatch.setattr(policy_service_module.select, "select", forbidden_select)

    assert client._read_response(timeout_s) == {"type": "prediction"}
    assert client._response_bytes == bytearray(b"partial")


def test_response_reader_reports_exited_worker_without_pipe_bytes(pipe_client):
    client, process, *_ = pipe_client
    process.returncode = 7
    with pytest.raises(RuntimeError, match="exited with code 7"):
        client._read_response(0)


def test_response_reader_accepts_exact_64_kib_line_before_newline(pipe_client):
    client, process, *_ = pipe_client
    prefix, suffix = b'{"marker":"', b'"}'
    marker = "x" * (65536 - len(prefix) - len(suffix))
    encoded = prefix + marker.encode() + suffix
    assert len(encoded) == 65536
    for offset in range(0, len(encoded), 4096):
        process.reply_writer.write(encoded[offset : offset + 4096])
        assert client._read_response(0) is None
    process.reply_writer.write(b"\n")
    assert client._read_response(0) == {"marker": marker}


@pytest.mark.parametrize("terminated", [False, True])
def test_response_reader_rejects_line_over_64_kib(pipe_client, terminated):
    client, process, *_ = pipe_client
    # Feed below pipe capacity so this test needs no concurrent writer and
    # verifies the limit across multiple separately arriving fragments.
    for _ in range(16):
        process.reply_writer.write(b" " * 4096)
        assert client._read_response(0) is None
    process.reply_writer.write(b"x" + (b"\n" if terminated else b""))
    with pytest.raises(RuntimeError):
        client._read_response(0)


@pytest.mark.parametrize("record", [b"not-json\n", b"[]\n", b"null\n", b"\xff\n"])
def test_response_reader_rejects_malformed_or_nonobject_records(pipe_client, record):
    client, process, *_ = pipe_client
    process.reply_writer.write(record)
    with pytest.raises((RuntimeError, ValueError)):
        client._read_response(0)


def _assert_no_new_inputs(action):
    assert action.keys_down == ()
    assert action.buttons_down == ()
    assert action.mouse_dx == action.mouse_dy == 0


@pytest.mark.parametrize(
    ("provider", "reply_kind"),
    [("openai-vpt", "prediction"), ("external", "prediction"), ("external", "error")],
)
@pytest.mark.parametrize("already_missed", [False, True])
def test_reset_drains_retired_response_before_reusing_frame_or_option_context(
    pipe_client, provider, reply_kind, already_missed
):
    client, process, frames, memory, closes = pipe_client
    if provider == "external":
        client.config = client.config.model_copy(
            update={
                "provider": "external",
                "external_module": "private_vendor.motor_worker",
                "external_architecture": "vendor.native-motor",
            }
        )
    board = PerceptionBlackboard()
    old = MotorIntent(skill_id="explore_forward", mode="explore", episode_id="run-old")
    new = old.model_copy(update={"episode_id": "run-new"})
    _assert_no_new_inputs(client.act(board, old, sequence=1))
    first = process.commands()
    assert len(first) == 1 and first[0]["type"] == "infer"
    old_request_id = first[0]["request_id"]
    assert bytes(memory.buf) == b"a" * 16
    client._held_keys, client._held_buttons = {"w"}, {"left"}
    client._pending_camera = (19, 7)

    reset = client.reset()
    assert reset.keys_up == ("w",) and reset.buttons_up == ("left",)
    _assert_no_new_inputs(reset)
    assert process.commands() == [{"type": "reset"}]
    assert client._pending_request_id == old_request_id
    assert client._pending_request_context is None
    assert client.status()["pending_request"] is None
    assert client.status()["transport_pending"] == {
        "request_id": old_request_id,
        "retired": True,
        "deadline_ns": first[0]["deadline_ns"],
    }
    assert client.metrics.invalidated_requests == 1
    frames[0] = CapturedFrame(frame_id=2, captured_ns=2, width=2, height=2, bgra=b"b" * 16)
    if already_missed:
        client._pending_deadline_ns = time.monotonic_ns() - 1
    for sequence in (3, 4):
        _assert_no_new_inputs(client.act(board, new, sequence=sequence))
        assert process.commands() == []
        assert client._pending_request_id == old_request_id
        assert client.status()["transport_pending"] == {
            "request_id": old_request_id,
            "retired": True,
            "deadline_ns": client._pending_deadline_ns,
        }
        assert client.status()["retired_responses"] == 0
        assert bytes(memory.buf) == b"a" * 16
    assert client.metrics.deadline_misses == int(already_missed)

    client._pending_deadline_ns = time.monotonic_ns() - 1
    process.send(
        {
            "type": reply_kind,
            "request_id": old_request_id,
            "error": "TimeoutError: request expired during perception; no action emitted",
            "output": {
                "keys": ["w"],
                "buttons": ["left"],
                "mouse_dx": 8,
                "mouse_dy": 3,
                "inference_ns": 1,
                "model_version": client.config.model_version,
            },
        }
    )
    _assert_no_new_inputs(client.act(board, new, sequence=5))
    assert client._accepted_predictions == 0
    assert client._consumed_request_context is None
    assert client._applied_request_context is None
    provenance = client.status()["last_action_provenance"]
    assert provenance["request_id"] is None and provenance["condition"] is None
    assert client.metrics.deadline_misses <= 1
    if already_missed or reply_kind == "error":
        assert client.metrics.deadline_misses == 1
    assert client.metrics.failures == 0 and not closes
    assert client._process is process
    assert client.status()["retired_responses"] == 1
    assert client.metrics.last_response_age_ms >= client.config.deadline_ms
    assert client.status()["last_response_age_ms"] >= client.config.deadline_ms

    # A timeout can defer successor submission until the following tick; a
    # discarded prediction may submit it immediately. Neither may submit twice.
    successor = process.commands()
    if not successor:
        _assert_no_new_inputs(client.act(board, new, sequence=6))
        successor = process.commands()
    assert len(successor) == 1 and successor[0]["type"] == "infer"
    fresh_request_id = successor[0]["request_id"]
    assert fresh_request_id != old_request_id
    assert successor[0]["intent"]["episode_id"] == "run-new"
    assert successor[0]["frame"]["captured_ns"] == 2
    assert client._pending_request_id == fresh_request_id
    assert client._pending_request_context.condition["episode_id"] == "run-new"
    assert client.status()["transport_pending"] == {
        "request_id": fresh_request_id,
        "retired": False,
        "deadline_ns": successor[0]["deadline_ns"],
    }
    assert bytes(memory.buf) == b"b" * 16
    assert client.metrics.requests == 2
    assert not client._discard_pending_response

    process.send(
        {
            "type": "prediction",
            "request_id": fresh_request_id,
            "output": {
                "keys": ["a"],
                "inference_ns": 1,
                "model_version": client.config.model_version,
            },
        }
    )
    fresh_action = client.act(board, new, sequence=7)
    assert fresh_action.keys_down == ("a",)
    assert fresh_action.buttons_down == ()
    assert fresh_action.mouse_dx == fresh_action.mouse_dy == 0
    assert client._accepted_predictions == 1
    provenance = client.status()["last_action_provenance"]
    assert provenance["request_id"] == fresh_request_id
    assert provenance["episode_id"] == "run-new"
    assert client.status()["transport_pending"] is None
    assert client.status()["retired_responses"] == 1
    assert 0 <= client.metrics.last_response_age_ms < client.config.deadline_ms
    assert client.metrics.failures == 0 and not closes
