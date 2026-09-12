"""Explicit terminal drain over real pipes; no model, child process or actuator."""

import copy
import json
import os
import select
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from dataclasses import dataclass, field

import pytest
from test_external_temporal_worker import external_config
from test_worker_drain_schema import ack as schema_ack
from test_worker_drain_schema import ready as schema_ready

import minecraft_ai.policy_service as policy_module
from minecraft_ai.motor import MotorIntent
from minecraft_ai.action_levels import ActionLevel
from minecraft_ai.perception import PerceptionBlackboard
from minecraft_ai.platforms.bedrock_x11 import CapturedFrame
from minecraft_ai.policy_service import GroundedPolicyRouter, TemporalPolicyClient


pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="select requires POSIX pipes")


class _Memory:
    def __init__(self, *, size, **_kwargs):
        self.name = "synthetic-drain-memory"
        self.buf = bytearray(size)
        self.closed = False
        self.unlinked = False

    def close(self):
        self.closed = True

    def unlink(self):
        self.unlinked = True


class _PipeProcess:
    """A serial fake worker uses actual descriptors, not mocked reply reads."""

    def __init__(self):
        command_read, command_write = os.pipe()
        reply_read, reply_write = os.pipe()
        self.stdin = os.fdopen(command_write, "w", buffering=1)
        self.stdout = os.fdopen(reply_read, "r", buffering=1)
        self.command_reader = os.fdopen(command_read, "rb", buffering=0)
        self.reply_writer = os.fdopen(reply_write, "wb", buffering=0)
        self.returncode = None
        self.commands = []
        self.errors = []
        self.signals = []
        self._finished = threading.Event()
        self._stop = threading.Event()
        self._thread = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if not self._finished.wait(timeout):
            raise subprocess.TimeoutExpired("fake-worker", timeout)
        return self.returncode

    def terminate(self):
        self.signals.append("terminate")
        self.exit(-15)

    def kill(self):
        self.signals.append("kill")
        self.exit(-9)

    def exit(self, code=0):
        self.returncode = code
        self._finished.set()

    def send(self, payload):
        self.send_bytes(json.dumps(payload, separators=(",", ":")).encode() + b"\n")

    def send_bytes(self, payload):
        assert self.reply_writer.write(payload) == len(payload)

    def start(self, handler):
        def respond():
            pending = bytearray()
            try:
                while not self._stop.is_set() and self.poll() is None:
                    if not select.select([self.command_reader], [], [], 0.01)[0]:
                        continue
                    chunk = os.read(self.command_reader.fileno(), 4096)
                    if not chunk:
                        return
                    pending.extend(chunk)
                    while b"\n" in pending:
                        line, _, rest = pending.partition(b"\n")
                        pending[:] = rest
                        request = json.loads(line)
                        self.commands.append(request)
                        handler(request)
            except BaseException as exc:
                self.errors.append(exc)

        self._thread = threading.Thread(target=respond, daemon=True)
        self._thread.start()

    def dispose(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            assert not self._thread.is_alive()
        for stream in (self.stdin, self.stdout, self.command_reader, self.reply_writer):
            if not stream.closed:
                stream.close()
        assert not self.errors


def _deadline(seconds=1.0):
    return time.monotonic_ns() + int(seconds * 1e9)


def _assert_release_only(action):
    assert not action.keys_down and not action.buttons_down
    assert action.mouse_dx == action.mouse_dy == 0


@dataclass
class _Harness:
    client: TemporalPolicyClient
    process: _PipeProcess
    memory: _Memory
    ready: dict
    replies: list[dict] = field(default_factory=list)
    outcome: str = "success"
    hold_inference: bool = True
    infer_reply_type: str = "prediction"
    pending_reply: dict | None = None
    last_infer: str | None = None
    last_completed: dict | None = None
    ack_mutator: object = None

    def submit(self):
        intent = MotorIntent(skill_id="explore_forward", mode="explore", episode_id="option-1")
        _assert_release_only(self.client.act(
            PerceptionBlackboard(), intent, sequence=self.client._last_sequence + 1,
        ))
        return self.client._last_submitted_request_id

    def await_commands(self, count):
        deadline = time.monotonic() + 1.0
        while len(self.process.commands) < count and time.monotonic() < deadline:
            self.process._stop.wait(0.001)
        assert len(self.process.commands) >= count

    def respond(self, request):
        if request["type"] == "infer":
            session = self.ready["session"]
            before = session["steps"]
            session["steps"] += 1
            session["episode_steps"] += 1
            self.last_infer = request["request_id"]
            self.last_completed = {
                "request_id": self.last_infer,
                "episode_id": session["episode_id"],
                "session_steps_before": before,
                "session_steps_after": session["steps"],
                "outcome": self.infer_reply_type,
                "prediction_written": self.infer_reply_type == "prediction",
            }
            self.pending_reply = {
                "type": self.infer_reply_type, "request_id": self.last_infer,
                "error": "TimeoutError: expired after recurrent work",
                "output": {
                    "keys": ["w"], "buttons": ["left"], "mouse_dx": 128,
                    "inference_ns": 1, "model_version": self.client.config.model_version,
                },
            }
            if not self.hold_inference:
                self.process.send(self.pending_reply)
                self.pending_reply = None
        elif request["type"] == "reset":
            if "episode_id" in request:
                self.ready["session"]["episode_id"] = request["episode_id"]
                self.ready["session"]["episode_index"] += 1
                self.ready["session"]["episode_steps"] = 0
        elif request["type"] == "stop":
            self.process.exit()
        elif request["type"] == "drain":
            if self.outcome == "timeout":
                return
            if self.outcome in {"eof", "partial_eof"}:
                if self.outcome == "partial_eof":
                    self.process.send_bytes(b'{"type":"drained"')
                self.process.reply_writer.close()
                return
            records = []
            if self.pending_reply is not None:
                pending = copy.deepcopy(self.pending_reply)
                if self.outcome == "pending_mismatch":
                    pending["request_id"] = "different-infer"
                if self.outcome != "ack_before_pending":
                    records.append(pending)
                if self.outcome == "duplicate_pending":
                    records.append(pending)
                self.pending_reply = None
            receipt = schema_ack()
            receipt["request_id"] = request["request_id"]
            receipt["session"] = copy.deepcopy(self.ready["session"])
            receipt["last_completed_inference"] = copy.deepcopy(self.last_completed)
            if self.outcome == "ack_mismatch":
                receipt["request_id"] = "different-drain"
            if self.ack_mutator is not None:
                self.ack_mutator(receipt)
            records.append(receipt)
            self.replies.extend(copy.deepcopy(records))
            # Deliberately put pending reply and ACK in one kernel write. The
            # second line must come from the client's retained byte buffer.
            self.process.send_bytes(b"".join(
                json.dumps(r, separators=(",", ":")).encode() + b"\n" for r in records
            ))
            if self.outcome != "exit_timeout":
                self.process.exit(7 if self.outcome == "exit_failure" else 0)


@pytest.fixture
def drain_harness(tmp_path, monkeypatch):
    frame = CapturedFrame(frame_id=1, captured_ns=1, width=2, height=2, bgra=b"a" * 16)
    client = TemporalPolicyClient(external_config(tmp_path), frame_provider=lambda: frame)
    process = _PipeProcess()
    memory = _Memory(size=16)
    ready = {
        **schema_ready(),
        "protocol": "minecraft-ai.temporal-policy.v1",
        "architecture": client.config.external_architecture,
        "model_sha256": client.config.model_sha256,
        "model_version": client.config.model_version,
        "goal_conditioned": False,
        "reset_scopes": ["actions", "world"],
    }
    h = _Harness(client, process, memory, ready)
    monkeypatch.setattr(policy_module.shared_memory, "SharedMemory", lambda **_kw: memory)
    monkeypatch.setattr(policy_module.subprocess, "Popen", lambda *_a, **_kw: process)
    process.send(ready)
    process.start(h.respond)
    try:
        client.warmup()
        assert client._drain_binding is not None
        assert client._startup_verified and client._worker_start_attempts == 1
        yield h
    finally:
        # Fixture cleanup closes only synthetic descriptors. It never invokes
        # the production lifecycle or turns a failed drain into a second stop.
        process.dispose()


@pytest.mark.parametrize("deadline_ns", [None, True, 0, -1, 1.5, 2**63])
def test_invalid_deadline_does_not_latch_or_contact_worker(tmp_path, deadline_ns):
    client = TemporalPolicyClient(external_config(tmp_path), frame_provider=lambda: None)
    with pytest.raises(ValueError):
        client.drain(deadline_ns=deadline_ns)
    assert not client._draining
    assert client._drain_receipt is None
    assert client._worker_start_attempts == 0


def test_unsupported_worker_is_not_started_or_latched(tmp_path):
    client = TemporalPolicyClient(external_config(tmp_path), frame_provider=lambda: None)
    with pytest.raises(RuntimeError, match="drain contract"):
        client.drain(deadline_ns=_deadline())
    assert not client._draining
    assert client._worker_start_attempts == 0


@pytest.mark.parametrize("reply_type", ["prediction", "error"])
def test_pending_reply_is_drained_as_metadata_before_ack_and_exit(drain_harness, reply_type):
    h = drain_harness
    h.infer_reply_type = reply_type
    request_id = h.submit()
    assert request_id
    # This is a test notification, not a claim that any real actuator was released.
    h.client.notify_inputs_released()
    before_pixels = bytes(h.memory.buf)
    deadline = _deadline()
    receipt = h.client.drain(deadline_ns=deadline)

    assert [r["type"] for r in h.process.commands] == ["infer", "drain"]
    request = h.process.commands[-1]
    assert request["after_request_id"] == request_id
    assert request["deadline_ns"] == deadline
    assert h.client._last_submitted_request_id == request_id
    assert receipt == h.client._drain_receipt
    assert h.process.returncode == 0
    assert h.client._pending_request_id is None
    assert h.client.metrics.retired_responses == 1
    assert h.client._accepted_predictions == 0
    assert h.client._last_prediction is None
    assert h.client.target_observation() is h.client.scene_observation() is None
    assert h.client._pending_camera == (0, 0)
    assert bytes(h.memory.buf) == before_pixels
    assert not h.memory.closed and not h.memory.unlinked and not h.process.signals


def test_drain_before_first_inference_has_null_barrier(drain_harness):
    h = drain_harness
    result = h.client.drain(deadline_ns=_deadline())
    assert [r["type"] for r in h.process.commands] == ["drain"]
    assert h.process.commands[0]["after_request_id"] is None
    assert result["last_completed_inference"] is None
    assert h.client.metrics.requests == 0


@pytest.mark.parametrize("held", ["keys", "buttons", "camera"])
def test_drain_requires_caller_reconciled_release(drain_harness, held):
    h = drain_harness
    if held == "keys":
        h.client._held_keys = {"w"}
    elif held == "buttons":
        h.client._held_buttons = {"left"}
    else:
        h.client._pending_camera = (8, 3)
    with pytest.raises(ValueError, match="release"):
        h.client.drain(deadline_ns=_deadline())
    assert not h.client._draining and not h.process.commands
    h.client.notify_inputs_released()
    assert h.client.drain(deadline_ns=_deadline())["status"] == "checkpointed"


@pytest.mark.parametrize("outcome", [
    "timeout", "eof", "partial_eof", "ack_mismatch", "pending_mismatch",
    "duplicate_pending", "ack_before_pending",
])
def test_failed_drain_remains_terminal_without_signals_or_shared_memory_cleanup(
    drain_harness, outcome,
):
    h = drain_harness
    h.outcome = outcome
    if outcome in {"pending_mismatch", "duplicate_pending", "ack_before_pending"}:
        h.submit()
    h.client.notify_inputs_released()
    with pytest.raises((RuntimeError, ValueError, TimeoutError)):
        h.client.drain(deadline_ns=_deadline(0.1))
    assert h.client._draining
    assert h.client._drain_receipt is None
    assert not h.process.signals and not h.memory.closed and not h.memory.unlinked
    assert sum(r["type"] == "drain" for r in h.process.commands) == 1
    with pytest.raises(RuntimeError, match="terminal"):
        h.client.drain(deadline_ns=_deadline())


@pytest.mark.parametrize("outcome", ["exit_timeout", "exit_failure"])
def test_valid_receipt_is_retained_when_normal_exit_is_unconfirmed(drain_harness, outcome):
    h = drain_harness
    h.outcome = outcome
    with pytest.raises((RuntimeError, subprocess.TimeoutExpired, TimeoutError)):
        h.client.drain(deadline_ns=_deadline(0.1))
    assert h.client._draining
    assert h.client._drain_receipt["status"] == "checkpointed"
    assert "drain_receipt" not in h.client.status()
    assert "/private/session.pt" not in json.dumps(h.client.status())
    assert not h.process.signals and not h.memory.closed and not h.memory.unlinked


def test_terminal_client_cannot_submit_reset_warm_or_restart(drain_harness, monkeypatch):
    h = drain_harness
    h.client.drain(deadline_ns=_deadline())
    before = copy.deepcopy(h.process.commands)

    def forbidden(*_args, **_kwargs):
        pytest.fail("terminal client attempted new work")

    monkeypatch.setattr(h.client, "frame_provider", forbidden)
    monkeypatch.setattr(h.client, "_start_worker", forbidden)
    _assert_release_only(h.client.act(
        PerceptionBlackboard(), MotorIntent(skill_id="walk", mode="walk"), sequence=1,
    ))
    _assert_release_only(h.client.reset())
    _assert_release_only(h.client.reset_world())
    with pytest.raises(RuntimeError):
        h.client.warmup()
    with pytest.raises(RuntimeError):
        h.client.drain(deadline_ns=_deadline())
    assert h.process.commands == before
    assert not h.process.signals


def test_close_after_successful_explicit_drain_sends_no_legacy_stop(drain_harness):
    h = drain_harness
    h.client.drain(deadline_ns=_deadline())
    h.client.close()
    h.client.close()
    assert [r["type"] for r in h.process.commands] == ["drain"]
    assert not h.process.signals
    assert h.memory.closed and h.memory.unlinked
    assert h.client._draining
    assert h.client._drain_receipt["status"] == "checkpointed"


def test_untrusted_transport_is_terminal_without_sending_a_barrier(drain_harness):
    h = drain_harness
    h.client._transport_trusted = False
    with pytest.raises(RuntimeError, match="untrusted"):
        h.client.drain(deadline_ns=_deadline())
    assert h.client._draining and not h.process.commands
    assert not h.process.signals and not h.memory.unlinked


def test_history_barrier_survives_consumed_response_and_world_reset(drain_harness):
    h = drain_harness
    h.hold_inference = False
    submitted = h.submit()
    h.await_commands(1)
    deadline = time.monotonic() + 1.0
    response = None
    while response is None and time.monotonic() < deadline:
        response = h.client._consume_pending_response()
        if response is None:
            h.process._stop.wait(0.001)
    assert response is not None
    assert h.client._pending_request_id is None
    assert h.client._last_submitted_request_id == submitted
    previous_episode = h.client._drain_binding.episode_id
    _assert_release_only(h.client.reset_world())
    h.await_commands(2)
    reset = h.process.commands[-1]
    assert reset["type"] == "reset" and reset["scope"] == "world"
    assert len(reset["episode_id"]) == 32
    int(reset["episode_id"], 16)
    assert reset["episode_id"] != previous_episode
    assert h.client._last_submitted_request_id == submitted
    h.client.notify_inputs_released()
    h.client.drain(deadline_ns=_deadline())
    request = h.process.commands[-1]
    assert request["after_request_id"] == submitted
    assert request["episode_id"] == reset["episode_id"]


def test_action_reset_preserves_world_identity_and_history_barrier(drain_harness):
    h = drain_harness
    submitted = h.submit()
    previous_episode = h.client._drain_binding.episode_id
    _assert_release_only(h.client.reset())
    h.await_commands(2)
    assert h.process.commands[-1] == {"type": "reset", "scope": "actions"}
    assert h.client._drain_binding.episode_id == previous_episode
    assert h.client._last_submitted_request_id == submitted
    h.client.drain(deadline_ns=_deadline())
    assert h.process.commands[-1]["after_request_id"] == submitted


def test_legacy_world_reset_and_close_wire_are_unchanged(drain_harness):
    h = drain_harness
    h.client._drain_binding = None
    _assert_release_only(h.client.reset_world())
    h.await_commands(1)
    assert h.process.commands[-1] == {"type": "reset", "scope": "world"}
    h.client.close()
    assert [r["type"] for r in h.process.commands] == ["reset", "stop"]
    assert not h.process.signals
    assert not h.client._draining


def test_backpressured_stdin_write_is_bounded_and_never_signals_child(drain_harness):
    h = drain_harness
    # Freeze only the fixture's responder thread, leaving the real pipe full
    # and open. There is no subprocess or runtime watchdog in this test.
    h.process._stop.set()
    h.process._thread.join(timeout=1.0)
    descriptor = h.process.stdin.fileno()
    blocking = os.get_blocking(descriptor)
    os.set_blocking(descriptor, False)
    try:
        while True:
            try:
                os.write(descriptor, b"x" * 4096)
            except BlockingIOError:
                break
    finally:
        os.set_blocking(descriptor, blocking)
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="command deadline"):
        h.client.drain(deadline_ns=_deadline(0.05))
    assert time.monotonic() - started < 0.5
    assert os.get_blocking(descriptor) == blocking
    assert h.client._draining and not h.client._transport_trusted
    assert not h.process.signals and not h.memory.closed and not h.memory.unlinked


@pytest.mark.parametrize("failure", ["short_write", "flush"])
def test_failed_infer_flush_does_not_advance_the_drain_barrier(drain_harness, failure):
    h = drain_harness
    original = h.process.stdin

    class Writer:
        closed = False

        def write(self, value):
            return len(value) - 1 if failure == "short_write" else original.write(value)

        def flush(self):
            raise OSError("synthetic write failure")

        def close(self):
            original.close()
            self.closed = True

    h.process.stdin = Writer()
    h.client._last_submitted_request_id = "previous-completed-request"
    with pytest.raises(OSError):
        h.client._submit(
            h.client.frame_provider(),
            MotorIntent(skill_id="walk", mode="walk"),
            PerceptionBlackboard(),
        )
    assert h.client._last_submitted_request_id == "previous-completed-request"
    assert h.client._pending_request_id is None
    assert not h.client._transport_trusted
    with pytest.raises(RuntimeError, match="untrusted"):
        h.client.drain(deadline_ns=_deadline())
    assert h.client._draining
    assert not any(r["type"] == "drain" for r in h.process.commands)


def test_close_after_failed_drain_does_not_retry_a_checkpoint(drain_harness):
    h = drain_harness
    h.outcome = "timeout"
    with pytest.raises(TimeoutError):
        h.client.drain(deadline_ns=_deadline(0.05))
    h.process.exit(1)
    h.client.close()
    assert [r["type"] for r in h.process.commands] == ["drain"]
    assert not h.process.signals
    assert h.client._drain_receipt is None and h.client._draining


def test_failed_forced_reap_retains_process_and_shared_memory(drain_harness, monkeypatch):
    h = drain_harness
    h.outcome = "timeout"
    with pytest.raises(TimeoutError):
        h.client.drain(deadline_ns=_deadline(0.05))
    waits = []

    def unconfirmed_wait(timeout):
        waits.append(timeout)
        raise subprocess.TimeoutExpired("unreaped-fake-worker", timeout)

    monkeypatch.setattr(h.process, "wait", unconfirmed_wait)
    monkeypatch.setattr(h.process, "terminate", lambda: h.process.signals.append("terminate"))
    monkeypatch.setattr(h.process, "kill", lambda: h.process.signals.append("kill"))
    with pytest.raises(subprocess.TimeoutExpired):
        h.client.close()
    assert waits == [1.0, 1.0, 1.0]
    assert h.process.signals == ["terminate", "kill"]
    assert h.client._process is h.process
    assert h.client._memory is h.memory
    assert not h.memory.closed and not h.memory.unlinked
    assert [r["type"] for r in h.process.commands] == ["drain"]
    assert h.client._draining and h.client._drain_receipt is None


def test_planned_retirement_drains_then_reaps_and_cleans_memory(drain_harness):
    h = drain_harness
    h.submit()
    h.client._held_keys = {"w"}
    result = h.client.retire(deadline_ns=_deadline(), inputs_released=True)
    assert result == {"complete": True, "workers": [{
        "policy_id": h.client.policy_id, "mode": "drain",
        "checkpoint_acknowledged": True, "process_exited": True, "error_code": None,
    }]}
    assert [r["type"] for r in h.process.commands] == ["infer", "drain"]
    assert h.memory.closed and h.memory.unlinked
    assert not h.process.signals
    assert h.client._drain_receipt is not None
    assert "checkpoint" not in result["workers"][0]
    with pytest.raises(RuntimeError, match="cannot be repeated"):
        h.client.retire(deadline_ns=_deadline(), inputs_released=True)
    with pytest.raises(RuntimeError, match="cannot warm"):
        h.client.warmup()


def test_retirement_legacy_stop_never_claims_a_checkpoint(drain_harness):
    h = drain_harness
    h.client._drain_binding = None
    result = h.client.retire(deadline_ns=_deadline(), inputs_released=True)
    assert result["complete"]
    assert result["workers"][0]["mode"] == "legacy"
    assert not result["workers"][0]["checkpoint_acknowledged"]
    assert [r["type"] for r in h.process.commands] == ["stop"]
    assert h.memory.closed and h.memory.unlinked
    assert h.client._drain_receipt is None


def test_retirement_legacy_abnormal_exit_is_not_cooperative_completion(drain_harness):
    h = drain_harness
    h.client._drain_binding = None
    h.process.exit(7)
    result = h.client.retire(deadline_ns=_deadline(), inputs_released=True)
    assert not result["complete"]
    assert result["workers"][0]["process_exited"]
    assert result["workers"][0]["error_code"] == "worker_exit_not_normal"
    assert not result["workers"][0]["checkpoint_acknowledged"]
    assert not h.process.commands


@pytest.mark.parametrize("capable", [True, False])
def test_windows_retirement_contains_without_posix_pipe_write(
    drain_harness, monkeypatch, capable,
):
    h = drain_harness
    if not capable:
        h.client._drain_binding = None
    monkeypatch.setattr(policy_module, "sys", SimpleNamespace(platform="win32"))

    def forbidden_write(*_args, **_kwargs):
        pytest.fail("Windows retirement must not call the POSIX writer")

    monkeypatch.setattr(h.client, "_write_drain_request", forbidden_write)
    result = h.client.retire(deadline_ns=_deadline(), inputs_released=True)
    assert not result["complete"]
    report = result["workers"][0]
    assert report["error_code"] == (
        "drain_platform_unsupported" if capable else "legacy_stop_unsupported"
    )
    assert report["process_exited"] and not report["checkpoint_acknowledged"]
    assert h.process.signals == ["terminate"] and not h.process.commands
    assert h.memory.closed and h.memory.unlinked


def test_windows_unbudgeted_close_retains_legacy_stop(drain_harness, monkeypatch):
    h = drain_harness
    h.client._drain_binding = None
    monkeypatch.setattr(policy_module, "sys", SimpleNamespace(platform="win32"))
    h.client.close()
    assert [r["type"] for r in h.process.commands] == ["stop"]
    assert not h.process.signals
    assert h.memory.closed and h.memory.unlinked


@pytest.mark.parametrize("capable", [True, False])
def test_no_release_proof_permits_containment_but_no_checkpoint(drain_harness, capable):
    h = drain_harness
    if not capable:
        h.client._drain_binding = None
    h.client._held_keys = {"w"}
    result = h.client.retire(deadline_ns=_deadline(0.02), inputs_released=False)
    assert not result["complete"]
    assert result["workers"][0]["error_code"] == "inputs_not_released"
    assert not result["workers"][0]["checkpoint_acknowledged"]
    assert result["workers"][0]["process_exited"]
    assert not h.process.commands
    assert h.process.signals == ["terminate"]


def test_legacy_reconciliation_failure_never_falls_back_to_stop(drain_harness, monkeypatch):
    h = drain_harness
    h.client._drain_binding = None

    def failed_release():
        raise RuntimeError("synthetic failed reconciliation")

    monkeypatch.setattr(h.client, "notify_inputs_released", failed_release)
    result = h.client.retire(deadline_ns=_deadline(0.02), inputs_released=True)
    assert not result["complete"]
    assert result["workers"][0]["error_code"] == "drain_RuntimeError"
    assert result["workers"][0]["process_exited"]
    assert not result["workers"][0]["checkpoint_acknowledged"]
    assert not h.process.commands and h.process.signals == ["terminate"]
    assert h.memory.closed and h.memory.unlinked and h.client._draining


@pytest.mark.parametrize("outcome,acknowledged", [
    ("ack_mismatch", False), ("exit_failure", True), ("exit_timeout", True), ("timeout", False),
])
def test_retirement_failure_is_separate_from_confirmed_reap(
    drain_harness, outcome, acknowledged,
):
    h = drain_harness
    h.outcome = outcome
    result = h.client.retire(deadline_ns=_deadline(0.05), inputs_released=True)
    assert not result["complete"]
    report = result["workers"][0]
    assert report["process_exited"]
    assert report["checkpoint_acknowledged"] is acknowledged
    assert report["error_code"].startswith("drain_")
    assert h.memory.closed and h.memory.unlinked
    assert [r["type"] for r in h.process.commands] == ["drain"]


def test_expired_retirement_budget_never_becomes_a_new_wait(drain_harness, monkeypatch):
    h = drain_harness
    waits = []
    original = h.process.wait

    def capture_wait(timeout):
        waits.append(timeout)
        return original(timeout=timeout)

    monkeypatch.setattr(h.process, "wait", capture_wait)
    result = h.client.retire(deadline_ns=time.monotonic_ns() - 1, inputs_released=True)
    assert not result["complete"]
    assert waits and all(timeout == 0 for timeout in waits)
    assert not h.process.commands
    assert result["workers"][0]["process_exited"]


def test_retirement_unconfirmed_exit_keeps_shared_memory_owned(drain_harness, monkeypatch):
    h = drain_harness

    def wait(timeout):
        raise subprocess.TimeoutExpired("synthetic-unreaped", timeout)

    monkeypatch.setattr(h.process, "wait", wait)
    monkeypatch.setattr(h.process, "terminate", lambda: h.process.signals.append("terminate"))
    monkeypatch.setattr(h.process, "kill", lambda: h.process.signals.append("kill"))
    result = h.client.retire(deadline_ns=time.monotonic_ns() - 1, inputs_released=True)
    assert not result["complete"]
    assert not result["workers"][0]["process_exited"]
    assert h.client._memory is h.memory and h.client._process is h.process
    assert not h.memory.closed and not h.memory.unlinked


@pytest.mark.parametrize("invalid", [True, 0, -1, 2**63, None])
def test_invalid_retirement_arguments_do_not_latch(drain_harness, invalid):
    h = drain_harness
    with pytest.raises(ValueError):
        h.client.retire(deadline_ns=invalid, inputs_released=True)
    assert not h.client._retiring and not h.process.commands
    with pytest.raises(ValueError):
        h.client.retire(deadline_ns=_deadline(), inputs_released="yes")
    assert not h.client._retiring and not h.process.commands


class _RetirementPolicy:
    def __init__(self, policy_id, log, *, fail=False, release_fail=False):
        self.policy_id = policy_id
        self.log = log
        self.fail = fail
        self.release_fail = release_fail

    def notify_inputs_released(self):
        self.log.append((self.policy_id, "release"))
        if self.release_fail:
            raise RuntimeError("synthetic reconciliation failure")

    def retire(self, *, deadline_ns, inputs_released):
        self.log.append((self.policy_id, "retire", deadline_ns, inputs_released))
        if self.fail:
            raise RuntimeError("private failure detail must never enter report")
        return {"complete": inputs_released, "workers": [{
            "policy_id": self.policy_id, "mode": "drain",
            "checkpoint_acknowledged": inputs_released, "process_exited": True,
            "error_code": None if inputs_released else "inputs_not_released",
        }]}


def test_router_reconciles_all_unique_workers_before_any_drain():
    log = []
    primary = _RetirementPolicy("primary", log)
    observer = _RetirementPolicy("observer", log)
    gui = _RetirementPolicy("gui", log)
    router = GroundedPolicyRouter(primary, grounded=observer, gui=gui, raw_motion=primary)
    deadline = _deadline()
    result = router.retire(deadline_ns=deadline, inputs_released=True)
    assert result["complete"]
    assert log[:3] == [("primary", "release"), ("gui", "release"), ("observer", "release")]
    assert log[3:] == [(name, "retire", deadline, True) for name in ("primary", "gui", "observer")]
    assert len(result["workers"]) == 3
    _assert_release_only(router.act(
        PerceptionBlackboard(), MotorIntent(skill_id="walk", mode="walk"), sequence=1,
    ))
    _assert_release_only(router.reset())
    _assert_release_only(router.reset_world())
    assert not router.specialist_ready(ActionLevel.GROUNDED)
    with pytest.raises(RuntimeError, match="cannot warm"):
        router.request_warm(ActionLevel.GROUNDED)
    with pytest.raises(RuntimeError, match="cannot be repeated"):
        router.retire(deadline_ns=deadline, inputs_released=True)


def test_router_continues_after_retirement_and_reconciliation_errors():
    log = []
    primary = _RetirementPolicy("primary", log, fail=True)
    observer = _RetirementPolicy("observer", log, release_fail=True)
    gui = _RetirementPolicy("gui", log)
    router = GroundedPolicyRouter(primary, grounded=observer, gui=gui)
    deadline = _deadline()
    result = router.retire(deadline_ns=deadline, inputs_released=True)
    assert not result["complete"]
    assert log[-3:] == [
        ("primary", "retire", deadline, True), ("gui", "retire", deadline, True),
        ("observer", "retire", deadline, False),
    ]
    assert result["workers"][0]["error_code"] == "retirement_RuntimeError"
    assert "private failure" not in json.dumps(result)


def test_warming_timeout_preserves_owner_and_still_retires_other_workers():
    log = []
    primary = _RetirementPolicy("primary", log)
    observer = _RetirementPolicy("observer", log)
    router = GroundedPolicyRouter(primary, grounded=observer)
    stop = threading.Event()
    thread = threading.Thread(target=stop.wait, daemon=True)
    thread.start()
    router._warming[id(observer)] = thread
    try:
        deadline = _deadline(0.02)
        result = router.retire(deadline_ns=deadline, inputs_released=True)
        assert not result["complete"]
        assert log == [("primary", "release"), ("primary", "retire", deadline, True)]
        assert result["workers"][1]["error_code"] == "warmup_deadline_exceeded"
        assert router._warming[id(observer)] is thread and thread.is_alive()
    finally:
        stop.set()
        thread.join(timeout=1)


@pytest.mark.parametrize("operation", ["retire", "warm"])
def test_inline_warmup_completion_does_not_wait_for_caller_lifetime(monkeypatch, operation):
    entered = threading.Event()
    finish_warm = threading.Event()
    caller_keepalive = threading.Event()
    log = []

    class InlinePolicy(_RetirementPolicy):
        def warmup(self):
            entered.set()
            finish_warm.wait()

    primary = InlinePolicy("primary", log)
    router = GroundedPolicyRouter(primary)

    def caller():
        router._ensure_warm(primary)
        caller_keepalive.wait()

    thread = threading.Thread(target=caller, daemon=True)
    thread.start()
    original_join = thread.join

    def finish_during_wait(timeout=None):
        finish_warm.set()
        assert timeout is not None, "warm ownership checks must never join caller lifetime"
        original_join(timeout=timeout)

    monkeypatch.setattr(thread, "join", finish_during_wait)
    try:
        assert entered.wait(timeout=1)
        if operation == "retire":
            deadline = _deadline(0.5)
            result = router.retire(deadline_ns=deadline, inputs_released=True)
            assert result["complete"]
            assert time.monotonic_ns() < deadline
            assert log[-1] == ("primary", "retire", deadline, True)
        else:
            router._ensure_warm(primary)
            assert id(primary) in router._warmed
        assert thread.is_alive()
        assert id(primary) not in router._warming
    finally:
        finish_warm.set()
        caller_keepalive.set()
        original_join(timeout=1)
        assert not thread.is_alive()


def test_current_thread_still_owning_warmup_is_not_drained():
    log = []
    primary = _RetirementPolicy("primary", log)
    router = GroundedPolicyRouter(primary)
    router._warming[id(primary)] = threading.current_thread()
    result = router.retire(deadline_ns=_deadline(), inputs_released=True)
    assert not result["complete"]
    assert result["workers"][0]["error_code"] == "warmup_deadline_exceeded"
    assert not log


def test_simple_policy_without_reconciliation_does_not_skip_native_observer():
    class Simple:
        policy_id = "simple"

    log = []
    router = GroundedPolicyRouter(Simple(), grounded=_RetirementPolicy("observer", log))
    result = router.retire(deadline_ns=_deadline(), inputs_released=True)
    assert result["complete"]
    assert result["workers"][0]["mode"] == "not_loaded"
    assert log[0] == ("observer", "release")
    assert log[1][1] == "retire"


def test_unbounded_third_party_close_is_reported_without_running_it():
    class Unknown:
        policy_id = "custom"

        def close(self):
            pytest.fail("unbounded custom close must not consume retirement budget")

    router = GroundedPolicyRouter(Unknown())
    result = router.retire(deadline_ns=_deadline(), inputs_released=False)
    assert not result["complete"]
    assert result["workers"][0]["error_code"] == "retirement_unsupported"
