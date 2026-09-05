"""Generic worker ownership, identity and absolute delivery deadlines."""

import io
import json
import os
import time

import pytest
from test_policy_service import _policy_config

from minecraft_ai.motor import MotorIntent
from minecraft_ai.perception import PerceptionBlackboard
from minecraft_ai.platforms.bedrock_x11 import CapturedFrame
import minecraft_ai.policy_service as policy_service_module
from minecraft_ai.policy_service import (
    LearnedPolicyOutput,
    TemporalPolicyClient,
    _external_worker_command,
    _external_worker_environment,
    _validate_external_ready,
    _validate_policy_config,
)


def external_config(tmp_path, **updates):
    return _policy_config(tmp_path).model_copy(
        update={
            "provider": "external",
            "external_module": "private_vendor.motor_worker",
            "external_args": ("serve", "--session-id", "example-session"),
            "external_architecture": "vendor.native-motor",
            **updates,
        }
    )


def test_external_command_keeps_parent_transport_and_identity(tmp_path):
    config = external_config(tmp_path)
    _validate_policy_config(config)
    command = _external_worker_command(config, "owned-memory")
    assert command[:6] == [
        config.python_path,
        "-m",
        config.external_module,
        "serve",
        "--session-id",
        "example-session",
    ]
    assert command.count("--shared-memory") == 1
    assert command[command.index("--shared-memory") + 1] == "owned-memory"
    assert command[command.index("--model-sha256") + 1] == config.model_sha256
    assert command[command.index("--camera-pitch-scale") + 1] == str(
        config.effective_camera_pitch_scale
    )
    assert "--source-path" not in command and "--weights-path" not in command


def test_external_import_root_precedes_editable_install_without_mutating_parent(
    tmp_path,
    monkeypatch,
):
    config = external_config(tmp_path)
    monkeypatch.setenv("PYTHONPATH", "parent-import-root")
    environment = _external_worker_environment(config)
    assert environment["PYTHONPATH"] == config.source_path + os.pathsep + "parent-import-root"
    assert os.environ["PYTHONPATH"] == "parent-import-root"
    monkeypatch.delenv("PYTHONPATH")
    assert _external_worker_environment(config)["PYTHONPATH"] == config.source_path


def test_external_goal_conditioning_is_not_silently_admitted(tmp_path):
    with pytest.raises(ValueError, match="raw motion only"):
        _validate_policy_config(external_config(tmp_path, external_goal_conditioned=True))


@pytest.mark.parametrize(
    "argument",
    [
        "--shared-memory",
        "--shared-memory=other",
        "--shared",
        "--model",
        "--seed=4",
        "--camera-pitch",
        "--stochastic",
        "--",
        "bad\0value",
    ],
)
def test_external_arguments_cannot_override_parent_options(tmp_path, argument):
    with pytest.raises(ValueError, match="parent-owned"):
        _validate_policy_config(external_config(tmp_path, external_args=("serve", argument)))


def test_external_identity_is_explicit_and_verified_at_ready(tmp_path):
    config = external_config(tmp_path)
    ready = {
        "protocol": "minecraft-ai.temporal-policy.v1",
        "architecture": config.external_architecture,
        "model_sha256": config.model_sha256,
        "model_version": config.model_version,
        "goal_conditioned": False,
    }
    _validate_external_ready(config, ready)
    for key in ready:
        with pytest.raises(RuntimeError, match="identity or protocol"):
            _validate_external_ready(config, {**ready, key: "wrong"})
    for update in (
        {"external_architecture": ""},
        {"external_module": "-m shell"},
        {"provider": "openai-vpt"},
    ):
        with pytest.raises(ValueError, match="external worker"):
            _validate_policy_config(config.model_copy(update=update))
    with pytest.raises(ValueError, match="unapproved"):
        _validate_policy_config(config.model_copy(update={"license": "unknown"}))


@pytest.mark.parametrize("reply_kind", ["prediction", "error"])
@pytest.mark.parametrize("already_missed", [False, True])
def test_late_response_releases_without_advancing_actions_or_restarting_worker(
    tmp_path, monkeypatch, reply_kind, already_missed
):
    frame = CapturedFrame(frame_id=1, captured_ns=1, width=1, height=1, bgra=b"\0" * 4)
    client = TemporalPolicyClient(config=external_config(tmp_path), frame_provider=lambda: frame)
    monkeypatch.setattr(client, "_ensure_started", lambda _size: None)
    closes = []
    monkeypatch.setattr(client, "close", lambda: closes.append(True))
    client._pending_request_id = "pending"
    client._pending_deadline_ns = time.monotonic_ns() - 1
    client._pending_miss_recorded = already_missed
    client.metrics.deadline_misses = int(already_missed)
    client._held_keys, client._held_buttons = {"w"}, {"left"}
    client._pending_camera = (9, 8)
    reply = {
        "type": reply_kind,
        "request_id": "pending",
        "error": "TimeoutError: request expired during perception; no action emitted",
        "output": {
            "keys": ["w"],
            "buttons": ["left"],
            "mouse_dx": 8,
            "inference_ns": 1,
            "model_version": client.config.model_version,
        },
    }
    monkeypatch.setattr(client, "_read_response", lambda _timeout: reply)
    action = client.act(
        PerceptionBlackboard(), MotorIntent(skill_id="explore", mode="explore"), sequence=1
    )
    assert action.keys_up == ("w",) and action.buttons_up == ("left",)
    assert (
        not action.keys_down and not action.buttons_down and action.mouse_dx == action.mouse_dy == 0
    )
    assert client.metrics.deadline_misses == 1 and client.metrics.failures == 0
    assert client._accepted_predictions == 0 and not closes
    assert client._pending_request_id is None and client._consumed_deadline_ns > 0
    submissions = []
    monkeypatch.setattr(client, "_submit", lambda *_args: submissions.append(True))
    client.act(PerceptionBlackboard(), MotorIntent(skill_id="explore", mode="explore"), sequence=2)
    assert submissions == [True] and not closes


class _StartupWorker:
    def __init__(self, ready):
        self.ready = ready
        self.stdin = io.StringIO()
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout):
        self.returncode = 0
        return 0


class _StartupMemory:
    def __init__(self, *, size, **_kwargs):
        self.name = "synthetic-worker-memory"
        self.buf = bytearray(size)
        self.closed = False
        self.unlinked = False

    def close(self):
        self.closed = True

    def unlink(self):
        self.unlinked = True


@pytest.fixture
def startup_client(tmp_path, monkeypatch):
    frame = CapturedFrame(frame_id=1, captured_ns=1, width=1, height=1, bgra=b"\0" * 4)
    client = TemporalPolicyClient(config=external_config(tmp_path), frame_provider=lambda: frame)
    ready = {
        "type": "ready",
        "protocol": "minecraft-ai.temporal-policy.v1",
        "architecture": client.config.external_architecture,
        "model_sha256": client.config.model_sha256,
        "model_version": client.config.model_version,
        "goal_conditioned": False,
    }
    replies, workers, memories = [], [], []

    def spawn(*_args, **_kwargs):
        worker = _StartupWorker(replies.pop(0))
        workers.append(worker)
        return worker

    def allocate(**kwargs):
        memory = _StartupMemory(**kwargs)
        memories.append(memory)
        return memory

    monkeypatch.setattr(policy_service_module.subprocess, "Popen", spawn)
    monkeypatch.setattr(policy_service_module.shared_memory, "SharedMemory", allocate)
    monkeypatch.setattr(client, "_read_response", lambda _timeout: client._process.ready)
    try:
        yield client, ready, replies, workers, memories
    finally:
        client.close()


@pytest.mark.parametrize("invalid_kind", ["identity", "protocol", "not-ready", "absent"])
def test_rejected_warmup_cannot_skip_handshake_and_infer_on_next_act(startup_client, invalid_kind):
    client, ready, replies, workers, memories = startup_client
    invalid = {
        "identity": {**ready, "model_sha256": "0" * 64},
        "protocol": {**ready, "protocol": "other-protocol"},
        "not-ready": {"type": "prediction"},
        "absent": None,
    }[invalid_kind]
    replies.extend((invalid, invalid))
    with pytest.raises(RuntimeError):
        client.warmup()

    action = client.act(
        PerceptionBlackboard(),
        MotorIntent(skill_id="explore", mode="explore"),
        sequence=1,
    )

    assert client.metrics.requests == 0
    assert action.keys_down == action.buttons_down == ()
    assert action.mouse_dx == action.mouse_dy == 0
    assert len(workers) == 2
    assert all(worker.poll() is not None for worker in workers)
    assert all("infer" not in worker.stdin.getvalue() for worker in workers)
    assert all(memory.closed and memory.unlinked for memory in memories)
    assert client._process is client._memory is None


def test_rejected_warmup_requires_a_fresh_successful_handshake_before_inference(startup_client):
    client, ready, replies, workers, memories = startup_client
    replies.extend(({**ready, "architecture": "unapproved"}, ready))
    with pytest.raises(RuntimeError):
        client.warmup()

    client.act(
        PerceptionBlackboard(),
        MotorIntent(skill_id="explore", mode="explore"),
        sequence=1,
    )

    assert len(workers) == 2
    assert workers[0].poll() is not None
    assert "infer" not in workers[0].stdin.getvalue()
    assert memories[0].closed and memories[0].unlinked
    assert workers[1].poll() is None
    assert json.loads(workers[1].stdin.getvalue())["type"] == "infer"
    assert client.metrics.requests == 1


def test_worker_spawn_failure_reclaims_allocated_frame_memory(startup_client, monkeypatch):
    client, _ready, _replies, workers, memories = startup_client

    def fail_spawn(*_args, **_kwargs):
        raise OSError("synthetic spawn failure")

    monkeypatch.setattr(policy_service_module.subprocess, "Popen", fail_spawn)
    with pytest.raises(OSError, match="synthetic spawn failure"):
        client.warmup()

    assert workers == []
    assert len(memories) == 1 and memories[0].closed and memories[0].unlinked
    assert client._process is client._memory is None
    assert client._startup_verified is False


def test_external_semantic_reply_releases_controls_without_publishing_facts(
    startup_client, monkeypatch
):
    monkeypatch.setattr(policy_service_module.time, "monotonic_ns", lambda: 1_000_000_000)
    client, ready, replies, workers, memories = startup_client
    replies.append(ready)
    client.warmup()
    board = PerceptionBlackboard()
    intent = MotorIntent(skill_id="explore", mode="explore")
    client.act(board, intent, sequence=1)
    request_id = client._pending_request_id
    assert request_id is not None
    client._held_keys, client._held_buttons = {"w"}, {"left"}
    response = {
        "type": "prediction",
        "request_id": request_id,
        "output": {
            "keys": ["w"],
            "buttons": ["left"],
            "mouse_dx": 10,
            "inference_ns": 1,
            "model_version": client.config.model_version,
            "target_exists_probability": 0.9,
            "scene_mode": "world",
            "scene_playable": True,
            "scene_confidence": 0.9,
        },
    }
    monkeypatch.setattr(client, "_read_response", lambda _timeout: response)

    action = client.act(board, intent, sequence=2)

    assert action.keys_up == ("w",) and action.buttons_up == ("left",)
    assert action.keys_down == action.buttons_down == ()
    assert action.mouse_dx == action.mouse_dy == 0
    assert client.metrics.failures == 1 and "raw-motion" in client.metrics.last_error
    assert client._accepted_predictions == 0
    assert client.target_observation() is None and client.scene_observation() is None
    assert not client.merge_perception(board)
    assert board.latest() is None
    assert workers[0].poll() is not None
    assert memories[0].closed and memories[0].unlinked


@pytest.mark.parametrize(
    "semantic_claim",
    [
        {"target_exists_probability": 0.0},
        {"target_exists_probability": 0.9},
        {"target_point_yx": (0.5, 0.5)},
        {"target_bbox_xyxy": (0.1, 0.1, 0.8, 0.8)},
        {"scene_mode": "unknown"},
        {"scene_playable": False},
        {"scene_confidence": 0.0},
        {"scene_class_probabilities": {"world": 0.9}},
        {"scene_model_version": "unapproved-scene"},
        {"scene_mode": "world", "scene_confidence": 0.9, "scene_playable": True},
        {"camera_semantics": "cursor"},
    ],
)
def test_external_raw_output_cannot_publish_semantics_or_cursor_actions(tmp_path, semantic_claim):
    client = TemporalPolicyClient(config=external_config(tmp_path), frame_provider=lambda: None)
    client._consumed_frame_captured_ns = time.monotonic_ns()
    output = LearnedPolicyOutput(
        keys=("w",),
        buttons=("left",),
        mouse_dx=8,
        inference_ns=1,
        model_version=client.config.model_version,
        **semantic_claim,
    )
    with pytest.raises(RuntimeError, match="raw.motion"):
        client._output_action(output, sequence=1)

    assert client._accepted_predictions == 0
    assert client._last_prediction is None
    assert client.target_observation() is None
    assert client.scene_observation() is None
    assert not client.merge_perception(PerceptionBlackboard())
    assert not client._held_keys and not client._held_buttons
    assert client._pending_camera == (0, 0)


@pytest.mark.parametrize("explicit_defaults", [False, True])
def test_external_raw_output_accepts_neutral_semantics_and_preserves_motor_tokens(
    tmp_path,
    explicit_defaults,
):
    client = TemporalPolicyClient(config=external_config(tmp_path), frame_provider=lambda: None)
    neutral = (
        {}
        if not explicit_defaults
        else {
            "target_exists_probability": None,
            "target_point_yx": None,
            "target_bbox_xyxy": None,
            "scene_mode": None,
            "scene_playable": None,
            "scene_confidence": None,
            "scene_class_probabilities": {},
            "scene_model_version": None,
        }
    )
    output = LearnedPolicyOutput(
        keys=("w", "space"),
        buttons=("left",),
        mouse_dx=8,
        mouse_dy=-3,
        camera_semantics="world",
        inference_ns=1,
        model_version=client.config.model_version,
        behavior_token=41,
        latent_id="z_041",
        suppressed_actions=("drop:restricted",),
        **neutral,
    )

    action = client._output_action(output, sequence=1)

    assert action.keys_down == ("space", "w") and action.buttons_down == ("left",)
    assert action.mouse_dx == 8 and action.mouse_dy == -3
    assert action.camera_semantics == "world"
    assert client._last_prediction == output
    assert client._accepted_predictions == 1
    provenance = client.status()["last_action_provenance"]
    assert provenance["behavior_token"] == 41 and provenance["latent_id"] == "z_041"
    assert client.target_observation() is None and client.scene_observation() is None
