"""Generic worker ownership, identity and absolute delivery deadlines."""

import time
import os

import pytest
from test_policy_service import _policy_config

from minecraft_ai.motor import MotorIntent
from minecraft_ai.perception import PerceptionBlackboard
from minecraft_ai.platforms.bedrock_x11 import CapturedFrame
from minecraft_ai.policy_service import (
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
    tmp_path, monkeypatch,
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
