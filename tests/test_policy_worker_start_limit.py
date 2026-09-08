"""Optional worker spawn budgets use fake processes and fake shared memory only."""

from unittest.mock import Mock

import pytest
from test_external_temporal_worker import startup_client as startup_client

from minecraft_ai.config import PolicyConfig
from minecraft_ai.motor import MotorIntent
from minecraft_ai.perception import PerceptionBlackboard
from minecraft_ai.policy_service import TemporalPolicyClient


@pytest.mark.parametrize("limit", [None, 1, 64])
def test_worker_start_limit_accepts_bounded_configuration(limit):
    assert PolicyConfig(max_worker_starts=limit).max_worker_starts == limit


@pytest.mark.parametrize("limit", [-1, 0, 65, 1.5])
def test_worker_start_limit_rejects_invalid_configuration(limit):
    with pytest.raises(ValueError):
        PolicyConfig(max_worker_starts=limit)


def test_default_allows_restarts_and_reports_lifetime_attempts(startup_client):
    client, ready, replies, workers, _memories = startup_client
    assert client.status()["max_worker_starts"] is None
    assert client.status()["worker_start_attempts"] == 0
    assert client.status()["startup_verified"] is False
    assert client.status()["process_alive"] is False
    replies.extend((ready, ready))
    client.warmup()
    client.close()
    client.warmup()
    assert len(workers) == 2
    assert client.status()["worker_start_attempts"] == 2
    assert client.status()["startup_verified"] is True
    assert client.status()["process_alive"] is True


def test_limit_one_allows_first_spawn_and_reuses_it_across_option_resets(startup_client):
    client, ready, replies, workers, _memories = startup_client
    client.config = client.config.model_copy(update={"max_worker_starts": 1})
    replies.append(ready)
    client.warmup()
    client.reset()
    client.reset_world()
    client.warmup()
    client._ensure_started(1)  # A smaller frame still fits the same worker allocation.
    assert len(workers) == 1
    assert client.status()["worker_start_attempts"] == 1
    assert client.status()["max_worker_starts"] == 1
    assert client.status()["startup_verified"] is True


@pytest.mark.parametrize("retirement", ["handshake-failure", "death", "resize", "close"])
def test_limit_one_blocks_second_spawn_and_preserves_policy_error_release(
    startup_client, retirement,
):
    client, ready, replies, workers, memories = startup_client
    client.config = client.config.model_copy(update={"max_worker_starts": 1})
    replies.append({**ready, "protocol": "invalid"} if retirement == "handshake-failure" else ready)
    if retirement == "handshake-failure":
        with pytest.raises(RuntimeError, match="identity or protocol"):
            client.warmup()
    else:
        client.warmup()
    if retirement == "death":
        workers[0].returncode = 1
        assert client.status()["process_alive"] is False
    elif retirement == "resize":
        frame = client.frame_provider()
        resized = type(frame)(frame_id=2, captured_ns=2, width=2, height=2, bgra=b"\0" * 16)
        client.frame_provider = lambda: resized
    elif retirement == "close":
        client.close()
        client.reset()
        assert client.status()["startup_verified"] is False

    client._held_keys, client._held_buttons = {"w"}, {"left"}
    client._pending_camera = (8, 8)
    action = client.act(
        PerceptionBlackboard(), MotorIntent(skill_id="explore", mode="explore"),
        sequence=client._last_sequence + 1,
    )

    assert len(workers) == 1
    assert client.status()["worker_start_attempts"] == 1
    assert client.status()["startup_verified"] is False
    assert client.status()["process_alive"] is False
    assert "worker-start limit exhausted (1)" in client.metrics.last_error
    assert client.metrics.failures == 1 and client.metrics.requests == 0
    assert client._process is client._memory is None
    assert len(memories) == 1  # Exhausted retries never allocate a replacement buffer.
    assert all(memory.closed and memory.unlinked for memory in memories)
    assert action.keys_up == ("w",) and action.buttons_up == ("left",)
    assert not action.keys_down and not action.buttons_down
    assert action.mouse_dx == action.mouse_dy == 0


def test_failed_popen_consumes_attempt_and_warmup_cannot_spawn_again(startup_client, monkeypatch):
    client, _ready, _replies, workers, memories = startup_client
    client.config = client.config.model_copy(update={"max_worker_starts": 1})
    spawn = Mock(side_effect=OSError("synthetic spawn failure"))
    monkeypatch.setattr("minecraft_ai.policy_service.subprocess.Popen", spawn)
    with pytest.raises(OSError, match="synthetic spawn failure"):
        client.warmup()
    client.reset()
    client.close()
    with pytest.raises(RuntimeError, match="worker-start limit exhausted"):
        client.warmup()
    assert spawn.call_count == 1 and workers == []
    assert client.status()["worker_start_attempts"] == 1
    assert client.status()["startup_verified"] is False
    assert client._process is client._memory is None
    assert len(memories) == 1
    assert all(memory.closed and memory.unlinked for memory in memories)


def test_current_prediction_error_cannot_spawn_replacement_worker(startup_client):
    client, ready, replies, workers, memories = startup_client
    client.config = client.config.model_copy(update={"max_worker_starts": 1})
    replies.append(ready)
    client.warmup()
    board, intent = PerceptionBlackboard(), MotorIntent(skill_id="explore", mode="explore")
    client.act(board, intent, sequence=1)
    workers[0].ready = {
        "type": "error", "request_id": client._pending_request_id,
        "error": "RuntimeError: synthetic inference failure",
    }
    failed = client.act(board, intent, sequence=2)
    assert "synthetic inference failure" in client.metrics.last_error
    assert client._process is None
    blocked = client.act(board, intent, sequence=3)
    assert "worker-start limit exhausted" in client.metrics.last_error
    assert len(workers) == len(memories) == 1
    assert client.status()["worker_start_attempts"] == 1
    assert client.metrics.failures == 2 and client.metrics.requests == 1
    for action in (failed, blocked):
        assert not action.keys_down and not action.buttons_down
        assert action.mouse_dx == action.mouse_dy == 0


def test_each_client_owns_its_worker_start_budget(startup_client, monkeypatch):
    first, ready, replies, workers, _memories = startup_client
    config = first.config.model_copy(update={"max_worker_starts": 1})
    first.config = config
    second = TemporalPolicyClient(config=config, frame_provider=first.frame_provider)
    monkeypatch.setattr(second, "_read_response", lambda _timeout: second._process.ready)
    replies.extend((ready, ready))
    try:
        first.warmup()
        first.close()
        with pytest.raises(RuntimeError, match="worker-start limit exhausted"):
            first.warmup()
        second.warmup()
        assert len(workers) == 2
        assert first.status()["worker_start_attempts"] == 1
        assert second.status()["worker_start_attempts"] == 1
        assert first.status()["process_alive"] is False
        assert second.status()["process_alive"] is True
    finally:
        second.close()
