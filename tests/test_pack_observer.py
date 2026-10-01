"""Fictional host envelopes, real delivery/revocation; no game or observer install."""

from dataclasses import replace
import json
import os
import sys
import time
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from minecraft_ai.agent.lifecycle import AgentProcess
from minecraft_ai.config import RuntimeConfig
from minecraft_ai.pack_observer import MAX_OBSERVATION_BYTES, RecipeObservationReader
from minecraft_ai.pack_scope import (
    ACTIVE_RECIPE_KEY, ACTIVE_RECIPE_SOURCE, active_recipe_identity,
    recipe_identity_matches,
)
from minecraft_ai.perception import PerceptionBlackboard
from minecraft_ai.platforms.frame_cache import owner_manifest
from minecraft_ai.runtime import AgentRuntime
from minecraft_ai.platforms.bedrock_x11 import ImageCaptureTimeout
from test_pack_recipe_scope import active_fact, publish, scope_catalog

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="Linux host boot/process observer contract",
)


@pytest.fixture
def observation(tmp_path, monkeypatch):
    catalog = scope_catalog()
    board = PerceptionBlackboard()
    owner = AgentProcess(
        pid=os.getpid(), started_ns=time.monotonic_ns(), display=":2", window_id=42,
        instance_id="bedrock:1.26.52.3:client-42", role="generalist",
        proc_start_ticks=456, command_sha256="a" * 64,
    )
    publish(board, instance=owner.instance_id)
    monkeypatch.setattr("minecraft_ai.pack_observer._boot_id", lambda: "fictional-boot")
    monkeypatch.setattr(AgentProcess, "load", lambda: owner)
    monkeypatch.setattr(
        "minecraft_ai.pack_observer._agent_process_state", lambda _owner: "verified-live",
    )
    path = tmp_path / "host-observation.json"
    envelope = {
        "schema_version": 1, "boot_id": "fictional-boot",
        "capture_owner": owner_manifest(owner), "observed_ns": time.monotonic_ns(),
        "expires_after_ms": 5000,
        "identity": json.loads(active_fact(catalog, owner.instance_id).value),
    }

    def write():
        path.write_text(json.dumps(envelope), encoding="utf-8")
        path.chmod(0o600)

    write()
    return board, catalog, owner, envelope, write, RecipeObservationReader(path, owner, catalog)


def test_reader_delivers_existing_identity_without_refreshing_evidence(observation):
    board, catalog, _owner, envelope, _write, reader = observation
    raw = board.raw_latest()
    reader.poll(board)
    first = board.fact(ACTIVE_RECIPE_KEY)
    assert first.observed_ns == envelope["observed_ns"]
    assert first.source == ACTIVE_RECIPE_SOURCE
    assert catalog.lookup_live("craft blue:beacon", board).chat_reply
    reader.poll(board)
    assert board.fact(ACTIVE_RECIPE_KEY).observed_ns == first.observed_ns
    assert board.raw_latest() is raw
    assert reader.status() == {"state": "delivered", "reason": "delivered"}
    assert board.fact("inventory.beacon") is None


@pytest.mark.parametrize("key,value", [
    ("schema_version", True), ("schema_version", 2), ("boot_id", "previous-boot"),
    ("expires_after_ms", 0), ("expires_after_ms", 5001), ("expires_after_ms", True),
    ("observed_ns", 0), ("observed_ns", True), ("observed_ns", 2**63),
    ("observed_ns", 1), ("identity", None), ("identity", {"state": "configured_only"}),
    ("capture_owner", []), ("extra", "unreviewed"),
])
def test_bad_or_ambiguous_envelope_revokes_previous_advice(observation, key, value):
    board, catalog, _owner, envelope, write, reader = observation
    reader.poll(board)
    prepared = catalog.active_identity(board)
    assert prepared is not None
    envelope[key] = value
    write()
    reader.poll(board)
    assert board.fact(ACTIVE_RECIPE_KEY) is None
    assert reader.status()["state"] == "unknown"
    assert not recipe_identity_matches(prepared, board)


@pytest.mark.parametrize("key,value", [
    ("pid", 999), ("pid", True), ("started_ns", 1), ("proc_start_ticks", 457),
    ("command_sha256", "b" * 64), ("display", ":0"), ("window_id", 43),
    ("instance_id", "bedrock:other"), ("allow_host_capture", 0),
])
def test_exact_owner_fields_cannot_select_discovery_candidates(observation, key, value):
    board, _catalog, _owner, envelope, write, reader = observation
    reader.poll(board)
    envelope["capture_owner"][key] = value
    write()
    reader.poll(board)
    assert active_recipe_identity(board) is None


@pytest.mark.parametrize("mode", ["missing", "oversized", "symlink", "public", "fifo"])
def test_lost_or_non_private_regular_file_revokes(observation, mode, tmp_path):
    board, _catalog, _owner, _envelope, _write, reader = observation
    reader.poll(board)
    if mode == "public":
        reader.path.chmod(0o644)
    elif mode == "oversized":
        reader.path.write_bytes(b"x" * (MAX_OBSERVATION_BYTES + 1))
    else:
        reader.path.unlink()
        if mode == "symlink":
            target = tmp_path / "target"
            target.write_text("{}")
            reader.path.symlink_to(target)
        elif mode == "fifo":
            os.mkfifo(reader.path, mode=0o600)
    reader.poll(board)
    assert board.fact(ACTIVE_RECIPE_KEY) is None


def test_selected_owner_replacement_or_unverified_process_revokes(observation, monkeypatch):
    board, _catalog, owner, _envelope, _write, reader = observation
    reader.poll(board)
    monkeypatch.setattr(AgentProcess, "load", lambda: replace(owner, proc_start_ticks=457))
    reader.poll(board)
    assert active_recipe_identity(board) is None
    monkeypatch.setattr(AgentProcess, "load", lambda: owner)
    monkeypatch.setattr("minecraft_ai.pack_observer._agent_process_state", lambda _: "mismatch")
    reader.poll(board)
    assert active_recipe_identity(board) is None
    reader.owner = None
    reader.poll(board)
    assert active_recipe_identity(board) is None


def test_server_change_heartbeat_and_out_of_order_replay(observation):
    board, catalog, _owner, envelope, write, reader = observation
    reader.poll(board)
    prepared = catalog.active_identity(board)
    old_ns = envelope["observed_ns"]
    envelope["observed_ns"] = time.monotonic_ns()
    write()
    reader.poll(board)
    assert recipe_identity_matches(prepared, board)
    envelope["observed_ns"] = time.monotonic_ns()
    envelope["identity"]["server_session"] = "fictional-restarted-server"
    write()
    reader.poll(board)
    assert not recipe_identity_matches(prepared, board)
    assert catalog.active_identity(board).server_session == "fictional-restarted-server"
    envelope["observed_ns"] = old_ns
    write()
    reader.poll(board)
    assert active_recipe_identity(board) is None


def test_same_timestamp_cannot_extend_ttl_or_have_duplicate_keys(observation):
    board, _catalog, _owner, envelope, write, reader = observation
    envelope["expires_after_ms"] = 1000
    write()
    reader.poll(board)
    assert active_recipe_identity(board) is not None
    envelope["expires_after_ms"] = 5000
    write()
    reader.poll(board)
    assert active_recipe_identity(board) is None
    envelope["expires_after_ms"] = 1000
    write()
    reader.poll(board)
    assert active_recipe_identity(board) is None  # Same-time ambiguity stays revoked.
    raw = reader.path.read_text()
    reader.path.write_text('{"schema_version":2,' + raw[1:])
    reader.poll(board)
    assert active_recipe_identity(board) is None


@pytest.mark.parametrize("revocation", [None, {"state": "configured_only"}])
def test_explicit_newer_revocation_blocks_still_fresh_positive_replay(observation, revocation):
    board, _catalog, _owner, envelope, write, reader = observation
    reader.poll(board)
    positive = envelope["identity"]
    t1 = envelope["observed_ns"]
    t2 = time.monotonic_ns()
    envelope["observed_ns"] = time.monotonic_ns()
    envelope["identity"] = revocation
    write()
    reader.poll(board)
    assert active_recipe_identity(board) is None
    for older in (t1, t2):
        envelope["identity"] = positive
        envelope["observed_ns"] = older
        write()
        reader.poll(board)
        assert active_recipe_identity(board) is None
    envelope["observed_ns"] = time.monotonic_ns()
    write()
    reader.poll(board)
    assert active_recipe_identity(board) is not None


def test_observation_before_owner_start_is_unverified(observation):
    board, _catalog, owner, envelope, write, reader = observation
    envelope["observed_ns"] = owner.started_ns - 1
    write()
    reader.poll(board)
    assert active_recipe_identity(board) is None


def test_immutable_snapshot_expires_despite_repeated_polling(observation, monkeypatch):
    board, _catalog, _owner, envelope, _write, reader = observation
    reader.poll(board)
    snapshot = board.cognition_snapshot()
    now = envelope["observed_ns"] + 5_000_000_001
    monkeypatch.setattr("time.monotonic_ns", lambda: now)
    reader.poll(board)
    assert active_recipe_identity(board) is None
    assert active_recipe_identity(snapshot) is None


def test_catalog_hash_and_world_mismatch_do_not_activate(observation):
    board, catalog, _owner, envelope, write, reader = observation
    for key, value in (("catalog_sha256", "d" * 64), ("world", "FictionalOtherWorld")):
        envelope["identity"][key] = value
        envelope["observed_ns"] = time.monotonic_ns()
        write()
        reader.poll(board)
        assert active_recipe_identity(board) is None
        assert reader.reason == "catalog_or_capture_scope_mismatch"
        assert catalog.lookup_live("craft blue:beacon", board) is None


def test_runtime_polls_before_stale_return_and_revokes_on_capture_timeout(observation):
    board, _catalog, _owner, _envelope, _write, reader = observation
    runtime = object.__new__(AgentRuntime)
    runtime.blackboard = board
    runtime.recipe_observer = reader
    runtime.perception = Mock()
    runtime.perception.capture_once.return_value = board.raw_latest()
    runtime.perception.stale.return_value = True
    runtime.metrics = Mock(frames=0, stale_frame_skips=0, consecutive_stale_frames=0)
    runtime.stale_frame_consecutive_limit = 3
    runtime._merge_operator_target = Mock()
    runtime._merge_policy_perception = Mock()
    runtime._release_and_reconcile_inputs = Mock()
    runtime.telemetry = Mock()
    runtime._telemetry_payload = Mock(return_value={})
    reader.poll = Mock(wraps=reader.poll)
    runtime.tick()
    reader.poll.assert_called_once_with(board)
    assert active_recipe_identity(board) is None
    reader.poll(board)
    assert active_recipe_identity(board) is not None
    runtime.perception.capture_once.side_effect = ImageCaptureTimeout("fictional timeout")
    runtime.tick()
    assert active_recipe_identity(board) is None
    assert reader.status()["state"] == "unknown"
    assert runtime._release_and_reconcile_inputs.call_count == 2


def test_reader_is_opt_in_and_requires_reviewed_catalog_pin():
    assert RuntimeConfig().pack_recipe_observation is None
    with pytest.raises(ValidationError, match="reviewed catalog and pin"):
        RuntimeConfig(pack_recipe_observation="/fictional/observation.json")
