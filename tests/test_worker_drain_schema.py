"""Schema-only drain tests: no worker, filesystem checkpoint or native imports."""
from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from minecraft_ai.worker_drain import DRAIN_CONTRACT, DrainBinding


def ready() -> dict[str, Any]:
    return {
        "type": "ready", "other_backend_field": "unchanged",
        "drain_contract": DRAIN_CONTRACT,
        "session": {
            "session_id": "worker-1", "episode_id": "world-1", "episode_index": 2,
            "steps": 80, "episode_steps": 16,
        },
        "checkpoint": {
            "path": "/private/session.pt", "payload_contract": "test.session.v1",
            "model_sha256": "a" * 64, "configuration_sha256": "b" * 64,
        },
    }


def binding() -> DrainBinding:
    result = DrainBinding.from_ready(ready(), model_sha256="a" * 64)
    assert result is not None
    return result


def ack() -> dict[str, Any]:
    metadata = ready()
    return {
        "type": "drained", "contract": DRAIN_CONTRACT, "request_id": "shutdown-1",
        "drained": True, "status": "checkpointed", "session": metadata["session"],
        "last_completed_inference": {
            "request_id": "infer-1", "episode_id": "world-1", "session_steps_before": 79,
            "session_steps_after": 80, "outcome": "prediction", "prediction_written": True,
        },
        "checkpoint": {
            **metadata["checkpoint"], "sha256": "c" * 64, "bytes": 4096,
            "file_fsync_completed": True, "directory_fsync_completed": True,
            "atomicity_scope": "one_native_session_file",
            "learning_state_embedded": False, "learning": None,
        },
    }


def validate(payload: dict[str, Any]) -> dict[str, Any]:
    return binding().validate_ack(payload, request_id="shutdown-1", after_request_id="infer-1")


def test_absent_capability_preserves_legacy_ready() -> None:
    assert DrainBinding.from_ready({"type": "ready"}, model_sha256="legacy") is None


@pytest.mark.parametrize("contract", [None, "", True, 1, [], "different.v1"])
def test_advertised_unknown_or_malformed_contract_is_refused(contract: Any) -> None:
    metadata = ready()
    metadata["drain_contract"] = contract
    with pytest.raises(ValueError):
        DrainBinding.from_ready(metadata, model_sha256="a" * 64)


def test_binding_and_returned_ack_are_detached() -> None:
    metadata = ready()
    frozen = DrainBinding.from_ready(metadata, model_sha256="a" * 64)
    assert frozen is not None
    metadata["session"]["steps"] = 999
    metadata["checkpoint"]["path"] = "/other/file.pt"
    receipt = ack()
    result = frozen.validate_ack(receipt, request_id="shutdown-1", after_request_id="infer-1")
    assert result == receipt and result is not receipt
    receipt["session"]["steps"] = 999
    receipt["checkpoint"]["path"] = "/other/file.pt"
    receipt["last_completed_inference"]["request_id"] = "changed"
    assert result["session"]["steps"] == frozen.steps == 80
    assert result["checkpoint"]["path"] == "/private/session.pt"
    assert result["last_completed_inference"]["request_id"] == "infer-1"
    with pytest.raises(FrozenInstanceError):
        frozen.steps = 90  # type: ignore[misc]
    assert "/private" not in repr(frozen)


@pytest.mark.parametrize("digest", ["", "a" * 63, "A" * 64, "z" * 64, True, None])
def test_supported_ready_requires_strict_configured_digest(digest: Any) -> None:
    with pytest.raises(ValueError):
        DrainBinding.from_ready(ready(), model_sha256=digest)


@pytest.mark.parametrize("path", [
    "relative.pt", "/", "//private/session.pt", "/private/../session.pt",
    "/private/./session.pt", "/private//session.pt", "/private/session.pt/",
    "/private/\x00session.pt", "/private/\nsession.pt", "x" * 4097,
    "C:\\private\\session.pt", None,
])
def test_ready_rejects_non_normalized_checkpoint_paths_without_io(path: Any) -> None:
    metadata = ready()
    metadata["checkpoint"]["path"] = path
    with pytest.raises(ValueError, match="checkpoint location"):
        DrainBinding.from_ready(metadata, model_sha256="a" * 64)


@pytest.mark.parametrize("key", ["session_id", "episode_id"])
@pytest.mark.parametrize("value", ["", "-invalid", "has space", "a" * 129, True, 7])
def test_ready_rejects_invalid_identities(key: str, value: Any) -> None:
    metadata = ready()
    metadata["session"][key] = value
    with pytest.raises(ValueError):
        DrainBinding.from_ready(metadata, model_sha256="a" * 64)


@pytest.mark.parametrize("key", ["episode_index", "steps", "episode_steps"])
@pytest.mark.parametrize("value", [True, False, -1, 2**63, 1.0, "1"])
def test_ready_rejects_invalid_counters(key: str, value: Any) -> None:
    metadata = ready()
    metadata["session"][key] = value
    with pytest.raises(ValueError):
        DrainBinding.from_ready(metadata, model_sha256="a" * 64)


def test_ready_null_checkpoint_is_capability_without_durable_ack_authority() -> None:
    metadata = ready()
    metadata["checkpoint"] = None
    frozen = DrainBinding.from_ready(metadata, model_sha256="a" * 64)
    assert frozen is not None
    with pytest.raises(ValueError, match="checkpoint binding"):
        frozen.validate_ack(ack(), request_id="shutdown-1", after_request_id="infer-1")


@pytest.mark.parametrize("key,value", [
    ("model_sha256", "d" * 64), ("configuration_sha256", "D" * 64),
    ("payload_contract", ""),
])
def test_ready_descriptor_validation(key: str, value: Any) -> None:
    metadata = ready()
    metadata["checkpoint"][key] = value
    with pytest.raises(ValueError):
        DrainBinding.from_ready(metadata, model_sha256="a" * 64)


@pytest.mark.parametrize("section", [None, "session", "checkpoint", "last_completed_inference"])
@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_ack_rejects_changed_key_sets(section: str | None, mutation: str) -> None:
    receipt = ack()
    target = receipt if section is None else receipt[section]
    if mutation == "missing":
        target.pop(next(iter(target)))
    else:
        target["unrequested"] = {"private": "payload"}
    with pytest.raises(ValueError):
        validate(receipt)


@pytest.mark.parametrize("key,value", [
    ("type", "drain_error"), ("contract", "other"), ("request_id", "other"),
    ("drained", False), ("drained", 1), ("status", "durable_after_deadline"),
])
def test_ack_requires_exact_success_identity(key: str, value: Any) -> None:
    receipt = ack()
    receipt[key] = value
    with pytest.raises(ValueError):
        validate(receipt)


@pytest.mark.parametrize("key,value", [
    ("session_id", "other"), ("episode_id", "other"), ("episode_index", 1),
    ("steps", 79), ("episode_steps", 15), ("episode_steps", 81), ("steps", True),
])
def test_ack_requires_matching_session_and_nondecreasing_counters(key: str, value: Any) -> None:
    receipt = ack()
    receipt["session"][key] = value
    with pytest.raises(ValueError):
        validate(receipt)


@pytest.mark.parametrize("key,value", [
    ("path", "/other/file.pt"), ("payload_contract", "other.v1"),
    ("model_sha256", "d" * 64), ("configuration_sha256", "e" * 64),
    ("sha256", "C" * 64), ("bytes", 0), ("bytes", 128 * 1024 * 1024 + 1),
    ("bytes", True), ("file_fsync_completed", False), ("directory_fsync_completed", False),
    ("file_fsync_completed", 1), ("directory_fsync_completed", 1),
    ("atomicity_scope", "whole_agent"), ("learning_state_embedded", 1),
])
def test_ack_rejects_checkpoint_mismatch_or_incomplete_durability(key: str, value: Any) -> None:
    receipt = ack()
    receipt["checkpoint"][key] = value
    with pytest.raises(ValueError):
        validate(receipt)


@pytest.mark.parametrize("key,value", [
    ("request_id", "other"), ("session_steps_before", 81), ("session_steps_after", 81),
    ("session_steps_before", True), ("outcome", "running"),
    ("prediction_written", 1), ("prediction_written", False),
])
def test_ack_requires_completed_exact_inference_barrier(key: str, value: Any) -> None:
    receipt = ack()
    receipt["last_completed_inference"][key] = value
    with pytest.raises(ValueError):
        validate(receipt)


def test_null_barrier_before_first_inference_allows_resumed_nonzero_steps() -> None:
    receipt = ack()
    receipt["last_completed_inference"] = None
    result = binding().validate_ack(receipt, request_id="shutdown-1", after_request_id=None)
    assert result["session"]["steps"] == 80
    assert result["last_completed_inference"] is None
    with pytest.raises(ValueError):
        validate(receipt)
    with pytest.raises(ValueError):
        binding().validate_ack(ack(), request_id="shutdown-1", after_request_id=None)


@pytest.mark.parametrize("after,written", [(79, False), (80, False), (80, True)])
def test_terminal_errors_may_preserve_native_progress_or_written_prediction(
    after: int, written: bool,
) -> None:
    receipt = ack()
    receipt["last_completed_inference"].update(
        outcome="error", session_steps_after=after, prediction_written=written,
    )
    assert validate(receipt)["last_completed_inference"]["outcome"] == "error"


def test_explicit_world_reset_preserves_lifetime_lower_bound_and_old_barrier() -> None:
    original = binding()
    reset = original.with_world_episode("world-2")
    assert (reset.session_id, reset.episode_id, reset.episode_index) == ("worker-1", "world-2", 3)
    assert (reset.steps, reset.episode_steps) == (80, 0)
    assert original.episode_id == "world-1" and original.episode_steps == 16
    receipt = ack()
    receipt["session"].update(episode_id="world-2", episode_index=3, steps=90, episode_steps=0)
    receipt["last_completed_inference"].update(session_steps_before=89, session_steps_after=90)
    result = reset.validate_ack(receipt, request_id="shutdown-1", after_request_id="infer-1")
    assert result["last_completed_inference"]["episode_id"] == "world-1"
    assert result["session"]["episode_steps"] == 0
    with pytest.raises(ValueError):
        original.validate_ack(receipt, request_id="shutdown-1", after_request_id="infer-1")


def test_reset_rejects_same_episode_or_counter_overflow() -> None:
    with pytest.raises(ValueError):
        binding().with_world_episode("world-1")
    metadata = ready()
    metadata["session"]["episode_index"] = 2**63 - 1
    frozen = DrainBinding.from_ready(metadata, model_sha256="a" * 64)
    assert frozen is not None
    with pytest.raises(ValueError):
        frozen.with_world_episode("world-2")


def learning_ack() -> dict[str, Any]:
    receipt = ack()
    receipt["checkpoint"].update(
        learning_state_embedded=True,
        learning={"update_algorithm": "custom.algorithm.v1", "prefix_cursor": 80,
                  "train_cursor": 70, "training_events": 4, "parameter_update_events": 3},
    )
    return receipt


def test_generic_learning_receipt_is_validated_and_detached() -> None:
    receipt = learning_ack()
    result = validate(receipt)
    assert result == receipt
    receipt["checkpoint"]["learning"]["training_events"] = 99
    assert result["checkpoint"]["learning"]["training_events"] == 4


@pytest.mark.parametrize("key,value", [
    ("update_algorithm", ""), ("prefix_cursor", 81), ("train_cursor", 81),
    ("training_events", True), ("parameter_update_events", 5),
    ("parameter_update_events", -1), ("train_cursor", 2**63),
])
def test_generic_learning_counter_refusals(key: str, value: Any) -> None:
    receipt = learning_ack()
    receipt["checkpoint"]["learning"][key] = value
    with pytest.raises(ValueError):
        validate(receipt)


@pytest.mark.parametrize("mutation", ["unexpected", "missing", "extra", "array"])
def test_learning_metadata_is_small_explicit_schema(mutation: str) -> None:
    receipt = learning_ack()
    if mutation == "unexpected":
        receipt["checkpoint"]["learning_state_embedded"] = False
    elif mutation == "missing":
        receipt["checkpoint"]["learning"] = None
    elif mutation == "extra":
        receipt["checkpoint"]["learning"]["weights"] = [1, 2, 3]
    else:
        receipt["checkpoint"]["learning"] = [1, 2, 3]
    with pytest.raises(ValueError):
        validate(receipt)


@pytest.mark.parametrize("identity", ["", "x" * 257, True, 1, None])
def test_request_identity_bounds_are_strict(identity: Any) -> None:
    with pytest.raises(ValueError):
        binding().validate_ack(ack(), request_id=identity, after_request_id="infer-1")
    with pytest.raises(ValueError):
        binding().validate_ack(ack(), request_id="shutdown-1", after_request_id=identity)


def test_boundaries_are_inclusive_and_request_ids_are_not_session_ids() -> None:
    receipt = ack()
    receipt["request_id"] = "request with spaces " + "x" * 236
    assert len(receipt["request_id"]) == 256
    receipt["session"]["steps"] = 2**63 - 1
    receipt["checkpoint"]["bytes"] = 128 * 1024 * 1024
    result = binding().validate_ack(
        receipt, request_id=receipt["request_id"], after_request_id="infer-1",
    )
    assert result["checkpoint"]["bytes"] == 128 * 1024 * 1024


def test_errors_never_echo_private_values() -> None:
    receipt = ack()
    receipt["checkpoint"]["path"] = "/sensitive/location/session.pt"
    before = copy.deepcopy(receipt)
    with pytest.raises(ValueError) as failure:
        validate(receipt)
    assert "sensitive" not in str(failure.value)
    assert "session.pt" not in str(failure.value)
    assert "a" * 64 not in str(failure.value)
    assert receipt == before
