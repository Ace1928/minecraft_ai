"""Validate optional worker-drain metadata without opening private checkpoints.

A matching receipt is a worker assertion about one session file, not independent
filesystem verification or evidence that the whole agent was durably saved.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import PurePosixPath
from typing import Any

DRAIN_CONTRACT = "erais.native-minecraft-drain.v1"
_INT64_MAX = 2**63 - 1
_MAX_CHECKPOINT_BYTES = 128 * 1024 * 1024
_IDENTITY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_SESSION_FIELDS = frozenset({"session_id", "episode_id", "episode_index", "steps", "episode_steps"})
_DESCRIPTOR_FIELDS = frozenset({"path", "payload_contract", "model_sha256", "configuration_sha256"})
_CHECKPOINT_FIELDS = _DESCRIPTOR_FIELDS | {
    "sha256", "bytes", "file_fsync_completed", "directory_fsync_completed",
    "atomicity_scope", "learning_state_embedded", "learning",
}
_ACK_FIELDS = frozenset({
    "type", "contract", "request_id", "drained", "status", "session",
    "last_completed_inference", "checkpoint",
})
_INFERENCE_FIELDS = frozenset({
    "request_id", "episode_id", "session_steps_before", "session_steps_after",
    "outcome", "prediction_written",
})
_LEARNING_FIELDS = frozenset({
    "update_algorithm", "prefix_cursor", "train_cursor", "training_events",
    "parameter_update_events",
})


def _object(value: Any, fields: frozenset[str]) -> dict[str, Any]:
    if type(value) is not dict or value.keys() != fields:
        raise ValueError("invalid drain metadata object")
    return value


def _identity(value: Any) -> str:
    if type(value) is not str or _IDENTITY.fullmatch(value) is None:
        raise ValueError("invalid drain identity")
    return value


def _request_id(value: Any) -> str:
    if type(value) is not str or not 1 <= len(value) <= 256:
        raise ValueError("invalid drain request identity")
    return value


def _counter(value: Any) -> int:
    if type(value) is not int or not 0 <= value <= _INT64_MAX:
        raise ValueError("invalid drain counter")
    return value


def _digest(value: Any) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError("invalid drain digest")
    return value


def _private_path(value: Any) -> str:
    if (type(value) is not str or not 1 <= len(value) <= 4096
            or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        raise ValueError("invalid drain checkpoint location")
    path = PurePosixPath(value)
    if (not path.is_absolute() or value.startswith("//") or value == "/"
            or ".." in path.parts or str(path) != value):
        raise ValueError("invalid drain checkpoint location")
    return value


def _session(value: Any) -> dict[str, Any]:
    data = _object(value, _SESSION_FIELDS)
    result: dict[str, Any] = {
        "session_id": _identity(data["session_id"]),
        "episode_id": _identity(data["episode_id"]),
        "episode_index": _counter(data["episode_index"]),
        "steps": _counter(data["steps"]),
        "episode_steps": _counter(data["episode_steps"]),
    }
    if result["episode_steps"] > result["steps"]:
        raise ValueError("inconsistent drain session counters")
    return result


@dataclass(frozen=True, slots=True)
class _CheckpointBinding:
    path: str = field(repr=False)
    payload_contract: str
    model_sha256: str = field(repr=False)
    configuration_sha256: str = field(repr=False)

    @classmethod
    def parse(cls, value: Any) -> _CheckpointBinding:
        data = _object(value, _DESCRIPTOR_FIELDS)
        return cls(
            _private_path(data["path"]), _identity(data["payload_contract"]),
            _digest(data["model_sha256"]), _digest(data["configuration_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class DrainBinding:
    """Frozen READY identity and conservative counters for a terminal barrier."""

    session_id: str
    episode_id: str
    episode_index: int
    steps: int
    episode_steps: int
    _checkpoint: _CheckpointBinding | None = field(repr=False)

    @classmethod
    def from_ready(cls, ready: dict[str, Any], *, model_sha256: str) -> DrainBinding | None:
        if type(ready) is not dict:
            raise ValueError("invalid worker readiness metadata")
        if "drain_contract" not in ready:
            return None
        if type(ready["drain_contract"]) is not str or ready["drain_contract"] != DRAIN_CONTRACT:
            raise ValueError("unsupported worker drain contract")
        configured_digest = _digest(model_sha256)
        session = _session(ready.get("session"))
        if "checkpoint" not in ready:
            raise ValueError("missing drain checkpoint descriptor")
        checkpoint = (
            None if ready["checkpoint"] is None
            else _CheckpointBinding.parse(ready["checkpoint"])
        )
        if checkpoint is not None and checkpoint.model_sha256 != configured_digest:
            raise ValueError("drain checkpoint model mismatch")
        return cls(**session, _checkpoint=checkpoint)

    def with_world_episode(self, episode_id: str) -> DrainBinding:
        """Track an explicit queued world reset; READY steps remain a lower bound."""
        episode_id = _identity(episode_id)
        if episode_id == self.episode_id:
            raise ValueError("world reset requires a new drain episode")
        return replace(
            self, episode_id=episode_id, episode_index=_counter(self.episode_index + 1),
            episode_steps=0,
        )

    def validate_ack(
        self, payload: dict[str, Any], *, request_id: str, after_request_id: str | None,
    ) -> dict[str, Any]:
        """Return detached private metadata; reject missing or mismatched evidence."""
        expected_request = _request_id(request_id)
        barrier = None if after_request_id is None else _request_id(after_request_id)
        data = _object(payload, _ACK_FIELDS)
        if (type(data["type"]) is not str or data["type"] != "drained"
                or type(data["contract"]) is not str or data["contract"] != DRAIN_CONTRACT
                or _request_id(data["request_id"]) != expected_request
                or data["drained"] is not True
                or type(data["status"]) is not str or data["status"] != "checkpointed"):
            raise ValueError("invalid drain acknowledgment")
        session = _session(data["session"])
        if (session["session_id"] != self.session_id or session["episode_id"] != self.episode_id
                or session["episode_index"] != self.episode_index
                or session["steps"] < self.steps or session["episode_steps"] < self.episode_steps):
            raise ValueError("drain session binding mismatch")
        inference = _inference(data["last_completed_inference"], barrier, session["steps"])
        checkpoint = _checkpoint(data["checkpoint"], self._checkpoint, session["steps"])
        return {
            "type": "drained", "contract": DRAIN_CONTRACT, "request_id": expected_request,
            "drained": True, "status": "checkpointed", "session": session,
            "last_completed_inference": inference, "checkpoint": checkpoint,
        }


def _inference(value: Any, barrier: str | None, steps: int) -> dict[str, Any] | None:
    if barrier is None:
        if value is not None:
            raise ValueError("unexpected drain inference barrier")
        return None
    data = _object(value, _INFERENCE_FIELDS)
    before, after = _counter(data["session_steps_before"]), _counter(data["session_steps_after"])
    if (_request_id(data["request_id"]) != barrier or not before <= after <= steps
            or type(data["outcome"]) is not str or data["outcome"] not in {"prediction", "error"}
            or type(data["prediction_written"]) is not bool
            or (data["outcome"] == "prediction" and not data["prediction_written"])):
        raise ValueError("invalid drain inference barrier")
    # A queued reset may follow the last inference. Its episode is intentionally
    # not required to equal the checkpoint's current world episode.
    return {
        "request_id": barrier, "episode_id": _identity(data["episode_id"]),
        "session_steps_before": before, "session_steps_after": after,
        "outcome": data["outcome"], "prediction_written": data["prediction_written"],
    }


def _checkpoint(
    value: Any, binding: _CheckpointBinding | None, steps: int,
) -> dict[str, Any]:
    data = _object(value, _CHECKPOINT_FIELDS)
    descriptor = _CheckpointBinding.parse({key: data[key] for key in _DESCRIPTOR_FIELDS})
    if binding is None or descriptor != binding:
        raise ValueError("drain checkpoint binding mismatch")
    size = _counter(data["bytes"])
    if (not 1 <= size <= _MAX_CHECKPOINT_BYTES
            or data["file_fsync_completed"] is not True
            or data["directory_fsync_completed"] is not True
            or type(data["atomicity_scope"]) is not str
            or data["atomicity_scope"] != "one_native_session_file"
            or type(data["learning_state_embedded"]) is not bool):
        raise ValueError("invalid drain checkpoint receipt")
    learning: dict[str, Any] | None = None
    if data["learning_state_embedded"]:
        row = _object(data["learning"], _LEARNING_FIELDS)
        learning = {
            "update_algorithm": _identity(row["update_algorithm"]),
            **{key: _counter(row[key]) for key in _LEARNING_FIELDS - {"update_algorithm"}},
        }
        if (not learning["train_cursor"] <= learning["prefix_cursor"] <= steps
                or learning["parameter_update_events"] > learning["training_events"]):
            raise ValueError("inconsistent drain learning counters")
    elif data["learning"] is not None:
        raise ValueError("unexpected drain learning metadata")
    return {
        "path": descriptor.path, "payload_contract": descriptor.payload_contract,
        "model_sha256": descriptor.model_sha256,
        "configuration_sha256": descriptor.configuration_sha256,
        "sha256": _digest(data["sha256"]), "bytes": size,
        "file_fsync_completed": True, "directory_fsync_completed": True,
        "atomicity_scope": "one_native_session_file",
        "learning_state_embedded": data["learning_state_embedded"], "learning": learning,
    }
