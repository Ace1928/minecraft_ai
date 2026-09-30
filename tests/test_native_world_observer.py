from __future__ import annotations

import hashlib
import json
import time

import pytest
from pydantic import ValidationError

from minecraft_ai.native_world_observer import (
    NativeGameObservationRequest, NativeGameObservationResult, NativeWorldGameObserver,
    TASK_OUTPUTS,
)
from minecraft_ai.platforms.bedrock_x11 import CapturedFrame


def _request(task="resource", **updates):
    frame = CapturedFrame(7, time.monotonic_ns() - 1_000_000, 2, 2, b"\0\0\0\xff" * 4)
    values = dict(
        request_id="bounded-query", instance_id="bedrock:1.26.52.3:x11:42",
        game_version="1.26.52.3", pack_catalog_sha256="4" * 64, task=task,
        frame_id=frame.frame_id, captured_ns=frame.captured_ns,
        width=frame.width, height=frame.height, pixel_sha256=hashlib.sha256(frame.bgra).hexdigest(),
        deadline_ns=frame.captured_ns + 5_000_000_000,
    )
    values.update(updates)
    return NativeGameObservationRequest(**values), frame


def _ready(tmp_path, **updates):
    tmp_path.chmod(0o700)
    path = tmp_path / "ready.json"
    path.write_text(json.dumps({
        "status": "private_ready", "runtime_id": "a" * 32,
        "vision_backend": {"scope": "native_clip_visual_matches", "native_execution": True},
        **updates,
    }))
    path.chmod(0o600)
    return path


@pytest.mark.parametrize("task", ("resource", "inventory", "chat"))
def test_real_clip_capability_explicitly_cannot_supply_game_observation_outputs(tmp_path, task):
    request, frame = _request(task)
    result = NativeWorldGameObserver(str(_ready(tmp_path))).inspect(request, frame)
    assert result.status == "unsupported"
    assert result.reason == "native_clip_has_no_qualified_game_outputs"
    assert result.unsupported_outputs == TASK_OUTPUTS[task]
    assert result.request_id == request.request_id and result.pixel_sha256 == request.pixel_sha256
    assert result.frame_id == frame.frame_id and result.captured_ns == frame.captured_ns
    assert result.runtime_id == "a" * 32
    assert not {"observations", "facts", "tracks", "chat"}.intersection(result.model_dump())


@pytest.mark.parametrize("vision", (None, {}, {"qualified": True}, {
    "scope": "native_clip_visual_matches", "native_execution": False, "object_detection": True,
}))
def test_flags_and_unfamiliar_backend_cannot_enable_semantic_authority(tmp_path, vision):
    request, frame = _request()
    ready = _ready(tmp_path, vision_backend=vision, minecraft_observer={"qualified": True})
    result = NativeWorldGameObserver(str(ready)).inspect(request, frame)
    assert result.status == "unsupported"
    assert result.reason == "qualified_observer_transport_unavailable"


@pytest.mark.parametrize("updates", ({"status": "public_ready"}, {"runtime_id": True},
                                     {"runtime_id": "other-owner"}))
def test_unbound_world_readiness_is_unavailable(tmp_path, updates):
    request, frame = _request()
    result = NativeWorldGameObserver(str(_ready(tmp_path, **updates))).inspect(request, frame)
    assert result.status == "unavailable" and result.runtime_id is None


def test_nonprivate_readiness_does_not_advertise_an_observer(tmp_path):
    request, frame = _request()
    ready = _ready(tmp_path)
    ready.chmod(0o644)
    result = NativeWorldGameObserver(str(ready)).inspect(request, frame)
    assert result.reason == "world_readiness_unavailable"


def test_deadline_is_checked_before_reading_world_assets(tmp_path, monkeypatch):
    request, frame = _request()
    monkeypatch.setattr("minecraft_ai.native_world_observer.time.monotonic_ns",
                        lambda: request.deadline_ns)
    result = NativeWorldGameObserver(str(tmp_path / "absent")).inspect(request, frame)
    assert result.status == "expired" and result.reason == "frame_deadline_expired"


@pytest.mark.parametrize("field,value", (("frame_id", 8), ("pixel_sha256", "f" * 64),
                                       ("width", 3), ("captured_ns", 1)))
def test_changed_frame_cannot_reuse_request_authority(tmp_path, field, value):
    request, frame = _request()
    # Bypass construction here to isolate the supplied-pixel binding guard.
    request = request.model_copy(update={field: value})
    with pytest.raises(ValueError, match="supplied game pixels"):
        NativeWorldGameObserver(str(_ready(tmp_path))).inspect(request, frame)


@pytest.mark.parametrize("values", (
    {"task": "detect_anything"}, {"game_version": "1.26"}, {"frame_id": True},
    {"instance_id": "java:1.26.52.3:x11:42"}, {"pack_catalog_sha256": "unknown"},
    {"width": 8192, "height": 8192}, {"deadline_ns": 1},
))
def test_request_refuses_unscoped_unbounded_or_untyped_inputs(values):
    with pytest.raises(ValidationError):
        _request(**values)


def test_positive_or_injected_outputs_are_not_part_of_unqualified_result_contract():
    request, _ = _request()
    values = dict(
        request_id=request.request_id, frame_id=request.frame_id, captured_ns=request.captured_ns,
        pixel_sha256=request.pixel_sha256, task=request.task, status="unsupported",
        reason="native_clip_has_no_qualified_game_outputs",
        unsupported_outputs=TASK_OUTPUTS["resource"],
    )
    with pytest.raises(ValidationError):
        NativeGameObservationResult(**{**values, "status": "observed"})
    with pytest.raises(ValidationError):
        NativeGameObservationResult(**values, facts={"target.kind": "oak_log"})
