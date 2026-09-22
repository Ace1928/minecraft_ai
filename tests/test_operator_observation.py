from __future__ import annotations

import base64
import hashlib
import io
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest
from PIL import Image, PngImagePlugin

from minecraft_ai.operator import observation, server


def packet():
    output = io.BytesIO()
    Image.new("RGB", (4, 3), (25, 70, 30)).save(output, format="PNG")
    image = dict(width=4, height=3, sha256=hashlib.sha256(output.getvalue()).hexdigest(),
                 png_base64=base64.b64encode(output.getvalue()).decode())
    return {
        "schema": observation.SCHEMA, "source_id": "native-policy", "stream_id": "a" * 32,
        "sequence": 8, "source_frame_id": 42, "source_captured_ns": "10000000000",
        "safety": {"game_only": True, "chat_free": True}, "input": image,
        "reconstruction": {**image, "supported_fraction": .5, "reprojection_rmse": .12},
        "receptive_fields": {"source_count": 40, "samples": [
            {"index": 7, "box": [.2, .1, .3, .5], "value": .8}]},
        "models": {
            "native_policy": {"available": True, "calls": 1, "binding_sha256": "b" * 64,
                              "source_unit_count": 16000, "population": {
                                  "identity": "c" * 64, "implementation_sha256": "d" * 64,
                                  "unit_count": 2, "layer_sizes": [1, 1],
                                  "positions": [[-.5, 0, 0, 0], [.5, 0, 0, 1]],
                                  "edges": [[0, 1]], "activity": [[1, .5]], "total_edges": 4}},
            "association_brain": {"available": False, "calls": 0, "binding_sha256": None,
                                  "source_unit_count": 0, "population": None},
        },
        "action": {"source_frame_id": 42, "source_captured_ns": "10000000000",
                   "kind": "prediction", "accepted": True, "buttons": ["forward"],
                   "camera": [0, .1], "outcome": "pending"},
    }


def test_projection_is_idempotent_allowlisted_and_source_owned():
    raw = packet()
    raw["private_configuration"] = "SECRET"
    raw["models"]["native_policy"]["weights"] = ["SECRET"]
    raw["input"]["path"] = "SECRET"
    raw["action"]["chat"] = "SECRET"
    result = observation.project_observation(raw, now_ns=11_000_000_000)
    assert "SECRET" not in json.dumps(result)
    assert result["frame_age_ms"] == 1000
    assert result["source_frame_id"] == 42
    assert result["models"]["association_brain"]["population"] is None
    assert result["models"]["native_policy"]["population"]["unit_count"] == 2
    assert observation.project_observation(result, now_ns=11_000_000_000) == result


@pytest.mark.parametrize("edit", [
    lambda r: r.update(source_captured_ns="15000000001"),
    lambda r: r.update(source_captured_ns="10000000000"),
    lambda r: r.update(source_frame_id=True),
    lambda r: r.update(stream_id="private"),
    lambda r: r["safety"].update(chat_free=False),
    lambda r: r["safety"].update(game_only=1),
    lambda r: r.update(online=False),
    lambda r: r["models"]["native_policy"].update(calls=0),
    lambda r: r["models"]["native_policy"].update(binding_sha256=None),
    lambda r: r["models"]["native_policy"].update(source_unit_count=1),
    lambda r: r["models"]["native_policy"]["population"].update(unit_count=1025),
    lambda r: r["models"]["association_brain"].update(calls=4),
    lambda r: r["input"].update(sha256="f" * 64),
    lambda r: r["input"].update(width=100),
    lambda r: r["receptive_fields"]["samples"][0].update(box=[.8, 0, .4, 1]),
    lambda r: r["receptive_fields"]["samples"][0].update(value=float("nan")),
    lambda r: r["action"].update(source_frame_id=9),
    lambda r: r["action"].update(buttons=["chat"]),
    lambda r: r["action"].update(outcome="succeeded"),
])
def test_invalid_or_unbound_evidence_fails_closed(edit):
    raw = packet()
    raw["source_captured_ns"] = "14000000000"
    raw["action"]["source_captured_ns"] = raw["source_captured_ns"]
    edit(raw)
    with pytest.raises((ValueError, TypeError)):
        observation.project_observation(raw, now_ns=15_000_000_000)


def test_png_metadata_and_wrong_affinity_are_rejected():
    raw = packet()
    for source, stream in [("association-brain", ""), ("native-policy", "b" * 32)]:
        with pytest.raises(ValueError):
            observation.project_observation(raw, now_ns=11_000_000_000,
                                            preferred_source=source, stream_id=stream)
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("private", "SECRET")
    output = io.BytesIO()
    Image.new("RGB", (4, 3)).save(output, format="PNG", pnginfo=metadata)
    raw["input"].update(sha256=hashlib.sha256(output.getvalue()).hexdigest(),
                        png_base64=base64.b64encode(output.getvalue()).decode())
    with pytest.raises(ValueError, match="metadata-free"):
        observation.project_observation(raw, now_ns=11_000_000_000)


def test_optional_hook_never_promotes_file_mtime_or_counter_activity(tmp_path, monkeypatch):
    path = tmp_path / "observation.json"
    monkeypatch.setattr(observation.time, "monotonic_ns", lambda: 11_000_000_000)
    assert observation.read_observation(path=path)["online"] is False
    observation.publish_observation(packet(), path=path)
    live = observation.read_observation(path=path, preferred_source="native-policy")
    assert live["online"] is True
    monkeypatch.setattr(observation.time, "monotonic_ns", lambda: 15_000_000_000)
    assert observation.read_observation(path=path)["online"] is False
    path.write_text(json.dumps({"frames": 100, "motor_actions": 200, "state": "running"}))
    assert observation.read_observation(path=path)["online"] is False
    path.write_text(" " * (observation.MAX_BYTES + 1))
    assert observation.read_observation(path=path)["online"] is False
    path.write_text("[" * 2000 + "]" * 2000)
    assert observation.read_observation(path=path)["online"] is False


def test_success_requires_source_recorded_outcome_evidence():
    raw = packet()
    raw["action"].update(outcome="succeeded", outcome_evidence_sha256="e" * 64)
    result = observation.project_observation(raw, now_ns=11_000_000_000)
    assert result["action"]["outcome"] == "succeeded"
    raw["action"]["accepted"] = False
    with pytest.raises(ValueError):
        observation.project_observation(raw, now_ns=11_000_000_000)


def test_phase_and_replay_fields_are_additive_and_default_honestly():
    raw = packet()
    legacy = observation.project_observation(raw, now_ns=11_000_000_000)
    assert legacy["state"] == "acting" and legacy["sample_replayed"] is False
    raw.update(state="reasoning", sample_replayed=True)
    projected = observation.project_observation(raw, now_ns=11_000_000_000)
    assert projected["state"] == "reasoning" and projected["sample_replayed"] is True
    # A projected packet remains projectable without rewriting its fields.
    assert observation.project_observation(projected, now_ns=11_000_000_000) == projected
    for edit in (lambda r: r.update(state="invented"),
                 lambda r: r.update(state=True),
                 lambda r: r.update(sample_replayed=1)):
        broken = packet()
        edit(broken)
        with pytest.raises(ValueError):
            observation.project_observation(broken, now_ns=11_000_000_000)


def test_expired_sample_reports_true_last_seen_gap_not_source_unavailable(tmp_path, monkeypatch):
    path = tmp_path / "observation.json"
    raw = packet()
    raw["source_id"] = "association-brain"
    # The producer's capture stamp is old; a stale packet must never look online.
    monkeypatch.setattr(observation.time, "monotonic_ns", lambda: 11_000_000_000)
    observation.publish_observation(raw, path=path)
    live = observation.read_observation(
        preferred_source="association-brain", path=path)
    assert live["online"] is True and live["state"] == "acting"
    # Now the same stored packet is read 31 s past its capture stamp.
    monkeypatch.setattr(observation.time, "monotonic_ns", lambda: 41_000_000_000)
    gap = observation.read_observation(
        preferred_source="association-brain", path=path)
    assert gap["online"] is False and gap["reason"] == "sample_expired"
    assert gap["last_seen_age_ms"] == 31_000.0
    assert gap["source_frame_id"] == raw["source_frame_id"]
    # A different source or pinned stream must not learn another source's gap.
    assert observation.read_observation(
        preferred_source="native-policy", path=path)["reason"] == "source_unavailable"
    assert observation.read_observation(
        preferred_source="association-brain", stream_id="b" * 32,
        path=path)["reason"] == "source_unavailable"


def test_runtime_phase_hook_tracks_planner_state_without_owning_a_readout():
    from types import SimpleNamespace

    from minecraft_ai.runtime import AgentRuntime

    seen: list[str] = []
    runtime = SimpleNamespace(
        executor=SimpleNamespace(policy=SimpleNamespace(note_observer_state=seen.append)),
        _pending_decision=None,
        _traversal_escalation_pending=False,
    )
    AgentRuntime._note_observer_phase(runtime, None)
    assert seen == ["idle"]
    runtime._traversal_escalation_pending = True
    AgentRuntime._note_observer_phase(runtime, None)
    assert seen[-1] == "replanning"
    AgentRuntime._note_observer_phase(runtime, object())
    assert seen[-1] == "acting"
    runtime._pending_decision = object()
    AgentRuntime._note_observer_phase(runtime, object())
    assert seen[-1] == "reasoning"
    # A producer fault must never fault the motor loop, and a policy without
    # the opt-in hook is simply skipped.
    runtime.executor.policy.note_observer_state = lambda phase: 1 / 0
    AgentRuntime._note_observer_phase(runtime, None)
    AgentRuntime._note_observer_phase(SimpleNamespace(executor=None), None)


def test_operator_route_is_private_get_only_and_does_not_capture_or_actuate(monkeypatch):
    calls = []
    def read(**kwargs):
        calls.append(kwargs)
        return {"schema": observation.SCHEMA, "online": False, "reason": "source_unavailable"}

    monkeypatch.setattr(server, "read_observation", read)
    def forbidden(*args, **kwargs):
        pytest.fail("observer must not invoke game operations")

    for name in ("operator_status", "send_command", "_get_live_bedrock_frame"):
        if hasattr(server, name):
            monkeypatch.setattr(server, name, forbidden)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.OperatorRequestHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{httpd.server_port}/api/observation"
    try:
        for suffix, code in [("", 503), ("?source=private", 400), ("?stream_id=wrong", 400),
                             ("?url=http://private", 400), ("?source=native-policy&source=x", 400)]:
            with pytest.raises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(url + suffix, timeout=2)
            assert caught.value.code == code
            assert caught.value.headers["Cache-Control"] == "no-store"
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(urllib.request.Request(url, method="POST", data=b"{}",
                                   headers={"Content-Type": "application/json"}), timeout=2)
        assert caught.value.code == 404
        assert calls == [{"preferred_source": "association-brain", "stream_id": ""}]
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(urllib.request.Request(url, headers={"Host": "public.example"}),
                                   timeout=2)
        assert caught.value.code == 400
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)
