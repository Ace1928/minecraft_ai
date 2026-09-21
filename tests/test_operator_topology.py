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
from PIL import Image

from minecraft_ai.operator import observation, server
from minecraft_ai.operator.topology import EDGES, PARTS, build_topology


def _observation_packet():
    output = io.BytesIO()
    Image.new("RGB", (4, 3), (25, 70, 30)).save(output, format="PNG")
    image = dict(width=4, height=3, sha256=hashlib.sha256(output.getvalue()).hexdigest(),
                 png_base64=base64.b64encode(output.getvalue()).decode())
    raw = {
        "schema": observation.SCHEMA, "source_id": "association-brain", "stream_id": "a" * 32,
        "sequence": 8, "source_frame_id": 42, "source_captured_ns": "10000000000",
        "safety": {"game_only": True, "chat_free": True}, "input": image,
        "reconstruction": {**image, "supported_fraction": .5, "reprojection_rmse": .12},
        "receptive_fields": {"source_count": 40, "samples": []},
        "models": {
            "native_policy": {"available": False, "calls": 0, "binding_sha256": None,
                              "source_unit_count": 0, "population": None},
            "association_brain": {"available": True, "calls": 4, "binding_sha256": "b" * 64,
                                  "source_unit_count": 4096, "source_kc_count": 512,
                                  "source_active_kcs": 64, "activity_basis": "observed-activation",
                                  "population": {
                                      "identity": "c" * 64, "implementation_sha256": "d" * 64,
                                      "unit_count": 2, "layer_sizes": [1, 1],
                                      "positions": [[-.5, 0, 0, 0], [.5, 0, 0, 1]],
                                      "edges": [[0, 1]], "activity": [[1, .5]], "total_edges": 4}},
        },
        "action": None,
    }
    return observation.project_observation(raw, now_ns=10_500_000_000, preferred_source="association-brain")


def test_offline_topology_still_maps_every_part_and_edge():
    topology = build_topology({}, {"schema": observation.SCHEMA, "online": False,
                                   "reason": "source_unavailable"}, now_ns=1)
    assert topology["schema"] == "minecraft.topology.v1"
    assert [part["id"] for part in topology["parts"]] == [part[0] for part in PARTS]
    assert all(part["state"] == "offline" for part in topology["parts"])
    known = {part["id"] for part in topology["parts"]}
    for edge in topology["edges"]:
        assert edge["source"] in known and edge["target"] in known
        assert edge["flow"] == 0.0
    assert len(topology["edges"]) == len(EDGES)
    assert topology["observation"]["online"] is False
    assert topology["populations"] == []


def test_live_topology_binds_model_populations_and_activity():
    topology = build_topology(
        {"telemetry": {"last_capture_ms": 20, "active_skill": "mine_log",
                       "trajectory_recording": {"enabled": True, "written_steps": 12},
                       "policy": {"primary": {"model_version": "rocket2", "last_inference_ms": 12}}},
         "bedrock": {"version": "1.21", "instances": ["bedrock-1"]},
         "supervisor_reachable": True, "supervisor": {"state": "RUNNING"},
         "agent": {"alive": True}},
        _observation_packet(), now_ns=2)
    parts = {part["id"]: part for part in topology["parts"]}
    assert parts["capture"]["state"] == "live"
    assert parts["bedrock"]["state"] == "live"
    assert parts["association_brain"]["state"] == "live"
    assert parts["association_brain"]["detail"].startswith("4 calls")
    assert "64/512 KC busy" in parts["association_brain"]["detail"]
    assert parts["native_policy"]["state"] == "offline"
    assert any(edge["flow"] > 0 for edge in topology["edges"])
    assert len(topology["populations"]) == 1
    population = topology["populations"][0]
    assert population["model"] == "association_brain"
    assert population["unit_count"] == 2 and population["calls"] == 4
    assert population["activity_basis"] == "observed-activation"
    assert topology["observation"]["source_id"] == "association-brain"


def test_dashboard_exposes_the_topology_panels():
    assert 'id="topoCanvas"' in server.DASHBOARD_HTML
    assert 'id="brainCanvas"' in server.DASHBOARD_HTML
    assert "/api/topology" in server.DASHBOARD_HTML


def test_topology_route_is_get_only_and_validates_preferences(monkeypatch):
    calls = []

    def read(**kwargs):
        calls.append(kwargs)
        return {"schema": observation.SCHEMA, "online": False, "reason": "source_unavailable"}

    monkeypatch.setattr(server, "read_observation", read)
    monkeypatch.setattr(server, "operator_status", lambda: {"telemetry": None})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.OperatorRequestHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{httpd.server_port}/api/topology"
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            payload = json.loads(response.read())
        assert response.status == 200 and payload["schema"] == "minecraft.topology.v1"
        for suffix in ("?source=private", "?stream_id=wrong", "?source=native-policy&source=x"):
            with pytest.raises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(url + suffix, timeout=2)
            assert caught.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(urllib.request.Request(
                url, method="POST", data=b"{}", headers={"Content-Type": "application/json"}), timeout=2)
        assert caught.value.code == 404
        assert calls == [{"preferred_source": "association-brain", "stream_id": ""}]
    finally:
        httpd.shutdown()
        httpd.server_close()
