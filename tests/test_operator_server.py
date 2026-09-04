from __future__ import annotations

import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_ai.config import AppPaths
from minecraft_ai.agent_lifecycle import AgentProcess
from minecraft_ai.operator_server import (
    OperatorRequestHandler,
    _capture_live_bedrock_frame,
    operator_readiness,
)
from minecraft_ai.perception_service import frame_dhash
from minecraft_ai.platforms.bedrock_x11 import CapturedFrame
from minecraft_ai.storage import StateDatabase


def test_operator_readiness_requires_fresh_matching_motor_telemetry(monkeypatch) -> None:
    now = time.monotonic_ns()
    session = SimpleNamespace(
        mode="weston",
        display=":71",
        host_display=":0",
        find_window=lambda: 42,
    )
    status = {
        "emergency_stop": {"latched": False},
        "supervisor_reachable": True,
        "supervisor": {
            "state": "RUNNING",
            "live_capable": True,
            "motor_lease_active": True,
            "motor_lease_id": "lease-1",
        },
        "agent": {"alive": True, "display": ":71", "window_id": 42},
        "telemetry": {
            "state": "running",
            "lease_id": "lease-1",
            "updated_monotonic_ns": now,
            "trajectory_recording": {"enabled": True},
        },
    }
    monkeypatch.setattr("minecraft_ai.operator_server.operator_status", lambda: status)
    monkeypatch.setattr(
        "minecraft_ai.operator_server.BedrockSession.load", lambda: session
    )
    monkeypatch.setattr(
        "minecraft_ai.operator_server.bedrock_session_alive", lambda _session: True
    )
    monkeypatch.setattr(
        "minecraft_ai.operator_server._capture_live_bedrock_frame",
        lambda: CapturedFrame(1, now, 1, 1, b"\x00\x00\x00\xff"),
    )
    monkeypatch.setattr(
        "minecraft_ai.operator_server.bedrock_in_world_hud_present", lambda _frame: True
    )
    monkeypatch.setattr("minecraft_ai.operator_server.time.monotonic_ns", lambda: now)

    ready, payload = operator_readiness()
    assert ready is True
    assert payload["checks"]["telemetry_fresh"] is True
    assert payload["checks"]["lease_consistent"] is True
    assert payload["checks"]["trajectory_recording"] is True
    assert payload["degradations"] == []

    status["telemetry"]["trajectory_recording"] = {
        "enabled": False,
        "disabled_reason": "disk reserve reached",
    }
    ready, payload = operator_readiness()
    assert ready is True
    assert payload["checks"]["trajectory_recording"] is False
    assert payload["degradations"] == [
        "trajectory learning is disabled: disk reserve reached"
    ]
    status["telemetry"]["trajectory_recording"] = {"enabled": True}

    status["telemetry"]["updated_monotonic_ns"] = now - 5_000_000_001
    ready, payload = operator_readiness()
    assert ready is False
    assert "telemetry" in " ".join(payload["reasons"])

    status["telemetry"]["updated_monotonic_ns"] = now
    status["telemetry"]["lease_id"] = "wrong-lease"
    ready, payload = operator_readiness()
    assert ready is False
    assert payload["checks"]["lease_consistent"] is False

    status["telemetry"]["lease_id"] = "lease-1"
    monkeypatch.setattr(
        "minecraft_ai.operator_server.bedrock_in_world_hud_present", lambda _frame: False
    )
    ready, payload = operator_readiness(require_playable_capture=False)
    assert ready is True
    assert payload["checks"]["live_capture"] is True
    assert payload["checks"]["playable_capture"] is False
    assert "outside" in " ".join(payload["degradations"])


def test_stale_agent_descriptor_cannot_authorize_host_capture(monkeypatch) -> None:
    process = AgentProcess(
        pid=999,
        started_ns=1,
        display=":0",
        window_id=123,
        instance_id="stale",
        role="generalist",
        allow_host_capture=True,
    )
    monkeypatch.setattr(
        "minecraft_ai.operator_server.BedrockSession.load",
        lambda: (_ for _ in ()).throw(FileNotFoundError()),
    )
    monkeypatch.setattr("minecraft_ai.operator_server.AgentProcess.load", lambda: process)
    monkeypatch.setattr("minecraft_ai.operator_server.agent_alive", lambda _process: False)
    monkeypatch.setattr(
        "minecraft_ai.operator_server.create_bedrock_capture",
        lambda **_kwargs: pytest.fail("stale descriptor must not create a capture"),
    )

    assert _capture_live_bedrock_frame() is None


def test_operator_pause_revokes_supervisor_before_waiting_for_agent(
    monkeypatch,
) -> None:
    calls: list[str] = []
    handler = object.__new__(OperatorRequestHandler)
    monkeypatch.setattr("minecraft_ai.operator_server.supervisor_alive", lambda: True)
    monkeypatch.setattr(
        "minecraft_ai.operator_server.send_command",
        lambda command, **_kwargs: calls.append(command)
        or {
            "state": "PAUSED",
            "operator_pause_persisted": True,
            "agent_containment_confirmed": True,
        },
    )
    monkeypatch.setattr(
        "minecraft_ai.operator_server.stop_agent_process",
        lambda: calls.append("stop-agent"),
    )
    monkeypatch.setattr(
        "minecraft_ai.operator_server.stop_persistent_agent_service",
        lambda: calls.append("stop-service") or True,
    )
    monkeypatch.setattr(
        "minecraft_ai.operator_server.latch_operator_pause",
        lambda: calls.append("latch-pause"),
    )
    monkeypatch.setattr(
        OperatorRequestHandler,
        "_send_json",
        lambda _self, status, _payload: calls.append(f"http-{status}"),
    )

    handler._pause_agent()

    assert calls == ["pause", f"http-{HTTPStatus.OK}"]


def test_operator_pause_still_revokes_when_durable_marker_write_fails(
    monkeypatch,
) -> None:
    calls: list[str] = []
    handler = object.__new__(OperatorRequestHandler)

    def fail_latch() -> None:
        raise OSError("read-only data directory")

    monkeypatch.setattr("minecraft_ai.operator_server.latch_operator_pause", fail_latch)
    monkeypatch.setattr("minecraft_ai.operator_server.supervisor_alive", lambda: True)
    monkeypatch.setattr(
        "minecraft_ai.operator_server.send_command",
        lambda command, **_kwargs: calls.append(command)
        or {
            "state": "PAUSED",
            "operator_pause_persisted": False,
            "agent_containment_confirmed": True,
        },
    )
    monkeypatch.setattr(
        "minecraft_ai.operator_server.stop_agent_process",
        lambda: calls.append("stop-agent"),
    )
    monkeypatch.setattr(
        "minecraft_ai.operator_server.stop_persistent_agent_service",
        lambda: calls.append("stop-service") or True,
    )
    monkeypatch.setattr(
        OperatorRequestHandler,
        "_send_json",
        lambda _self, status, _payload: calls.append(f"http-{status}"),
    )

    handler._pause_agent()

    assert calls == [
        "pause",
        "stop-service",
        f"http-{HTTPStatus.SERVICE_UNAVAILABLE}",
    ]


def test_operator_pause_fails_closed_when_local_agent_survives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    handler = object.__new__(OperatorRequestHandler)
    agent_file = tmp_path / "agent-process.json"
    agent_file.write_text("{malformed", encoding="utf-8")
    monkeypatch.setattr("minecraft_ai.operator_server.AGENT_FILE", agent_file)
    monkeypatch.setattr("minecraft_ai.operator_server.supervisor_alive", lambda: False)
    monkeypatch.setattr(
        "minecraft_ai.operator_server.current_control_owner_state", lambda: "absent"
    )
    monkeypatch.setattr("minecraft_ai.operator_server.latch_operator_pause", lambda: None)
    monkeypatch.setattr(
        "minecraft_ai.operator_server.terminate_registered_supervisor", lambda: False
    )
    monkeypatch.setattr(
        "minecraft_ai.operator_server.stop_agent_process", lambda: False
    )
    monkeypatch.setattr(
        OperatorRequestHandler,
        "_send_json",
        lambda _self, status, _payload: calls.append(f"http-{status}"),
    )

    handler._pause_agent()

    assert calls == [f"http-{HTTPStatus.SERVICE_UNAVAILABLE}"]


def test_operator_http_message_roundtrip(tmp_path: Path, monkeypatch) -> None:
    paths = AppPaths(
        config_dir=tmp_path,
        data_dir=tmp_path,
        config_file=tmp_path / "config.yaml",
        state_db=tmp_path / "state.sqlite3",
        wiki_cache=tmp_path / "wiki",
        knowledge_dir=tmp_path / "knowledge",
    )
    monkeypatch.setattr("minecraft_ai.operator_server.app_paths", lambda: paths)
    reference_frame = CapturedFrame(
        frame_id=17,
        captured_ns=123_456,
        width=4,
        height=4,
        bgra=bytes([0, 128, 255, 255]) * 16,
    )
    monkeypatch.setattr(
        "minecraft_ai.operator_server._capture_live_bedrock_frame",
        lambda: reference_frame,
    )
    monkeypatch.setattr(
        "minecraft_ai.operator_server.operator_readiness",
        lambda: (
            True,
            {"status": "ready", "checks": {"agent_attached": True}, "reasons": []},
        ),
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), OperatorRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        for hostile_host in (
            f"untrusted.example:{port}",
            f"127.0.0.2:{port}",
            f"{host}:{port + 1}",
        ):
            rejected_get = urllib.request.Request(
                f"http://{host}:{port}/api/status",
                headers={"Host": hostile_host},
            )
            with pytest.raises(urllib.error.HTTPError) as rejected_error:
                urllib.request.urlopen(rejected_get, timeout=2)
            assert rejected_error.value.code == HTTPStatus.BAD_REQUEST

        rebound_post = urllib.request.Request(
            f"http://{host}:{port}/api/messages",
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Host": f"untrusted.example:{port}",
                "Origin": f"http://untrusted.example:{port}",
            },
            data=b"{}",
        )
        with pytest.raises(urllib.error.HTTPError) as rejected_error:
            urllib.request.urlopen(rebound_post, timeout=2)
        assert rejected_error.value.code == HTTPStatus.BAD_REQUEST

        for headers in (
            {"Content-Type": "text/plain"},
            {
                "Content-Type": "application/json",
                "Origin": "https://untrusted.example",
            },
        ):
            rejected = urllib.request.Request(
                f"http://{host}:{port}/api/messages",
                method="POST",
                headers=headers,
                data=b"{}",
            )
            with pytest.raises(urllib.error.HTTPError) as rejected_error:
                urllib.request.urlopen(rejected, timeout=2)
            assert rejected_error.value.code == HTTPStatus.BAD_REQUEST

        request = urllib.request.Request(
            f"http://{host}:{port}/api/messages",
            method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps(
                {"kind": "instruction", "priority": 0.9, "text": "Inspect the workshop."}
            ).encode(),
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            assert response.status == 201
        with StateDatabase(paths.state_db) as database:
            messages = database.load_operator_messages()
        assert messages[0].text == "Inspect the workshop."

        with urllib.request.urlopen(f"http://{host}:{port}/api/frame.png", timeout=2) as response:
            assert response.status == 200
            assert response.headers["Content-Type"] == "image/jpeg"
            frame_token = response.headers["X-Minecraft-Frame-Token"]
            assert response.headers["X-Minecraft-Frame-Width"] == "4"
            assert response.headers["X-Minecraft-Frame-Height"] == "4"
            assert response.headers["X-Minecraft-HUD-Complete"] == "false"
            assert "img-src 'self' blob:" in response.headers["Content-Security-Policy"]
            assert len(response.read()) > 0

        target_request = urllib.request.Request(
            f"http://{host}:{port}/api/target",
            method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps(
                {
                    "label": "oak_log",
                    "x": 0.4,
                    "y": 0.2,
                    "width": 0.15,
                    "height": 0.5,
                    "frame_token": frame_token,
                }
            ).encode(),
        )
        with urllib.request.urlopen(target_request, timeout=2) as response:
            target = json.load(response)
            assert response.status == 201
        assert target["label"] == "oak_log"
        with StateDatabase(paths.state_db) as database:
            stored_target = database.load_operator_target()
        assert stored_target is not None
        assert stored_target.region.height == 0.5
        assert stored_target.attributes["reference_dhash"] == frame_dhash(reference_frame)
        assert stored_target.attributes["reference_frame_id"] == 17
        assert stored_target.attributes["reference_captured_ns"] == 123_456
        reference_path = Path(str(stored_target.attributes["reference_image_path"]))
        assert reference_path.is_file()
        assert (
            hashlib.sha256(reference_path.read_bytes()).hexdigest()
            == (stored_target.attributes["reference_image_sha256"])
        )

        replay_request = urllib.request.Request(
            f"http://{host}:{port}/api/target",
            method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps(
                {
                    "label": "replayed",
                    "x": 0.1,
                    "y": 0.1,
                    "width": 0.1,
                    "height": 0.1,
                    "frame_token": frame_token,
                }
            ).encode(),
        )
        try:
            urllib.request.urlopen(replay_request, timeout=2)
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
            assert "expired" in json.load(exc)["error"]
        else:
            raise AssertionError("consumed frame token was accepted twice")

        with urllib.request.urlopen(f"http://{host}:{port}/healthz", timeout=2) as response:
            assert json.load(response) == {"status": "ok"}
        with urllib.request.urlopen(f"http://{host}:{port}/readyz", timeout=2) as response:
            assert json.load(response)["status"] == "ready"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
