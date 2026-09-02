from __future__ import annotations

import hashlib
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from minecraft_ai.config import AppPaths
from minecraft_ai.operator_server import OperatorRequestHandler
from minecraft_ai.perception_service import frame_dhash
from minecraft_ai.platforms.bedrock_x11 import CapturedFrame
from minecraft_ai.storage import StateDatabase


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
    server = ThreadingHTTPServer(("127.0.0.1", 0), OperatorRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
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
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
