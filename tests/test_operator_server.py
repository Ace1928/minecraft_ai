from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from minecraft_ai.config import AppPaths
from minecraft_ai.operator_server import OperatorRequestHandler
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

        with urllib.request.urlopen(f"http://{host}:{port}/healthz", timeout=2) as response:
            assert json.load(response) == {"status": "ok"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
