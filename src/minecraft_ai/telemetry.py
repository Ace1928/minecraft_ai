from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from platformdirs import user_runtime_dir


TELEMETRY_FILE = Path(user_runtime_dir("minecraft-ai")) / "telemetry.json"


@dataclass
class TelemetryPublisher:
    path: Path = TELEMETRY_FILE
    interval_ms: int = 250
    _last_publish_ns: int = field(default=0, init=False)

    def publish(self, payload: dict[str, object], *, force: bool = False) -> bool:
        now = time.monotonic_ns()
        if not force and now - self._last_publish_ns < self.interval_ms * 1_000_000:
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        staged = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        staged.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        staged.replace(self.path)
        self._last_publish_ns = now
        return True


def read_telemetry(path: Path = TELEMETRY_FILE) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None
