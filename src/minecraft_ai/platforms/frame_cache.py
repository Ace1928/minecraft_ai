"""Private, bounded spectator frames from the agent's existing capture owner.

The live player captures once. Spectators read an atomic pixel snapshot, rather
than making another large XGetImage request on the private game display.
"""

from __future__ import annotations

import json
import os
import stat
import struct
import tempfile
import time
from pathlib import Path
from typing import Protocol

from minecraft_ai.agent_lifecycle import AgentProcess, RUNTIME_DIR
from minecraft_ai.platforms.bedrock_x11 import CapturedFrame

FRAME_CACHE_FILE = RUNTIME_DIR / "agent-frame.bgra"
MAX_FRAME_BYTES = 32 * 1024 * 1024
MAX_HEADER_BYTES = 4096
FRAME_CACHE_INTERVAL_NS = 250_000_000
FRAME_CACHE_MAX_AGE_NS = 500_000_000


class CaptureSource(Protocol):
    def capture(self) -> CapturedFrame: ...

    def close(self) -> None: ...


def _owner_payload(process: AgentProcess) -> dict[str, object]:
    return {
        "pid": process.pid,
        "started_ns": process.started_ns,
        "proc_start_ticks": process.proc_start_ticks,
        "command_sha256": process.command_sha256,
        "display": process.display,
        "window_id": process.window_id,
        "instance_id": process.instance_id,
        "allow_host_capture": process.allow_host_capture,
    }


def _pixel_count(width: object, height: object) -> int | None:
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        return None
    size = width * height * 4
    return size if size <= MAX_FRAME_BYTES else None


class PublishedFrameCapture:
    """Best-effort spectator publication; capture/input authority is unchanged."""

    def __init__(
        self, source: CaptureSource, owner: AgentProcess, *, path: Path = FRAME_CACHE_FILE,
    ) -> None:
        self.source = source
        self.owner = owner
        self.path = path
        self._last_publish_ns = 0
        self._published_identity: tuple[int, int] | None = None

    def capture(self) -> CapturedFrame:
        frame = self.source.capture()
        now = time.monotonic_ns()
        if now - self._last_publish_ns >= FRAME_CACHE_INTERVAL_NS:
            try:
                self._publish(frame)
            except (OSError, ValueError):
                # A spectator/storage failure cannot disable gameplay.
                pass
            self._last_publish_ns = now
        return frame

    def _publish(self, frame: CapturedFrame) -> None:
        size = _pixel_count(frame.width, frame.height)
        if size is None or len(frame.bgra) != size:
            raise ValueError("invalid spectator frame geometry")
        metadata = {
            "schema_version": 1,
            "owner": _owner_payload(self.owner),
            "frame_id": frame.frame_id,
            "captured_ns": frame.captured_ns,
            "width": frame.width,
            "height": frame.height,
        }
        header = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
        if len(header) > MAX_HEADER_BYTES:
            raise ValueError("spectator frame header exceeds bound")
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        parent = self.path.parent.lstat()
        if not stat.S_ISDIR(parent.st_mode) or parent.st_uid != os.getuid():
            raise ValueError("spectator frame directory is not privately owned")
        fd, staged_name = tempfile.mkstemp(prefix=".agent-frame-", dir=self.path.parent)
        staged = Path(staged_name)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(struct.pack("!I", len(header)))
                stream.write(header)
                stream.write(frame.bgra)
                stream.flush()
                info = os.fstat(stream.fileno())
            staged.replace(self.path)
            self._published_identity = (info.st_dev, info.st_ino)
        finally:
            staged.unlink(missing_ok=True)

    def close(self) -> None:
        try:
            if self._published_identity is not None:
                try:
                    info = self.path.lstat()
                    if (info.st_dev, info.st_ino) == self._published_identity:
                        self.path.unlink()
                except OSError:
                    pass
        finally:
            self.source.close()


def read_agent_frame(
    process: AgentProcess, *, path: Path = FRAME_CACHE_FILE,
    max_age_ns: int = FRAME_CACHE_MAX_AGE_NS,
) -> CapturedFrame | None:
    """Read only exact-owner, complete, fresh pixels; never start a capture."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) & 0o077
                or info.st_size < 4 or info.st_size > 4 + MAX_HEADER_BYTES + MAX_FRAME_BYTES
            ):
                return None
            size_raw = stream.read(4)
            if len(size_raw) != 4:
                return None
            header_size = struct.unpack("!I", size_raw)[0]
            if not 0 < header_size <= MAX_HEADER_BYTES:
                return None
            metadata = json.loads(stream.read(header_size))
            expected_owner = _owner_payload(process)
            owner = metadata.get("owner") if type(metadata) is dict else None
            if (
                type(metadata) is not dict or type(metadata.get("schema_version")) is not int
                or metadata.get("schema_version") != 1 or type(owner) is not dict
                or owner != expected_owner
                or any(type(owner.get(key)) is not type(value)
                       for key, value in expected_owner.items())
            ):
                return None
            width, height = metadata.get("width"), metadata.get("height")
            pixel_size = _pixel_count(width, height)
            captured_ns, frame_id = metadata.get("captured_ns"), metadata.get("frame_id")
            now = time.monotonic_ns()
            if (
                pixel_size is None or info.st_size != 4 + header_size + pixel_size
                or type(captured_ns) is not int or not 0 < captured_ns <= now
                or now - captured_ns > max_age_ns
                or type(frame_id) is not int or frame_id < 0
            ):
                return None
            pixels = stream.read(pixel_size + 1)
            if len(pixels) != pixel_size:
                return None
            assert type(width) is int and type(height) is int
            return CapturedFrame(frame_id, captured_ns, width, height, pixels)
    except (OSError, ValueError, UnicodeError):
        return None
