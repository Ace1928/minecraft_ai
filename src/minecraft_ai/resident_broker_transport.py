"""Same-user private socket transport; no direct-model fallback or credentials."""

from __future__ import annotations

import json
import os
import re
import select
import socket
import stat
import struct
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

CONTRACT = "erais.private.minecraft-resident.v1"
MODEL_ID = "erais-dense-gemma4-e4b"
UPSTREAM_MODEL = "gemma-4-e4b-vl"


class BrokerError(RuntimeError):
    def __init__(self, code: str, status: int = 503):
        self.response = SimpleNamespace(status_code=status)
        super().__init__(code)


def _parent(path: Path) -> tuple[int, str]:
    if not path.is_absolute() or any(p in {".", ".."} for p in path.parts[1:]):
        raise ValueError("broker socket must be an absolute canonical path")
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", path.name):
        raise ValueError("invalid broker socket name")
    flags = os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open("/", flags)
    try:
        for component in path.parts[1:-1]:
            child = os.open(component, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        metadata = os.fstat(fd)
        if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
            raise ValueError("broker directory must be owned by this user with mode 0700")
        return fd, path.name
    except BaseException:
        os.close(fd)
        raise


def exchange(
    path: Path,
    *,
    operation: str,
    deadline_ns: int,
    cancel_requested: Callable[[], bool],
    purpose: str | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request_id = uuid.uuid4().hex
    message = {
        "contract": CONTRACT,
        "request_id": request_id,
        "operation": operation,
        "deadline_ns": deadline_ns,
    }
    if operation == "infer":
        message.update(purpose=purpose, payload=payload)
    raw = json.dumps(message, separators=(",", ":"), allow_nan=False).encode()
    if len(raw) > 4_194_304:
        raise BrokerError("private model request exceeds bound", 400)

    def check() -> None:
        if cancel_requested() or time.monotonic_ns() >= deadline_ns:
            raise TimeoutError("private model request cancelled or expired")

    def wait(connection: socket.socket, *, write: bool = False) -> None:
        while True:
            check()
            seconds = min(0.025, max(0, (deadline_ns - time.monotonic_ns()) / 1e9))
            ready = select.select(
                [] if write else [connection], [connection] if write else [], [], seconds
            )
            if ready[1] if write else ready[0]:
                return

    check()
    fd, name = _parent(path)
    try:
        metadata = os.stat(name, dir_fd=fd, follow_symlinks=False)
        if (
            not stat.S_ISSOCK(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise BrokerError("broker socket ownership or mode changed")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            # A local connect has a short bound within the absolute request budget.
            connection.settimeout(max(0.001, min(0.25, (deadline_ns - time.monotonic_ns()) / 1e9)))
            connection.connect(f"/proc/self/fd/{fd}/{name}")
            _, uid, _ = struct.unpack(
                "3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
            )
            current = os.stat(name, dir_fd=fd, follow_symlinks=False)
            if uid != os.getuid() or (metadata.st_dev, metadata.st_ino) != (
                current.st_dev,
                current.st_ino,
            ):
                raise BrokerError("broker peer changed")
            connection.setblocking(False)
            outgoing = memoryview(struct.pack("!I", len(raw)) + raw)
            while outgoing:
                wait(connection, write=True)
                try:
                    sent = connection.send(outgoing)
                except BlockingIOError:
                    continue
                if sent <= 0:
                    raise BrokerError("broker disconnected")
                outgoing = outgoing[sent:]

            def read(size: int) -> bytes:
                data = bytearray()
                while len(data) < size:
                    wait(connection)
                    try:
                        chunk = connection.recv(size - len(data))
                    except BlockingIOError:
                        continue
                    if not chunk:
                        raise BrokerError("broker disconnected")
                    data.extend(chunk)
                return bytes(data)

            size = struct.unpack("!I", read(4))[0]
            if not 0 < size <= 1_049_600:
                raise BrokerError("broker response exceeds bound")
            result = json.loads(read(size))
            check()
    finally:
        os.close(fd)
    if (
        type(result) is not dict
        or result.get("contract") != CONTRACT
        or result.get("request_id") != request_id
        or result.get("model_id") != MODEL_ID
        or result.get("upstream_model") != UPSTREAM_MODEL
        or type(result.get("ok")) is not bool
    ):
        raise BrokerError("broker identity or response changed")
    if not result["ok"]:
        status = result.get("http_status")
        raise BrokerError(
            "private resident request refused", status if status in {400, 404, 422} else 503
        )
    return result


def checked_completion(result: object, *, max_tokens: int) -> dict[str, Any]:
    if type(result) is not dict or result.get("model") != UPSTREAM_MODEL:
        raise BrokerError("resident model identity changed")
    choices = result.get("choices")
    usage = result.get("usage")
    if (
        type(choices) is not list
        or len(choices) != 1
        or type(choices[0]) is not dict
        or choices[0].get("finish_reason") not in {"stop", "length"}
        or choices[0].get("index", 0) != 0
    ):
        raise BrokerError("invalid resident completion")
    message = choices[0].get("message")
    if (
        type(message) is not dict
        or type(message.get("content")) is not str
        or not message["content"].strip()
        or "\0" in message["content"]
        or message.get("tool_calls")
        or len(message["content"].encode()) > 524_288
    ):
        raise BrokerError("invalid resident text")
    if (
        type(usage) is not dict
        or any(
            type(usage.get(k)) is not int or not 0 <= usage[k] <= 9_007_199_254_740_991
            for k in ("prompt_tokens", "completion_tokens")
        )
        or usage["completion_tokens"] > max_tokens
    ):
        raise BrokerError("invalid resident usage")
    if "total_tokens" in usage and (
        type(usage["total_tokens"]) is not int
        or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]
    ):
        raise BrokerError("invalid resident usage total")
    return result
