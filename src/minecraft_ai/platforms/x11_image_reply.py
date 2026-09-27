"""Bounded coalescing of large replies on one privately owned Xlib connection.

python-xlib concatenates immutable buffers for every partial reply. Accumulate
the *already framed* large reply here once, without consuming the next packet.
This bounds only the added recv loop: Xlib's preheader select wait is unchanged.
No thread, global library patch, timestamp adjustment or connection retry exists.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator

MAX_IMAGE_REPLY_BYTES = 32 * 1024 * 1024
MIN_COALESCED_REPLY_BYTES = 1024 * 1024


class ImageReplyError(RuntimeError):
    """The image cannot be safely assembled inside its acquisition budget."""


class _ImageReplySocket:
    def __init__(self, protocol: Any, raw: Any, deadline_ns: int) -> None:
        self.protocol = protocol
        self.raw = raw
        self.deadline_ns = deadline_ns

    def __getattr__(self, name: str) -> Any:
        return getattr(self.raw, name)

    def check_deadline(self) -> None:
        if time.monotonic_ns() >= self.deadline_ns:
            raise ImageReplyError("image reply acquisition budget expired")

    def _read(self, count: int, flags: int) -> bytes:
        self.check_deadline()
        original = self.raw.gettimeout()
        remaining = (self.deadline_ns - time.monotonic_ns()) / 1e9
        if remaining <= 0:
            raise ImageReplyError("image reply acquisition budget expired")
        # Preserve nonblocking behavior and any shorter existing socket limit.
        bounded = remaining if original is None else min(original, remaining)
        try:
            self.raw.settimeout(bounded)
            result = self.raw.recv(count, flags)
        except TimeoutError:
            raise ImageReplyError("image reply read timed out") from None
        finally:
            self.raw.settimeout(original)
        self.check_deadline()
        if not isinstance(result, bytes):
            raise ImageReplyError("image reply read returned invalid bytes")
        return result

    def recv(self, count: int, flags: int = 0) -> bytes:
        try:
            return self._recv(count, flags)
        except ImageReplyError:
            raise
        except Exception as exc:
            # This includes reset, timeout-restoration and allocation errors.
            # Even if Xlib catches OSError internally, it must never turn an
            # unsafe partial accumulation into an ordinary drawable fallback.
            raise ImageReplyError("image reply assembly failed") from exc

    def _recv(self, count: int, flags: int) -> bytes:
        self.check_deadline()
        size = self.protocol.recv_packet_len
        # python-xlib also uses its private bytesview (not a memoryview), so
        # inspect only the length; Xlib retains authority over packet parsing.
        buffered_size = len(self.protocol.data_recv)
        if type(size) is not int or size < 0 or buffered_size < 0:
            raise ImageReplyError("invalid image reply framing")
        if size > MAX_IMAGE_REPLY_BYTES:
            raise ImageReplyError("image reply exceeds acquisition bound")
        remaining = size - buffered_size
        if (flags or size < MIN_COALESCED_REPLY_BYTES or remaining <= 0
                or count < remaining or self.raw.gettimeout() == 0):
            # Headers, interleaved small packets and special flags retain Xlib's
            # original single-read behavior. Framing remains Xlib's authority.
            return self._read(min(count, MAX_IMAGE_REPLY_BYTES), flags)
        result = bytearray()
        while len(result) < remaining:
            chunk = self._read(remaining - len(result), flags)
            if not chunk:
                raise ImageReplyError("image reply ended before its declared length")
            if len(chunk) > remaining - len(result):
                raise ImageReplyError("image reply read exceeded its bound")
            result.extend(chunk)
        return bytes(result)


@contextmanager
def coalesce_image_reply(protocol: Any, *, deadline_ns: int) -> Iterator[None]:
    """Install only on this connection, restoring ownership on every exit path."""
    raw = protocol.socket
    if isinstance(raw, _ImageReplySocket):
        raise ImageReplyError("nested image reply acquisition is not supported")
    reader = _ImageReplySocket(protocol, raw, deadline_ns)
    reader.check_deadline()
    protocol.socket = reader
    try:
        yield
        reader.check_deadline()
    finally:
        if protocol.socket is not reader:
            # Do not overwrite an unexpected replacement connection.
            raise ImageReplyError("image reply connection ownership changed")
        protocol.socket = raw
