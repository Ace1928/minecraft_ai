"""Optional same-host delivery of reviewed recipe scope, without a new worker.

This reader does not infer a connection or loaded stack. An admitted host
writer must prove both before writing the envelope. Re-reading an envelope
never refreshes its evidence timestamp or grants input authority.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import time

from .agent.lifecycle import AgentProcess, _agent_process_state
from .pack_scope import (
    ACTIVE_RECIPE_KEY, ACTIVE_RECIPE_SOURCE, decode_active_recipe_identity,
)
from .pack_recipes import PackRecipeCatalog
from .perception import PerceptionBlackboard, PerceptionFact
from .platforms.frame_cache import FRAME_CACHE_MAX_AGE_NS, owner_manifest

MAX_OBSERVATION_BYTES = 8192


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("ambiguous observation field")
        result[key] = value
    return result


def _boot_id() -> str | None:
    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="ascii",
        ).strip()
        return value if 0 < len(value) <= 64 else None
    except (OSError, UnicodeError):
        return None


def _read_envelope(path: Path) -> dict[str, object] | None:
    """Reuse the frame reader's private regular-file/non-symlink policy."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) & 0o077
                or not 0 < info.st_size <= MAX_OBSERVATION_BYTES
            ):
                return None
            value = json.loads(
                stream.read(MAX_OBSERVATION_BYTES + 1), object_pairs_hook=_unique_object,
            )
            return value if type(value) is dict else None
    except (OSError, ValueError, UnicodeError, RecursionError, AttributeError):
        return None


class RecipeObservationReader:
    """Bounded reader bound to the agent's actual PublishedFrameCapture owner."""

    def __init__(
        self, path: Path, owner: AgentProcess | None, catalog: PackRecipeCatalog,
    ) -> None:
        self.path = path
        self.owner = owner
        self.catalog = catalog
        self.reason = "not_observed"
        self._last_observed_ns = 0
        self._last_payload: str | None = None
        self._last_ttl_ms = 0

    def _fact(self, board: PerceptionBlackboard) -> PerceptionFact | None:
        owner = self.owner
        latest = board.raw_latest()
        now = time.monotonic_ns()
        self.reason = "capture_owner_unverified"
        if (
            owner is None or latest is None or owner.pid != os.getpid()
            or owner.proc_start_ticks is None or owner.command_sha256 is None
            or latest.instance_id != owner.instance_id
            or not 0 <= now - latest.captured_ns <= FRAME_CACHE_MAX_AGE_NS
        ):
            return None
        try:
            if AgentProcess.load() != owner or _agent_process_state(owner) != "verified-live":
                return None
        except (OSError, ValueError, TypeError, KeyError):
            return None
        self.reason = "observation_missing_or_invalid"
        value = _read_envelope(self.path)
        if value is None:
            return None
        self.reason = "observation_scope_unverified"
        boot = _boot_id()
        expected_owner = owner_manifest(owner)
        recorded_owner = value.get("capture_owner")
        observed_ns = value.get("observed_ns")
        ttl_ms = value.get("expires_after_ms")
        if (
            set(value) != {"schema_version", "boot_id", "capture_owner",
                           "observed_ns", "expires_after_ms", "identity"}
            or type(value["schema_version"]) is not int or value["schema_version"] != 1
            or boot is None or value["boot_id"] != boot
            or type(recorded_owner) is not dict or recorded_owner != expected_owner
            or any(type(recorded_owner.get(key)) is not type(item)
                   for key, item in expected_owner.items())
            or type(observed_ns) is not int or type(ttl_ms) is not int
            or not 0 < ttl_ms <= 5000 or not 0 < owner.started_ns <= observed_ns <= now
            or now - observed_ns >= ttl_ms * 1_000_000
        ):
            return None
        # Reuse the exact existing scalar contract. Null/unknown/configured
        # descriptors are revocations, never an invitation to infer active state.
        payload = json.dumps(value["identity"], separators=(",", ":"))
        if observed_ns < self._last_observed_ns:
            return None
        if (observed_ns == self._last_observed_ns
                and (payload != self._last_payload or ttl_ms != self._last_ttl_ms)):
            # Ambiguous same-time evidence tombstones this timestamp; replaying
            # its former positive value must not undo the revocation.
            self._last_payload, self._last_ttl_ms = None, 0
            return None
        # Valid envelopes advance the watermark even when identity is unknown,
        # null or malformed. A later explicit revocation blocks old positives.
        self._last_observed_ns, self._last_payload = observed_ns, payload
        self._last_ttl_ms = ttl_ms
        if decode_active_recipe_identity(payload, owner.instance_id) is None:
            return None
        self.reason = "delivered"
        return PerceptionFact(
            key=ACTIVE_RECIPE_KEY, value=payload, source=ACTIVE_RECIPE_SOURCE,
            confidence=1.0, observed_ns=observed_ns, expires_after_ms=ttl_ms,
        )

    def poll(self, board: PerceptionBlackboard) -> None:
        """Poll on each captured cycle; revoke immediately when evidence is lost."""
        fact = self._fact(board)
        # Remove even a newer rejected/stale envelope from this same producer;
        # an older successful envelope must not preserve another session's fact.
        board.remove_semantic_facts(
            (ACTIVE_RECIPE_KEY,), expected_source=ACTIVE_RECIPE_SOURCE,
        )
        if fact is not None and self.owner is not None:
            merged = board.merge_semantics(instance_id=self.owner.instance_id, facts=(fact,))
            if not merged or self.catalog.active_scope_status(board) != "verified":
                self.revoke(board)
                self.reason = "catalog_or_capture_scope_mismatch"

    def revoke(self, board: PerceptionBlackboard) -> None:
        self.reason = "capture_unavailable"
        board.remove_semantic_facts(
            (ACTIVE_RECIPE_KEY,), expected_source=ACTIVE_RECIPE_SOURCE,
        )

    def status(self) -> dict[str, str]:
        # Public runtime telemetry exposes the result, not private owner fields.
        return {"state": "delivered" if self.reason == "delivered" else "unknown",
                "reason": self.reason}
