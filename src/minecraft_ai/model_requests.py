"""Generic model work accounting, separate from synchronous decision publication.

Future completion may finish accounting or claim a discard notice. Publication
belongs exclusively to the runtime's final authority check and ``accept`` call.
Detailed model receipts remain owned by the adapter.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from .perception import CognitionBlackboardSnapshot


def _text(value: str, name: str) -> None:
    if type(value) is not str or not value.strip() or len(value) > 256:
        raise ValueError(f"{name} must be a nonempty bounded string")


def _sha256(value: str, name: str) -> None:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _integer(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class RequestBinding:
    """Identity of the semantic snapshot and authority submitted for model work.

    ``source_sha256`` is the snapshot's existing semantic-source digest. It is
    neither a pixel digest nor proof that current publication authority matches.
    """

    request_id: str
    source_sha256: str
    instance_id: str
    frame_id: int
    captured_ns: int
    operator_revision: int
    execution_revision: int
    submitted_ns: int
    deadline_ns: int

    def __post_init__(self) -> None:
        _text(self.request_id, "request_id")
        _text(self.instance_id, "instance_id")
        _sha256(self.source_sha256, "source_sha256")
        for name in (
            "frame_id", "captured_ns", "operator_revision", "execution_revision",
            "submitted_ns", "deadline_ns",
        ):
            _integer(getattr(self, name), name)
        if not self.captured_ns <= self.submitted_ns <= self.deadline_ns:
            raise ValueError("request capture, submission and deadline must be ordered")

    @classmethod
    def from_snapshot(
        cls,
        snapshot: CognitionBlackboardSnapshot,
        *,
        request_id: str,
        operator_revision: int,
        execution_revision: int,
        submitted_ns: int,
        deadline_ns: int,
    ) -> RequestBinding:
        return cls(
            request_id=request_id, source_sha256=snapshot.source_sha256,
            instance_id=snapshot.instance_id, frame_id=snapshot.frame_id,
            captured_ns=snapshot.captured_ns, operator_revision=operator_revision,
            execution_revision=execution_revision, submitted_ns=submitted_ns,
            deadline_ns=deadline_ns,
        )


@dataclass(frozen=True, slots=True)
class ModelAttempt:
    attempt_id: str
    name: str
    started_ns: int
    finished_ns: int | None = None
    error_type: str | None = None


@dataclass(frozen=True, slots=True)
class RequestLifecycleSnapshot:
    binding: RequestBinding
    attempts: tuple[ModelAttempt, ...]
    computation_completed_ns: int | None
    disposition: Literal["pending", "accepted", "rejected", "publication_failed"]
    disposition_ns: int | None
    reason: str | None
    selected_attempt_id: str | None
    final_decision_sha256: str | None
    publication_error_type: str | None


class ModelRequestLifecycle:
    """Account actual attempts without treating completion as publication.

    ``accept`` serializes one short publication callback with rejection. The
    runtime owns its final source/revision/deadline check, and the callback owns
    any external transaction or rollback. No completion callback is registered.
    """

    def __init__(
        self, binding: RequestBinding, *, clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if type(binding) is not RequestBinding or not callable(clock_ns):
            raise TypeError("request lifecycle requires a binding and clock")
        self._binding = binding
        self._clock_ns = clock_ns
        self._lock = threading.RLock()
        self._attempts: dict[str, ModelAttempt] = {}
        self._computation_completed_ns: int | None = None
        self._disposition: Literal["pending", "accepted", "rejected", "publication_failed"] = (
            "pending"
        )
        self._disposition_ns: int | None = None
        self._reason: str | None = None
        self._selected_attempt_id: str | None = None
        self._final_decision_sha256: str | None = None
        self._publication_error_type: str | None = None
        self._publishing = False
        self._discard_notice_taken = False

    @property
    def binding(self) -> RequestBinding:
        return self._binding

    def _now(self) -> int:
        value = self._clock_ns()
        _integer(value, "clock_ns")
        if value < self.binding.submitted_ns:
            raise ValueError("request clock precedes submission")
        return value

    def start_attempt(self, name: str) -> str:
        _text(name, "attempt name")
        with self._lock:
            if self._disposition != "pending" or self._computation_completed_ns is not None:
                raise RuntimeError("request cannot start another model attempt")
            now = self._now()
            if now >= self.binding.deadline_ns:
                # Refuse new work, but preserve the real completion/error of
                # earlier attempts. Runtime still owns terminal disposition.
                raise RuntimeError("request deadline prevents another model attempt")
            attempt_id = f"{self.binding.request_id}:attempt-{len(self._attempts) + 1}"
            self._attempts[attempt_id] = ModelAttempt(attempt_id, name, now)
            return attempt_id

    def finish_attempt(self, attempt_id: str, error_type: str | None = None) -> None:
        if error_type is not None:
            _text(error_type, "error_type")
        with self._lock:
            if attempt_id not in self._attempts:
                raise ValueError("unknown model attempt")
            attempt = self._attempts[attempt_id]
            if attempt.finished_ns is not None:
                if attempt.error_type != error_type:
                    raise ValueError("completed attempt error cannot be rewritten")
                return
            now = self._now()
            if now < attempt.started_ns:
                raise ValueError("attempt finish precedes its start")
            self._attempts[attempt_id] = replace(attempt, finished_ns=now, error_type=error_type)

    def mark_computation_complete(self) -> None:
        with self._lock:
            if self._computation_completed_ns is not None:
                return
            if any(attempt.finished_ns is None for attempt in self._attempts.values()):
                raise RuntimeError("model attempts are still running")
            now = self._now()
            if any(now < cast(int, attempt.finished_ns) for attempt in self._attempts.values()):
                raise ValueError("computation completion precedes an attempt finish")
            self._computation_completed_ns = now

    def reject(self, reason: str) -> bool:
        """Record the first rejection without callbacks or invented completion."""
        _text(reason, "rejection reason")
        with self._lock:
            if self._disposition != "pending" or self._publishing:
                return False
            self._disposition_ns = self._now()
            self._disposition, self._reason = "rejected", reason
            return True

    def accept(
        self,
        selected_attempt_id: str,
        final_decision_sha256: str,
        publication_callback: Callable[[], None],
    ) -> bool:
        """Publish once after runtime admission; exceptions never mean accepted.

        Repeating the exact accepted identity returns True without invoking a
        callback again. A failed publication is terminal and cannot be retried.
        """
        _sha256(final_decision_sha256, "final_decision_sha256")
        if not callable(publication_callback):
            raise TypeError("publication_callback must be callable")
        with self._lock:
            if self._publishing:
                raise RuntimeError("publication callback cannot reenter acceptance")
            if self._disposition == "accepted":
                return (
                    self._selected_attempt_id == selected_attempt_id
                    and self._final_decision_sha256 == final_decision_sha256
                )
            if self._disposition != "pending":
                return False
            if self._computation_completed_ns is None:
                raise RuntimeError("computation is not complete")
            attempt = self._attempts.get(selected_attempt_id)
            if attempt is None or attempt.finished_ns is None or attempt.error_type is not None:
                raise ValueError("publication must select a completed successful model attempt")
            self._selected_attempt_id = selected_attempt_id
            self._final_decision_sha256 = final_decision_sha256
            self._publishing = True
            try:
                publication_callback()
            except BaseException as error:
                self._disposition = "publication_failed"
                self._reason = "publication_failed"
                self._publication_error_type = type(error).__name__
                raise
            else:
                self._disposition = "accepted"
                return True
            finally:
                self._publishing = False
                try:
                    self._disposition_ns = self._now()
                except Exception:
                    # An unavailable accounting clock must never reopen an
                    # already executed publication for another callback.
                    self._disposition_ns = None

    def take_discard_notice(self) -> str | None:
        """Claim the generic discard reason once, only after work really ended."""
        with self._lock:
            if (
                self._discard_notice_taken or self._computation_completed_ns is None
                or self._disposition not in {"rejected", "publication_failed"}
            ):
                return None
            self._discard_notice_taken = True
            return self._reason

    def snapshot(self) -> RequestLifecycleSnapshot:
        with self._lock:
            return RequestLifecycleSnapshot(
                binding=self.binding, attempts=tuple(self._attempts.values()),
                computation_completed_ns=self._computation_completed_ns,
                disposition=self._disposition, disposition_ns=self._disposition_ns,
                reason=self._reason, selected_attempt_id=self._selected_attempt_id,
                final_decision_sha256=self._final_decision_sha256,
                publication_error_type=self._publication_error_type,
            )
