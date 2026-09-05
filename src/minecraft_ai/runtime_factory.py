"""Trusted optional runtime construction under an existing, bounded lease.

This is not a plugin sandbox. Factories may replace cognition only and must not
start inference or inputs during assembly. Cancellation of import/native calls
is cooperative; the guard bounds renewal, not arbitrary Python/C execution.
"""

from __future__ import annotations

import importlib
import signal
import threading
import time
from typing import Any

from .config import RuntimeFactoryConfig
from .emergency import emergency_stop_latched
from .runtime import AgentRuntime
from .supervisor import operator_pause_latched, send_command


class RuntimeStartupCancelled(RuntimeError):
    """Construction lost its authority or received sticky cancellation."""


class RuntimeStartupCleanupIncomplete(RuntimeError):
    """Do not close the main-owned database while constructed owners remain."""


class _ConstructionLease:
    def __init__(
        self, lease_id: str, cancel_event: threading.Event,
        *, timeout_s: float, interval_s: float,
    ) -> None:
        self.lease_id = lease_id
        self.cancel_event = cancel_event
        self.deadline = time.monotonic() + timeout_s
        self.interval_s = interval_s
        self._finished = threading.Event()
        self._thread: threading.Thread | None = None
        self._fault: Exception | None = None

    def check(self) -> None:
        if self._fault is not None:
            raise RuntimeStartupCancelled("runtime construction lease failed") from self._fault
        if (self.cancel_event.is_set() or emergency_stop_latched()
                or operator_pause_latched() or time.monotonic() >= self.deadline):
            self.cancel_event.set()
            raise RuntimeStartupCancelled("runtime construction cancelled or deadline reached")

    def _renew(self) -> None:
        self.check()
        send_command("renew", lease_id=self.lease_id, ttl_ms=5_000, timeout_s=0.5)
        self.check()

    def start(self) -> None:
        # A rejected lease must not be revived or even import an external module.
        self._renew()
        thread = threading.Thread(
            target=self._heartbeat, name="minecraft-ai-construction-lease", daemon=True,
        )
        thread.start()
        self._thread = thread

    def _heartbeat(self) -> None:
        while not self._finished.wait(min(self.interval_s, max(
            0.0, self.deadline - time.monotonic(),
        ))):
            try:
                self._renew()
            except Exception as exc:
                self._fault = exc
                self.cancel_event.set()
                return

    def close(self) -> None:
        self._finished.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                self.cancel_event.set()
                raise RuntimeStartupCleanupIncomplete("construction lease worker did not stop")


def _close_supplied_resources(runtime_kwargs: dict[str, Any]) -> bool:
    """Factory failed before returning a valid cleanup owner; never introspect it."""
    deadline = time.monotonic() + 2.0
    for name in ("perception", "executor", "trajectory"):
        resource = runtime_kwargs[name]
        if resource is None:
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        try:
            if name == "trajectory":
                resource.close(timeout_s=remaining)
            else:
                resource.close()
        except Exception:
            return False
    return True


def run_agent_runtime(
    runtime_kwargs: dict[str, Any], *, factory_config: RuntimeFactoryConfig | None = None,
) -> None:
    """Keep canonical process identity and legacy construction unless opted in.

    External callable: factory(*, runtime_kwargs, cancel_event) -> AgentRuntime.
    All supplied fields except high_level must be preserved by identity. A valid
    result also owns close_constructed_runtime() -> bool: stop/drain private
    owners, then call close_before_run(). True means all pre-run owners closed;
    False/exception prevents further teardown, including main's database close.
    Retention lasts until process exit, not an indefinite background guardian.

    A factory that raises or returns an invalid object owns cleanup of private
    partial construction. No arbitrary cleanup method on an invalid result is
    invoked. If partial cleanup cannot finish, it must instead raise
    RuntimeStartupCleanupIncomplete to retain the borrowed resources too.
    Constructors must check cancel_event between expensive stages.
    """
    runtime: AgentRuntime | None = None
    entered_run = False
    cancel_event = threading.Event()
    originals: dict[signal.Signals, Any] = {}

    def stop(_signum: int, _frame: object) -> None:
        cancel_event.set()
        if runtime is not None:
            runtime.stop()

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            originals[sig] = signal.signal(sig, stop)
        if factory_config is None:
            runtime = AgentRuntime(**runtime_kwargs)
        else:
            supplied_policy = runtime_kwargs["executor"].policy
            guard = _ConstructionLease(
                runtime_kwargs["lease_id"], cancel_event,
                timeout_s=factory_config.startup_timeout_s,
                interval_s=runtime_kwargs["lease_renew_ms"] / 1000.0,
            )
            try:
                guard.start()
                guard.check()
                module_name, attribute = factory_config.reference.split(":")
                module = importlib.import_module(module_name)
                guard.check()
                factory = getattr(module, attribute)
                if not callable(factory):
                    raise TypeError("configured runtime factory is not callable")
                result = factory(runtime_kwargs=dict(runtime_kwargs), cancel_event=cancel_event)
                if not isinstance(result, AgentRuntime) or not callable(
                    getattr(result, "close_constructed_runtime", None)
                ):
                    raise TypeError("runtime factory must return AgentRuntime with cleanup hook")
                runtime = result
                for name, supplied in runtime_kwargs.items():
                    if name != "high_level" and getattr(runtime, name) is not supplied:
                        raise ValueError(f"runtime factory replaced protected field: {name}")
                if runtime.executor.policy is not supplied_policy:
                    raise ValueError("runtime factory replaced protected executor policy")
            finally:
                # No simultaneous construction/runtime renewal owners at handoff.
                guard.close()
            guard.check()
        if cancel_event.is_set() or emergency_stop_latched() or operator_pause_latched():
            raise RuntimeStartupCancelled("runtime startup cancelled")
        entered_run = True
        # The runtime's synchronous initial renewal remains authoritative.
        runtime.run_forever()
    except BaseException as exc:
        if not entered_run:
            cancel_event.set()
            # A partial factory can explicitly retain its borrowed owners when
            # private teardown is incomplete. Never guess a cleanup fallback.
            if isinstance(exc, RuntimeStartupCleanupIncomplete):
                raise
            try:
                if runtime is None:
                    closed = _close_supplied_resources(runtime_kwargs)
                else:
                    runtime.stop()
                    if factory_config is None:
                        closed = runtime.close_before_run()
                    else:
                        closed = runtime.close_constructed_runtime()  # type: ignore[attr-defined]
                if closed is not True:
                    raise RuntimeStartupCleanupIncomplete("pre-run runtime cleanup incomplete")
            except BaseException as cleanup_exc:
                raise RuntimeStartupCleanupIncomplete(
                    "pre-run cleanup incomplete; database retained until process exit"
                ) from cleanup_exc
        raise
    finally:
        for sig, handler in originals.items():
            signal.signal(sig, handler)
