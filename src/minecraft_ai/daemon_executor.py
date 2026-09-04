"""A minimal single-worker executor whose worker cannot hold process exit open."""

from __future__ import annotations

import threading
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Generic, ParamSpec, TypeVar


_P = ParamSpec("_P")
_T = TypeVar("_T")


@dataclass(slots=True)
class _WorkItem(Generic[_T]):
    future: Future[_T]
    callback: Callable[[], _T]

    def run(self) -> None:
        if not self.future.set_running_or_notify_cancel():
            return
        try:
            result = self.callback()
        except BaseException as exc:
            self.future.set_exception(exc)
        else:
            self.future.set_result(result)


class SingleWorkerDaemonExecutor:
    """Execute submitted calls serially on one daemon thread.

    Unlike :class:`concurrent.futures.ThreadPoolExecutor`, the worker is a daemon,
    so a blocked external call cannot keep interpreter shutdown waiting. Queued
    work remains cancellable, and submitted calls use ordinary ``Future`` objects.
    """

    def __init__(self, *, thread_name: str = "minecraft-ai-daemon-worker") -> None:
        self._condition = threading.Condition()
        self._work: deque[_WorkItem[Any]] = deque()
        self._shutdown = False
        self._thread = threading.Thread(
            target=self._worker,
            name=thread_name,
            daemon=True,
        )
        self._thread.start()

    def submit(
        self,
        fn: Callable[_P, _T],
        /,
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> Future[_T]:
        future: Future[_T] = Future()
        item = _WorkItem(future=future, callback=partial(fn, *args, **kwargs))
        with self._condition:
            if self._shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")
            self._work.append(item)
            self._condition.notify()
        return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        with self._condition:
            self._shutdown = True
            if cancel_futures:
                while self._work:
                    self._work.popleft().future.cancel()
            self._condition.notify_all()
        if wait:
            self._thread.join()

    def _worker(self) -> None:
        while True:
            with self._condition:
                while not self._work:
                    if self._shutdown:
                        return
                    self._condition.wait()
                item = self._work.popleft()
            item.run()
