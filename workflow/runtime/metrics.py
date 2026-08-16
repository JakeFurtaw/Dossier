"""Named counters for defensive / fallback paths.

Each path that exists to paper over a small-model failure mode should call
``record(name)`` when it actually fires. The snapshot is written into the
run report so you can see later whether a given fallback is still earning
its keep.
"""

from __future__ import annotations

import logging
import threading
from contextvars import ContextVar

logger = logging.getLogger(__name__)


class Counters:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}

    def record(self, name: str) -> int:
        with self._lock:
            self._counts[name] = self._counts.get(name, 0) + 1
            return self._counts[name]

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)

    def clear(self) -> None:
        with self._lock:
            self._counts.clear()


_active: ContextVar[Counters | None] = ContextVar("workflow_counters", default=None)
_process = Counters()


def get_counters() -> Counters:
    return _active.get() or _process


def install_counters(counters: Counters):
    return _active.set(counters)


def uninstall_counters(token) -> None:
    _active.reset(token)


def record(name: str) -> int:
    n = get_counters().record(name)
    logger.info("counter %s = %d", name, n)
    return n


def snapshot() -> dict[str, int]:
    return get_counters().snapshot()


def reset_counters() -> None:
    get_counters().clear()
