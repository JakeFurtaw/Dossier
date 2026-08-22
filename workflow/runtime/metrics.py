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
    """Thread-safe named counter map for one run (or the process fallback)."""

    def __init__(self) -> None:
        """Lock-protected name → count map."""
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}

    def record(self, name: str) -> int:
        """Increment ``name`` and return its new total."""
        with self._lock:
            self._counts[name] = self._counts.get(name, 0) + 1
            return self._counts[name]

    def snapshot(self) -> dict[str, int]:
        """Copy of all counts (written into the run report)."""
        with self._lock:
            return dict(self._counts)

    def clear(self) -> None:
        """Drop all counts (test fixture helper)."""
        with self._lock:
            self._counts.clear()


_active: ContextVar[Counters | None] = ContextVar("workflow_counters", default=None)
_process = Counters()


def get_counters() -> Counters:
    """This run's counters, or the process-wide ones outside a trace."""
    return _active.get() or _process


def install_counters(counters: Counters):
    """Bind ``counters`` to this context (start_trace)."""
    return _active.set(counters)


def uninstall_counters(token) -> None:
    """Undo an install_counters token."""
    _active.reset(token)


def record(name: str) -> int:
    """Bump a defensive-path counter (the one call site used across the codebase)."""
    n = get_counters().record(name)
    logger.info("counter %s = %d", name, n)
    return n


def snapshot() -> dict[str, int]:
    """Counts so far (used by the run report and tests)."""
    return get_counters().snapshot()


def reset_counters() -> None:
    """Clear the active counters (test fixture helper)."""
    get_counters().clear()
