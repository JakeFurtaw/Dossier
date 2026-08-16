from __future__ import annotations

import pytest

from workflow.runtime.metrics import reset_counters
from workflow.runtime.tracing import _agent_stack


@pytest.fixture(autouse=True)
def _clean_counters() -> None:
    reset_counters()
    yield
    reset_counters()


@pytest.fixture(autouse=True)
def _clean_agent_stack() -> None:
    token = _agent_stack.set(())
    yield
    _agent_stack.reset(token)
