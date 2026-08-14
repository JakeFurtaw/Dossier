"""Small shared helpers (message text + parallel fan-out)."""

from __future__ import annotations

import contextvars
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypeVar

from langchain_core.messages import BaseMessage

T = TypeVar("T")
R = TypeVar("R")


def message_text(message: BaseMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        return "\n".join(part for part in parts if part).strip()
    return str(content or "").strip()


def thought_text(message: BaseMessage) -> str:
    extra = getattr(message, "additional_kwargs", None) or {}
    reasoning = str(extra.get("reasoning_content") or extra.get("reasoning") or "").strip()
    content = message_text(message)
    if reasoning and content:
        return f"{reasoning}\n\n{content}"
    return reasoning or content or "(no explicit thought)"


def run_in_threads(fn: Callable[[T], R], items: Sequence[T]) -> list[R]:
    """Run fn(item) for each item. One item stays on this thread; many use a pool.

    Each worker gets a copy of the current ContextVar state so parallel
    researchers nest under the planner in the live tree.
    """
    if not items:
        return []
    if len(items) == 1:
        return [fn(items[0])]

    results: list[R | None] = [None] * len(items)

    def _run(index: int, item: T) -> None:
        results[index] = fn(item)

    with ThreadPoolExecutor(max_workers=len(items)) as pool:
        futures = []
        for index, item in enumerate(items):
            ctx = contextvars.copy_context()
            futures.append(pool.submit(ctx.run, _run, index, item))
        for future in as_completed(futures):
            future.result()
    return [results[i] for i in range(len(items))]  # type: ignore[misc]
