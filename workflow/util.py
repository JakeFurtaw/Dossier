"""Small shared helpers (message text + parallel fan-out)."""

from __future__ import annotations

import contextvars
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, TypeVar

from langchain_core.messages import BaseMessage

T = TypeVar("T")
R = TypeVar("R")


def message_text(message: BaseMessage) -> str:
    """Readable text of a LangChain message (handles list content blocks).

    Used by the ReAct loop, citation index builder, fallback tiers, and the evaluator.
    """
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


def is_spawn_tool(name: str) -> bool:
    """Planner spawn tools (spawn_researcher, spawn_agents, …) run concurrently."""
    return (name or "").startswith("spawn_")


def invoke_tool(tool: Any, args: dict[str, Any]) -> str:
    """Invoke a LangChain tool, returning an error string instead of raising."""
    try:
        result = tool.invoke(args)
    except Exception as exc:
        return f"Error running {getattr(tool, 'name', 'tool')}: {exc}"
    if result is None:
        return ""
    return result if isinstance(result, str) else str(result)


def thought_text(message: BaseMessage) -> str:
    """The model's visible reasoning for a step (reasoning_content + text), for tracing.

    Used by the ReAct loop when emitting Thought events.
    """
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
        """Pool worker: run fn(item) and stash the result at its index."""
        results[index] = fn(item)

    with ThreadPoolExecutor(max_workers=len(items)) as pool:
        futures = []
        for index, item in enumerate(items):
            ctx = contextvars.copy_context()
            futures.append(pool.submit(ctx.run, _run, index, item))
        for future in as_completed(futures):
            future.result()
    return [results[i] for i in range(len(items))]  # type: ignore[misc]
