"""Fallbacks when an agent stops without its stop tool."""

from __future__ import annotations

from collections.abc import Iterable

from langchain_core.messages import BaseMessage, ToolMessage

from workflow.util import message_text

_USEFUL_TOOLS = {"web_search", "browse_page"}
_SPAWN_TOOLS = {"spawn_researcher", "spawn_researchers"}
_BLOCKED_MARKERS = (
    "blocked:",
    "captcha",
    "check for humans",
    "failed to extract",
    "timeout while",
    "error fetching",
    "no search results",
)


def _blocked(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in _BLOCKED_MARKERS)


def _tool_chunks(
    messages: Iterable[BaseMessage],
    names: set[str],
    limit: int,
    *,
    headed: bool = False,
) -> list[str]:
    chunks: list[str] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        name = getattr(message, "name", "") or "tool"
        if name not in names:
            continue
        text = message_text(message)
        if not text or text.startswith("Error") or _blocked(text):
            continue
        if len(text) > limit:
            text = text[:limit].rstrip() + f"\n… [{len(text) - limit} more chars truncated]"
        chunks.append(f"### {name}\n{text}" if headed else text)
    return chunks


def compile_researcher_reports(messages: Iterable[BaseMessage], limit: int = 12000) -> str:
    return "\n\n---\n\n".join(_tool_chunks(messages, _SPAWN_TOOLS, limit))


def planner_fallback_answer(messages: Iterable[BaseMessage], goal: str = "") -> str:
    del goal
    reports = compile_researcher_reports(messages)
    if not reports:
        return ""
    return (
        "Planner did not call final_answer. Evidence gathered by researchers:\n\n"
        + reports
    )


def researcher_fallback(messages: Iterable[BaseMessage], reason: str) -> str:
    notes = "\n\n".join(_tool_chunks(messages, _USEFUL_TOOLS, 2500, headed=True))
    if not notes:
        return f"Researcher stopped without findings ({reason})."
    return (
        f"Researcher stopped without report_findings ({reason}). "
        "Compiled evidence from tool results:\n\n"
        f"{notes}"
    )
