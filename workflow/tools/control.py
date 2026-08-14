"""Stop-signal tools: planner final_answer and researcher report_findings."""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def final_answer(answer: str) -> str:
    """End the run and return the answer to the user.

    This is the only way the planner finishes. A Thought is not enough.
    Structure the answer for THIS goal (not a fixed population template).
    Include evidence/sources, any requested calculation, and a confidence note.
    """
    text = (answer or "").strip()
    if not text:
        return "Error: final_answer requires a non-empty answer."
    return text


@tool
def report_findings(summary: str) -> str:
    """Finish this researcher sub-task and return evidence to the planner.

    This is a tool call, not answering from memory. Include facts, numbers,
    source titles and URLs, and any uncertainty. Call this as soon as you
    have two useful sources — do not keep searching.
    """
    text = (summary or "").strip()
    if not text:
        return "Error: report_findings requires a non-empty summary."
    return text
