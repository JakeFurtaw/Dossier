"""Salvage usable findings when an agent hits a cap without its stop tool."""

from __future__ import annotations

import re
from typing import Iterable

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

_USEFUL_TOOLS = {"web_search", "browse_page"}
_BLOCKED_MARKERS = (
    "blocked:",
    "captcha",
    "check for humans",
    "failed to extract",
    "timeout while",
    "error fetching",
    "no search results",
)
_META_MARKERS = (
    "call final_answer",
    "call report_findings",
    "you already drafted",
    "does not complete the run",
    "put that same answer",
    "answer argument",
    "this is your last step",
    "do not spawn another",
    "available tools:",
    "we need to output final_answer",
    "we should call final_answer",
    "we'll output final_answer",
    "let's copy the answer we drafted",
    "the user says",
    "two consecutive",
    "stopped: two consecutive",
)
_ANSWER_STARTS = (
    "## final answer",
    "## answer",
    "### quick recommendation",
    "## estimated population",
    "## recommendation",
    "### recommendation",
)
_PLAN_MARKERS = (
    "i'll delegate",
    "i will delegate",
    "delegate them in parallel",
    "spawn_researchers",
    "spawn_researcher",
    "these are independent",
    "**plan:**",
    "plan:\n",
    "plan: ",
)
_NOTES_MARKERS = (
    "let me now compose",
    "let me compose",
    "let me synthesize",
    "let me structure",
    "i have enough to write",
    "i have enough to synthesize",
    "i now have comprehensive research",
    "i have comprehensive research",
    "key findings:",
)
_GOAL_STOP = {
    "what",
    "the",
    "best",
    "to",
    "build",
    "new",
    "on",
    "looking",
    "and",
    "but",
    "open",
    "trying",
    "others",
    "each",
    "how",
    "many",
    "than",
    "with",
    "for",
    "about",
    "from",
    "this",
    "that",
    "have",
    "been",
    "into",
    "your",
    "their",
    "them",
    "you",
    "are",
    "was",
    "were",
}


def _content(message: BaseMessage) -> str:
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
        return "\n".join(p for p in parts if p).strip()
    return str(content or "").strip()


def _combined_thought(message: AIMessage) -> str:
    extra = getattr(message, "additional_kwargs", None) or {}
    reasoning = str(extra.get("reasoning_content") or extra.get("reasoning") or "").strip()
    text = _content(message)
    return "\n\n".join(part for part in (reasoning, text) if part)


def _looks_blocked(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in _BLOCKED_MARKERS)


def goal_terms(goal: str) -> set[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9+\-]{2,}", (goal or "").lower())
    return {word for word in words if word not in _GOAL_STOP}


def mentions_goal(text: str, terms: Iterable[str], min_hits: int = 2) -> bool:
    terms = list(terms)
    if not terms:
        return True
    lower = (text or "").lower()
    hits = sum(1 for term in terms if term in lower)
    return hits >= min(min_hits, len(terms))


def is_meta_thought(text: str) -> bool:
    """True when the model is talking about calling tools, not answering the goal."""
    lower = (text or "").lower()
    hits = sum(1 for marker in _META_MARKERS if marker in lower)
    if hits == 0:
        return False
    if _has_answer_body(text) and mentions_goal(extract_answer_body(text), goal_terms(text)):
        return False
    return True


def is_plan_or_notes(text: str) -> bool:
    """True for 'here is my plan / let me compose' scratch text, not a user answer."""
    if not text:
        return False
    if _has_answer_body(text) and ("http://" in text.lower() or "https://" in text.lower()):
        return False
    lower = text.lower()
    plan_hits = sum(1 for marker in _PLAN_MARKERS if marker in lower)
    if plan_hits >= 2:
        return True
    if any(marker in lower for marker in _NOTES_MARKERS) and not _has_answer_body(text):
        return True
    numbered = len(re.findall(r"^\s*\d+\.\s+research\b", text, flags=re.I | re.M))
    return numbered >= 2 and "http" not in lower


def _has_answer_body(text: str) -> bool:
    lower = (text or "").lower()
    if any(marker in lower for marker in _ANSWER_STARTS):
        return True
    return text.count("|") >= 6 and text.count("\n") >= 4


def extract_answer_body(text: str) -> str:
    """Prefer the markdown answer section, not the preceding plan/tool chatter."""
    if not text:
        return ""
    lower = text.lower()
    cut = -1
    for marker in _ANSWER_STARTS:
        idx = lower.find(marker)
        if idx >= 0 and (cut < 0 or idx < cut):
            cut = idx
    return text[cut:].strip() if cut >= 0 else text.strip()


def looks_like_answer(text: str, goal: str = "") -> bool:
    if not text or len(text.strip()) < 80:
        return False
    if is_plan_or_notes(text):
        return False
    if is_meta_thought(text) and not _has_answer_body(text):
        return False
    body = extract_answer_body(text)
    if goal and not mentions_goal(body, goal_terms(goal)):
        return False
    lower = body.lower()
    has_url = "http://" in lower or "https://" in lower
    has_table = body.count("|") >= 6
    has_headings = body.count("## ") >= 2
    if _has_answer_body(body) or has_url or has_table or has_headings:
        return True
    signals = (
        "confidence",
        "in summary",
        "conclusion",
        "recommendation",
    )
    return any(token in lower for token in signals) and len(body) > 500


def looks_incomplete(text: str) -> bool:
    stripped = (text or "").rstrip()
    if not stripped:
        return True
    if stripped.endswith("|") or stripped.endswith("(") or stripped.endswith(","):
        return True
    return len(stripped) < 400 and not stripped.endswith((".", "!", "?", "`", "*"))


def compile_tool_notes(messages: Iterable[BaseMessage], limit: int = 2500) -> str:
    """Join useful tool observations so a failed researcher still returns evidence."""
    chunks: list[str] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        name = getattr(message, "name", "") or "tool"
        if name not in _USEFUL_TOOLS:
            continue
        text = _content(message)
        if not text or _looks_blocked(text):
            continue
        if len(text) > limit:
            text = text[:limit].rstrip() + f"\n… [{len(text) - limit} more chars truncated]"
        chunks.append(f"### {name}\n{text}")
    return "\n\n".join(chunks)


def compile_researcher_reports(messages: Iterable[BaseMessage], limit: int = 12000) -> str:
    chunks: list[str] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        if getattr(message, "name", "") not in {"spawn_researcher", "spawn_researchers"}:
            continue
        text = _content(message)
        if not text or text.startswith("Error") or _looks_blocked(text):
            continue
        if len(text) > limit:
            text = text[:limit].rstrip() + f"\n… [{len(text) - limit} more chars truncated]"
        chunks.append(text)
    return "\n\n---\n\n".join(chunks)


def last_substantial_thought(messages: Iterable[BaseMessage], min_chars: int = 200) -> str:
    """Return the last long assistant thought (used for nudges, not salvage)."""
    last = ""
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        combined = _combined_thought(message)
        if len(combined) >= min_chars:
            last = combined
    return last


def best_answer_draft(messages: Iterable[BaseMessage], goal: str = "") -> str:
    """Pick the thought that actually answers the goal, ignoring tool-call meta."""
    terms = goal_terms(goal)
    candidates: list[tuple[int, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, AIMessage):
            continue
        combined = _combined_thought(message)
        if len(combined) < 80:
            continue
        body = extract_answer_body(combined)
        if is_meta_thought(combined) and not _has_answer_body(combined):
            continue
        if is_meta_thought(body) and not _has_answer_body(body):
            continue
        if goal and not mentions_goal(body, terms):
            continue
        if is_plan_or_notes(combined) or is_plan_or_notes(body):
            continue
        if not looks_like_answer(body, goal):
            continue
        score = len(body) // 4
        if _has_answer_body(body):
            score += 5000
        if "http://" in body.lower() or "https://" in body.lower():
            score += 2000
        score += 200 * sum(1 for term in terms if term in body.lower())
        score += index * 50
        candidates.append((score, body))
    if not candidates:
        return ""
    return max(candidates, key=lambda item: item[0])[1]


def planner_fallback_answer(messages: Iterable[BaseMessage], goal: str = "") -> str:
    draft = best_answer_draft(messages, goal)
    reports = compile_researcher_reports(messages)
    if draft and not is_plan_or_notes(draft):
        if looks_incomplete(draft) and reports:
            return (
                draft.rstrip()
                + "\n\n---\n\nAdditional evidence from researcher reports:\n\n"
                + reports
            )
        return draft
    if reports:
        return (
            "Planner did not call final_answer. Evidence gathered by researchers:\n\n"
            + reports
        )
    return draft


def researcher_fallback(messages: Iterable[BaseMessage], reason: str) -> str:
    notes = compile_tool_notes(messages)
    if not notes:
        return f"Researcher stopped without findings ({reason})."
    return (
        f"Researcher stopped without report_findings ({reason}). "
        "Compiled evidence from tool results:\n\n"
        f"{notes}"
    )
