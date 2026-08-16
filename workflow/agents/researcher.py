"""Researcher sub-agent, parallel spawn, and planner-facing tools."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool

from workflow.agents.evaluator import evaluate_findings
from workflow.runtime.citations import build_evidence_index, citation_check_line, extract_urls
from workflow.runtime.ledger import LedgerEntry, brief_text
from workflow.runtime.react import run_react
from workflow.runtime.recovery import researcher_fallback
from workflow.runtime.tracing import try_get_bus
from workflow.config import (
    CITATION_CHECK,
    EVALUATOR_ENABLED,
    EVALUATOR_RETRY,
    MAX_PARALLEL_RESEARCHERS,
    RESEARCHER_MAX_ITERS,
    make_llm,
)
from workflow.prompts import RESEARCHER_SYSTEM
from workflow.tools import browse_page, report_findings, web_search
from workflow.util import run_in_threads


def researcher_user_message(task: str, digest: str = "") -> str:
    body = (
        f"Assigned task:\n{task.strip()}\n\n"
        "Use web_search and browse_page as needed, then call report_findings. "
        "report_findings is a tool. Two useful sources is enough."
    )
    if digest.strip():
        body += f"\n\n{digest.strip()}"
    return body


def _peer_digest() -> str:
    bus = try_get_bus()
    if bus is None:
        return ""
    return bus.peer_digest("researcher")


def _publish_report(agent_id: str, task: str, text: str) -> None:
    if not (text or "").strip():
        return
    bus = try_get_bus()
    if bus is None:
        return
    assign = " ".join((task or "").split())
    if len(assign) > 80:
        assign = assign[:79] + "…"
    summary = brief_text(text)
    title = f"{assign}: {summary}" if assign else summary
    bus.publish_entry(
        LedgerEntry(
            role="researcher",
            agent_id=agent_id,
            kind="report",
            title=title,
            urls=extract_urls(text),
            queries=[],
        )
    )


def _gather_findings(task: str) -> tuple[str, str, list[BaseMessage]]:
    result = run_react(
        make_llm(role="researcher"),
        [web_search, browse_page, report_findings],
        RESEARCHER_SYSTEM,
        researcher_user_message(task, _peer_digest()),
        role="researcher",
        max_iterations=RESEARCHER_MAX_ITERS,
        stop_tools={"report_findings"},
        indent="  ",
    )
    text = result.payload or researcher_fallback(
        result.messages, result.stopped_reason or "stopped"
    )
    return text, result.agent_id, result.messages


def _with_citation_check(text: str, messages: list[BaseMessage]) -> str:
    """Append a deterministic citation verdict, checked against this
    researcher's own tool output (no extra model calls)."""
    if not CITATION_CHECK or not text.strip():
        return text
    line = citation_check_line(text, build_evidence_index(messages))
    return f"{text.strip()}\n\n{line}"


def run_researcher(task: str, indent: str = "  ", *, allow_retry: bool = True) -> str:
    """Run a researcher, then evaluate the report (optional one retry on FAIL)."""
    del indent  # nesting comes from the live trace tree
    findings, agent_id, messages = _gather_findings(task)
    findings = _with_citation_check(findings, messages)
    _publish_report(agent_id, task, findings)
    if not EVALUATOR_ENABLED:
        return findings

    review = evaluate_findings(task, findings, parent_id=agent_id)
    package = f"{findings.strip()}\n\n---\n\n{review.text}"
    if review.failed and allow_retry and EVALUATOR_RETRY:
        retry_task = (
            f"{task.strip()}\n\n"
            "The evaluator REJECTED the previous report. Address these issues "
            "and return better sourced findings:\n"
            f"{review.text}"
        )
        second, second_id, second_messages = _gather_findings(retry_task)
        second = _with_citation_check(second, second_messages)
        _publish_report(second_id, task, second)
        second_review = evaluate_findings(task, second, parent_id=second_id)
        return (
            f"{second.strip()}\n\n---\n\n{second_review.text}\n\n"
            f"(Retried once after evaluator {review.verdict}.)"
        )
    return package


def _normalize_tasks(tasks: Any) -> list[str]:
    if tasks is None:
        return []
    if isinstance(tasks, str):
        text = tasks.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [text]
        return _normalize_tasks(parsed)
    if isinstance(tasks, dict):
        inner = tasks.get("tasks") or tasks.get("task") or tasks.get("items")
        if inner is not None:
            return _normalize_tasks(inner)
        return []
    if isinstance(tasks, (list, tuple)):
        out: list[str] = []
        for item in tasks:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                value = item.get("task") or item.get("text") or item.get("query") or ""
                if str(value).strip():
                    out.append(str(value).strip())
        return out
    return []


def run_researchers_parallel(tasks: list[str]) -> str:
    """Run up to MAX_PARALLEL_RESEARCHERS researchers at the same time."""
    cleaned = _normalize_tasks(tasks)[: max(1, MAX_PARALLEL_RESEARCHERS)]
    if not cleaned:
        return "Error: spawn_researchers requires at least one non-empty task."

    def _one(assigned: str) -> str:
        try:
            return run_researcher(assigned)
        except Exception as exc:
            return f"Researcher failed: {exc}"

    bodies = run_in_threads(_one, cleaned)
    if len(cleaned) == 1:
        return bodies[0]
    parts = []
    for index, (assigned, body) in enumerate(zip(cleaned, bodies), start=1):
        parts.append(f"### Parallel researcher {index}\n**Task:** {assigned}\n\n{body}")
    return "\n\n".join(parts)


@tool
def spawn_researcher(task: str) -> str:
    """Spawn one researcher sub-agent that can search the web and browse pages.

    Give ONE focused research task. For several independent tasks in the same
    turn, prefer spawn_researchers or emit multiple spawn_researcher calls.
    The report includes an evaluator verdict (PASS / WEAK / FAIL).
    """
    focused = (task or "").strip()
    if not focused:
        return "Error: spawn_researcher requires a non-empty task."
    return run_researchers_parallel([focused])


@tool
def spawn_researchers(tasks: list[str]) -> str:
    """Spawn several researcher sub-agents in parallel.

    Pass 2–3 independent focused tasks (they run at the same time). Use this
    when the goal splits naturally (e.g. LangChain overview AND LlamaIndex
    overview). Each report is evaluated before it is returned.
    """
    try:
        return run_researchers_parallel(tasks)
    except Exception as exc:
        return f"Researchers failed: {exc}"
