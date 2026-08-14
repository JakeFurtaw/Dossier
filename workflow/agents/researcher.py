"""Researcher sub-agent, parallel spawn, and planner-facing tools."""

from __future__ import annotations

import contextvars
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langchain_core.tools import tool

from workflow.agents.evaluator import evaluate_findings
from workflow.agents.react import run_react
from workflow.agents.recovery import researcher_fallback
from workflow.config import (
    EVALUATOR_ENABLED,
    EVALUATOR_RETRY,
    MAX_PARALLEL_RESEARCHERS,
    RESEARCHER_MAX_ITERS,
    make_llm,
)
from workflow.prompts import RESEARCHER_SYSTEM
from workflow.tools import browse_page, report_findings, web_search


def _gather_findings(task: str) -> str:
    result = run_react(
        make_llm(),
        [web_search, browse_page, report_findings],
        RESEARCHER_SYSTEM,
        (
            f"Assigned task:\n{task.strip()}\n\n"
            "Use web_search and browse_page as needed, then call report_findings. "
            "report_findings is a tool. Two useful sources is enough."
        ),
        role="researcher",
        max_iterations=RESEARCHER_MAX_ITERS,
        stop_tools={"report_findings"},
        indent="  ",
    )
    if result.payload:
        return result.payload
    return researcher_fallback(result.messages, result.stopped_reason or "stopped")


def run_researcher(task: str, indent: str = "  ", *, allow_retry: bool = True) -> str:
    """Run a researcher, then evaluate the report (optional one retry on FAIL)."""
    del indent  # nesting comes from the live trace tree
    findings = _gather_findings(task)
    if not EVALUATOR_ENABLED:
        return findings

    review = evaluate_findings(task, findings)
    package = f"{findings.strip()}\n\n---\n\n{review.text}"
    if review.failed and allow_retry and EVALUATOR_RETRY:
        retry_task = (
            f"{task.strip()}\n\n"
            "The evaluator REJECTED the previous report. Address these issues "
            "and return better sourced findings:\n"
            f"{review.text}"
        )
        second = _gather_findings(retry_task)
        second_review = evaluate_findings(task, second)
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
    if len(cleaned) == 1:
        return run_researcher(cleaned[0])

    results: dict[int, str] = {}

    def _run(index: int, assigned: str) -> tuple[int, str]:
        try:
            return index, run_researcher(assigned)
        except Exception as exc:
            return index, f"Researcher failed: {exc}"

    with ThreadPoolExecutor(max_workers=len(cleaned)) as pool:
        futures = []
        for index, assigned in enumerate(cleaned):
            ctx = contextvars.copy_context()
            futures.append(pool.submit(ctx.run, _run, index, assigned))
        for future in as_completed(futures):
            index, text = future.result()
            results[index] = text

    parts = []
    for index, assigned in enumerate(cleaned):
        body = results.get(index, "Researcher produced no output.")
        parts.append(f"### Parallel researcher {index + 1}\n**Task:** {assigned}\n\n{body}")
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
    try:
        return run_researcher(focused)
    except Exception as exc:
        return f"Researcher failed: {exc}"


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
