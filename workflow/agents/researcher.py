"""Specialist sub-agents, parallel spawn, and planner-facing tools."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from workflow.agents.evaluator import evaluate_findings
from workflow.recipes import Recipe, SpecialistSpec, active_recipe
from workflow.recipes.research import RESEARCHER as DEFAULT_RESEARCHER
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
from workflow.tools import TOOL_REGISTRY, calculator, final_answer
from workflow.util import run_in_threads


def researcher_user_message(
    task: str,
    digest: str = "",
    spec: SpecialistSpec | None = None,
) -> str:
    """Compose the specialist's first user message: task + instructions + peer notes.

    Used by _gather_findings before each ReAct run.
    """
    instructions = (
        spec.user_instructions
        if spec is not None
        else (
            "Use web_search and browse_page as needed, then call report_findings. "
            "report_findings is a tool. Two useful sources is enough."
        )
    )
    body = f"Assigned task:\n{task.strip()}\n\n{instructions}"
    if digest.strip():
        body += f"\n\n{digest.strip()}"
    return body


def _peer_digest(role: str) -> str:
    """Ledger digest of what other same-role agents gathered this run (or "")."""
    bus = try_get_bus()
    if bus is None:
        return ""
    return bus.peer_digest(role)


def _publish_report(agent_id: str, role: str, task: str, text: str) -> None:
    """Record the finished report in the shared ledger so siblings can skip it."""
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
            role=role,
            agent_id=agent_id,
            kind="report",
            title=title,
            urls=extract_urls(text),
            queries=[],
        )
    )


def _tools_for_spec(spec: SpecialistSpec) -> list[Any]:
    """Resolve the specialist's declared tool names to tool objects.

    ``report_findings`` is the specialist's stop tool and must always be
    present; unknown names fail fast with the list of known tools.
    """
    if "report_findings" not in spec.tools:
        raise ValueError(
            f"specialist {spec.name!r} must include 'report_findings' in its tools "
            f"(got {list(spec.tools)})"
        )
    tools: list[Any] = []
    for name in spec.tools:
        tool = TOOL_REGISTRY.get(name)
        if tool is None:
            known = ", ".join(sorted(TOOL_REGISTRY))
            raise ValueError(
                f"specialist {spec.name!r} uses unknown tool {name!r} (known: {known})"
            )
        tools.append(tool)
    return tools


def _gather_findings(task: str, spec: SpecialistSpec) -> tuple[str, str, list[BaseMessage]]:
    """Run one specialist ReAct loop; returns (report text, agent_id, messages)."""
    result = run_react(
        make_llm(role="researcher"),
        _tools_for_spec(spec),
        spec.system_prompt,
        researcher_user_message(task, _peer_digest(spec.name), spec),
        role=spec.name,
        max_iterations=RESEARCHER_MAX_ITERS,
        stop_tools={"report_findings"},
    )
    text = result.payload or researcher_fallback(
        result.messages, result.stopped_reason or "stopped"
    )
    return text, result.agent_id, result.messages


def _with_citation_check(text: str, messages: list[BaseMessage]) -> str:
    """Append a deterministic citation verdict, checked against this
    specialist's own tool output (no extra model calls)."""
    if not CITATION_CHECK or not text.strip():
        return text
    line = citation_check_line(text, build_evidence_index(messages))
    return f"{text.strip()}\n\n{line}"


def run_researcher(
    task: str,
    *,
    allow_retry: bool = True,
    spec: SpecialistSpec | None = None,
) -> str:
    """Run a specialist, then evaluate the report (optional one retry on FAIL)."""
    chosen = spec or _researcher_spec()
    findings, agent_id, messages = _gather_findings(task, chosen)
    findings = _with_citation_check(findings, messages)
    _publish_report(agent_id, chosen.name, task, findings)
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
        second, second_id, second_messages = _gather_findings(retry_task, chosen)
        second = _with_citation_check(second, second_messages)
        _publish_report(second_id, chosen.name, task, second)
        second_review = evaluate_findings(task, second, parent_id=second_id)
        return (
            f"{second.strip()}\n\n---\n\n{second_review.text}\n\n"
            f"(Retried once after evaluator {review.verdict}.)"
        )
    return package


def _researcher_spec() -> SpecialistSpec:
    """The active recipe's 'researcher' specialist, or the research default."""
    recipe = active_recipe()
    try:
        return recipe.specialist("researcher")
    except KeyError:
        return DEFAULT_RESEARCHER


def _normalize_tasks(tasks: Any) -> list[str]:
    """Coerce a spawn_researchers-style ``tasks`` argument (str/JSON/dict/list) to task strings."""
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


def _run_one(spec: SpecialistSpec, assigned: str) -> str:
    """Run one specialist, converting any exception into an error report string."""
    try:
        return run_researcher(assigned, spec=spec)
    except Exception as exc:
        return f"{spec.name} failed: {exc}"


def run_agents_parallel(pairs: list[tuple[SpecialistSpec, str]]) -> str:
    """Run up to MAX_PARALLEL_RESEARCHERS specialists at once and join the reports."""
    cleaned = [(spec, task.strip()) for spec, task in pairs if (task or "").strip()]
    cleaned = cleaned[: max(1, MAX_PARALLEL_RESEARCHERS)]
    if not cleaned:
        return "Error: spawn requires at least one non-empty assignment."

    bodies = run_in_threads(lambda pair: _run_one(pair[0], pair[1]), cleaned)
    if len(cleaned) == 1:
        return bodies[0]
    parts = []
    for index, ((spec, assigned), body) in enumerate(zip(cleaned, bodies), start=1):
        parts.append(f"### Parallel {spec.name} {index}\n**Task:** {assigned}\n\n{body}")
    return "\n\n".join(parts)


def _normalize_assignments(raw: Any, recipe: Recipe) -> list[tuple[SpecialistSpec, str]]:
    """Coerce spawn_agents assignments (str/JSON/dict/list of {agent, task}) to (spec, task) pairs."""
    if raw is None:
        return []
    if hasattr(raw, "agent") and hasattr(raw, "task") and not isinstance(raw, (str, dict, list, tuple)):
        return _normalize_assignments({"agent": raw.agent, "task": raw.task}, recipe)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return []
        return _normalize_assignments(parsed, recipe)
    if isinstance(raw, dict):
        inner = raw.get("assignments") or raw.get("tasks") or raw.get("items")
        if inner is not None:
            return _normalize_assignments(inner, recipe)
        name = str(raw.get("agent") or raw.get("name") or raw.get("specialist") or "").strip()
        task = str(raw.get("task") or raw.get("text") or raw.get("query") or "").strip()
        if not task:
            return []
        if not name and len(recipe.specialists) == 1:
            return [(recipe.specialists[0], task)]
        try:
            return [(recipe.specialist(name), task)]
        except KeyError:
            return []
    if isinstance(raw, (list, tuple)):
        out: list[tuple[SpecialistSpec, str]] = []
        for item in raw:
            out.extend(_normalize_assignments(item, recipe))
        return out
    return []


class _SpawnTaskInput(BaseModel):
    task: str = Field(description="ONE focused assignment for this specialist.")


class _SpawnManyInput(BaseModel):
    tasks: list[str] = Field(description="Independent focused tasks (they run at the same time).")


class _AssignmentInput(BaseModel):
    agent: str = Field(description="Specialist name (listing, geo, amenities, …).")
    task: str = Field(description="Focused assignment for that specialist.")


class _SpawnAgentsInput(BaseModel):
    assignments: list[_AssignmentInput] = Field(
        description="Which specialist to run and what to assign each one."
    )


def planner_tools(recipe: Recipe | None = None) -> list[Any]:
    """Spawn tools for this recipe, plus calculator and final_answer."""
    chosen = recipe or active_recipe()
    tools: list[Any] = []
    for spec in chosen.specialists:
        tools.append(_make_spawn_one(spec))
        if spec.batch_name:
            tools.append(_make_spawn_many(spec))
    if len(chosen.specialists) > 1:
        tools.append(_make_spawn_agents(chosen))
    tools.extend([calculator, final_answer])
    return tools


def _make_spawn_one(spec: SpecialistSpec) -> StructuredTool:
    """Build the spawn_<specialist> tool (one task) for the planner."""
    def _run(task: str) -> str:
        """Tool body: run the single assignment through run_agents_parallel."""
        focused = (task or "").strip()
        if not focused:
            return f"Error: spawn_{spec.name} requires a non-empty task."
        return run_agents_parallel([(spec, focused)])

    return StructuredTool.from_function(
        name=f"spawn_{spec.name}",
        description=spec.description,
        func=_run,
        args_schema=_SpawnTaskInput,
    )


def _make_spawn_many(spec: SpecialistSpec) -> StructuredTool:
    """Build the batch spawn tool (e.g. spawn_researchers) for a specialist."""
    def _run(tasks: list[str]) -> str:
        """Tool body: normalize tasks and run them in parallel."""
        try:
            return run_agents_parallel([(spec, task) for task in _normalize_tasks(tasks)])
        except Exception as exc:
            return f"{spec.name} agents failed: {exc}"

    return StructuredTool.from_function(
        name=spec.batch_name,
        description=spec.batch_description or spec.description,
        func=_run,
        args_schema=_SpawnManyInput,
    )


def _make_spawn_agents(recipe: Recipe) -> StructuredTool:
    """Build spawn_agents for multi-specialist recipes (apartments)."""
    names = ", ".join(spec.name for spec in recipe.specialists)

    def _run(assignments: list[Any]) -> str:
        """Tool body: normalize assignments, then run the specialists in parallel."""
        try:
            pairs = _normalize_assignments(assignments, recipe)
            if not pairs:
                return (
                    f"Error: spawn_agents requires assignments "
                    f"like [{{'agent': '<name>', 'task': '...'}}]. Known agents: {names}."
                )
            return run_agents_parallel(pairs)
        except Exception as exc:
            return f"Agents failed: {exc}"

    return StructuredTool.from_function(
        name="spawn_agents",
        description=(
            f"Run several specialist agents in parallel. "
            f"Pass assignments as [{{agent, task}}, …]. Known agents: {names}."
        ),
        func=_run,
        args_schema=_SpawnAgentsInput,
    )
