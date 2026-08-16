"""Replay a saved runs/*.md file without calling the LLM.

Reconstructs the event tree, re-renders it through the same TraceBus /
RichRenderer path a live run uses, and can re-run citation.py against the
saved final answer with the current audit logic.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import ToolMessage

from workflow.runtime.citations import (
    audit_citations,
    audit_to_markdown,
    build_evidence_index,
    summarize_audit,
)
from workflow.runtime.report import TraceEvent
from workflow.runtime.tracing import start_trace

_HEADING_RE = re.compile(r"^#{3,4}\s+(\S+)\s+·\s+step\s+(\S+)\s*$")
_ACTION_RE = re.compile(r"^\*\*Action\*\*\s+`([^`]+)`\s*$")
_SPAWN_RE = re.compile(r"^\*\*Spawned\*\*\s+`([^`]+)`\s+—\s+(.*)\s*$")
_FINISH_RE = re.compile(r"^\*\*Finished\*\*\s+`([^`]+)`\s+\((.*)\)\s*$")
_NOTE_RE = re.compile(r"^\*([a-z][a-z0-9_]*-\d+):\s*(.*)\*\s*$")
_META_RE = re.compile(r"^-\s+\*\*([^*]+):\*\*\s+(.*)\s*$")
_SECTION_RE = re.compile(r"^## (?!#)(.*)\s*$")
_TOP_SECTIONS = {
    "Goal",
    "Trace",
    "Shared context",
    "Citation audit (final answer)",
    "Defensive counters",
    "Final answer",
}


@dataclass
class ReplayRun:
    goal: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    events: list[TraceEvent] = field(default_factory=list)
    final: str = ""
    reason: str = ""
    started: datetime | None = None
    citation_audit_md: str = ""


def parse_run_markdown(text: str) -> ReplayRun:
    """Turn a report written by ``events_to_markdown`` back into structured data."""
    run = ReplayRun()
    sections = _split_sections(text)
    _parse_header(sections.get("", ""), run)
    run.goal = (sections.get("Goal") or "").strip()
    run.citation_audit_md = (sections.get("Citation audit (final answer)") or "").strip()
    run.final = (sections.get("Final answer") or "").strip()
    if run.final == "_(no final_answer)_":
        run.final = ""
    run.events = _parse_trace(sections.get("Trace") or "")
    return run


def replay_from_path(
    path: str | Path,
    *,
    verbose: bool = False,
    reaudit: bool = False,
    save: bool = False,
    report_dir: str | Path | None = None,
    render: bool = True,
) -> ReplayRun:
    """Load ``path``, push events through a TraceBus, optionally re-audit citations."""
    source = Path(path)
    run = parse_run_markdown(source.read_text(encoding="utf-8"))
    citation_md = run.citation_audit_md
    citation_summary = ""
    if reaudit:
        messages = tool_messages_from_events(run.events)
        audit = audit_citations(
            run.final,
            build_evidence_index(messages),
            stage="final answer (replay)",
            grounding=False,
        )
        citation_md = audit_to_markdown(audit)
        citation_summary = summarize_audit(audit)

    kwargs: dict[str, Any] = {
        "goal": run.goal,
        "verbose": verbose,
        "save": save,
        "config": run.config or None,
        "browser": False,
        "render": render,
    }
    if report_dir is not None:
        kwargs["report_dir"] = report_dir

    with start_trace(**kwargs) as bus:
        for event in run.events:
            bus.ingest_event(event)
        bus.complete(
            final=run.final,
            reason=run.reason or "replay",
            citation_audit_md=citation_md,
            citation_summary=citation_summary,
        )
        run.citation_audit_md = citation_md
    return run


def tool_messages_from_events(events: list[TraceEvent]) -> list[ToolMessage]:
    """Rebuild ToolMessages so citation audit can run against a saved trace."""
    last_tool: dict[str, str] = {}
    messages: list[ToolMessage] = []
    for event in events:
        if event.kind == "action":
            last_tool[event.agent_id] = event.tool
        elif event.kind == "observation":
            name = last_tool.get(event.agent_id, "")
            messages.append(
                ToolMessage(
                    content=event.text,
                    tool_call_id=f"replay-{len(messages)}",
                    name=name,
                )
            )
    return messages


def _split_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {"": []}
    current = ""
    for line in text.splitlines():
        match = _SECTION_RE.match(line)
        if match:
            name = match.group(1).strip()
            if name in _TOP_SECTIONS:
                current = name
                sections.setdefault(current, [])
                continue
        if line.startswith("# ") and current == "" and not sections[""]:
            continue
        sections.setdefault(current, []).append(line)
    return {name: "\n".join(body).strip("\n") for name, body in sections.items()}


def _parse_header(header: str, run: ReplayRun) -> None:
    for line in header.splitlines():
        match = _META_RE.match(line)
        if not match:
            continue
        key, value = match.group(1).strip(), match.group(2).strip()
        if key == "Model":
            run.config["model"] = value
        elif key == "Host":
            run.config["host"] = value
        elif key == "Status":
            run.reason = value
        elif key == "Started":
            run.config["started"] = value
        elif key == "Models":
            for part in value.split(","):
                if "=" in part:
                    role, name = part.split("=", 1)
                    run.config[f"model_{role.strip()}"] = name.strip()


def _parse_trace(trace: str) -> list[TraceEvent]:
    events: list[TraceEvent] = []
    agent_id = ""
    role = ""
    step = 0
    role_counts: dict[str, int] = {}
    lines = trace.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        heading = _HEADING_RE.match(line)
        if heading:
            agent_id = heading.group(1)
            role = agent_id.rsplit("-", 1)[0] if "-" in agent_id else agent_id
            try:
                step = int(heading.group(2))
            except ValueError:
                step = 0
            i += 1
            continue

        spawn = _SPAWN_RE.match(line)
        if spawn:
            spawned_role = spawn.group(1)
            n = role_counts.get(spawned_role, 0) + 1
            role_counts[spawned_role] = n
            events.append(
                TraceEvent(
                    kind="spawn",
                    role=spawned_role,
                    agent_id=f"{spawned_role}-{n}",
                    step=0,
                    text=spawn.group(2).strip(),
                )
            )
            i += 1
            continue

        finish = _FINISH_RE.match(line)
        if finish:
            finished_id = finish.group(1)
            finished_role = finished_id.rsplit("-", 1)[0] if "-" in finished_id else ""
            events.append(
                TraceEvent(
                    kind="finish",
                    role=finished_role,
                    agent_id=finished_id,
                    step=0,
                    text=finish.group(2).strip(),
                )
            )
            i += 1
            continue

        note = _NOTE_RE.match(line)
        if note:
            note_agent = note.group(1).strip()
            note_role = note_agent.rsplit("-", 1)[0] if "-" in note_agent else role
            events.append(
                TraceEvent(
                    kind="note",
                    role=note_role,
                    agent_id=note_agent,
                    step=step,
                    text=note.group(2).strip(),
                )
            )
            i += 1
            continue

        if line.strip() == "**Thought**":
            body, i = _take_body(lines, i + 1)
            events.append(
                TraceEvent(kind="thought", role=role, agent_id=agent_id, step=step, text=body)
            )
            continue

        if line.strip() == "**Observation**":
            body, i = _take_body(lines, i + 1)
            events.append(
                TraceEvent(
                    kind="observation", role=role, agent_id=agent_id, step=step, text=body
                )
            )
            continue

        action = _ACTION_RE.match(line)
        if action:
            args, i = _take_json_args(lines, i + 1)
            events.append(
                TraceEvent(
                    kind="action",
                    role=role,
                    agent_id=agent_id,
                    step=step,
                    tool=action.group(1),
                    args=args,
                    text="",
                )
            )
            continue

        i += 1
    return events


_BODY_STOP = (
    "**Thought**",
    "**Observation**",
    "**Action**",
)


def _is_boundary(line: str) -> bool:
    if _HEADING_RE.match(line) or _SPAWN_RE.match(line) or _FINISH_RE.match(line):
        return True
    if _NOTE_RE.match(line):
        return True
    stripped = line.strip()
    if stripped in _BODY_STOP or stripped.startswith("**Action**"):
        return True
    section = _SECTION_RE.match(line)
    if section and section.group(1).strip() in _TOP_SECTIONS:
        return True
    return False


def _take_body(lines: list[str], start: int) -> tuple[str, int]:
    i = start
    if i < len(lines) and lines[i].strip() == "":
        i += 1
    body: list[str] = []
    while i < len(lines) and not _is_boundary(lines[i]):
        body.append(lines[i])
        i += 1
    text = "\n".join(body).strip()
    return text.replace("``\\`", "```"), i


def _take_json_args(lines: list[str], start: int) -> tuple[dict[str, Any], int]:
    i = start
    while i < len(lines) and lines[i].strip() == "":
        i += 1
    if i < len(lines) and lines[i].strip().startswith("```"):
        i += 1
        blob: list[str] = []
        while i < len(lines) and not lines[i].strip().startswith("```"):
            blob.append(lines[i])
            i += 1
        if i < len(lines) and lines[i].strip().startswith("```"):
            i += 1
        try:
            parsed = json.loads("\n".join(blob))
        except json.JSONDecodeError:
            return {}, i
        return parsed if isinstance(parsed, dict) else {"value": parsed}, i
    return {}, i
