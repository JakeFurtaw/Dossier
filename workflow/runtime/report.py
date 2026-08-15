"""Serialize a run to Markdown."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class TraceEvent:
    kind: str
    role: str
    agent_id: str
    step: int
    text: str = ""
    tool: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    ts: str = ""

    def __post_init__(self) -> None:
        if not self.ts:
            self.ts = datetime.now(timezone.utc).isoformat(timespec="seconds")


def _md_fence(text: str) -> str:
    return text.replace("```", "``\\`")


def events_to_markdown(
    *,
    goal: str,
    config: dict[str, Any],
    events: list[TraceEvent],
    final: str,
    reason: str,
    started: datetime,
    ended: datetime,
    citation_audit_md: str = "",
) -> str:
    duration = max((ended - started).total_seconds(), 0.0)
    lines = [
        "# Agentic workflow run",
        "",
        f"- **Started:** {started.astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}".rstrip(),
        f"- **Duration:** {duration:.1f}s",
        f"- **Model:** {config.get('model', '')}",
        f"- **Host:** {config.get('host', '')}",
        f"- **Status:** {reason or 'unknown'}",
        "",
        "## Goal",
        "",
        goal.strip() or "(none)",
        "",
        "## Trace",
        "",
    ]

    last_heading = ""
    for event in events:
        depth = 3 if event.role == "planner" else 4
        hashes = "#" * depth
        heading = f"{hashes} {event.agent_id} · step {event.step}"
        if event.kind in {"thought", "action", "observation"} and heading != last_heading:
            lines.append(heading)
            lines.append("")
            last_heading = heading

        if event.kind == "thought":
            lines.extend(["**Thought**", "", _md_fence(event.text), ""])
        elif event.kind == "action":
            args_json = json.dumps(event.args, indent=2, ensure_ascii=False, default=str)
            lines.extend(
                [
                    f"**Action** `{event.tool}`",
                    "",
                    "```json",
                    args_json,
                    "```",
                    "",
                ]
            )
        elif event.kind == "observation":
            lines.extend(["**Observation**", "", _md_fence(event.text), ""])
        elif event.kind == "note":
            lines.extend([f"*{event.agent_id}: {_md_fence(event.text)}*", ""])
        elif event.kind == "spawn":
            lines.extend([f"**Spawned** `{event.role}` — {_md_fence(event.text)}", ""])
        elif event.kind == "finish":
            lines.extend([f"**Finished** `{event.agent_id}` ({event.text})", ""])

    if citation_audit_md.strip():
        lines.extend(["## Citation audit (final answer)", "", citation_audit_md.strip(), ""])

    lines.extend(["## Final answer", "", final.strip() or "_(no final_answer)_", ""])
    return "\n".join(lines)


def write_reports(
    report_dir: Path,
    stem: str,
    *,
    goal: str,
    config: dict[str, Any],
    events: list[TraceEvent],
    final: str,
    reason: str,
    started: datetime,
    ended: datetime,
    citation_audit_md: str = "",
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    md_path = report_dir / f"{stem}.md"
    md_path.write_text(
        events_to_markdown(
            goal=goal,
            config=config,
            events=events,
            final=final,
            reason=reason,
            started=started,
            ended=ended,
            citation_audit_md=citation_audit_md,
        ),
        encoding="utf-8",
    )
    return md_path
