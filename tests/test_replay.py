from __future__ import annotations

from datetime import datetime, timezone

from workflow.runtime.replay import parse_run_markdown, replay_from_path, tool_messages_from_events
from workflow.runtime.report import TraceEvent, events_to_markdown
from workflow.runtime.tracing import TraceBus


SAMPLE = """# Agentic workflow run

- **Started:** 2026-08-14 16:32:55 EDT
- **Duration:** 12.0s
- **Model:** qwen3.8:latest
- **Host:** http://localhost:11434
- **Status:** stop_tool

## Goal

Compare LangChain and LlamaIndex.

## Trace

**Spawned** `planner` — planner

### planner-1 · step 1

**Thought**

Plan: delegate.

**Action** `spawn_researchers`

```json
{
  "tasks": [
    "LangChain",
    "LlamaIndex"
  ]
}
```

**Spawned** `researcher` — researcher

#### researcher-1 · step 1

**Thought**

Search first.

**Action** `web_search`

```json
{
  "query": "LangChain 2026"
}
```

**Observation**

Search results for "LangChain 2026" (1 hits):

1. Docs
   URL: https://www.langchain.com/blog

**Finished** `researcher-1` (stop_tool)

### planner-1 · step 1

**Observation**

### Parallel researcher 1
**Task:** LangChain

LangChain is an agent framework. https://www.langchain.com/blog

*planner-1: Fallback [planner.raw_evidence] fired.*

**Finished** `planner-1` (stop_tool)

## Citation audit (final answer)

_No source URLs cited._

## Final answer

Use LangChain. https://www.langchain.com/blog
"""


def test_parse_run_markdown_round_trip_shape() -> None:
    run = parse_run_markdown(SAMPLE)
    assert run.goal == "Compare LangChain and LlamaIndex."
    assert run.config["model"] == "qwen3.8:latest"
    assert run.reason == "stop_tool"
    assert run.final.startswith("Use LangChain")
    kinds = [event.kind for event in run.events]
    assert kinds[0] == "spawn"
    assert run.events[0].agent_id == "planner-1"
    assert "thought" in kinds
    assert "action" in kinds
    assert "observation" in kinds
    assert "finish" in kinds
    action = next(event for event in run.events if event.kind == "action" and event.tool == "spawn_researchers")
    assert action.args["tasks"] == ["LangChain", "LlamaIndex"]
    note = next(event for event in run.events if event.kind == "note")
    assert "planner.raw_evidence" in note.text


def test_ingest_rebuilds_agent_tree() -> None:
    run = parse_run_markdown(SAMPLE)
    bus = TraceBus(goal=run.goal, save=False)
    for event in run.events:
        bus.ingest_event(event)
    assert "planner-1" in bus.nodes
    assert "researcher-1" in bus.nodes
    planner = bus.nodes["planner-1"]
    assert any(child.agent_id == "researcher-1" for child in planner.children)
    assert bus.nodes["researcher-1"].status == "done"
    assert bus.nodes["planner-1"].status == "done"


def test_tool_messages_from_events_for_reaudit() -> None:
    run = parse_run_markdown(SAMPLE)
    messages = tool_messages_from_events(run.events)
    names = [getattr(msg, "name", "") for msg in messages]
    assert "web_search" in names
    assert "spawn_researchers" in names


NESTED_HEADINGS = """# Agentic workflow run

- **Started:** 2026-08-14 16:32:55 EDT
- **Duration:** 1.0s
- **Model:** m
- **Host:** h
- **Status:** stop_tool

## Goal

A goal.

## Trace

**Spawned** `planner` — planner

### planner-1 · step 1

**Thought**

**Plan:**
1. Do research.

**Action** `spawn_researcher`

```json
{"task": "x"}
```

**Observation**

## Verdict
PASS

## Issues
- None

**Finished** `planner-1` (stop_tool)

## Final answer

## Direct Recommendation

Use LangGraph.
"""


def test_parse_keeps_embedded_markdown_headings() -> None:
    run = parse_run_markdown(NESTED_HEADINGS)
    assert run.goal == "A goal."
    assert "## Direct Recommendation" in run.final
    assert "Use LangGraph." in run.final
    kinds = [event.kind for event in run.events]
    assert "note" not in kinds
    thought = next(event for event in run.events if event.kind == "thought")
    assert "**Plan:**" in thought.text
    observation = next(event for event in run.events if event.kind == "observation")
    assert "## Verdict" in observation.text
    assert "PASS" in observation.text


def test_replay_from_path_reaudit_without_renderer(tmp_path) -> None:
    path = tmp_path / "run.md"
    path.write_text(SAMPLE, encoding="utf-8")
    run = replay_from_path(path, verbose=False, reaudit=True, save=False, render=False)
    assert "LangChain" in run.final
    assert run.citation_audit_md
    assert "langchain.com" in run.citation_audit_md.lower() or "verified" in run.citation_audit_md.lower()


def test_events_to_markdown_includes_counters() -> None:
    started = datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc)
    ended = datetime(2026, 8, 14, 12, 1, tzinfo=timezone.utc)
    md = events_to_markdown(
        goal="g",
        config={"model": "m", "host": "h"},
        events=[
            TraceEvent(kind="spawn", role="planner", agent_id="planner-1", step=0, text="planner"),
        ],
        final="done",
        reason="stop_tool",
        started=started,
        ended=ended,
        counters={"parse_text_tool_calls": 2, "blocked_page": 1},
        shared_context_md="- researcher-1 searched “langchain 2026”\n",
    )
    assert "## Defensive counters" in md
    assert "`parse_text_tool_calls`: 2" in md
    assert "## Shared context" in md
    assert "langchain 2026" in md
    parsed = parse_run_markdown(md)
    assert parsed.goal == "g"
    assert parsed.final == "done"
    assert parsed.events[0].kind == "spawn"
