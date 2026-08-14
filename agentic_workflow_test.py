#!/usr/bin/env python3
"""
Multi-agent ReAct demo (Ollama + LangChain).

How to extend later
-------------------
- Add a worker role: write a system prompt, give it tools, and wrap it in a
  LangChain `@tool` the same way `spawn_researcher` wraps `run_researcher`.
- Add a real tool: implement a function in `workflow/tools/`, decorate it with
  `@tool`, and attach it to the planner or researcher tool list.
- Swap models: `OLLAMA_MODEL=gemma4:31b python agentic_workflow_test.py`
- Point at a remote Ollama host: `OLLAMA_HOST=http://host:11434 ...`

This file is the spec entry point. The loop lives in `workflow/agents/react.py`
so Thought / Action / Observation stay visible instead of hidden inside a graph.
"""

from __future__ import annotations

import argparse
import sys

from workflow.agents.planner import run_planner
from workflow.agents.tracing import start_trace
from workflow.config import (
    DEFAULT_GOAL,
    HOST,
    MODEL,
    NUM_PREDICT,
    PLANNER_MAX_ITERS,
    REPORT_DIR,
    TEMPERATURE,
    VERBOSE,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the multi-agent ReAct demo.")
    parser.add_argument("goal", nargs="*", help="Goal to solve (default: Lisbon population).")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=VERBOSE,
        help="Print full Thought / Action / Observation panels in the terminal.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write runs/<timestamp>.md.",
    )
    parser.add_argument(
        "--report-dir",
        default=REPORT_DIR,
        help=f"Directory for run reports (default: {REPORT_DIR}).",
    )
    return parser.parse_args(argv)


def run(goal: str, *, verbose: bool, save: bool, report_dir: str) -> int:
    config = {
        "model": MODEL,
        "host": HOST,
        "temperature": TEMPERATURE,
        "num_predict": NUM_PREDICT,
        "planner_max_iters": PLANNER_MAX_ITERS,
    }
    with start_trace(
        goal=goal,
        verbose=verbose,
        save=save,
        report_dir=report_dir,
        config=config,
    ) as session:
        result = run_planner(goal)
        final = result.payload if result.stop_tool == "final_answer" else ""
        session.complete(final=final, reason=result.stopped_reason)
        return 0 if final else 1


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    goal = " ".join(args.goal).strip() if args.goal else DEFAULT_GOAL
    return run(goal, verbose=args.verbose, save=not args.no_save, report_dir=args.report_dir)


if __name__ == "__main__":
    raise SystemExit(main())
