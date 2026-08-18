#!/usr/bin/env python3
"""Dossier — run or replay a multi-agent research session."""

from __future__ import annotations

import argparse
import sys

from workflow.agents.planner import run_planner
from workflow.recipes import Recipe, get_recipe, list_recipes, use_recipe
from workflow.runtime.citations import (
    audit_citations,
    audit_to_markdown,
    build_evidence_index,
    summarize_audit,
)
from workflow.runtime.replay import replay_from_path
from workflow.runtime.tracing import start_trace
from workflow.config import (
    CITATION_CHECK,
    CITATION_STRICT,
    HOST,
    MODEL,
    MODEL_EVALUATOR,
    MODEL_PLANNER,
    MODEL_RESEARCHER,
    NUM_PREDICT,
    PLANNER_MAX_ITERS,
    REPORT_DIR,
    TEMPERATURE,
    VERBOSE,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Dossier research session.")
    parser.add_argument("goal", nargs="*", help="Research goal (default depends on --workflow).")
    parser.add_argument(
        "-w",
        "--workflow",
        default="research",
        metavar="NAME",
        help="Recipe to run: research (default) or apartments. See --list-workflows.",
    )
    parser.add_argument(
        "--list-workflows",
        action="store_true",
        help="List available workflows and exit.",
    )
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
    parser.add_argument(
        "--replay",
        metavar="PATH",
        help="Replay a saved runs/*.md file without calling the LLM.",
    )
    parser.add_argument(
        "--reaudit",
        action="store_true",
        help="With --replay, re-run citation audit on the saved final answer.",
    )
    return parser.parse_args(argv)


def run(
    goal: str,
    *,
    verbose: bool,
    save: bool,
    report_dir: str,
    recipe: Recipe | None = None,
) -> int:
    chosen = recipe or get_recipe("research")
    config = {
        "workflow": chosen.name,
        "model": MODEL,
        "model_planner": MODEL_PLANNER,
        "model_researcher": MODEL_RESEARCHER,
        "model_evaluator": MODEL_EVALUATOR,
        "host": HOST,
        "temperature": TEMPERATURE,
        "num_predict": NUM_PREDICT,
        "planner_max_iters": PLANNER_MAX_ITERS,
    }
    with use_recipe(chosen), start_trace(
        goal=goal,
        verbose=verbose,
        save=save,
        report_dir=report_dir,
        config=config,
    ) as session:
        result = run_planner(goal, recipe=chosen)
        final = result.payload if result.stop_tool == "final_answer" else ""
        citation_md = ""
        citation_summary = ""
        strict_fail = False
        if final and CITATION_CHECK:
            audit = audit_citations(
                final,
                build_evidence_index(result.messages),
                stage="final answer",
                grounding=False,
            )
            citation_md = audit_to_markdown(audit)
            citation_summary = summarize_audit(audit)
            strict_fail = CITATION_STRICT and not audit.all_verified
        session.complete(
            final=final,
            reason=result.stopped_reason,
            citation_audit_md=citation_md,
            citation_summary=citation_summary,
        )
        if not final:
            return 1
        return 1 if strict_fail else 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.list_workflows:
        for recipe in list_recipes():
            print(f"{recipe.name:12} {recipe.description}")
        return 0
    if args.replay:
        replay_from_path(
            args.replay,
            verbose=args.verbose,
            reaudit=args.reaudit,
            save=not args.no_save and args.reaudit,
            report_dir=args.report_dir,
        )
        return 0
    try:
        recipe = get_recipe(args.workflow)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    goal = " ".join(args.goal).strip() if args.goal else recipe.default_goal
    return run(
        goal,
        verbose=args.verbose,
        save=not args.no_save,
        report_dir=args.report_dir,
        recipe=recipe,
    )


if __name__ == "__main__":
    raise SystemExit(main())
