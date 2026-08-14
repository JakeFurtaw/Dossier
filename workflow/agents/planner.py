"""Planner / supervisor agent."""

from __future__ import annotations

from workflow.runtime.react import AgentResult, run_react
from workflow.agents.researcher import spawn_researcher, spawn_researchers
from workflow.config import PLANNER_MAX_ITERS, make_llm
from workflow.prompts import PLANNER_SYSTEM
from workflow.tools import calculator, final_answer


def run_planner(goal: str) -> AgentResult:
    """Run the planner ReAct loop for a high-level user goal."""
    return run_react(
        make_llm(),
        [spawn_researchers, spawn_researcher, calculator, final_answer],
        PLANNER_SYSTEM,
        (
            f"GOAL:\n{goal.strip()}\n\n"
            "Begin with a short Plan. Delegate independent research in parallel "
            "(spawn_researchers or multiple spawn_researcher calls in one turn), "
            "then finish with final_answer."
        ),
        role="planner",
        max_iterations=PLANNER_MAX_ITERS,
        stop_tools={"final_answer"},
        indent="",
    )
