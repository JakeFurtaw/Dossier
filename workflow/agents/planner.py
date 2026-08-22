"""Planner / supervisor agent."""

from __future__ import annotations

from workflow.agents.researcher import planner_tools
from workflow.recipes import Recipe, active_recipe, use_recipe
from workflow.runtime.react import AgentResult, run_react
from workflow.config import PLANNER_MAX_ITERS, make_llm


def run_planner(goal: str, recipe: Recipe | None = None) -> AgentResult:
    """Run the planner ReAct loop for a high-level user goal."""
    chosen = recipe or active_recipe()

    def _run() -> AgentResult:
        """Planner ReAct loop with the recipe's tools, prompts, and max iterations."""
        return run_react(
            make_llm(role="planner"),
            planner_tools(chosen),
            chosen.planner_system,
            f"GOAL:\n{goal.strip()}\n\n{chosen.planner_kickoff}",
            role="planner",
            max_iterations=PLANNER_MAX_ITERS,
            stop_tools={"final_answer"},
        )

    if recipe is None:
        return _run()
    with use_recipe(recipe):
        return _run()
