"""A recipe is a named bundle of prompts and specialist agents.

The ReAct loop, tools, tracing, citations, and salvage stay in ``workflow.runtime``.
A recipe only changes who the planner can spawn and what each agent is told.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SpecialistSpec:
    """One planner-spawnable sub-agent (researcher, listing, geo, …)."""

    name: str
    system_prompt: str
    description: str
    user_instructions: str = (
        "Use web_search and browse_page as needed, then call report_findings. "
        "report_findings is a tool. Two useful sources is enough."
    )
    color: str = "magenta"
    batch_name: str = ""
    batch_description: str = ""


@dataclass(frozen=True)
class Recipe:
    """Everything that differs between general research and a custom workflow."""

    name: str
    description: str
    default_goal: str
    planner_system: str
    planner_kickoff: str
    evaluator_system: str
    synthesis_system: str
    specialists: tuple[SpecialistSpec, ...]
    role_colors: dict[str, str] = field(default_factory=dict)

    def specialist(self, name: str) -> SpecialistSpec:
        for spec in self.specialists:
            if spec.name == name:
                return spec
        known = ", ".join(spec.name for spec in self.specialists) or "(none)"
        raise KeyError(f"recipe {self.name!r} has no specialist {name!r} (have {known})")
