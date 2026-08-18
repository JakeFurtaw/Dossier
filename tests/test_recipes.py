from __future__ import annotations

import pytest
from langchain_core.messages import ToolMessage

from workflow.agents.researcher import _normalize_assignments, planner_tools
from workflow.recipes import get_recipe, list_recipes, use_recipe
from workflow.runtime.citations import build_evidence_index
from workflow.runtime.recovery import (
    FallbackContext,
    compile_researcher_reports,
    run_fallback_chain,
)
from workflow.runtime.report import TraceEvent, events_to_markdown
from workflow.runtime.tracing import TraceBus


def test_list_and_resolve_recipes() -> None:
    names = {recipe.name for recipe in list_recipes()}
    assert names == {"research", "apartments"}
    assert get_recipe("research").specialist("researcher").name == "researcher"
    assert get_recipe("general").name == "research"
    apartments = get_recipe("apartment")
    assert apartments.name == "apartments"
    assert [spec.name for spec in apartments.specialists] == ["listing", "geo", "amenities"]


def test_unknown_recipe_lists_available() -> None:
    with pytest.raises(ValueError, match="unknown workflow"):
        get_recipe("spaceships")


def test_planner_tools_match_recipe() -> None:
    research = {tool.name for tool in planner_tools(get_recipe("research"))}
    assert research == {
        "spawn_researcher",
        "spawn_researchers",
        "calculator",
        "final_answer",
    }
    apartments = {tool.name for tool in planner_tools(get_recipe("apartments"))}
    assert apartments == {
        "spawn_listing",
        "spawn_geo",
        "spawn_amenities",
        "spawn_agents",
        "calculator",
        "final_answer",
    }


def test_normalize_assignments_accepts_objects_and_json() -> None:
    recipe = get_recipe("apartments")
    pairs = _normalize_assignments(
        [
            {"agent": "listing", "task": "Reston 1-beds under 2800"},
            {"name": "geo", "text": "Wiehle vs Innovation Center"},
        ],
        recipe,
    )
    assert [spec.name for spec, _ in pairs] == ["listing", "geo"]
    from_json = _normalize_assignments(
        '{"assignments": [{"agent": "amenities", "task": "score these units"}]}',
        recipe,
    )
    assert from_json[0][0].name == "amenities"
    assert _normalize_assignments([{"agent": "nope", "task": "x"}], recipe) == []


def test_spawn_listing_counts_as_evidence_and_salvage() -> None:
    messages = [
        ToolMessage(
            content="Unit at 12000 Market St, $2400. https://example.com/listing",
            tool_call_id="1",
            name="spawn_listing",
        ),
        ToolMessage(
            content="Wiehle-Reston East is on the Silver Line. https://example.com/metro",
            tool_call_id="2",
            name="spawn_geo",
        ),
    ]
    compiled = compile_researcher_reports(messages)
    assert "12000 Market St" in compiled
    assert "Wiehle-Reston East" in compiled
    index = build_evidence_index(messages)
    assert "example.com/listing" in index
    assert "example.com/metro" in index

    outcome = run_fallback_chain(
        FallbackContext(
            role="listing",
            stop_tools={"report_findings"},
            messages=[
                ToolMessage(
                    content="### Content from: https://example.com/a\n\n$2,400 / mo",
                    tool_call_id="1",
                    name="browse_page",
                )
            ],
            stopped_reason="max_iterations",
        )
    )
    assert outcome is not None
    assert outcome.tier == "researcher.raw_evidence"
    assert "$2,400" in outcome.payload


def test_use_recipe_sets_active_and_restores() -> None:
    from workflow.recipes import active_recipe

    assert active_recipe().name == "research"
    with use_recipe(get_recipe("apartments")):
        assert active_recipe().name == "apartments"
    assert active_recipe().name == "research"


def test_replay_nests_listing_under_planner() -> None:
    bus = TraceBus(save=False)
    bus.ingest_event(TraceEvent(kind="spawn", role="planner", agent_id="planner-1", step=0, text="planner"))
    bus.ingest_event(TraceEvent(kind="spawn", role="listing", agent_id="listing-1", step=0, text="listing"))
    bus.ingest_event(TraceEvent(kind="spawn", role="evaluator", agent_id="evaluator-1", step=0, text="evaluator"))
    assert bus.nodes["listing-1"] in bus.nodes["planner-1"].children
    assert bus.nodes["evaluator-1"] in bus.nodes["listing-1"].children


def test_report_includes_workflow_name() -> None:
    from datetime import datetime, timezone

    started = datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc)
    ended = datetime(2026, 8, 17, 12, 1, tzinfo=timezone.utc)
    md = events_to_markdown(
        goal="g",
        config={"model": "m", "host": "h", "workflow": "apartments"},
        events=[],
        final="done",
        reason="stop_tool",
        started=started,
        ended=ended,
    )
    assert "- **Workflow:** apartments" in md


def test_cli_lists_and_selects_workflow(capsys) -> None:
    from dossier import main, _parse_args

    assert main(["--list-workflows"]) == 0
    out = capsys.readouterr().out
    assert "research" in out
    assert "apartments" in out

    args = _parse_args(["--workflow", "apartments", "2 bed near Wiehle"])
    assert args.workflow == "apartments"
    assert " ".join(args.goal) == "2 bed near Wiehle"
    assert main(["--workflow", "spaceships"]) == 2


def test_specialist_default_tools() -> None:
    default = ("web_search", "browse_page", "report_findings")
    assert get_recipe("research").specialist("researcher").tools == default
    for name in ("listing", "geo", "amenities"):
        assert get_recipe("apartments").specialist(name).tools == default


def test_tools_for_spec_resolves_and_validates() -> None:
    from workflow.agents.researcher import _tools_for_spec
    from workflow.recipes import SpecialistSpec

    names = [t.name for t in _tools_for_spec(get_recipe("research").specialist("researcher"))]
    assert names == ["web_search", "browse_page", "report_findings"]

    custom = SpecialistSpec(
        name="api",
        system_prompt="s",
        description="d",
        tools=("web_search", "fetch_raw", "report_findings"),
    )
    assert [t.name for t in _tools_for_spec(custom)] == [
        "web_search",
        "fetch_raw",
        "report_findings",
    ]

    with pytest.raises(ValueError, match="report_findings"):
        _tools_for_spec(
            SpecialistSpec(name="bad1", system_prompt="s", description="d", tools=("web_search",))
        )
    with pytest.raises(ValueError, match="unknown tool"):
        _tools_for_spec(
            SpecialistSpec(
                name="bad2", system_prompt="s", description="d", tools=("nope", "report_findings")
            )
        )
