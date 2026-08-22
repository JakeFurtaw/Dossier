from __future__ import annotations

from langchain_core.messages import HumanMessage, ToolMessage

from workflow.runtime.metrics import snapshot
from workflow.runtime.recovery import (
    FALLBACK_CHAINS,
    FallbackContext,
    compile_researcher_reports,
    is_blocked,
    planner_fallback_answer,
    researcher_fallback,
    run_fallback_chain,
    should_fallback_early,
)


def test_is_blocked_markers() -> None:
    assert is_blocked("Blocked: https://x returned a captcha")
    assert is_blocked("Please check for humans before continuing")
    assert is_blocked("Failed to extract usable content from: https://x")
    assert is_blocked("Timeout while fetching the page at https://x")
    assert is_blocked("Error fetching https://x: boom")
    assert is_blocked("No search results found for \"q\"")
    assert not is_blocked("### Content from: https://x\n\nA real article about robots.")


def test_compile_skips_blocked_and_errors() -> None:
    messages = [
        ToolMessage(content="Blocked: https://a captcha", tool_call_id="1", name="spawn_researcher"),
        ToolMessage(content="Error: unknown tool", tool_call_id="2", name="spawn_researcher"),
        ToolMessage(content="Useful report about Lisbon.", tool_call_id="3", name="spawn_researcher"),
        HumanMessage(content="ignore me"),
    ]
    compiled = compile_researcher_reports(messages)
    assert "Useful report about Lisbon." in compiled
    assert "captcha" not in compiled
    assert "unknown tool" not in compiled


def test_planner_fallback_answer() -> None:
    messages = [
        ToolMessage(content="LangChain notes", tool_call_id="1", name="spawn_researchers"),
    ]
    text = planner_fallback_answer(messages)
    assert text.startswith("Planner did not call final_answer")
    assert "LangChain notes" in text
    assert planner_fallback_answer([]) == ""


def test_researcher_fallback() -> None:
    empty = researcher_fallback([], "no_tool_calls")
    assert empty == "Researcher stopped without findings (no_tool_calls)."
    messages = [
        ToolMessage(
            content="Search results for \"x\" (1 hits):\n   URL: https://ok.example",
            tool_call_id="1",
            name="web_search",
        )
    ]
    dumped = researcher_fallback(messages, "max_iterations")
    assert "without report_findings (max_iterations)" in dumped
    assert "### web_search" in dumped


def test_chain_planner_raw_evidence_then_give_up() -> None:
    roles = {tier.name for tier in FALLBACK_CHAINS["planner"]}
    assert roles == {"planner.llm_synthesis", "planner.raw_evidence", "planner.give_up"}

    reports = [
        ToolMessage(content="Evidence A", tool_call_id="1", name="spawn_researcher"),
    ]
    ctx = FallbackContext(
        role="planner",
        stop_tools={"final_answer"},
        messages=reports,
        stopped_reason="no_tool_calls",
        tool_map={},  # no final_answer tool → synthesis cannot run
    )
    outcome = run_fallback_chain(ctx)
    assert outcome is not None
    assert outcome.tier == "planner.raw_evidence"
    assert "Evidence A" in outcome.payload
    assert outcome.stop_tool == "final_answer"
    assert snapshot()["planner.raw_evidence"] == 1


def test_chain_planner_give_up_when_no_reports() -> None:
    ctx = FallbackContext(
        role="planner",
        stop_tools={"final_answer"},
        messages=[],
        stopped_reason="max_iterations",
        tool_map={},
    )
    outcome = run_fallback_chain(ctx)
    assert outcome is not None
    assert outcome.tier == "planner.give_up"
    assert outcome.payload == ""
    assert outcome.stop_tool == ""


def test_chain_researcher_raw_then_give_up() -> None:
    names = [tier.name for tier in FALLBACK_CHAINS["researcher"]]
    assert names == ["researcher.raw_evidence", "researcher.give_up"]

    ctx = FallbackContext(
        role="researcher",
        stop_tools={"report_findings"},
        messages=[
            ToolMessage(content="### Content from: https://x\n\nHello", tool_call_id="1", name="browse_page")
        ],
        stopped_reason="max_iterations",
    )
    outcome = run_fallback_chain(ctx)
    assert outcome is not None
    assert outcome.tier == "researcher.raw_evidence"
    assert "Hello" in outcome.payload

    empty = run_fallback_chain(
        FallbackContext(
            role="researcher",
            stop_tools={"report_findings"},
            messages=[],
            stopped_reason="no_tool_calls",
        )
    )
    assert empty is not None
    assert empty.tier == "researcher.give_up"
    assert "without findings" in empty.payload


def test_usable_synthesis_rejects_planning_chatter() -> None:
    from workflow.runtime.recovery import _usable_synthesis

    assert not _usable_synthesis("too short")
    assert not _usable_synthesis("I will call final_answer after I think more about this plan.")
    assert not _usable_synthesis("Let me compose the final write-up next.")
    good = "## Recommendation\n\n" + ("Use LangChain. " * 20) + "https://example.com"
    assert _usable_synthesis(good)


def test_should_fallback_early() -> None:
    reports = [ToolMessage(content="ok", tool_call_id="1", name="spawn_researcher")]
    assert should_fallback_early({"final_answer"}, reports)
    assert not should_fallback_early({"final_answer"}, [])
    assert not should_fallback_early({"report_findings"}, reports)
