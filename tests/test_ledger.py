from __future__ import annotations

import threading
import time

from workflow.agents.researcher import researcher_user_message
from workflow.runtime.ledger import (
    LedgerEntry,
    SharedContext,
    brief_text,
    normalize_search_query,
    search_cache_key,
)
from workflow.runtime.tracing import TraceBus, start_trace


def test_normalize_search_query_collapses_case_and_space() -> None:
    assert normalize_search_query("  LangChain   2026 ") == "langchain 2026"
    assert search_cache_key("LangChain 2026", 5) == search_cache_key("langchain   2026", 5)


def test_brief_text_strips_citation_and_evaluator() -> None:
    text = (
        "## LangChain\n\nIt reached 1.0 in October 2025.\n\n"
        "**Citation check:** 2/2 URLs verified.\n\n"
        "---\n\n## Evaluator\n**Verdict:** PASS"
    )
    brief = brief_text(text)
    assert "1.0" in brief
    assert "Citation check" not in brief
    assert "Evaluator" not in brief


def test_digest_is_empty_without_peers() -> None:
    ctx = SharedContext()
    assert ctx.digest("researcher") == ""
    ctx.publish(
        LedgerEntry(role="researcher", agent_id="researcher-1", kind="search", queries=["foo"])
    )
    assert ctx.digest("researcher", exclude_agent="researcher-1") == ""
    digest = ctx.digest("researcher", exclude_agent="researcher-2")
    assert "researcher-1 searched" in digest
    assert "Already gathered" in digest
    assert ctx.digest("planner") == ""


def test_digest_and_markdown_cover_all_kinds() -> None:
    ctx = SharedContext()
    ctx.publish(
        LedgerEntry(
            role="researcher",
            agent_id="researcher-1",
            kind="search",
            queries=["langchain 2026"],
        )
    )
    ctx.publish(
        LedgerEntry(
            role="researcher",
            agent_id="researcher-1",
            kind="browse",
            urls=["https://www.langchain.com/blog"],
        )
    )
    ctx.publish(
        LedgerEntry(
            role="researcher",
            agent_id="researcher-1",
            kind="report",
            title="LangChain 1.0 shipped October 2025",
            urls=["https://www.langchain.com/blog", "https://example.com/a"],
        )
    )
    digest = ctx.digest("researcher")
    assert "searched “langchain 2026”" in digest
    assert "browsed https://www.langchain.com/blog" in digest
    assert "report: LangChain 1.0" in digest
    assert "sources:" in digest
    md = ctx.to_markdown()
    assert md.startswith("## Shared context")
    assert "searched" in md


def test_search_inflight_single_writer() -> None:
    ctx = SharedContext()
    started = threading.Event()

    def owner() -> None:
        assert ctx.acquire_search("q", 5)
        started.set()
        time.sleep(0.05)
        ctx.put_search("q", 5, "RESULT")
        ctx.release_search("q", 5)

    def waiter() -> None:
        started.wait(timeout=1)
        assert not ctx.acquire_search("q", 5)
        ctx.wait_search("q", 5, timeout=1)
        assert ctx.get_search("q", 5) == "RESULT"

    first = threading.Thread(target=owner)
    second = threading.Thread(target=waiter)
    first.start()
    second.start()
    first.join()
    second.join()


def test_researcher_user_message_appends_digest() -> None:
    bare = researcher_user_message("Cover LangChain")
    assert "Assigned task:\nCover LangChain" in bare
    assert "Already gathered" not in bare
    with_digest = researcher_user_message("Cover LangChain", "Already gathered by other researchers this run.")
    assert "Already gathered by other researchers this run." in with_digest


def test_web_search_uses_run_cache(monkeypatch) -> None:
    from workflow.runtime.metrics import snapshot
    from workflow.tools.web import web_search

    calls: list[str] = []

    def fake_search(query: str, max_results: int | None = None) -> list[dict]:
        calls.append(query)
        return [{"title": "Docs", "href": "https://example.com/a", "body": "hello"}]

    monkeypatch.setattr("workflow.tools.web.sync_search", fake_search)
    with start_trace(goal="g", save=False, render=False, browser=False) as bus:
        bus.start_agent("researcher")
        first = web_search.invoke({"query": "LangChain 2026", "max_results": 5})
        second = web_search.invoke({"query": "langchain   2026", "max_results": 5})
        assert snapshot()["search_cache_hit"] == 1
        assert any(entry.kind == "search" for entry in bus.shared.entries)
    assert first == second
    assert len(calls) == 1


def test_bus_peer_digest_after_report() -> None:
    bus = TraceBus(save=False)
    bus.publish_entry(
        LedgerEntry(
            role="researcher",
            agent_id="researcher-1",
            kind="report",
            title="Lisbon is about 545,000",
            urls=["https://example.com/lisbon"],
        )
    )
    digest = bus.peer_digest("researcher", exclude_agent="researcher-2")
    assert "Lisbon" in digest
    assert "https://example.com/lisbon" in digest
