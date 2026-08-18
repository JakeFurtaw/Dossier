from __future__ import annotations

from langchain_core.messages import ToolMessage

from workflow.runtime.citations import (
    audit_citations,
    audit_to_markdown,
    build_evidence_index,
    citation_check_line,
    extract_urls,
    normalize_url,
    summarize_audit,
)


def test_normalize_url_strips_www_query_fragment_and_slash() -> None:
    assert normalize_url("https://www.Example.com/path/?q=1#frag") == "example.com/path"


def test_normalize_url_collapses_old_reddit() -> None:
    assert normalize_url("https://old.reddit.com/r/foo/") == "reddit.com/r/foo"


def test_normalize_url_keeps_nonstandard_port() -> None:
    assert normalize_url("http://localhost:8080/x/") == "localhost:8080/x"


def test_normalize_url_strips_trailing_punctuation() -> None:
    assert normalize_url("https://example.com/a).") == "example.com/a"


def test_extract_urls_unique_in_order() -> None:
    text = "See https://www.example.com/a and https://example.com/a/ plus https://other.test/b."
    assert extract_urls(text) == ["https://www.example.com/a", "https://other.test/b"]


def test_build_evidence_index_parses_search_browse_and_spawn() -> None:
    messages = [
        ToolMessage(
            content='Search results for "x" (1 hits):\n\n1. Title\n   URL: https://www.example.com/page\n',
            tool_call_id="1",
            name="web_search",
        ),
        ToolMessage(
            content="### Content from: https://example.com/page\n\nThe figure is 42,000.",
            tool_call_id="2",
            name="browse_page",
        ),
        ToolMessage(
            content="Report cites https://other.test/src",
            tool_call_id="3",
            name="spawn_researcher",
        ),
    ]
    index = build_evidence_index(messages)
    assert "example.com/page" in index
    assert "other.test/src" in index
    page = index["example.com/page"]
    assert "web_search" in page.sources
    assert "browse_page" in page.sources
    assert "42,000" in page.body


def test_build_evidence_index_blocked_page_still_counts_as_provenance() -> None:
    messages = [
        ToolMessage(
            content="Blocked: https://walled.example/x returned a CAPTCHA or bot-check, not article text.",
            tool_call_id="1",
            name="browse_page",
        )
    ]
    index = build_evidence_index(messages)
    assert "walled.example/x" in index
    assert index["walled.example/x"].sources == ["browse_page (no content)"]
    assert index["walled.example/x"].body == ""


def test_audit_verifies_cited_urls() -> None:
    messages = [
        ToolMessage(
            content="### Content from: https://example.com/stats\n\nPopulation is 504,718 in 2021.",
            tool_call_id="1",
            name="browse_page",
        )
    ]
    index = build_evidence_index(messages)
    text = "Lisbon has 504,718 people (https://example.com/stats) but not https://made-up.test/nope"
    audit = audit_citations(text, index, stage="researcher report", grounding=True)
    assert audit.total == 2
    assert audit.verified_count == 1
    assert not audit.all_verified
    bad = audit.unverified[0]
    assert "made-up.test" in bad.url
    good = [row for row in audit.rows if row.verified][0]
    assert good.grounding_checked
    assert good.numbers_total >= 1
    assert good.numbers_found >= 1


def test_audit_skips_number_grounding_without_body() -> None:
    messages = [
        ToolMessage(
            content="See https://example.com/a",
            tool_call_id="1",
            name="spawn_researcher",
        )
    ]
    index = build_evidence_index(messages)
    audit = audit_citations(
        "fact (https://example.com/a)",
        index,
        stage="final answer",
        grounding=False,
    )
    assert audit.all_verified
    assert not audit.rows[0].grounding_checked


def test_citation_check_line_and_summarize() -> None:
    messages = [
        ToolMessage(
            content="   URL: https://ok.example/a",
            tool_call_id="1",
            name="web_search",
        )
    ]
    index = build_evidence_index(messages)
    ok = citation_check_line("source https://ok.example/a", index)
    assert "1/1 URLs verified" in ok
    empty = citation_check_line("no urls here", index)
    assert "no source URLs cited" in empty
    bad = citation_check_line("https://ok.example/a and https://fake.example/x", index)
    assert "NOT in any tool output" in bad

    audit = audit_citations("https://ok.example/a", index, stage="t", grounding=False)
    assert summarize_audit(audit) == "citations: 1/1 verified"
    assert summarize_audit(audit_citations("", index, stage="t")) == "citations: none cited"


def test_audit_to_markdown_table() -> None:
    messages = [
        ToolMessage(
            content="### Content from: https://example.com/a\n\n42 units",
            tool_call_id="1",
            name="browse_page",
        )
    ]
    index = build_evidence_index(messages)
    audit = audit_citations(
        "42 units at https://example.com/a",
        index,
        stage="researcher report",
        grounding=True,
    )
    md = audit_to_markdown(audit)
    assert "| URL | Verified | Seen in | Numbers |" in md
    assert "1/1 URLs traced" in md
    assert audit_to_markdown(audit_citations("", index, stage="t")) == "_No source URLs cited._"


def test_evidence_index_counts_fetch_raw_output() -> None:
    messages = [
        ToolMessage(
            content='### Content from: https://api.example/units\n\n{"rent": 2400}',
            tool_call_id="1",
            name="fetch_raw",
        )
    ]
    index = build_evidence_index(messages)
    entry = index.get("api.example/units")
    assert entry is not None
    assert entry.sources == ["fetch_raw"]
    assert "2400" in entry.body
