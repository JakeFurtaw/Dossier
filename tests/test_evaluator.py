from __future__ import annotations

from workflow.agents.evaluator import parse_verdict


def test_parse_verdict_heading() -> None:
    assert parse_verdict("## Verdict\nPASS\n\n## Issues\n- None") == "PASS"
    assert parse_verdict("## Verdict\n**FAIL**\n\n## Issues\n- empty") == "FAIL"
    assert parse_verdict("## verdict\nweak") == "WEAK"


def test_parse_verdict_inline_fallback() -> None:
    assert parse_verdict("Verdict: FAIL because no sources") == "FAIL"
    assert parse_verdict("The VERDICT is PASS overall") == "PASS"
    assert parse_verdict("verdict looks WEAK to me") == "WEAK"


def test_parse_verdict_default_weak() -> None:
    assert parse_verdict("") == "WEAK"
    assert parse_verdict("No structured output at all") == "WEAK"
