from __future__ import annotations

from langchain_core.messages import AIMessage

from workflow.runtime.metrics import snapshot
from workflow.runtime.react import _parse_text_tool_calls, _tool_calls


def test_parse_fenced_json_tool_call() -> None:
    text = """I'll search now.

```json
{"name": "web_search", "arguments": {"query": "Lisbon population"}}
```
"""
    parsed = _parse_text_tool_calls(text)
    assert parsed == [
        {"name": "web_search", "args": {"query": "Lisbon population"}, "id": "parsed-0"}
    ]


def test_parse_bare_json_and_openai_function_shape() -> None:
    bare = '{"tool": "browse_page", "args": {"url": "https://example.com"}}'
    parsed = _parse_text_tool_calls(bare)
    assert parsed[0]["name"] == "browse_page"
    assert parsed[0]["args"]["url"] == "https://example.com"

    fn = '{"function": {"name": "report_findings", "arguments": {"summary": "done"}}}'
    parsed_fn = _parse_text_tool_calls(fn)
    assert parsed_fn[0]["name"] == "report_findings"
    assert parsed_fn[0]["args"]["summary"] == "done"


def test_parse_ignores_non_tool_json() -> None:
    assert _parse_text_tool_calls('{"foo": 1}') == []
    assert _parse_text_tool_calls("not json") == []
    assert _parse_text_tool_calls("") == []


def test_tool_calls_prefers_native_then_records_text_fallback() -> None:
    native = AIMessage(content="", tool_calls=[{"name": "web_search", "args": {"query": "x"}, "id": "1"}])
    assert _tool_calls(native)[0]["name"] == "web_search"
    assert "parse_text_tool_calls" not in snapshot()

    text_only = AIMessage(content='{"name": "final_answer", "arguments": {"answer": "ok"}}')
    parsed = _tool_calls(text_only)
    assert parsed[0]["name"] == "final_answer"
    assert snapshot()["parse_text_tool_calls"] == 1
