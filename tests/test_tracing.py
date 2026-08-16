from __future__ import annotations

from workflow.runtime.report import TraceEvent
from workflow.runtime.tracing import TraceBus, TraceListener, get_bus, start_trace


class _Collector(TraceListener):
    def __init__(self) -> None:
        self.kinds: list[str] = []

    def on_event(self, bus: TraceBus, event: TraceEvent) -> None:
        del bus
        self.kinds.append(event.kind)


def test_trace_bus_records_events_without_rich() -> None:
    bus = TraceBus(goal="goal", save=False)
    collector = _Collector()
    bus.subscribe(collector)
    agent = bus.start_agent("planner", max_iterations=3)
    bus.thought(agent, "planner", 1, "thinking")
    bus.action(agent, "planner", 1, "calculator", {"expression": "1+1"})
    bus.observation(agent, "planner", 1, "1+1 = 2")
    bus.end_agent(agent, "stop_tool")
    assert [event.kind for event in bus.events] == [
        "spawn",
        "thought",
        "action",
        "observation",
        "finish",
    ]
    assert collector.kinds == [event.kind for event in bus.events]
    assert bus.nodes[agent].role == "planner"
    assert bus.roots[0].agent_id == agent


def test_url_cache_is_normalized_and_thread_scoped() -> None:
    bus = TraceBus(save=False)
    bus.put_cached_url("https://www.Example.com/path/?q=1", "body")
    assert bus.get_cached_url("https://example.com/path") == "body"
    assert bus.get_cached_url("https://other.example/x") is None


def test_browse_page_uses_run_url_cache() -> None:
    from workflow.runtime.metrics import snapshot
    from workflow.tools.web import browse_page

    with start_trace(goal="g", save=False, render=False, browser=False) as bus:
        bus.put_cached_url("https://example.com/page", "### Content from: https://example.com/page\n\ncached body")
        result = browse_page.invoke({"url": "https://www.example.com/page", "instructions": "dates"})
        assert "cached body" in result
        assert result.startswith("Focus requested: dates")
        assert snapshot()["url_cache_hit"] == 1


def test_start_trace_without_browser_or_renderer() -> None:
    with start_trace(goal="g", save=False, render=False, browser=False) as bus:
        assert get_bus() is bus
        bus.put_cached_url("https://www.example.com/a/", "cached")
        assert bus.get_cached_url("https://example.com/a") == "cached"
        bus.complete(final="ok", reason="stop_tool")
        assert bus.final == "ok"


def test_researcher_nests_under_planner() -> None:
    bus = TraceBus(save=False)
    planner = bus.start_agent("planner")
    researcher = bus.start_agent("researcher", parent_id=planner)
    assert bus.nodes[researcher] in bus.nodes[planner].children
    assert bus.depth(researcher) == 1
