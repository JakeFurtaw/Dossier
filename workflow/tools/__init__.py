"""LangChain tools used by the planner and researcher agents."""

from workflow.tools.calc import calculator
from workflow.tools.control import final_answer, report_findings
from workflow.tools.web import browse_page, fetch_raw, web_search

TOOL_REGISTRY: dict[str, object] = {
    "web_search": web_search,
    "browse_page": browse_page,
    "fetch_raw": fetch_raw,
    "calculator": calculator,
    "report_findings": report_findings,
    "final_answer": final_answer,
}

__all__ = [
    "TOOL_REGISTRY",
    "browse_page",
    "calculator",
    "fetch_raw",
    "final_answer",
    "report_findings",
    "web_search",
]
