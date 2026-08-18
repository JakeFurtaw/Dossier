"""Tiered fallbacks when an agent stops without its stop tool.

Each role has an ordered chain. The first tier that returns a payload wins.
New tiers (e.g. a cheaper synthesis model) slot in by appending to the
role's list — no hunting through react.py.

    planner:     llm synthesis → raw evidence dump → give-up message
    researcher:  raw evidence dump → give-up message
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from workflow.config import make_llm
from workflow.runtime.metrics import record
from workflow.util import message_text, thought_text

_USEFUL_TOOLS = {"web_search", "browse_page"}


def _is_spawn_tool(name: str) -> bool:
    return (name or "").startswith("spawn_")
_BLOCKED_MARKERS = (
    "blocked:",
    "captcha",
    "check for humans",
    "failed to extract",
    "timeout while",
    "error fetching",
    "no search results",
)


def is_blocked(text: str) -> bool:
    """True when tool output is a blocked/empty/failed page, not usable evidence."""
    lower = (text or "").lower()
    return any(marker in lower for marker in _BLOCKED_MARKERS)


def _tool_chunks(
    messages: Iterable[BaseMessage],
    names: set[str],
    limit: int,
    *,
    headed: bool = False,
) -> list[str]:
    chunks: list[str] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        name = getattr(message, "name", "") or "tool"
        if name not in names:
            continue
        text = message_text(message)
        if not text or text.startswith("Error") or is_blocked(text):
            continue
        if len(text) > limit:
            text = text[:limit].rstrip() + f"\n… [{len(text) - limit} more chars truncated]"
        chunks.append(f"### {name}\n{text}" if headed else text)
    return chunks


def compile_researcher_reports(messages: Iterable[BaseMessage], limit: int = 12000) -> str:
    collected = list(messages)
    names = {
        getattr(message, "name", "") or ""
        for message in collected
        if _is_spawn_tool(getattr(message, "name", "") or "")
    }
    if not names:
        return ""
    return "\n\n---\n\n".join(_tool_chunks(collected, names, limit))


def planner_fallback_answer(messages: Iterable[BaseMessage], goal: str = "") -> str:
    del goal
    reports = compile_researcher_reports(messages)
    if not reports:
        return ""
    return (
        "Planner did not call final_answer. Evidence gathered by researchers:\n\n"
        + reports
    )


def researcher_fallback(messages: Iterable[BaseMessage], reason: str) -> str:
    notes = "\n\n".join(_tool_chunks(messages, _USEFUL_TOOLS, 2500, headed=True))
    if not notes:
        return f"Researcher stopped without findings ({reason})."
    return (
        f"Researcher stopped without report_findings ({reason}). "
        "Compiled evidence from tool results:\n\n"
        f"{notes}"
    )


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------


@dataclass
class FallbackOutcome:
    payload: str
    stop_tool: str = ""
    tier: str = ""
    stopped_reason: str = ""


@dataclass
class FallbackContext:
    role: str
    stop_tools: set[str]
    messages: list[BaseMessage]
    stopped_reason: str = ""
    goal: str = ""
    tool_map: dict[str, Any] = field(default_factory=dict)
    tracer: Any = None


class FallbackTier(Protocol):
    name: str

    def applies(self, ctx: FallbackContext) -> bool: ...

    def try_recover(self, ctx: FallbackContext) -> FallbackOutcome | None: ...


def _usable_synthesis(text: str) -> bool:
    """True when the model wrote a user-facing answer, not more planning chatter."""
    if not text or len(text.strip()) < 200:
        return False
    lower = text.lower()
    if "call final_answer" in lower and "## " not in text and "http" not in lower:
        return False
    planning = ("let me compose", "let me structure", "let me write the final", "i'll delegate")
    if any(marker in lower for marker in planning) and text.count("## ") < 2:
        return False
    return True


def _invoke_tool(tool: Any, args: dict[str, Any]) -> str:
    try:
        result = tool.invoke(args)
    except Exception as exc:
        return f"Error running {getattr(tool, 'name', 'tool')}: {exc}"
    if result is None:
        return ""
    return result if isinstance(result, str) else str(result)


def _commit_final_answer(tool: Any, answer: str, tracer: Any) -> str | None:
    if tracer is not None:
        tracer.action("final_answer", {"answer": answer})
    observation = _invoke_tool(tool, {"answer": answer})
    if tracer is not None:
        tracer.observation(observation)
    if observation and not observation.startswith("Error:"):
        return observation
    return None


class PlannerLlmSynthesis:
    """No-tools write step, then submit final_answer ourselves.

    Ollama ignores tool_choice, and reasoning models often burn the token
    budget thinking about the call instead of emitting it.
    """

    name = "planner.llm_synthesis"

    def applies(self, ctx: FallbackContext) -> bool:
        return "final_answer" in ctx.stop_tools and bool(compile_researcher_reports(ctx.messages))

    def try_recover(self, ctx: FallbackContext) -> FallbackOutcome | None:
        tool = (ctx.tool_map or {}).get("final_answer")
        reports = compile_researcher_reports(ctx.messages)
        if tool is None or not reports:
            return None

        compact = reports if len(reports) <= 12000 else reports[:12000] + "\n… [truncated]"
        writer = make_llm(role="planner", reasoning=False, num_predict=2048)
        prompt = [
            SystemMessage(content=_synthesis_system()),
            HumanMessage(
                content=(
                    f"Original goal:\n{ctx.goal.strip()}\n\n"
                    f"Research evidence:\n{compact}\n\n"
                    "Write the complete markdown answer now."
                )
            ),
        ]

        tracer = ctx.tracer
        if tracer is not None:
            tracer.note("Writing final_answer from researcher reports (no tool call from planner).")
        try:
            thinking = tracer.thinking() if tracer is not None else _null_cm()
            with thinking:
                response = writer.invoke(prompt)
        except Exception as exc:
            if tracer is not None:
                tracer.note(f"Final-answer synthesis failed: {exc}")
            return None

        if not isinstance(response, AIMessage):
            response = AIMessage(content=str(response))
        prose = message_text(response) or thought_text(response)
        if tracer is not None:
            tracer.thought(prose)
        if not _usable_synthesis(prose):
            return None
        committed = _commit_final_answer(tool, prose, tracer)
        if not committed:
            return None
        return FallbackOutcome(
            payload=committed,
            stop_tool="final_answer",
            tier=self.name,
            stopped_reason="stop_tool",
        )


class PlannerRawEvidence:
    name = "planner.raw_evidence"

    def applies(self, ctx: FallbackContext) -> bool:
        return "final_answer" in ctx.stop_tools

    def try_recover(self, ctx: FallbackContext) -> FallbackOutcome | None:
        draft = planner_fallback_answer(ctx.messages, ctx.goal)
        if not draft:
            return None
        return FallbackOutcome(
            payload=draft,
            stop_tool="final_answer",
            tier=self.name,
            stopped_reason=f"salvaged_final_answer({ctx.stopped_reason})",
        )


class PlannerGiveUp:
    name = "planner.give_up"

    def applies(self, ctx: FallbackContext) -> bool:
        return "final_answer" in ctx.stop_tools

    def try_recover(self, ctx: FallbackContext) -> FallbackOutcome | None:
        reason = ctx.stopped_reason or "stopped"
        return FallbackOutcome(
            payload="",
            stop_tool="",
            tier=self.name,
            stopped_reason=reason,
        )


class ResearcherRawEvidence:
    name = "researcher.raw_evidence"

    def applies(self, ctx: FallbackContext) -> bool:
        return "report_findings" in ctx.stop_tools

    def try_recover(self, ctx: FallbackContext) -> FallbackOutcome | None:
        notes = "\n\n".join(_tool_chunks(ctx.messages, _USEFUL_TOOLS, 2500, headed=True))
        if not notes:
            return None
        return FallbackOutcome(
            payload=(
                f"Researcher stopped without report_findings "
                f"({ctx.stopped_reason or 'stopped'}). "
                "Compiled evidence from tool results:\n\n"
                f"{notes}"
            ),
            stop_tool="",
            tier=self.name,
            stopped_reason=ctx.stopped_reason,
        )


class ResearcherGiveUp:
    name = "researcher.give_up"

    def applies(self, ctx: FallbackContext) -> bool:
        return "report_findings" in ctx.stop_tools

    def try_recover(self, ctx: FallbackContext) -> FallbackOutcome | None:
        reason = ctx.stopped_reason or "stopped"
        return FallbackOutcome(
            payload=f"Researcher stopped without findings ({reason}).",
            stop_tool="",
            tier=self.name,
            stopped_reason=reason,
        )


FALLBACK_CHAINS: dict[str, list[FallbackTier]] = {
    "planner": [PlannerLlmSynthesis(), PlannerRawEvidence(), PlannerGiveUp()],
    "researcher": [ResearcherRawEvidence(), ResearcherGiveUp()],
}


def should_fallback_early(role: str, stop_tools: set[str], messages: list[BaseMessage]) -> bool:
    """True when the agent went quiet but already has enough evidence to salvage."""
    del role
    if "final_answer" in stop_tools:
        return bool(compile_researcher_reports(messages))
    return False


def _chain_for(ctx: FallbackContext) -> list[FallbackTier]:
    if ctx.role in FALLBACK_CHAINS:
        return FALLBACK_CHAINS[ctx.role]
    if "final_answer" in ctx.stop_tools:
        return FALLBACK_CHAINS["planner"]
    if "report_findings" in ctx.stop_tools:
        return FALLBACK_CHAINS["researcher"]
    return []


def _synthesis_system() -> str:
    try:
        from workflow.recipes import active_recipe

        return active_recipe().synthesis_system
    except Exception:
        from workflow.recipes.research import SYNTHESIS_SYSTEM

        return SYNTHESIS_SYSTEM


def run_fallback_chain(ctx: FallbackContext) -> FallbackOutcome | None:
    """Walk the role's tiers. First non-None payload (or give-up) wins."""
    chain = _chain_for(ctx)
    for tier in chain:
        if not tier.applies(ctx):
            continue
        outcome = tier.try_recover(ctx)
        if outcome is None:
            continue
        record(tier.name)
        if ctx.tracer is not None:
            ctx.tracer.note(f"Fallback [{tier.name}] fired.")
        return outcome
    return None


class _null_cm:
    def __enter__(self):
        return None

    def __exit__(self, *exc: object) -> bool:
        return False
