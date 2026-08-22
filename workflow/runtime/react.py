"""A small, explicit ReAct loop on top of LangChain ChatOllama + tools."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from workflow.runtime.metrics import record
from workflow.runtime.recovery import (
    FallbackContext,
    run_fallback_chain,
    should_fallback_early,
)
from workflow.runtime.tracing import TracePrinter
from workflow.config import LLM_RETRIES
from workflow.util import invoke_tool, is_spawn_tool, message_text, run_in_threads, thought_text



@dataclass
class AgentResult:
    payload: str = ""
    stop_tool: str = ""
    stopped_reason: str = ""
    messages: list[BaseMessage] = field(default_factory=list)
    agent_id: str = ""


def _as_args(raw: Any) -> dict[str, Any]:
    """Normalize a tool call's raw args (None/dict/JSON string/other) to a dict."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {"input": text}
        return parsed if isinstance(parsed, dict) else {"input": parsed}
    return {"input": raw}


def _parse_text_tool_calls(text: str) -> list[dict[str, Any]]:
    """Fallback when the model writes a JSON tool call instead of using native tools."""
    if not text:
        return []
    candidates = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if not candidates:
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            candidates = [stripped]
    parsed: list[dict[str, Any]] = []
    for blob in candidates:
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name") or obj.get("tool") or ""
        if not name:
            fn = obj.get("function")
            if isinstance(fn, dict):
                name = fn.get("name") or ""
                args = fn.get("arguments") or fn.get("parameters") or {}
            else:
                continue
        else:
            args = obj.get("arguments") or obj.get("parameters") or obj.get("args") or {}
        if name:
            parsed.append({"name": str(name), "args": _as_args(args), "id": f"parsed-{len(parsed)}"})
    return parsed


def _tool_calls(message: AIMessage) -> list[dict[str, Any]]:
    """Extract this step's tool calls (native first, JSON-in-text as fallback)."""
    native = getattr(message, "tool_calls", None) or []
    calls: list[dict[str, Any]] = []
    for i, call in enumerate(native):
        if isinstance(call, dict):
            name = call.get("name") or ""
            args = _as_args(call.get("args") or call.get("arguments"))
            call_id = call.get("id") or f"call-{i}"
        else:
            name = getattr(call, "name", "") or ""
            args = _as_args(getattr(call, "args", None) or getattr(call, "arguments", None))
            call_id = getattr(call, "id", None) or f"call-{i}"
        if name:
            calls.append({"name": name, "args": args, "id": str(call_id)})
    if calls:
        return calls
    parsed = _parse_text_tool_calls(message_text(message))
    if parsed:
        record("parse_text_tool_calls")
    return parsed


def _invoke_one(call: dict[str, Any], tool_map: dict[str, BaseTool], tracer: TracePrinter) -> str:
    """Run one tool call serially under the acting() trace span."""
    tool = tool_map.get(call["name"])
    if tool is None:
        return f"Error: unknown tool '{call['name']}'. Available tools: {', '.join(tool_map)}."
    with tracer.acting(call["name"], call["args"]):
        return invoke_tool(tool, call["args"])


def _invoke_parallel(calls: list[dict[str, Any]], tool_map: dict[str, BaseTool]) -> list[str]:
    """Run a batch of spawn calls concurrently; results stay in call order."""
    names = ", ".join(tool_map)

    def _run(call: dict[str, Any]) -> str:
        """Pool worker: resolve and invoke one tool (error string if unknown)."""
        tool = tool_map.get(call["name"])
        if tool is None:
            return f"Error: unknown tool '{call['name']}'. Available tools: {names}."
        return invoke_tool(tool, call["args"])

    return run_in_threads(_run, calls)


def run_react(
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    system_prompt: str,
    user_message: str,
    *,
    role: str,
    max_iterations: int,
    stop_tools: Iterable[str],
) -> AgentResult:
    """Run Thought → Action → Observation until a stop tool or iteration cap."""
    stop = set(stop_tools)
    tool_map = {t.name: t for t in tools}
    names = ", ".join(tool_map)
    tracer = TracePrinter(role, max_iterations=max_iterations)
    llm_with_tools = llm.bind_tools(list(tools))

    messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]
    result = AgentResult(messages=messages, agent_id=tracer.agent_id)
    empty_actions = 0
    format_failures = 0

    def _finish() -> AgentResult:
        """Run this role's fallback chain at a stop point and return the result."""
        apply_stop_fallback(
            result,
            messages,
            stop,
            role=role,
            goal=user_message,
            tool_map=tool_map,
            tracer=tracer,
        )
        return result

    try:
        for iteration in range(1, max_iterations + 1):
            tracer.next_step()
            if iteration == max_iterations:
                record("last_step_nudge")
                messages.append(HumanMessage(content=_last_step_nudge(stop, names)))
            try:
                with tracer.thinking():
                    response = llm_with_tools.invoke(messages)
            except Exception as exc:
                format_failures += 1
                tracer.note(f"LLM error ({format_failures}/{LLM_RETRIES + 1}): {exc}")
                if format_failures > LLM_RETRIES:
                    result.stopped_reason = f"llm_error: {_friendly_llm_error(exc)}"
                    return _finish()
                messages.append(
                    HumanMessage(
                        content=(
                            f"Your previous response failed with: {exc}. "
                            f"Call exactly one of these tools: {names}."
                        )
                    )
                )
                continue

            if not isinstance(response, AIMessage):
                response = AIMessage(content=str(response))

            messages.append(response)
            tracer.thought(thought_text(response))
            calls = _tool_calls(response)

            if not calls:
                empty_actions += 1
                if should_fallback_early(stop, messages):
                    result.stopped_reason = result.stopped_reason or "no_tool_calls"
                    return _finish()
                if empty_actions >= 2:
                    result.stopped_reason = "no_tool_calls"
                    tracer.note("Stopped: two consecutive responses with no tool call.")
                    return _finish()
                record("missing_tool_nudge")
                messages.append(HumanMessage(content=_missing_tool_nudge(stop, names)))
                continue

            empty_actions = 0
            format_failures = 0
            for call in calls:
                tracer.action(call["name"], call["args"])

            if len(calls) > 1 and all(is_spawn_tool(call["name"]) for call in calls):
                observations = _invoke_parallel(calls, tool_map)
            else:
                observations = [_invoke_one(call, tool_map, tracer) for call in calls]

            for call, observation in zip(calls, observations):
                tracer.observation(observation)
                messages.append(
                    ToolMessage(
                        content=observation,
                        tool_call_id=call["id"],
                        name=call["name"],
                    )
                )
                if call["name"] in stop and not observation.startswith("Error:"):
                    result.payload = observation
                    result.stop_tool = call["name"]
                    result.stopped_reason = "stop_tool"
                    return result

        result.stopped_reason = "max_iterations"
        tracer.note(f"Reached max iterations ({max_iterations}).")
        return _finish()
    finally:
        tracer.finish(result.stopped_reason or "stopped")


def apply_stop_fallback(
    result: AgentResult,
    messages: list[BaseMessage],
    stop: set[str],
    *,
    role: str,
    goal: str = "",
    tool_map: dict[str, Any] | None = None,
    tracer: TracePrinter | None = None,
) -> bool:
    """Run the role's fallback chain and copy any payload onto ``result``.

    Defined here (and re-exported from recovery conceptually) so AgentResult
    stays local. Thin wrapper around ``run_fallback_chain``.
    """
    if result.stop_tool or result.payload:
        return False
    ctx = FallbackContext(
        role=role,
        stop_tools=set(stop),
        messages=messages,
        stopped_reason=result.stopped_reason,
        goal=goal,
        tool_map=tool_map or {},
        tracer=tracer,
    )
    outcome = run_fallback_chain(ctx)
    if outcome is None:
        return False
    if outcome.payload:
        result.payload = outcome.payload
    if outcome.stop_tool:
        result.stop_tool = outcome.stop_tool
    if outcome.stopped_reason:
        result.stopped_reason = outcome.stopped_reason
    return bool(outcome.payload)


def _last_step_nudge(stop: set[str], names: str) -> str:
    """Final-iteration reminder telling the agent to call its stop tool now."""
    if "report_findings" in stop:
        return (
            "This is your LAST step. Call report_findings NOW with every fact, number, "
            "and source URL you already have. Do not search or browse again. "
            "report_findings is a tool call, not answering from memory."
        )
    if "final_answer" in stop:
        return (
            "This is your LAST step. Call final_answer NOW using the research already "
            "gathered. Do not spawn another researcher. Writing the answer in a Thought "
            "does not finish the run."
        )
    return f"This is your last step. Call one of: {names}."


def _missing_tool_nudge(stop: set[str], names: str) -> str:
    """Nudge after a step with no tool call (per stop-tool flavor)."""
    if "final_answer" in stop:
        return (
            "Call final_answer now with the researched answer. "
            "A Thought does not finish the run. Do not spawn another researcher "
            "unless a required fact is still missing."
        )
    if "report_findings" in stop:
        return (
            "Call report_findings now with the evidence you already collected. "
            "It is a tool call — not answering from memory. Do not search again "
            "unless you have zero usable facts."
        )
    return f"You must call a tool. Available tools: {names}."


def _friendly_llm_error(exc: Exception) -> str:
    """Augment 'model not found'-style Ollama errors with fix hints."""
    text = str(exc)
    lower = text.lower()
    if "not found" in lower and "model" in lower:
        return (
            f"{text}. Check OLLAMA_MODEL (no trailing spaces) and that "
            "`ollama list` shows the tag. Pull it with `ollama pull <name>` if needed."
        )
    return text
