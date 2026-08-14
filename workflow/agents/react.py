"""A small, explicit ReAct loop on top of LangChain ChatOllama + tools.

create_agent / create_react_agent would hide the trace. This runner keeps
Thought → Action → Observation first-class, stops on named tools, and
recovers from malformed model output.
"""

from __future__ import annotations

import contextvars
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from workflow.agents.recovery import (
    compile_researcher_reports,
    looks_like_answer,
    planner_fallback_answer,
)
from workflow.agents.tracing import TracePrinter
from workflow.config import LLM_RETRIES

_PARALLEL_TOOLS = {"spawn_researcher", "spawn_researchers"}


@dataclass
class AgentResult:
    payload: str = ""
    stop_tool: str = ""
    last_text: str = ""
    iterations: int = 0
    stopped_reason: str = ""
    messages: list[BaseMessage] = field(default_factory=list)


def _message_text(message: BaseMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        return "\n".join(p for p in parts if p).strip()
    return str(content or "").strip()


def _thought_text(message: BaseMessage) -> str:
    extra = getattr(message, "additional_kwargs", None) or {}
    reasoning = str(extra.get("reasoning_content") or extra.get("reasoning") or "").strip()
    content = _message_text(message)
    if reasoning and content:
        return f"{reasoning}\n\n{content}"
    return reasoning or content or "(no explicit thought)"


def _as_args(raw: Any) -> dict[str, Any]:
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
    return _parse_text_tool_calls(_message_text(message))


def _invoke_tool(tool: BaseTool, args: dict[str, Any]) -> str:
    try:
        result = tool.invoke(args)
    except Exception as exc:
        return f"Error running {tool.name}: {exc}"
    if result is None:
        return ""
    return result if isinstance(result, str) else str(result)


def _should_run_parallel(calls: list[dict[str, Any]]) -> bool:
    if len(calls) < 2:
        return False
    return all(call["name"] in _PARALLEL_TOOLS for call in calls)


def _invoke_one(call: dict[str, Any], tool_map: dict[str, BaseTool], tracer: TracePrinter) -> str:
    tool = tool_map.get(call["name"])
    if tool is None:
        return f"Error: unknown tool '{call['name']}'. Available tools: {', '.join(tool_map)}."
    with tracer.acting(call["name"], call["args"]):
        return _invoke_tool(tool, call["args"])


def _invoke_parallel(calls: list[dict[str, Any]], tool_map: dict[str, BaseTool]) -> list[str]:
    """Run independent spawn_* tool calls at the same time."""
    observations = [""] * len(calls)

    def _run(index: int) -> tuple[int, str]:
        call = calls[index]
        tool = tool_map.get(call["name"])
        if tool is None:
            return index, (
                f"Error: unknown tool '{call['name']}'. Available tools: {', '.join(tool_map)}."
            )
        return index, _invoke_tool(tool, call["args"])

    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        futures = []
        for index in range(len(calls)):
            ctx = contextvars.copy_context()
            futures.append(pool.submit(ctx.run, _run, index))
        for future in as_completed(futures):
            index, text = future.result()
            observations[index] = text
    return observations


def run_react(
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    system_prompt: str,
    user_message: str,
    *,
    role: str,
    max_iterations: int,
    stop_tools: Iterable[str],
    indent: str = "",
) -> AgentResult:
    """Run Thought → Action → Observation until a stop tool or iteration cap."""
    stop = set(stop_tools)
    tool_map = {t.name: t for t in tools}
    names = ", ".join(tool_map)
    tracer = TracePrinter(role, indent=indent, max_iterations=max_iterations)
    llm_with_tools = llm.bind_tools(list(tools))

    messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]
    result = AgentResult(messages=messages)
    empty_actions = 0
    format_failures = 0

    try:
        for iteration in range(1, max_iterations + 1):
            result.iterations = iteration
            tracer.next_step()
            if iteration == max_iterations:
                messages.append(HumanMessage(content=_last_step_nudge(stop, names)))
            try:
                with tracer.thinking():
                    response = llm_with_tools.invoke(messages)
            except Exception as exc:
                format_failures += 1
                tracer.note(f"LLM error ({format_failures}/{LLM_RETRIES + 1}): {exc}")
                if format_failures > LLM_RETRIES:
                    result.stopped_reason = f"llm_error: {_friendly_llm_error(exc)}"
                    result.last_text = result.stopped_reason
                    _salvage_stop(result, messages, stop, user_message)
                    return result
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
            tracer.thought(_thought_text(response))
            calls = _tool_calls(response)

            if not calls:
                empty_actions += 1
                result.last_text = _thought_text(response)
                # After research is in, a thought-only turn means the model
                # "answered in prose" instead of calling final_answer. Nudging
                # usually produces another thought. Force a finish turn instead.
                if (
                    "final_answer" in stop
                    and compile_researcher_reports(messages)
                    and empty_actions >= 1
                ):
                    forced = _try_forced_final_answer(
                        llm, tool_map, messages, user_message, tracer
                    )
                    if forced is not None:
                        result.payload = forced
                        result.stop_tool = "final_answer"
                        result.stopped_reason = "stop_tool"
                        result.last_text = forced
                        return result
                    tracer.note(
                        "Planner wrote thoughts instead of calling final_answer; "
                        "using researcher reports."
                    )
                    _salvage_stop(result, messages, stop, user_message)
                    return result
                if empty_actions >= 2:
                    result.stopped_reason = "no_tool_calls"
                    tracer.note("Stopped: two consecutive responses with no tool call.")
                    _salvage_stop(result, messages, stop, user_message)
                    return result
                messages.append(
                    HumanMessage(content=_missing_tool_nudge(stop, result.last_text, names, user_message))
                )
                continue

            empty_actions = 0
            format_failures = 0
            for call in calls:
                tracer.action(call["name"], call["args"])

            if _should_run_parallel(calls):
                observations = _invoke_parallel(calls, tool_map)
            else:
                observations = []
                for call in calls:
                    observations.append(_invoke_one(call, tool_map, tracer))

            for call, observation in zip(calls, observations):
                tracer.observation(observation)
                messages.append(
                    ToolMessage(
                        content=observation,
                        tool_call_id=call["id"],
                        name=call["name"],
                    )
                )
                result.last_text = observation
                if call["name"] in stop and not observation.startswith("Error:"):
                    result.payload = observation
                    result.stop_tool = call["name"]
                    result.stopped_reason = "stop_tool"
                    return result

        result.stopped_reason = "max_iterations"
        tracer.note(f"Reached max iterations ({max_iterations}).")
        _salvage_stop(result, messages, stop, user_message)
        return result
    finally:
        tracer.finish(result.stopped_reason or "stopped")


def _last_step_nudge(stop: set[str], names: str) -> str:
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


def _missing_tool_nudge(stop: set[str], last_text: str, names: str, goal: str = "") -> str:
    if "final_answer" in stop and looks_like_answer(last_text, goal):
        return (
            "You already drafted the answer in text. That does not complete the run. "
            "Call final_answer now and put that researched answer (not these instructions) "
            "in the answer argument. Do not spawn another researcher unless a required "
            "fact is still missing."
        )
    if "report_findings" in stop:
        return (
            "Call report_findings now with the evidence you already collected. "
            "It is a tool call — not answering from memory. Do not search again "
            "unless you have zero usable facts."
        )
    return f"You must call a tool. Available tools: {names}."


def _try_forced_final_answer(
    llm: BaseChatModel,
    tool_map: dict[str, BaseTool],
    messages: list[BaseMessage],
    goal: str,
    tracer: TracePrinter,
) -> str | None:
    """One compact retry with only final_answer bound, after a thought-only turn."""
    tool = tool_map.get("final_answer")
    if tool is None:
        return None
    reports = compile_researcher_reports(messages)
    if not reports:
        return None
    tracer.note("Forcing a final_answer tool call (research is done; no tool was called).")
    compact = reports if len(reports) <= 9000 else reports[:9000] + "\n… [truncated]"
    forced_llm = llm.bind_tools([tool])
    forced_messages: list[BaseMessage] = [
        SystemMessage(
            content=(
                "You are finishing a multi-agent research run. "
                "You already have researcher reports. "
                "Call the final_answer tool now with a complete markdown answer. "
                "Do not write a plan. Do not call any other tool. "
                "A Thought without a tool call does not finish the run."
            )
        ),
        HumanMessage(
            content=(
                f"Original goal:\n{goal.strip()}\n\n"
                f"Research evidence:\n{compact}\n\n"
                "Call final_answer now."
            )
        ),
    ]
    try:
        with tracer.thinking():
            response = forced_llm.invoke(forced_messages)
    except Exception as exc:
        tracer.note(f"Forced final_answer invoke failed: {exc}")
        return None
    if not isinstance(response, AIMessage):
        response = AIMessage(content=str(response))
    tracer.thought(_thought_text(response))
    calls = _tool_calls(response)
    for call in calls:
        if call["name"] != "final_answer":
            continue
        tracer.action("final_answer", call["args"])
        observation = _invoke_tool(tool, call["args"])
        tracer.observation(observation)
        if observation and not observation.startswith("Error:"):
            return observation
    return None


def _salvage_stop(
    result: AgentResult,
    messages: list[BaseMessage],
    stop: set[str],
    goal: str = "",
) -> None:
    """Keep a goal-matching draft (or researcher reports), never tool-call meta."""
    if result.stop_tool or result.payload:
        return
    if "final_answer" not in stop:
        return
    draft = planner_fallback_answer(messages, goal)
    if not draft:
        return
    result.payload = draft
    result.stop_tool = "final_answer"
    result.stopped_reason = f"salvaged_final_answer({result.stopped_reason})"
    result.last_text = draft


def _friendly_llm_error(exc: Exception) -> str:
    text = str(exc)
    lower = text.lower()
    if "not found" in lower and "model" in lower:
        return (
            f"{text}. Check OLLAMA_MODEL (no trailing spaces) and that "
            "`ollama list` shows the tag. Pull it with `ollama pull <name>` if needed."
        )
    return text
