"""Post-research evaluator: one LLM pass over a researcher's findings."""

from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from workflow.agents.tracing import TracePrinter
from workflow.config import make_llm
from workflow.prompts import EVALUATOR_SYSTEM


@dataclass
class Evaluation:
    verdict: str
    text: str

    @property
    def failed(self) -> bool:
        return self.verdict == "FAIL"


def parse_verdict(text: str) -> str:
    match = re.search(r"(?im)^\s*##\s*verdict\s*$", text or "")
    if match:
        tail = text[match.end() :]
        line = next((ln.strip() for ln in tail.splitlines() if ln.strip()), "")
        token = line.split()[0].upper().strip("*_:") if line else ""
        if token in {"PASS", "WEAK", "FAIL"}:
            return token
    upper = (text or "").upper()
    if re.search(r"\bVERDICT\b[^\n]*\bFAIL\b", upper):
        return "FAIL"
    if re.search(r"\bVERDICT\b[^\n]*\bPASS\b", upper):
        return "PASS"
    if re.search(r"\bVERDICT\b[^\n]*\bWEAK\b", upper):
        return "WEAK"
    return "WEAK"


def evaluate_findings(task: str, findings: str) -> Evaluation:
    """Score a researcher's report against its assigned task."""
    printer = TracePrinter("evaluator", max_iterations=1)
    printer.next_step()
    llm = make_llm(reasoning=False, num_predict=600)
    prompt = (
        f"Assigned task:\n{task.strip()}\n\n"
        f"Findings to evaluate:\n{findings.strip() or '(empty)'}"
    )
    try:
        with printer.thinking():
            response = llm.invoke(
                [SystemMessage(content=EVALUATOR_SYSTEM), HumanMessage(content=prompt)]
            )
    except Exception as exc:
        printer.note(f"Evaluator LLM error: {exc}")
        printer.finish("llm_error")
        text = f"## Verdict\nWEAK\n\n## Issues\n- Evaluator failed: {exc}\n\n## Notes\nCould not validate."
        return Evaluation(verdict="WEAK", text=text)

    content = getattr(response, "content", "")
    if isinstance(content, list):
        content = "\n".join(
            block if isinstance(block, str) else str(block.get("text") or "")
            for block in content
        )
    body = str(content or "").strip() or "## Verdict\nWEAK\n\n## Issues\n- Empty evaluator response."
    extra = getattr(response, "additional_kwargs", None) or {}
    reasoning = str(extra.get("reasoning_content") or "").strip()
    printer.thought(reasoning or body)
    verdict = parse_verdict(body)
    formatted = f"## Evaluator\n**Verdict:** {verdict}\n\n{body}"
    printer.note(f"verdict {verdict}")
    printer.finish(verdict.lower())
    return Evaluation(verdict=verdict, text=formatted)
