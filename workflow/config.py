"""Runtime configuration for the multi-agent demo."""

from __future__ import annotations

import os

from langchain_ollama import ChatOllama

DEFAULT_GOAL = (
    "What is the approximate current population of Lisbon, Portugal? "
    "How many times larger is it than a town of 5,000 people? "
    "Provide a short summary with sources or estimates and the calculation."
)


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


MODEL = _env_str("OLLAMA_MODEL", "qwen3.8:latest").strip()
HOST = _env_str("OLLAMA_HOST", "http://localhost:11434")
TEMPERATURE = float(_env_str("OLLAMA_TEMPERATURE", "0.2"))
NUM_PREDICT = int(_env_str("OLLAMA_NUM_PREDICT", "2048"))
REASONING = _env_bool("OLLAMA_REASONING", True)
PLANNER_MAX_ITERS = int(_env_str("PLANNER_MAX_ITERS", "30"))
RESEARCHER_MAX_ITERS = int(_env_str("RESEARCHER_MAX_ITERS", "6"))
OBS_TRUNCATE = int(_env_str("OBS_TRUNCATE", "1500"))
LLM_RETRIES = int(_env_str("LLM_RETRIES", "2"))
REPORT_DIR = _env_str("REPORT_DIR", "runs")
VERBOSE = _env_bool("VERBOSE", False)
MAX_PARALLEL_RESEARCHERS = int(_env_str("MAX_PARALLEL_RESEARCHERS", "3"))
EVALUATOR_ENABLED = _env_bool("EVALUATOR_ENABLED", True)
EVALUATOR_RETRY = _env_bool("EVALUATOR_RETRY", True)


def make_llm(*, reasoning: bool | None = None, num_predict: int | None = None) -> ChatOllama:
    """Build the shared Ollama chat model used by planner, researchers, and evaluator."""
    return ChatOllama(
        model=MODEL.strip(),
        base_url=HOST,
        temperature=TEMPERATURE,
        num_predict=NUM_PREDICT if num_predict is None else num_predict,
        reasoning=REASONING if reasoning is None else reasoning,
    )
