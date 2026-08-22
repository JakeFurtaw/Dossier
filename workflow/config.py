"""Runtime configuration."""

from __future__ import annotations

import os

from langchain_ollama import ChatOllama

def _env_str(name: str, default: str) -> str:
    """Read a string env var, falling back to ``default`` when unset or blank."""
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean env var (0/false/no/off are False), else ``default``."""
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


MODEL = _env_str("OLLAMA_MODEL", "qwen3.8:latest")
MODEL_PLANNER = _env_str("OLLAMA_MODEL_PLANNER", "") or MODEL
MODEL_RESEARCHER = _env_str("OLLAMA_MODEL_RESEARCHER", "") or MODEL
MODEL_EVALUATOR = _env_str("OLLAMA_MODEL_EVALUATOR", "") or MODEL
HOST = _env_str("OLLAMA_HOST", "http://localhost:11434")
TEMPERATURE = float(_env_str("OLLAMA_TEMPERATURE", "0.2"))
NUM_PREDICT = int(_env_str("OLLAMA_NUM_PREDICT", "2048"))
REASONING = _env_bool("OLLAMA_REASONING", True)
PLANNER_MAX_ITERS = int(_env_str("PLANNER_MAX_ITERS", "8"))
RESEARCHER_MAX_ITERS = int(_env_str("RESEARCHER_MAX_ITERS", "6"))
OBS_TRUNCATE = int(_env_str("OBS_TRUNCATE", "1500"))
LLM_RETRIES = int(_env_str("LLM_RETRIES", "2"))
REPORT_DIR = _env_str("REPORT_DIR", "runs")
VERBOSE = _env_bool("VERBOSE", False)
MAX_PARALLEL_RESEARCHERS = int(_env_str("MAX_PARALLEL_RESEARCHERS", "3"))
EVALUATOR_ENABLED = _env_bool("EVALUATOR_ENABLED", True)
EVALUATOR_RETRY = _env_bool("EVALUATOR_RETRY", True)
CITATION_CHECK = _env_bool("CITATION_CHECK", True)
CITATION_STRICT = _env_bool("CITATION_STRICT", False)

_ROLE_MODELS = {
    "planner": MODEL_PLANNER,
    "researcher": MODEL_RESEARCHER,
    "evaluator": MODEL_EVALUATOR,
}


def model_for(role: str | None = None) -> str:
    """Resolve the Ollama model tag for a role, falling back to OLLAMA_MODEL."""
    if role and role in _ROLE_MODELS:
        return _ROLE_MODELS[role].strip()
    return MODEL.strip()


def make_llm(
    *,
    role: str | None = None,
    reasoning: bool | None = None,
    num_predict: int | None = None,
) -> ChatOllama:
    """Build an Ollama chat model. ``role`` selects OLLAMA_MODEL_<ROLE> when set."""
    return ChatOllama(
        model=model_for(role),
        base_url=HOST,
        temperature=TEMPERATURE,
        num_predict=NUM_PREDICT if num_predict is None else num_predict,
        reasoning=REASONING if reasoning is None else reasoning,
    )
