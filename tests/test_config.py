from __future__ import annotations

from workflow.config import MODEL, MODEL_EVALUATOR, MODEL_PLANNER, MODEL_RESEARCHER, model_for


def test_model_for_falls_back_to_default() -> None:
    assert model_for(None) == MODEL.strip()
    assert model_for("unknown") == MODEL.strip()
    assert model_for("planner") == MODEL_PLANNER.strip()
    assert model_for("researcher") == MODEL_RESEARCHER.strip()
    assert model_for("evaluator") == MODEL_EVALUATOR.strip()
