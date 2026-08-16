from __future__ import annotations

import pytest

from workflow.tools.calc import calculator, evaluate_expression


def test_evaluate_basic_ops() -> None:
    assert evaluate_expression("1 + 2 * 3") == 7
    assert evaluate_expression("(1 + 2) * 3") == 9
    assert evaluate_expression("2 ** 8") == 256
    assert evaluate_expression("10 // 3") == 3
    assert evaluate_expression("10 % 3") == 1
    assert evaluate_expression("-4 + 1") == -3


def test_evaluate_rejects_names_and_calls() -> None:
    with pytest.raises(ValueError):
        evaluate_expression("abs(-1)")
    with pytest.raises(ValueError):
        evaluate_expression("x + 1")


def test_evaluate_division_by_zero() -> None:
    with pytest.raises(ZeroDivisionError):
        evaluate_expression("1 / 0")


def test_calculator_tool_formats_and_errors() -> None:
    assert calculator.invoke({"expression": "2 + 2"}) == "2 + 2 = 4"
    assert calculator.invoke({"expression": "4 / 2"}) == "4 / 2 = 2"
    assert calculator.invoke({"expression": ""}).startswith("Error:")
    assert "division by zero" in calculator.invoke({"expression": "1 / 0"})
    assert calculator.invoke({"expression": "foo("}).startswith("Error: could not evaluate")
