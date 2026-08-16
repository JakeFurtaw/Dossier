"""Safe arithmetic tool. Parses + - * / ** and parentheses via ast — no eval()."""

from __future__ import annotations

import ast
import operator
from typing import Union

from langchain_core.tools import tool

Number = Union[int, float]

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def evaluate_expression(expression: str) -> Number:
    """Parse and evaluate a restricted arithmetic expression."""
    tree = ast.parse(expression, mode="eval")
    return _eval_node(tree)


def _eval_node(node: ast.AST) -> Number:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(
        node.value, bool
    ):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ZeroDivisionError("division by zero")
        return _BIN_OPS[type(node.op)](left, right)
    raise ValueError(f"unsupported expression: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression and return the numeric result.

    Allowed: numbers, parentheses, and + - * / // % **. No variables or function calls.
    Use this for multipliers, ratios, and any other calculation. Do not guess arithmetic.
    """
    text = (expression or "").strip()
    if not text:
        return "Error: calculator requires a non-empty expression."
    try:
        result = evaluate_expression(text)
    except ZeroDivisionError as exc:
        return f"Error: {exc}"
    except Exception as exc:
        return f"Error: could not evaluate {expression!r}: {exc}"
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return f"{expression} = {result}"
