"""Built-in example tools (FRAMEWORK_DESIGN.md §7.4): both safe, read-only and public.

  product_catalogue_lookup  a fictional product catalogue read from a local JSON fixture
                            (tools/data/product_catalogue.json by default)
  calculator                arithmetic on numbers only, evaluated over a parsed AST,
                            never with eval()

Importing sdgf.tools registers both in tools/registry.REGISTRY, so a spec can list them
by name. Both are deterministic, so a record's tool trace replays it exactly (see
ToolCache.from_trace).

An unknown product code is a normal result ({"found": false, ...}), not a failure, so the
model can be told and the answer is cached. A bad expression raises ValueError, which
the gateway returns to the model as tool_failed.
"""

from __future__ import annotations

import ast
import json
import math
import operator
from pathlib import Path
from typing import Any

from sdgf.tools.registry import REGISTRY, ToolDefinition, ToolRegistry

CATALOGUE_LOOKUP = "product_catalogue_lookup"
CALCULATOR = "calculator"
DEFAULT_CATALOGUE = Path(__file__).resolve().parent / "data" / "product_catalogue.json"

MAX_EXPRESSION_CHARS = 200
MAX_EXPONENT = 100
MAX_MAGNITUDE = 1e18


def load_catalogue(path: str | Path = DEFAULT_CATALOGUE) -> dict[str, dict[str, Any]]:
    """The catalogue keyed by product_code; a missing or duplicate code raises."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    products = data.get("products") if isinstance(data, dict) else None
    if not isinstance(products, list):
        raise ValueError(f"{path}: expected an object with a 'products' list")
    catalogue: dict[str, dict[str, Any]] = {}
    for i, product in enumerate(products):
        code = product.get("product_code") if isinstance(product, dict) else None
        if not isinstance(code, str) or not code:
            raise ValueError(f"{path}: product {i} has no product_code")
        if code in catalogue:
            raise ValueError(f"{path}: duplicate product_code {code!r}")
        catalogue[code] = product
    return catalogue


def catalogue_lookup_tool(path: str | Path = DEFAULT_CATALOGUE) -> ToolDefinition:
    catalogue = load_catalogue(path)

    def lookup(arguments: dict[str, Any]) -> dict[str, Any]:
        code = arguments["product_code"].strip().upper()
        product = catalogue.get(code)
        if product is None:
            return {"found": False, "product_code": code, "known_codes": sorted(catalogue)}
        return {"found": True, "product": product}

    return ToolDefinition(
        name=CATALOGUE_LOOKUP,
        description=(
            "Look up a product in the (fictional) product catalogue by its product code. "
            "Returns the product's name, category, fees, rates and features."
        ),
        input_schema={
            "type": "object",
            "properties": {"product_code": {"type": "string", "minLength": 1, "maxLength": 32}},
            "required": ["product_code"],
            "additionalProperties": False,
        },
        sensitivity="public",
        handler=lookup,
    )


_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _evaluate(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"unsupported constant {node.value!r}")
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError(f"exponent {right} is larger than {MAX_EXPONENT}")
        try:
            value = _BINARY[type(node.op)](left, right)
        except ZeroDivisionError:
            raise ValueError("division by zero") from None
        if isinstance(value, complex):
            raise ValueError("result is not a real number")
        if not math.isfinite(value) or abs(value) > MAX_MAGNITUDE:
            raise ValueError(f"result is larger than {MAX_MAGNITUDE:g}")
        return value
    raise ValueError(f"unsupported expression element: {type(node).__name__}")


def calculate(expression: str) -> float | int:
    """Evaluate +, -, *, /, //, %, ** and parentheses over numbers; anything else raises."""
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise ValueError(f"expression is longer than {MAX_EXPRESSION_CHARS} characters")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        raise ValueError(f"not an arithmetic expression: {expression!r}") from None
    value = _evaluate(tree)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def calculator_tool() -> ToolDefinition:
    def run(arguments: dict[str, Any]) -> dict[str, Any]:
        expression = arguments["expression"]
        return {"expression": expression, "result": calculate(expression)}

    return ToolDefinition(
        name=CALCULATOR,
        description=(
            "Evaluate an arithmetic expression over numbers, for example '1200 * 0.045 / 12'. "
            "Supports + - * / // % ** and parentheses."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "minLength": 1, "maxLength": MAX_EXPRESSION_CHARS}
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
        sensitivity="public",
        handler=run,
    )


def builtin_tools(catalogue_path: str | Path = DEFAULT_CATALOGUE) -> list[ToolDefinition]:
    return [catalogue_lookup_tool(catalogue_path), calculator_tool()]


def register_builtin_tools(
    registry: ToolRegistry = REGISTRY,
    *,
    catalogue_path: str | Path = DEFAULT_CATALOGUE,
    replace: bool = False,
) -> list[ToolDefinition]:
    return [registry.register(t, replace=replace) for t in builtin_tools(catalogue_path)]


register_builtin_tools(REGISTRY, replace=True)
