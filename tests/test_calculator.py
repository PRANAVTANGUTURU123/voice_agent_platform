"""Calculator correctness and, more importantly, safety."""

import math

import pytest

from app.agent.tools import calculator, safe_eval


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("2 + 3 * 4", 14),
        ("(2 + 3) * 4", 20),
        ("2 ^ 10", 1024),
        ("10 / 4", 2.5),
        ("7 // 2", 3),
        ("-5 + 2", -3),
        ("sqrt(144)", 12.0),
        ("round(pi, 2)", 3.14),
        ("15/100*240", 36.0),
        ("factorial(5)", 120),
        ("max(3, 9, 4)", 9),
    ],
)
def test_valid_expressions(expr: str, expected: float) -> None:
    assert math.isclose(safe_eval(expr), expected)


@pytest.mark.parametrize(
    "malicious",
    [
        "__import__('os').system('echo pwned')",
        "open('secrets.txt').read()",
        "().__class__.__bases__[0].__subclasses__()",
        "eval('1+1')",
        "lambda: 1",
        "[1, 2, 3]",
        "x + 1",
        "'a' * 10",
        "sqrt.__globals__",
        "print(1)",
    ],
)
def test_rejects_code_execution(malicious: str) -> None:
    with pytest.raises((ValueError, SyntaxError)):
        safe_eval(malicious)


def test_rejects_resource_exhaustion() -> None:
    with pytest.raises(ValueError):
        safe_eval("9 ** 9 ** 9")
    with pytest.raises(ValueError):
        safe_eval("factorial(100000)")
    with pytest.raises(ValueError):
        safe_eval("1+" * 200 + "1")


def test_tool_returns_errors_as_text() -> None:
    # The tool must never raise: errors become strings the LLM can read.
    assert calculator.invoke({"expression": "1/0"}) == "calculator error: division by zero"
    assert calculator.invoke({"expression": "import os"}).startswith("calculator error")
    assert calculator.invoke({"expression": "6*7"}) == "42"
    assert calculator.invoke({"expression": "10/4"}) == "2.5"
