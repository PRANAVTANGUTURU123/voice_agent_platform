"""The agent's three tools: pdf_qa, web_search, calculator.

Every tool catches its own exceptions and returns a readable error string, so
a failing tool never crashes the agent — the LLM sees the error and can tell
the user (or try another tool).
"""

from __future__ import annotations

import ast
import logging
import math
import operator
from typing import Any, Callable

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app.config import get_settings
from app.rag.ingest import get_session_index
from app.rag.retriever import format_results, retrieve

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 1. pdf_qa
# --------------------------------------------------------------------------- #
@tool
def pdf_qa(query: str, config: RunnableConfig) -> str:
    """Search the PDF documents the user uploaded in this session.

    Use this for ANY question about the uploaded documents/papers/files.
    Returns the 4 most relevant passages, each labelled with file name and
    page number so you can cite them.
    """
    # `config` is injected by LangGraph (hidden from the LLM). The thread_id is
    # the session id, which keys both memory and this session's FAISS index.
    session_id = config.get("configurable", {}).get("thread_id", "")
    try:
        if get_session_index(session_id).is_empty:
            return "No PDFs have been uploaded in this session yet. Ask the user to upload one."
        results = retrieve(session_id, query)
        if not results:
            return "No relevant passages found in the uploaded PDFs."
        return format_results(results)
    except Exception as exc:  # noqa: BLE001 - tool must never raise
        logger.exception("pdf_qa failed")
        return f"pdf_qa error: {exc}"


# --------------------------------------------------------------------------- #
# 2. web_search (Tavily, falling back to DuckDuckGo)
# --------------------------------------------------------------------------- #
def _tavily_search(query: str) -> str:
    from langchain_tavily import TavilySearch

    search = TavilySearch(max_results=3, tavily_api_key=get_settings().tavily_api_key)
    response: dict[str, Any] = search.invoke({"query": query})
    results = response.get("results", []) if isinstance(response, dict) else []
    if not results:
        return ""
    return "\n\n".join(
        f"- {r.get('title', '')} ({r.get('url', '')}): {r.get('content', '')[:500]}"
        for r in results
    )


def _duckduckgo_search(query: str) -> str:
    from ddgs import DDGS

    results = DDGS().text(query, max_results=3)
    return "\n\n".join(
        f"- {r.get('title', '')} ({r.get('href', '')}): {r.get('body', '')[:500]}"
        for r in results
    )


@tool
def web_search(query: str) -> str:
    """Search the web for current events or general knowledge that is NOT in
    the uploaded PDFs (news, prices, weather, people, recent facts)."""
    backends: list[tuple[str, Callable[[str], str]]] = []
    if get_settings().tavily_api_key:
        backends.append(("tavily", _tavily_search))
    backends.append(("duckduckgo", _duckduckgo_search))

    errors: list[str] = []
    for name, backend in backends:
        try:
            text = backend(query)
            if text:
                return text
            errors.append(f"{name}: no results")
        except Exception as exc:  # noqa: BLE001 - fall through to next backend
            logger.warning("web_search backend %s failed: %s", name, exc)
            errors.append(f"{name}: {exc}")
    return "web_search failed: " + "; ".join(errors)


# --------------------------------------------------------------------------- #
# 3. calculator (safe AST evaluator — never calls eval())
# --------------------------------------------------------------------------- #
_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCS: dict[str, Callable[..., float]] = {
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "log2": math.log2,
    "exp": math.exp, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "abs": abs, "round": round, "floor": math.floor, "ceil": math.ceil,
    "factorial": math.factorial, "min": min, "max": max,
}
_CONSTS: dict[str, float] = {"pi": math.pi, "e": math.e}

_MAX_EXPONENT = 1000  # blocks DoS like 9**9**9
_MAX_FACTORIAL = 1000
_MAX_EXPR_LEN = 200


def safe_eval(expression: str) -> float | int:
    """Evaluate an arithmetic expression by walking its AST.

    Only numbers, + - * / // % **, parentheses, and a whitelist of math
    functions/constants are allowed. Names, attributes, calls to anything
    else, subscripts, lambdas, etc. raise ValueError — so input like
    `__import__('os').system('...')` can never execute.
    """
    if len(expression) > _MAX_EXPR_LEN:
        raise ValueError("expression too long")
    expr = expression.replace("^", "**").replace("×", "*").replace("÷", "/")
    tree = ast.parse(expr, mode="eval")

    def _eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left, right = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
                raise ValueError("exponent too large")
            return _BIN_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.Name) and node.id in _CONSTS:
            return _CONSTS[node.id]
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS
            and not node.keywords
        ):
            args = [_eval(a) for a in node.args]
            if node.func.id == "factorial" and args and args[0] > _MAX_FACTORIAL:
                raise ValueError("factorial argument too large")
            return _FUNCS[node.func.id](*args)
        raise ValueError(f"unsupported expression element: {type(node).__name__}")

    return _eval(tree)


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression, e.g. '(12.5 * 4) / 3', 'sqrt(144)', '2**10',
    '15/100*240'. Use this for any arithmetic instead of doing it in your head."""
    try:
        result = safe_eval(expression)
        if isinstance(result, float):
            result = round(result, 6)
            if result.is_integer():
                result = int(result)
        return str(result)
    except ZeroDivisionError:
        return "calculator error: division by zero"
    except Exception as exc:  # noqa: BLE001 - SyntaxError, ValueError, OverflowError...
        return f"calculator error: {exc}"


ALL_TOOLS = [pdf_qa, web_search, calculator]
