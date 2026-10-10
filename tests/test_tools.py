"""web_search fallback behaviour (network calls are mocked)."""

import pytest

from app.agent import tools


def test_web_search_falls_back_to_duckduckgo(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_tavily(query: str) -> str:
        raise RuntimeError("tavily down")

    monkeypatch.setattr(tools, "_tavily_search", broken_tavily)
    monkeypatch.setattr(tools, "_duckduckgo_search", lambda q: f"- ddg result for {q}")
    out = tools.web_search.invoke({"query": "weather"})
    assert out == "- ddg result for weather"


def test_web_search_reports_total_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(query: str) -> str:
        raise RuntimeError("offline")

    monkeypatch.setattr(tools, "_tavily_search", boom)
    monkeypatch.setattr(tools, "_duckduckgo_search", boom)
    out = tools.web_search.invoke({"query": "x"})
    assert out.startswith("web_search failed")  # never raises


def test_tool_schemas_hide_injected_config() -> None:
    # The LLM should only see `query`; the RunnableConfig is injected by LangGraph.
    assert set(tools.pdf_qa.args) == {"query"}
    assert set(tools.calculator.args) == {"expression"}
    assert [t.name for t in tools.ALL_TOOLS] == ["pdf_qa", "web_search", "calculator"]
