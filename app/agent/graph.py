"""LangGraph ReAct agent with per-session memory.

Flow for one user turn:
    user text -> [agent node: LLM decides] -> (tool call? -> [tools node] -> agent)* -> answer

`create_react_agent` builds that loop for us. The `MemorySaver` checkpointer
stores the message history keyed by `thread_id`; we use the session id as the
thread id, so each browser session has its own conversation memory (and its
own PDF index — see tools.pdf_qa).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, AsyncIterator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from app.agent.tools import ALL_TOOLS
from app.config import get_settings
from app.rag.ingest import get_session_index

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful voice assistant. Your replies are spoken aloud, so:
- Answer in 2-3 short, natural spoken sentences. No markdown, bullet points, tables, or URLs.
- Use plain words instead of math symbols or Greek letters (say "beta two of 0.98", not "β₂ = 0.98"),
  but keep numbers as digits (say "391", not "three hundred ninety-one").

Tools:
- pdf_qa: search the user's uploaded PDFs. Prefer it whenever the question could be about the uploaded documents.
  When you use it, cite the exact file name and page shown in the result, e.g. "(report.pdf, page 3)".
- web_search: for current events or general facts that are not in the PDFs.
- calculator: for any arithmetic. Never do math in your head.

If the tools don't give you the answer, say "I don't know" rather than guessing.
{doc_status}"""

# Keep roughly the last N messages of history in the prompt. Memory still
# stores everything; this only bounds prompt size (and therefore latency/cost).
MAX_HISTORY_MESSAGES = 24


def build_llm(streaming: bool = True) -> BaseChatModel:
    """Create the chat model from config. Swap providers via LLM_PROVIDER/LLM_MODEL."""
    s = get_settings()
    if s.llm_provider == "groq":
        if not s.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and add it.")
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=s.llm_model,
            temperature=s.llm_temperature,
            api_key=s.groq_api_key,
            streaming=streaming,
            reasoning_effort=s.llm_reasoning_effort or None,
            # The Groq client retries 429 rate-limit errors with backoff.
            max_retries=4,
        )
    # Any other provider supported by LangChain, e.g. LLM_PROVIDER=openai,
    # LLM_MODEL=gpt-4o-mini (requires `pip install langchain langchain-openai`).
    from langchain.chat_models import init_chat_model  # type: ignore[import-not-found]

    return init_chat_model(s.llm_model, model_provider=s.llm_provider, temperature=s.llm_temperature)


def _drop_dangling_tool_calls(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Remove AI tool-call messages whose tool results never arrived.

    This happens when a turn is cancelled by barge-in while a tool is running:
    the checkpoint holds the tool call but not its result, and LLM APIs reject
    a history like that. Dropping the orphaned call keeps memory usable.
    """
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    return [
        m
        for m in messages
        if not (isinstance(m, AIMessage) and m.tool_calls and any(tc["id"] not in answered for tc in m.tool_calls))
    ]


def _prompt(state: dict[str, Any], config: RunnableConfig) -> list[AnyMessage]:
    """Build the messages sent to the LLM on every agent step.

    Adds the system prompt (with which PDFs are loaded in this session) and
    trims old history. `start_on="human"` ensures we never cut between a tool
    call and its result, which would make the LLM API reject the request.
    """
    session_id = config.get("configurable", {}).get("thread_id", "")
    files = get_session_index(session_id).files
    doc_status = (
        f"Uploaded PDFs in this session: {', '.join(files)}."
        if files
        else "No PDFs are uploaded yet; if asked about a document, tell the user to upload one."
    )
    history = trim_messages(
        _drop_dangling_tool_calls(state["messages"]),
        strategy="last",
        token_counter=len,  # count messages, not tokens: cheap and predictable
        max_tokens=MAX_HISTORY_MESSAGES,
        start_on="human",
        include_system=False,
    )
    return [SystemMessage(SYSTEM_PROMPT.format(doc_status=doc_status)), *history]


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """One shared LLM client, so its HTTPS connection pool is reused across turns."""
    return build_llm()


@lru_cache(maxsize=1)
def get_agent():  # -> CompiledStateGraph
    """Build the agent graph once per process; memory is shared across calls."""
    return create_react_agent(
        model=get_llm(),
        tools=ALL_TOOLS,
        prompt=_prompt,
        checkpointer=MemorySaver(),
    )


async def warm_up() -> None:
    """Build the graph and open the HTTPS connection to the LLM at startup.

    Measured cold cost on the first user turn: ~0.9 s graph build + ~0.4 s
    TLS handshake. A tiny request now (~50 tokens) moves that off the user's turn.
    """
    get_agent()
    try:
        await get_llm().ainvoke("Reply with OK.", max_tokens=16)
    except Exception as exc:  # noqa: BLE001 - warm-up is best effort (e.g. no network yet)
        logger.warning("LLM warm-up failed: %s", exc)


def _config(session_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": session_id}, "recursion_limit": 12}


@dataclass
class AgentResult:
    answer: str
    tools_used: list[str] = field(default_factory=list)


async def ask(session_id: str, text: str) -> AgentResult:
    """Run one full turn (non-streaming). Used by the CLI and evaluation."""
    state = await get_agent().ainvoke({"messages": [HumanMessage(text)]}, _config(session_id))
    # Only look at messages produced in this turn (after the last human message).
    messages: list[AnyMessage] = state["messages"]
    last_human = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    turn = messages[last_human + 1 :]
    tools_used = [tc["name"] for m in turn if isinstance(m, AIMessage) for tc in m.tool_calls]
    answer = next((m.content for m in reversed(turn) if isinstance(m, AIMessage) and m.content), "")
    return AgentResult(answer=str(answer), tools_used=tools_used)


async def stream_answer(session_id: str, text: str) -> AsyncIterator[dict[str, str]]:
    """Stream one turn as events: {"type": "token", "text": ...} for answer
    tokens and {"type": "tool", "name": ...} when a tool is called.

    `stream_mode="messages"` yields LLM tokens as they are generated, so the
    caller can start TTS on the first complete sentence.
    """
    async for chunk, metadata in get_agent().astream(
        {"messages": [HumanMessage(text)]}, _config(session_id), stream_mode="messages"
    ):
        if metadata.get("langgraph_node") != "agent":
            continue  # skip tool outputs; we only stream what the LLM says
        if isinstance(chunk, AIMessageChunk):
            for tc in chunk.tool_call_chunks or []:
                if tc.get("name"):
                    yield {"type": "tool", "name": tc["name"]}
            if chunk.content and isinstance(chunk.content, str):
                yield {"type": "token", "text": chunk.content}
