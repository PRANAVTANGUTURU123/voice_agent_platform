"""Similarity search over a session's FAISS index."""

from __future__ import annotations

from dataclasses import dataclass

from app.config import get_settings
from app.rag.ingest import Chunk, embed, get_session_index


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float  # cosine similarity in [-1, 1]; higher is more relevant


def retrieve(session_id: str, query: str, k: int | None = None) -> list[RetrievedChunk]:
    """Return the top-k chunks for `query` in this session (empty if no PDFs)."""
    session = get_session_index(session_id)
    if session.is_empty or not query.strip():
        return []
    k = min(k or get_settings().top_k, session.index.ntotal)  # type: ignore[union-attr]
    query_vec = embed([query])
    with session.lock:
        scores, ids = session.index.search(query_vec, k)  # type: ignore[union-attr]
        return [
            RetrievedChunk(chunk=session.chunks[i], score=float(s))
            for s, i in zip(scores[0], ids[0])
            if i != -1
        ]


def format_results(results: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as text the LLM can read and cite."""
    blocks = []
    for n, r in enumerate(results, start=1):
        c = r.chunk
        blocks.append(f"[{n}] (file: {c.file_name}, page {c.page})\n{c.text}")
    return "\n\n".join(blocks)
