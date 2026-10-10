"""PDF ingestion: PDF -> page text -> chunks -> embeddings -> FAISS index.

Each session gets its own in-memory `SessionIndex`, so documents uploaded by
one user are never visible to another. Uploading more PDFs to the same
session appends to that session's index.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """One retrievable piece of text plus where it came from (for citations)."""

    text: str
    file_name: str
    page: int  # 1-based page number, as a human would cite it


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Load the embedding model once per process (~90 MB, CPU).

    Try the local Hugging Face cache first: otherwise every startup makes
    ~20 HTTP checks (~5 s) and fails when offline. Download only if missing.
    """
    name = get_settings().embedding_model
    try:
        return SentenceTransformer(name, device="cpu", local_files_only=True)
    except OSError:
        logger.info("Embedding model not cached; downloading %s", name)
        return SentenceTransformer(name, device="cpu")


def embed(texts: list[str]) -> np.ndarray:
    """Embed texts into L2-normalised float32 vectors.

    Normalising lets us use inner product (IndexFlatIP) as cosine similarity.
    """
    vectors = get_embedder().encode(
        texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
    )
    return np.asarray(vectors, dtype="float32")


def load_pdf_chunks(pdf_path: Path, file_name: str | None = None) -> list[Chunk]:
    """Read a PDF page by page and split each page into overlapping chunks.

    Splitting per page (rather than the whole document) keeps an exact page
    number on every chunk so the agent can cite it.
    """
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    name = file_name or pdf_path.name
    reader = PdfReader(str(pdf_path))
    chunks: list[Chunk] = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue  # scanned/image-only page: nothing to index
        for piece in splitter.split_text(text):
            chunks.append(Chunk(text=piece, file_name=name, page=page_no))
    return chunks


@dataclass
class SessionIndex:
    """A FAISS index plus the chunk metadata aligned with its vector ids."""

    index: faiss.IndexFlatIP | None = None
    chunks: list[Chunk] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add_chunks(self, new_chunks: list[Chunk]) -> None:
        if not new_chunks:
            return
        vectors = embed([c.text for c in new_chunks])
        with self.lock:
            if self.index is None:
                # Exact (brute-force) search: perfect recall, and fast enough
                # for the few thousand chunks a handful of PDFs produce.
                self.index = faiss.IndexFlatIP(vectors.shape[1])
            self.index.add(vectors)
            self.chunks.extend(new_chunks)

    @property
    def is_empty(self) -> bool:
        return self.index is None or self.index.ntotal == 0


# session_id -> SessionIndex. In-memory: indexes vanish when the server restarts.
_SESSIONS: dict[str, SessionIndex] = {}
_SESSIONS_LOCK = threading.Lock()


def get_session_index(session_id: str) -> SessionIndex:
    """Return (creating if needed) the index for a session."""
    with _SESSIONS_LOCK:
        return _SESSIONS.setdefault(session_id, SessionIndex())


def ingest_pdf(session_id: str, pdf_path: Path, file_name: str | None = None) -> int:
    """Add one PDF to a session's index. Returns the number of chunks added.

    Raises ValueError if the PDF has no extractable text.
    """
    chunks = load_pdf_chunks(pdf_path, file_name)
    if not chunks:
        raise ValueError(
            f"No extractable text in {file_name or pdf_path.name} (is it a scanned PDF?)"
        )
    session = get_session_index(session_id)
    session.add_chunks(chunks)
    session.files.append(file_name or pdf_path.name)
    logger.info("Session %s: indexed %d chunks from %s", session_id, len(chunks), pdf_path.name)
    return len(chunks)


def share_index(source_session_id: str, target_session_id: str) -> None:
    """Make `target` use the same index object as `source` (no re-embedding).

    Used by evaluation: each question gets a fresh conversation thread but
    reuses one index built from the sample PDFs.
    """
    with _SESSIONS_LOCK:
        _SESSIONS[target_session_id] = _SESSIONS[source_session_id]


def clear_session(session_id: str) -> None:
    """Drop a session's index (e.g. on explicit reset)."""
    with _SESSIONS_LOCK:
        _SESSIONS.pop(session_id, None)
