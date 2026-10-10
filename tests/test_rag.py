"""RAG ingestion + retrieval on the sample PDFs, and the pdf_qa tool wrapper."""

from pathlib import Path

import pytest

from app.agent.tools import pdf_qa
from app.config import get_settings
from app.rag.ingest import clear_session, ingest_pdf
from app.rag.retriever import retrieve

SAMPLE = get_settings().sample_docs_dir
ATTENTION = SAMPLE / "attention_is_all_you_need.pdf"
NIST = SAMPLE / "nist_ai_rmf_1.0.pdf"


@pytest.fixture(scope="module")
def session_id() -> str:
    sid = "test-rag"
    ingest_pdf(sid, ATTENTION)
    ingest_pdf(sid, NIST)
    yield sid
    clear_session(sid)


def test_chunks_have_citations(session_id: str) -> None:
    results = retrieve(session_id, "BLEU score on WMT 2014 English-to-German")
    assert len(results) == 4  # top-k
    for r in results:
        assert r.chunk.file_name.endswith(".pdf")
        assert r.chunk.page >= 1


@pytest.mark.parametrize(
    ("query", "expected_file", "expected_text"),
    [
        ("scaled dot-product attention divide by square root of dk", "attention_is_all_you_need.pdf", "dk"),
        ("number of attention heads h in the base model", "attention_is_all_you_need.pdf", "h = 8"),
        ("four core functions govern map measure manage", "nist_ai_rmf_1.0.pdf", "MANAGE"),
        ("characteristics of trustworthy AI systems", "nist_ai_rmf_1.0.pdf", "trustworthy"),
    ],
)
def test_retrieval_finds_right_document(
    session_id: str, query: str, expected_file: str, expected_text: str
) -> None:
    results = retrieve(session_id, query)
    assert results[0].chunk.file_name == expected_file
    joined = " ".join(r.chunk.text for r in results)
    assert expected_text.lower() in joined.lower()


def test_sessions_are_isolated(session_id: str) -> None:
    assert retrieve("some-other-session", "attention heads") == []


def test_pdf_qa_without_upload() -> None:
    out = pdf_qa.invoke({"query": "anything"}, config={"configurable": {"thread_id": "empty"}})
    assert "No PDFs have been uploaded" in out


def test_pdf_qa_formats_citations(session_id: str) -> None:
    out = pdf_qa.invoke(
        {"query": "Adam optimizer warmup steps"},
        config={"configurable": {"thread_id": session_id}},
    )
    assert "file: attention_is_all_you_need.pdf, page" in out
    assert out.count("[") >= 4


def test_ingest_rejects_textless_pdf(tmp_path: Path) -> None:
    from pypdf import PdfWriter

    blank = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with blank.open("wb") as f:
        writer.write(f)
    with pytest.raises(ValueError, match="No extractable text"):
        ingest_pdf("blank-session", blank)
