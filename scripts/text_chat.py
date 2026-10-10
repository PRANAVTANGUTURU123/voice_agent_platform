"""Phase 1: text-only agent in the terminal.

Usage:
    python -m scripts.text_chat                 # interactive, loads data/sample_docs/*.pdf
    python -m scripts.text_chat --demo          # run 5 scripted questions (incl. memory follow-up)
    python -m scripts.text_chat --pdf my.pdf    # use your own PDF(s)
"""

from __future__ import annotations

import argparse
import asyncio
import time
import uuid
from pathlib import Path

from app.agent.graph import ask
from app.config import get_settings
from app.rag.ingest import ingest_pdf

DEMO_QUESTIONS = [
    "How many attention heads does the base Transformer model use?",
    "And what is the dimension of each of those heads?",  # needs memory of "heads"
    "What are the four functions of the NIST AI Risk Management Framework core?",
    "What is 512 divided by 8, times 6?",
    "Who is the current CEO of Nvidia?",
]


def load_pdfs(session_id: str, pdfs: list[Path]) -> None:
    for pdf in pdfs:
        t0 = time.perf_counter()
        n = ingest_pdf(session_id, pdf)
        print(f"Indexed {pdf.name}: {n} chunks in {time.perf_counter() - t0:.1f}s")


async def run_turn(session_id: str, question: str) -> None:
    t0 = time.perf_counter()
    try:
        result = await ask(session_id, question)
    except Exception as exc:  # noqa: BLE001 - keep the REPL alive on API errors
        print(f"Agent error: {exc}\n")
        return
    tools = ", ".join(result.tools_used) or "none"
    print(f"Agent [{tools}; {time.perf_counter() - t0:.1f}s]: {result.answer}\n")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--pdf", type=Path, nargs="*", help="PDF files to index")
    parser.add_argument("--demo", action="store_true", help="run the scripted demo questions")
    args = parser.parse_args()

    session_id = f"cli-{uuid.uuid4().hex[:8]}"
    pdfs = args.pdf or sorted(get_settings().sample_docs_dir.glob("*.pdf"))
    load_pdfs(session_id, pdfs)

    if args.demo:
        for q in DEMO_QUESTIONS:
            print(f"You: {q}")
            await run_turn(session_id, q)
        return

    print("Type a question (empty line or Ctrl+C to quit).")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        await run_turn(session_id, q)


if __name__ == "__main__":
    asyncio.run(main())
