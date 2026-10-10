"""Evaluate answer accuracy and tool-selection accuracy.

For every question in eval_questions.json:
  1. Run the agent in a fresh conversation (sample PDFs indexed).
  2. Tool accuracy:
       - "tool used"   : the expected tool was called at some point in the turn
       - "first choice": the FIRST tool called was the expected one
  3. Answer accuracy:
       - "judge"   : an LLM judge (a different model from the agent) decides
                     whether the answer matches the expected answer
       - "keywords": every keyword group appears in the answer (each group is a
                     list of acceptable alternatives) — deterministic cross-check

Usage:
    python -m eval.run_eval                 # all questions
    python -m eval.run_eval --limit 5       # quick smoke run
    python -m eval.run_eval --delay 10      # seconds between questions (rate limits)

Results: eval/results/eval_results.json and a summary printed to stdout.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.graph import ask
from app.config import get_settings
from app.rag.ingest import ingest_pdf, share_index

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"

JUDGE_PROMPT = """You grade a voice assistant's answer against a reference answer.
Reply with exactly one word: CORRECT if the assistant's answer contains the key facts of
the reference answer (wording and extra detail don't matter), otherwise INCORRECT."""


def build_judge():
    """The judge uses a different model than the agent (separate rate-limit bucket,
    and avoids a model grading its own output)."""
    from langchain_groq import ChatGroq

    s = get_settings()
    return ChatGroq(model="openai/gpt-oss-20b", temperature=0, api_key=s.groq_api_key,
                    reasoning_effort="low", max_retries=4)


async def judge_answer(judge: Any, question: str, expected: str, answer: str) -> bool:
    msg = (f"Question: {question}\nReference answer: {expected}\nAssistant answer: {answer}\n"
           "Verdict:")
    response = await judge.ainvoke([SystemMessage(JUDGE_PROMPT), HumanMessage(msg)])
    return "INCORRECT" not in str(response.content).upper() and "CORRECT" in str(response.content).upper()


_DASHES = str.maketrans({c: "-" for c in "‐‑‒–—−"})


def keywords_match(answer: str, groups: list[list[str]]) -> bool:
    # LLMs often emit Unicode hyphens (e.g. U+2011 in "thirty‑six"); normalise them.
    norm = re.sub(r"[,\s  ]+", " ", answer.lower().translate(_DASHES))
    return all(any(alt.lower() in norm for alt in group) for group in groups)


def rescore_keywords(questions_path: Path) -> None:
    """Recompute the keyword metric from saved answers; judge/tool results are untouched."""
    out = RESULTS_DIR / "eval_results.json"
    data = json.loads(out.read_text(encoding="utf-8"))
    keywords = {q["id"]: q["keywords"] for q in json.loads(questions_path.read_text(encoding="utf-8"))}
    for row in data["rows"]:
        row["keywords_ok"] = keywords_match(row["answer"], keywords[row["id"]])
    n = len(data["rows"])
    data["summary"]["answer_accuracy_keywords_pct"] = round(100 * sum(r["keywords_ok"] for r in data["rows"]) / n, 1)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(data["summary"], indent=2))


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--delay", type=float, default=6.0, help="pause between questions (Groq TPM limits)")
    parser.add_argument("--questions", type=Path, default=EVAL_DIR / "eval_questions.json")
    parser.add_argument("--rescore-keywords", action="store_true",
                        help="re-grade keywords on the saved answers (no agent or judge calls)")
    args = parser.parse_args()

    if args.rescore_keywords:
        rescore_keywords(args.questions)
        return

    questions: list[dict[str, Any]] = json.loads(args.questions.read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]

    base = "eval-base"
    for pdf in sorted(get_settings().sample_docs_dir.glob("*.pdf")):
        ingest_pdf(base, pdf)

    judge = build_judge()
    rows: list[dict[str, Any]] = []
    for i, q in enumerate(questions, start=1):
        sid = f"eval-{uuid.uuid4().hex[:8]}"
        share_index(base, sid)
        t0 = time.perf_counter()
        try:
            result = await ask(sid, q["question"])
            answer, tools, error = result.answer, result.tools_used, None
        except Exception as exc:  # noqa: BLE001 - record and continue
            answer, tools, error = "", [], str(exc)
        latency = time.perf_counter() - t0

        judged = await judge_answer(judge, q["question"], q["expected_answer"], answer) if answer else False
        row = {
            "id": q["id"],
            "question": q["question"],
            "expected_tool": q["expected_tool"],
            "tools_used": tools,
            "tool_used_ok": q["expected_tool"] in tools,
            "first_choice_ok": bool(tools) and tools[0] == q["expected_tool"],
            "expected_answer": q["expected_answer"],
            "answer": answer,
            "judge_ok": judged,
            "keywords_ok": keywords_match(answer, q["keywords"]),
            "latency_s": round(latency, 2),
            "error": error,
        }
        rows.append(row)
        mark = lambda ok: "ok" if ok else "XX"  # noqa: E731 - ASCII: Windows consoles may be cp1252
        print(f"[{i:2}/{len(questions)}] tool {mark(row['tool_used_ok'])} {','.join(tools) or '-':<22} "
              f"judge {mark(judged)} kw {mark(row['keywords_ok'])}  {q['question'][:60]}")
        if i < len(questions):
            await asyncio.sleep(args.delay)

    n = len(rows)
    pct = lambda key: round(100 * sum(r[key] for r in rows) / n, 1)  # noqa: E731
    summary = {
        "model": get_settings().llm_model,
        "n_questions": n,
        "answer_accuracy_judge_pct": pct("judge_ok"),
        "answer_accuracy_keywords_pct": pct("keywords_ok"),
        "tool_accuracy_pct": pct("tool_used_ok"),
        "tool_first_choice_accuracy_pct": pct("first_choice_ok"),
        "by_tool": {},
    }
    for tool_name in sorted({r["expected_tool"] for r in rows}):
        subset = [r for r in rows if r["expected_tool"] == tool_name]
        summary["by_tool"][tool_name] = {
            "n": len(subset),
            "answer_judge_pct": round(100 * sum(r["judge_ok"] for r in subset) / len(subset), 1),
            "tool_pct": round(100 * sum(r["tool_used_ok"] for r in subset) / len(subset), 1),
        }

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "eval_results.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + json.dumps(summary, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    asyncio.run(main())
