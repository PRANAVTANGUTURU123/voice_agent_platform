"""Per-stage latency benchmark for the full voice pipeline.

For each benchmark question we synthesise a spoken WAV (edge-tts, a different
voice from the agent's), stream it through Silero VAD in 100 ms chunks exactly
like the browser does, and run the same `run_turn` the WebSocket server uses.

Timeline per turn, all measured from the end-of-turn decision (VAD has seen
500 ms of silence):

    end of speech --> STT done --> first LLM token --> first TTS audio ready

Note: the user-perceived delay also includes the 500 ms VAD silence window
(the time needed to *decide* the user stopped talking). That is a fixed,
configurable constant and is reported separately.

Usage:
    python -m eval.latency_bench               # default: 3 rounds x 6 questions
    python -m eval.latency_bench --rounds 1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
import uuid
from pathlib import Path
from typing import Any

import miniaudio
import numpy as np

from app.agent.graph import warm_up
from app.audio.stt import get_whisper, transcribe
from app.audio.tts import synthesize
from app.audio.vad import SpeechEnd, StreamingVAD
from app.config import get_settings
from app.pipeline import run_turn
from app.rag.ingest import ingest_pdf, share_index

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
SPEAKER_VOICE = "en-US-GuyNeural"  # the "user" voice, distinct from the agent's

BENCH_QUESTIONS = [
    "How many layers does the Transformer encoder have?",           # pdf_qa
    "What optimizer was used to train the Transformer?",            # pdf_qa
    "What are the four functions of the NIST AI risk management framework?",  # pdf_qa
    "What is seventeen times twenty three?",                         # calculator
    "What is the square root of one thousand and twenty four?",      # calculator
    "Who won the most recent FIFA World Cup?",                       # web_search
]


async def speech_audio(text: str) -> np.ndarray:
    """TTS a question and decode it to 16 kHz mono float32, plus 1 s of silence."""
    mp3 = await synthesize(text, voice=SPEAKER_VOICE)
    decoded = miniaudio.decode(mp3, output_format=miniaudio.SampleFormat.FLOAT32,
                               nchannels=1, sample_rate=get_settings().sample_rate)
    audio = np.frombuffer(decoded.samples, dtype=np.float32)
    return np.concatenate([np.zeros(4000, np.float32), audio, np.zeros(16000, np.float32)])


def vad_segment(audio: np.ndarray) -> np.ndarray | None:
    """Stream audio through VAD in 100 ms chunks; return the detected utterance."""
    vad = StreamingVAD()
    for i in range(0, len(audio), 1600):
        for event in vad.process(audio[i : i + 1600]):
            if isinstance(event, SpeechEnd):
                return event.audio
    return None


def pct(values: list[float], q: float) -> float:
    """Percentile with linear interpolation (q in [0, 100])."""
    if len(values) == 1:
        return values[0]
    return float(np.percentile(np.asarray(values), q))


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--delay", type=float, default=6.0, help="pause between turns (Groq TPM limits)")
    args = parser.parse_args()

    base = "bench-base"
    for pdf in sorted(get_settings().sample_docs_dir.glob("*.pdf")):
        ingest_pdf(base, pdf)
    # Same warm-up as the server's startup, so we measure steady-state turns.
    get_whisper()
    transcribe(np.zeros(8000, dtype=np.float32))
    await warm_up()

    print("Synthesising spoken questions...")
    clips = [(q, await speech_audio(q)) for q in BENCH_QUESTIONS]

    async def ignore_json(_: dict[str, Any]) -> None: ...
    async def ignore_audio(_seq: int, _mp3: bytes) -> None: ...

    runs: list[dict[str, Any]] = []
    total = args.rounds * len(clips)
    for r in range(args.rounds):
        for question, audio in clips:
            utterance = vad_segment(audio)
            if utterance is None:
                print(f"VAD found no speech in: {question}")
                continue
            sid = f"bench-{uuid.uuid4().hex[:8]}"
            share_index(base, sid)
            t_end = time.perf_counter()  # == the moment VAD emitted SpeechEnd
            m = await run_turn(sid, ignore_json, ignore_audio, audio=utterance, t_start=t_end)
            row = {"question": question, **m.as_dict()}
            runs.append(row)
            print(f"[{len(runs):2}/{total}] STT {m.stt_ms:>6} ms | first token {m.first_token_ms!s:>7} ms | "
                  f"first audio {m.first_audio_ms!s:>7} ms | {','.join(m.tools) or '-'}")
            await asyncio.sleep(args.delay)

    ok = [r for r in runs if r["first_audio_ms"] is not None and r["first_token_ms"] is not None]
    stages = {
        "stt_done_ms": [r["stt_ms"] for r in ok],
        "first_llm_token_ms": [r["first_token_ms"] for r in ok],
        "first_tts_audio_ms": [r["first_audio_ms"] for r in ok],
        # per-stage deltas
        "llm_ttft_after_stt_ms": [r["first_token_ms"] - r["stt_ms"] for r in ok],
        "tts_first_chunk_after_token_ms": [r["first_audio_ms"] - r["first_token_ms"] for r in ok],
    }
    summary = {
        "n_turns": len(ok),
        "model": get_settings().llm_model,
        "whisper": get_settings().whisper_model,
        "vad_silence_window_ms": get_settings().vad_min_silence_ms,
        "stages": {k: {"p50": round(pct(v, 50)), "p95": round(pct(v, 95))} for k, v in stages.items() if v},
    }
    if ok:
        summary["mean_first_tts_audio_ms"] = round(statistics.mean(stages["first_tts_audio_ms"]))

    print("\nCumulative from end of speech (ms):")
    print(f"{'stage':<34}{'p50':>8}{'p95':>8}")
    for k, v in summary["stages"].items():
        print(f"{k:<34}{v['p50']:>8}{v['p95']:>8}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "latency_results.json"
    out.write_text(json.dumps({"summary": summary, "runs": runs}, indent=2), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    asyncio.run(main())
