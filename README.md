# Real-Time Voice Agent

Upload any PDF in the browser and **talk** to an agent about it. The agent listens (VAD + Whisper), reasons with tools (PDF RAG, web search, calculator), remembers the conversation, and **speaks the answer back while it is still being generated**. You can interrupt it mid-sentence (barge-in).

## Architecture

```
 BROWSER (frontend/index.html + app.js)                         SERVER (FastAPI, app/main.py)
 ┌───────────────────────────────────────┐                     ┌─────────────────────────────────────────────────────┐
 │ Mic ─► AudioWorklet (pcm-worklet.js)  │  16 kHz int16 PCM   │ /ws/{session_id}                                    │
 │        48k→16k downsample, 100 ms     ├────── binary ──────►│  StreamingVAD (Silero, 32 ms frames)                │
 │                                       │                     │   ├─ SpeechStart ─► cancel current turn (barge-in)  │
 │ Status / transcript / reply text  ◄───┼──── JSON events ────┤   └─ SpeechEnd (500 ms silence) ─► run_turn()       │
 │                                       │                     │                                                     │
 │ MediaSource <audio> player        ◄───┼── binary: turn id ──┤  app/pipeline.py: run_turn()                        │
 │  plays from the FIRST MP3 chunk       │     + MP3 chunk     │   1. faster-whisper STT (base.en, int8, CPU)        │
 │  stop() instantly on speech_start     │                     │   2. LangGraph ReAct agent ── stream tokens ──┐     │
 │                                       │                     │        │ tools:                               │     │
 │ PDF upload ───────────────────────────┼── POST /upload/{id}►│        ├─ pdf_qa ─► FAISS (per session)       │     │
 └───────────────────────────────────────┘                     │        ├─ web_search ─► Tavily → DuckDuckGo   │     │
                                                               │        └─ calculator ─► safe AST evaluator    │     │
                                                               │        memory: MemorySaver[thread_id=session] │     │
                                                               │   3. SentenceBuffer ◄────────────────────────-┘     │
                                                               │        sentence queue ─► edge-tts (stream chunks)   │
                                                               └─────────────────────────────────────────────────────┘
 Upload path:  PDF ─► pypdf (per page) ─► RecursiveCharacterTextSplitter (800/120) ─► all-MiniLM-L6-v2 ─► FAISS IndexFlatIP
```

**Why it feels real-time.** Three stages overlap instead of running one after another:
1. The LLM streams tokens. `SentenceBuffer` releases each complete sentence immediately (the first chunk may even end at a comma).
2. A TTS worker synthesises sentence *n* while the LLM is still writing sentence *n+1*.
3. Each MP3 chunk is forwarded to the browser as soon as edge-tts produces it, and the browser starts playback on the **first chunk** via MediaSource.

**Barge-in.** Server-side VAD fires `SpeechStart` → the server cancels the turn's asyncio task (LLM stream + TTS) and sends `speech_start`, so the browser stops playback instantly. Every audio frame carries a 4-byte turn id, so late frames from a cancelled turn are dropped. If a turn is cancelled mid-tool-call, the orphaned tool call is removed from memory before the next LLM request (otherwise the API rejects the history).

## Project structure

```
app/
  main.py            FastAPI: WebSocket /ws/{session_id}, POST /upload/{session_id}, static frontend
  pipeline.py        One turn: STT -> streamed agent -> sentence split -> streamed TTS (+ latency metrics)
  config.py          All settings from .env (LLM is swappable: LLM_PROVIDER / LLM_MODEL)
  audio/vad.py       Silero VAD streaming wrapper (speech start/end events, pre-roll)
  audio/stt.py       faster-whisper wrapper (+ hallucination filter for silence)
  audio/tts.py       edge-tts streaming + SentenceBuffer + symbol-to-speech cleanup
  agent/graph.py     LangGraph create_react_agent + MemorySaver + prompt/history trimming
  agent/tools.py     pdf_qa, web_search, calculator
  rag/ingest.py      PDF -> chunks -> embeddings -> FAISS index per session
  rag/retriever.py   top-k search + citation formatting
frontend/            index.html, app.js, pcm-worklet.js (AudioWorklet downsampler)
scripts/
  text_chat.py       Phase 1: terminal text agent (--demo runs 5 scripted questions)
  voice_loop.py      Phase 2: local mic -> VAD -> STT -> agent -> TTS -> speakers (--wav for files)
  ws_smoke_test.py   End-to-end WebSocket test acting like the browser (--barge-in)
eval/
  eval_questions.json   30 questions (18 PDF, 6 calculator, 6 web) with expected answer + tool
  run_eval.py           answer accuracy (LLM judge + keywords) and tool-selection accuracy
  latency_bench.py      per-stage latency p50/p95 through the real pipeline
  results/              JSON output of the runs reported below
tests/               pytest: calculator safety, tools, RAG retrieval, sentence splitting
data/sample_docs/    attention_is_all_you_need.pdf (arXiv 1706.03762), nist_ai_rmf_1.0.pdf (NIST AI 100-1, public domain)
```

## Setup (Windows, Python 3.11+)

```bat
git clone <this repo> && cd "voice agent plaform"
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
notepad .env          &:: add GROQ_API_KEY (required) and TAVILY_API_KEY (optional)
```

- `requirements.txt` installs **CPU-only PyTorch** (~200 MB instead of ~2 GB).
- First run downloads Whisper `base.en` (~145 MB) and `all-MiniLM-L6-v2` (~90 MB) from Hugging Face; after that they load from the local cache (no network checks).
- **Slow `pip install`?** Some ISPs have a slow route to PyPI's CDN (we measured 70 KB/s to PyPI vs 1.9 MB/s elsewhere on the same Wi-Fi). Use a mirror:
  `pip install -r requirements.txt -i https://asia-south1-python.pkg.dev/pypi-mirror/pypi/simple --extra-index-url https://download.pytorch.org/whl/cpu`
- **Laptop on battery?** Plug it in and use the "Best performance" power mode. On an i5-1235U, STT and PDF indexing were ~2x slower on battery.

Keys: get a free Groq key at https://console.groq.com/keys. Without `TAVILY_API_KEY`, web search falls back to DuckDuckGo automatically.

## Run

```bat
:: Web app (Phases 3-5)
uvicorn app.main:app --port 8000
:: open http://127.0.0.1:8000 -> upload PDF(s) -> click the mic -> talk
```

| Phase | Command |
|---|---|
| 1. Text agent | `python -m scripts.text_chat --demo` (or interactive: `python -m scripts.text_chat`) |
| 2. Local voice loop | `python -m scripts.voice_loop` (`--list-devices`, `--wav file.wav`) |
| 3-5. Web app | `uvicorn app.main:app --port 8000` |
| End-to-end check | `python -m scripts.ws_smoke_test --barge-in` (server running) |
| Tests | `pytest -q` |
| Eval | `python -m eval.run_eval` then `python -m eval.latency_bench` |

**Headphones vs speakers.** The page has a "🎧 I'm using headphones" toggle. With speakers, the browser's echo cancellation (AEC) must stay on, or the agent hears itself and interrupts itself. But we measured that AEC also mutes the first ~1 s of the user's speech while the agent is talking, so interruptions can lose their first word. With headphones there is no echo, so the toggle turns AEC off and barge-in is clean.

## Results

Measured on a laptop (Intel i5-1235U, CPU only, plugged in), Groq `openai/gpt-oss-120b` (free tier), 2026-10-10. Raw outputs: [eval/results/](eval/results/).

**Latency**: `python -m eval.latency_bench` (18 spoken turns: 6 questions × 3 rounds, mixing pdf_qa / calculator / web_search). Times are cumulative from the end-of-turn decision (VAD saw 500 ms of silence):

| Stage (from end of speech) | p50 | p95 |
|---|---|---|
| STT done (faster-whisper base.en) | 1.80 s | 3.33 s |
| First LLM token | 3.40 s | 6.39 s |
| **First TTS audio** | **4.56 s** | **8.00 s** |
| ↳ LLM time after STT (incl. tool calls) | 1.35 s | 4.74 s |
| ↳ TTS first chunk after first token | 0.84 s | 1.98 s |

User-perceived delay = the above + the 500 ms VAD silence window. The p95 tail comes from web-search turns (Tavily + a second LLM call) and turns where the model called `pdf_qa` twice.

**Accuracy**: `python -m eval.run_eval` (30 questions: 18 on the sample PDFs, 6 calculator, 6 web search):

| Metric | Result |
|---|---|
| **Tool-selection accuracy** (expected tool called) | **100%** (30/30) |
| Tool first-choice accuracy (first tool = expected) | 100% (30/30) |
| **Answer accuracy** (LLM judge: `gpt-oss-20b`, a different model) | **90.0%** (27/30) |
| Answer accuracy (strict keyword match, secondary) | 80.0% (24/30) |

| By tool | n | Answer (judge) | Tool |
|---|---|---|---|
| pdf_qa | 18 | 88.9% | 100% |
| calculator | 6 | 100% | 100% |
| web_search | 6 | 83.3% | 100% |

Error analysis of the 3 judge failures (manually reviewed):
- **pdf-06** (real error): said English-French BLEU was "about 41.0"; the paper says 41.8.
- **web-01** (real error, but safe): "Who won the 2026 FIFA World Cup?" → "I don't know." The truncated Tavily snippet didn't state the winner plainly; the agent declined to guess, as instructed.
- **pdf-04** (judge false negative): the answer correctly said 2048, but the reference also mentioned d_model = 512 and the judge penalised its absence. Scores are reported unadjusted.

The keyword check misses answers where the model spells numbers out ("twenty-eight point four" for 28.4), so the LLM judge is the primary metric.

**Before → after latency work** (same question, end of speech → first audio): 14.5 s (first working version, on battery) → 8.8 s (sentence streaming) → 5.5 s (plugged in, 8 Whisper threads, cached models) → 3.7 s (startup warm-up, chunk-level TTS streaming), measured in a real browser.

**Tests**: `pytest -q` → 41 passed (calculator safety incl. code-injection and DoS attempts, web-search fallback, RAG retrieval/citations/session isolation, sentence splitting).

## Design decisions (and what I'd say in an interview)

| Decision | Why |
|---|---|
| Server-side VAD (Silero) instead of browser VAD | One source of truth for turn-taking and barge-in; the browser stays a thin audio pipe. Silero costs <1 ms per 32 ms frame. |
| 500 ms silence = end of turn | Shorter cuts people off mid-thought; longer adds dead air. It is a config value (`VAD_MIN_SILENCE_MS`). |
| `beam_size=1`, `int8`, `base.en` Whisper | Greedy decoding is ~2x faster than beam search, with little accuracy loss on short commands. 8 CPU threads measured fastest on this laptop. |
| Sentence-level TTS + chunk streaming | Time to first audio no longer depends on answer length. The first chunk may end at a comma to start speaking sooner. |
| One FAISS `IndexFlatIP` per session | Exact cosine search (normalised vectors). Session isolation for free. A few thousand chunks don't need approximate search. |
| Chunk per page (800 chars / 120 overlap) | Every chunk keeps an exact page number for citations. |
| Calculator walks the AST | Never calls `eval`. Only numbers, operators and whitelisted math functions; caps on exponent/factorial/length block DoS like `9**9**9`. |
| Tools never raise | Errors come back as text, so the LLM can recover or say "I don't know" instead of crashing the turn. |
| History trimmed to last 24 messages (`start_on="human"`) | Bounds prompt size and latency without splitting a tool call from its result. |
| `gpt-oss-120b` on Groq | `llama-3.3-70b-versatile` was retired on Groq. Of the available models, gpt-oss-120b chose tools reliably (~0.5 s per decision). `qwen3.8-27b` was faster but answered a PDF question without calling `pdf_qa`. |
| Startup warm-up | Graph build (0.9 s) + Groq TLS handshake (0.4 s) were landing on the user's first turn; now done at server start. |

## Error handling

- **No mic / permission denied** → message in the chat; typed questions still work.
- **Empty or noise-only transcript** → Whisper hallucinations ("Thank you.") are filtered; the agent says it didn't catch that.
- **No PDF uploaded** → `pdf_qa` says so and the agent asks for an upload.
- **Non-PDF, >25 MB, or scanned PDF without text** → HTTP 400 with a clear reason.
- **Tool failure** (search down, bad expression) → error text to the LLM; web search falls back Tavily → DuckDuckGo.
- **Groq rate limit (429)** → retried with backoff by the client (free tier is 8K tokens/min).
- **TTS failure** → text still streams; the user sees an info message.
- **WebSocket drop** → the browser reconnects automatically; server cancels the in-flight turn.

## Limitations

- Memory and FAISS indexes are in-process; they vanish on restart and don't scale across workers (a production version would use Redis/Postgres checkpointers and a vector DB).
- CPU-only STT is the largest latency stage; a GPU or Groq's hosted Whisper would cut it substantially.
- Barge-in with speakers depends on browser AEC (see above).
