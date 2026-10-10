"""One conversational turn: audio/text in -> streamed text + audio out.

    end of speech --STT--> transcript --LLM stream--> tokens --sentence split-->
        sentence queue --edge-tts stream--> MP3 chunks --> emit_audio()

The LLM producer and the TTS consumer run concurrently, connected by an
asyncio.Queue: while sentence 1 is being synthesised, the LLM is already
writing sentence 2. This is what makes time-to-first-audio roughly
"STT + time to first sentence + TTS of one sentence" instead of
"STT + whole answer + TTS of whole answer".

Used by both the WebSocket server (app/main.py) and eval/latency_bench.py,
so benchmark numbers measure the real code path.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable

import numpy as np

from app.agent.graph import stream_answer
from app.audio.stt import transcribe
from app.audio.tts import SentenceBuffer, synthesize_stream

logger = logging.getLogger(__name__)

# emit(event_dict) for JSON events; emit_audio(sentence_seq, mp3_chunk) for audio.
EmitJson = Callable[[dict[str, Any]], Awaitable[None]]
EmitAudio = Callable[[int, bytes], Awaitable[None]]


@dataclass
class TurnMetrics:
    """Latencies in milliseconds, all measured from the end-of-turn decision
    (VAD saw ~500 ms of silence) — or from text submission for typed turns."""

    stt_ms: float | None = None
    first_token_ms: float | None = None
    first_audio_ms: float | None = None
    total_ms: float | None = None
    tools: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


async def run_turn(
    session_id: str,
    emit: EmitJson,
    emit_audio: EmitAudio,
    *,
    audio: np.ndarray | None = None,
    text: str | None = None,
    t_start: float | None = None,
) -> TurnMetrics:
    """Run a full turn. Cancel the surrounding task to abort it (barge-in)."""
    t0 = t_start or time.perf_counter()
    since = lambda: round((time.perf_counter() - t0) * 1000, 1)  # noqa: E731
    metrics = TurnMetrics()

    # ---- 1. Speech-to-text -------------------------------------------------
    if audio is not None:
        text = await asyncio.to_thread(transcribe, audio)
        metrics.stt_ms = since()
    text = (text or "").strip()
    await emit({"type": "transcript", "text": text})
    if not text:
        await emit({"type": "info", "message": "Sorry, I didn't catch that. Please try again."})
        return metrics

    # ---- 2+3. LLM stream -> sentences -> TTS, concurrently ---------------
    sentences: asyncio.Queue[str | None] = asyncio.Queue()

    async def tts_worker() -> None:
        # Each MP3 chunk is forwarded the moment edge-tts produces it (~0.75 s
        # after the request), not after the whole sentence is synthesised.
        # The browser appends chunks to one MediaSource stream, so playback
        # starts on the first chunk and sentences play back to back.
        seq = 0
        while (sentence := await sentences.get()) is not None:
            try:
                async for chunk in synthesize_stream(sentence):
                    if metrics.first_audio_ms is None:
                        metrics.first_audio_ms = since()
                    await emit_audio(seq, chunk)
            except Exception as exc:  # noqa: BLE001 - keep text flowing even if TTS fails
                logger.warning("TTS failed for %r: %s", sentence, exc)
                await emit({"type": "info", "message": "Voice synthesis failed; showing text only."})
            # Marks a complete MP3 sentence, for clients that decode per sentence.
            await emit({"type": "audio_sentence_end", "seq": seq})
            seq += 1

    tts_task = asyncio.create_task(tts_worker())
    buffer = SentenceBuffer()
    try:
        async for event in stream_answer(session_id, text):
            if event["type"] == "tool":
                metrics.tools.append(event["name"])
                await emit({"type": "tool", "name": event["name"]})
                continue
            if metrics.first_token_ms is None:
                metrics.first_token_ms = since()
            await emit({"type": "agent_token", "text": event["text"]})
            for sentence in buffer.push(event["text"]):
                sentences.put_nowait(sentence)
        if rest := buffer.flush():
            sentences.put_nowait(rest)
        sentences.put_nowait(None)  # tell the TTS worker we're done
        await tts_task
    except asyncio.CancelledError:
        tts_task.cancel()
        raise
    except Exception as exc:  # noqa: BLE001 - LLM/API errors become a spoken-safe message
        tts_task.cancel()
        logger.exception("Agent turn failed")
        await emit({"type": "error", "message": f"The agent hit an error: {exc}"})
        return metrics

    metrics.total_ms = since()
    await emit({"type": "agent_done", "metrics": metrics.as_dict()})
    return metrics
