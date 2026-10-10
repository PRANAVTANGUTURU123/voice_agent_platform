"""FastAPI app: PDF upload, voice WebSocket, and the static frontend.

WebSocket protocol (/ws/{session_id})
-------------------------------------
Client -> server
  binary : 16 kHz mono int16 PCM chunks from the mic (AudioWorklet)
  text   : {"type": "text", "text": "..."}   typed question (no-mic fallback)
           {"type": "interrupt"}             user pressed stop

Server -> client
  binary : 4-byte big-endian turn id + an MP3 chunk, streamed as edge-tts
           produces it (playback starts on the first chunk)
  text   : {"type": "speech_start"}          user began talking -> stop playback
           {"type": "turn_start", "turn": n}
           {"type": "transcript", "text"}    what STT heard
           {"type": "tool", "name"}          agent called a tool
           {"type": "agent_token", "text"}   streamed answer text
           {"type": "audio_sentence_end"}    previous chunks form a whole MP3 sentence
           {"type": "agent_done", "metrics"} per-stage latencies
           {"type": "info" | "error", "message"}

Barge-in: whenever server-side VAD detects the start of speech, the server
cancels the in-flight turn task (LLM stream + TTS) and tells the browser to
stop playback. The turn id on audio frames lets the browser drop any stale
audio from the cancelled turn that was already in flight.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.agent.graph import warm_up as warm_up_agent
from app.audio.stt import get_whisper, transcribe
from app.audio.vad import SpeechEnd, SpeechStart, StreamingVAD, pcm16_to_float32
from app.config import PROJECT_ROOT, get_settings
from app.pipeline import run_turn
from app.rag.ingest import get_embedder, get_session_index, ingest_pdf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("voice-agent")
# Log every outbound HTTP call with its status (e.g. Groq 429 rate-limit retries).
logging.getLogger("httpx").setLevel(logging.INFO)

FRONTEND_DIR = PROJECT_ROOT / "frontend"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI):
    # Load heavy models at startup so the first user turn isn't slow.
    logger.info("Warming up models...")
    await asyncio.to_thread(get_embedder)
    await asyncio.to_thread(get_whisper)
    await asyncio.to_thread(transcribe, np.zeros(8000, dtype=np.float32))
    await warm_up_agent()  # graph build + open the Groq HTTPS connection
    logger.info("Ready.")
    yield


app = FastAPI(title="Real-Time Voice Agent", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


def _check_session_id(session_id: str) -> None:
    # The id is used in a file path, so restrict it to safe characters.
    if not _SESSION_ID_RE.match(session_id):
        raise HTTPException(400, "Invalid session id")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload/{session_id}")
async def upload(session_id: str, files: list[UploadFile] = File(...)) -> dict[str, Any]:
    """Save uploaded PDFs and add them to this session's FAISS index."""
    _check_session_id(session_id)
    upload_dir = get_settings().upload_dir / session_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    added: list[dict[str, Any]] = []
    errors: list[str] = []
    for f in files:
        name = Path(f.filename or "upload.pdf").name  # strip any client path
        data = await f.read()
        if not name.lower().endswith(".pdf") or not data.startswith(b"%PDF"):
            errors.append(f"{name}: not a PDF")
            continue
        if len(data) > MAX_UPLOAD_BYTES:
            errors.append(f"{name}: larger than 25 MB")
            continue
        path = upload_dir / name
        path.write_bytes(data)
        try:
            t0 = time.perf_counter()
            n_chunks = await asyncio.to_thread(ingest_pdf, session_id, path, name)
            added.append({"file": name, "chunks": n_chunks, "seconds": round(time.perf_counter() - t0, 2)})
        except Exception as exc:  # noqa: BLE001 - corrupt/scanned PDFs
            logger.warning("Ingest failed for %s: %s", name, exc)
            errors.append(f"{name}: {exc}")

    if not added and errors:
        raise HTTPException(400, "; ".join(errors))
    return {"added": added, "errors": errors, "files": get_session_index(session_id).files}


class VoiceSession:
    """State for one WebSocket connection: VAD + the currently running turn."""

    def __init__(self, ws: WebSocket, session_id: str) -> None:
        self.ws = ws
        self.session_id = session_id
        self.vad = StreamingVAD()
        self.turn_task: asyncio.Task[Any] | None = None
        self.turn_id = 0
        self.send_lock = asyncio.Lock()  # one writer at a time on the socket

    async def send_json(self, payload: dict[str, Any]) -> None:
        async with self.send_lock:
            await self.ws.send_text(json.dumps(payload))

    async def send_audio(self, turn_id: int, mp3: bytes) -> None:
        async with self.send_lock:
            await self.ws.send_bytes(struct.pack(">I", turn_id) + mp3)

    async def cancel_turn(self) -> None:
        """Barge-in: stop the LLM stream and TTS of the current turn."""
        if self.turn_task and not self.turn_task.done():
            self.turn_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self.turn_task
            logger.info("Session %s: turn %d cancelled (barge-in)", self.session_id, self.turn_id)
        self.turn_task = None

    async def start_turn(self, *, audio: np.ndarray | None = None, text: str | None = None, t_start: float) -> None:
        await self.cancel_turn()
        self.turn_id += 1
        turn_id = self.turn_id
        await self.send_json({"type": "turn_start", "turn": turn_id})

        async def emit_audio(_seq: int, mp3: bytes) -> None:
            await self.send_audio(turn_id, mp3)

        async def runner() -> None:
            metrics = await run_turn(
                self.session_id, self.send_json, emit_audio, audio=audio, text=text, t_start=t_start
            )
            logger.info("Session %s turn %d metrics: %s", self.session_id, turn_id, metrics.as_dict())

        self.turn_task = asyncio.create_task(runner())

    async def on_audio(self, data: bytes) -> None:
        if len(data) % 2:
            return  # malformed frame
        for event in self.vad.process(pcm16_to_float32(data)):
            if isinstance(event, SpeechStart):
                # Tell the browser first so playback stops immediately.
                await self.send_json({"type": "speech_start"})
                await self.cancel_turn()
            elif isinstance(event, SpeechEnd):
                await self.start_turn(audio=event.audio, t_start=time.perf_counter())

    async def on_text(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return
        if msg.get("type") == "text" and str(msg.get("text", "")).strip():
            await self.start_turn(text=str(msg["text"]), t_start=time.perf_counter())
        elif msg.get("type") == "interrupt":
            await self.cancel_turn()


@app.websocket("/ws/{session_id}")
async def voice_ws(ws: WebSocket, session_id: str) -> None:
    if not _SESSION_ID_RE.match(session_id):
        await ws.close(code=1008)
        return
    await ws.accept()
    session = VoiceSession(ws, session_id)
    logger.info("Session %s connected", session_id)
    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                await session.on_audio(message["bytes"])
            elif message.get("text") is not None:
                await session.on_text(message["text"])
    except WebSocketDisconnect:
        pass
    finally:
        await session.cancel_turn()
        logger.info("Session %s disconnected", session_id)
