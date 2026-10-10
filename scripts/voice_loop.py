"""Phase 2: local voice loop to verify the audio pipeline end to end.

    mic -> Silero VAD -> faster-whisper -> LangGraph agent -> edge-tts -> speakers

Usage:
    python -m scripts.voice_loop                    # talk to the agent (sample PDFs loaded)
    python -m scripts.voice_loop --list-devices     # show audio devices
    python -m scripts.voice_loop --wav question.wav # feed a WAV file instead of the mic

This loop is half-duplex: the mic is ignored while the agent is speaking, since
there's no echo cancellation outside the browser. Barge-in is in the web app.
"""

from __future__ import annotations

import argparse
import asyncio
import queue
import sys
import time
import uuid
import wave
from pathlib import Path

import miniaudio
import numpy as np
import sounddevice as sd

from app.agent.graph import stream_answer
from app.audio.stt import transcribe
from app.audio.tts import SentenceBuffer, synthesize
from app.audio.vad import SpeechEnd, SpeechStart, StreamingVAD, pcm16_to_float32
from app.config import get_settings
from app.rag.ingest import ingest_pdf

SR = get_settings().sample_rate


def decode_mp3(data: bytes) -> tuple[np.ndarray, int]:
    """Decode an MP3 blob to float32 mono samples for sounddevice."""
    decoded = miniaudio.decode(data, output_format=miniaudio.SampleFormat.FLOAT32, nchannels=1)
    return np.frombuffer(decoded.samples, dtype=np.float32), decoded.sample_rate


async def respond(session_id: str, text: str, t_end: float) -> None:
    """Stream the agent's answer, speaking each sentence as soon as it's ready."""
    buffer = SentenceBuffer()
    first_audio = True

    async def speak(sentence: str) -> None:
        nonlocal first_audio
        mp3 = await synthesize(sentence)
        if not mp3:
            return
        samples, rate = decode_mp3(mp3)
        if first_audio:
            print(f"  [first audio {time.perf_counter() - t_end:.2f}s after end of speech]")
            first_audio = False
        sd.play(samples, rate)
        await asyncio.to_thread(sd.wait)

    print("Agent: ", end="", flush=True)
    async for event in stream_answer(session_id, text):
        if event["type"] == "tool":
            print(f"[{event['name']}] ", end="", flush=True)
            continue
        print(event["text"], end="", flush=True)
        for sentence in buffer.push(event["text"]):
            await speak(sentence)
    if rest := buffer.flush():
        await speak(rest)
    print()


async def handle_utterance(session_id: str, audio: np.ndarray) -> None:
    t_end = time.perf_counter()
    text = await asyncio.to_thread(transcribe, audio)
    print(f"You: {text or '(nothing recognised)'}  [STT {time.perf_counter() - t_end:.2f}s]")
    if not text:
        return
    try:
        await respond(session_id, text, t_end)
    except Exception as exc:  # noqa: BLE001 - keep the loop alive
        print(f"\nAgent error: {exc}")


async def run_wav(session_id: str, path: Path) -> None:
    """Push a WAV file through VAD exactly like mic audio (useful for testing)."""
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != SR or w.getnchannels() != 1 or w.getsampwidth() != 2:
            sys.exit(f"{path} must be 16 kHz mono 16-bit PCM")
        audio = pcm16_to_float32(w.readframes(w.getnframes()))
    audio = np.concatenate([audio, np.zeros(SR, dtype=np.float32)])  # trailing silence
    vad = StreamingVAD()
    for i in range(0, len(audio), 1600):  # 100 ms chunks, like a mic
        for event in vad.process(audio[i : i + 1600]):
            if isinstance(event, SpeechEnd):
                await handle_utterance(session_id, event.audio)


async def run_mic(session_id: str) -> None:
    mic_q: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata: np.ndarray, frames: int, t: object, status: sd.CallbackFlags) -> None:
        mic_q.put(indata[:, 0].copy())

    try:
        stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=1600, callback=callback)
    except Exception as exc:  # noqa: BLE001 - PortAudioError and friends
        sys.exit(f"Could not open microphone: {exc}\nCheck Windows mic privacy settings or use --wav.")

    vad = StreamingVAD()
    print("Listening... speak a question (Ctrl+C to quit).")
    with stream:
        while True:
            chunk = await asyncio.to_thread(mic_q.get)
            for event in vad.process(chunk):
                if isinstance(event, SpeechStart):
                    print("  (speech detected)")
                elif isinstance(event, SpeechEnd):
                    stream.stop()  # half duplex: don't hear ourselves
                    await handle_utterance(session_id, event.audio)
                    with mic_q.mutex:
                        mic_q.queue.clear()
                    vad.reset()
                    stream.start()
                    print("Listening...")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=Path, help="16 kHz mono WAV to use instead of the mic")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()
    if args.list_devices:
        print(sd.query_devices())
        return

    session_id = f"voice-{uuid.uuid4().hex[:8]}"
    for pdf in sorted(get_settings().sample_docs_dir.glob("*.pdf")):
        ingest_pdf(session_id, pdf)
    print("Warming up Whisper...")
    await asyncio.to_thread(transcribe, np.zeros(SR // 2, dtype=np.float32))

    if args.wav:
        await run_wav(session_id, args.wav)
    else:
        await run_mic(session_id)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
