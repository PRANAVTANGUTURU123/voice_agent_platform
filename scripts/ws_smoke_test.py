"""End-to-end test of the running server, acting like the browser.

1. POST a PDF to /upload/{session}
2. Stream a spoken question (TTS-generated) to /ws/{session} as 16 kHz int16
   PCM in 100 ms chunks at real-time pace, like the AudioWorklet does
3. Print every event, and check transcript / tool / tokens / audio arrive
4. --barge-in: start a second question while the first answer is streaming
   and check the server interrupts (speech_start + new turn, old audio stops)

Usage (server must be running: uvicorn app.main:app):
    python -m scripts.ws_smoke_test
    python -m scripts.ws_smoke_test --barge-in
"""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
import time
import uuid

import httpx
import miniaudio
import numpy as np
import websockets

from app.audio.tts import synthesize
from app.config import get_settings

BASE = "127.0.0.1:8000"


async def spoken_pcm(text: str) -> bytes:
    """Synthesise `text` with a 'user' voice -> 16 kHz mono int16 PCM bytes."""
    mp3 = await synthesize(text, voice="en-US-GuyNeural")
    d = miniaudio.decode(mp3, output_format=miniaudio.SampleFormat.SIGNED16, nchannels=1, sample_rate=16000)
    silence = np.zeros(16000, dtype="<i2").tobytes()  # 1 s so VAD sees end of turn
    return np.asarray(d.samples, dtype="<i2").tobytes() + silence


async def send_realtime(ws: websockets.ClientConnection, pcm: bytes) -> None:
    chunk = 3200  # 100 ms of int16 @ 16 kHz
    for i in range(0, len(pcm), chunk):
        await ws.send(pcm[i : i + chunk])
        await asyncio.sleep(0.1)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--barge-in", action="store_true")
    args = parser.parse_args()
    sid = uuid.uuid4().hex[:12]

    pdf = get_settings().sample_docs_dir / "attention_is_all_you_need.pdf"
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(f"http://{BASE}/upload/{sid}", files={"files": (pdf.name, pdf.read_bytes(), "application/pdf")})
        print("upload:", r.status_code, r.json())
        bad = await client.post(f"http://{BASE}/upload/{sid}", files={"files": ("notes.txt", b"hello", "text/plain")})
        print("non-PDF upload:", bad.status_code, bad.json())

    q1 = await spoken_pcm("What optimizer was used to train the Transformer, and what learning rate schedule did it use?")
    q2 = await spoken_pcm("What is seventeen times twenty three?")

    audio_frames: dict[int, int] = {}
    events: list[tuple[float, dict]] = []
    async with websockets.connect(f"ws://{BASE}/ws/{sid}", max_size=None) as ws:
        t0 = time.perf_counter()

        async def reader() -> None:
            async for msg in ws:
                t = round(time.perf_counter() - t0, 2)
                if isinstance(msg, bytes):
                    turn = struct.unpack(">I", msg[:4])[0]
                    audio_frames[turn] = audio_frames.get(turn, 0) + 1
                    print(f"{t:6.2f}s  [audio turn={turn} {len(msg) - 4} bytes]")
                    continue
                ev = json.loads(msg)
                events.append((t, ev))
                if ev["type"] != "agent_token":
                    print(f"{t:6.2f}s  {ev}")
                if ev["type"] == "agent_done" and (not args.barge_in or ev_turns() >= 2):
                    return

        def ev_turns() -> int:
            return sum(1 for _, e in events if e["type"] == "turn_start")

        read_task = asyncio.create_task(reader())
        await send_realtime(ws, q1)
        if args.barge_in:
            # Wait until the first answer's audio is streaming, then talk over it.
            while not audio_frames:
                await asyncio.sleep(0.05)
            print("---- user barges in ----")
            await send_realtime(ws, q2)
        await asyncio.wait_for(read_task, timeout=120)

    answer = "".join(e["text"] for _, e in events if e["type"] == "agent_token")
    print("\nfull streamed text:", answer)
    print("audio frames per turn:", audio_frames)


if __name__ == "__main__":
    asyncio.run(main())
