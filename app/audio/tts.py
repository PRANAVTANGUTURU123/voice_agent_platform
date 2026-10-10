"""Text-to-speech with edge-tts, plus a sentence splitter for streaming.

Latency trick: instead of waiting for the full LLM answer, `SentenceBuffer`
collects streamed tokens and releases each sentence as soon as it is
complete, so TTS for sentence 1 runs while the LLM is still writing
sentence 2.

edge-tts returns MP3 (24 kHz mono). Each sentence is synthesised to one
self-contained MP3 blob, which the browser can decode independently.
"""

from __future__ import annotations

import re
from typing import AsyncIterator

import edge_tts

from app.config import get_settings

# A sentence ends at . ! ? (optionally followed by quotes/brackets) + whitespace.
_SENTENCE_END = re.compile(r"""(?<=[.!?])["')\]]*\s+""")
# Don't split after common abbreviations or single initials ("e.g. ", "Dr. ", "J. ").
_NO_SPLIT_BEFORE = re.compile(r"(?:\b(?:e\.g|i\.e|etc|vs|Dr|Mr|Mrs|Ms|St|No|Fig|Eq)\.|\b[A-Z]\.)$")
_MIN_SENTENCE_CHARS = 12  # merge very short fragments ("Yes.") with the next
# The FIRST chunk may also end at a clause boundary (comma/semicolon/colon/dash)
# once it is long enough: shorter text -> faster first TTS -> earlier first audio.
_CLAUSE_END = re.compile(r"(?<=[,;:])\s+|\s+[–—-]\s+")
_FIRST_CLAUSE_MIN_CHARS = 30


# Symbols the TTS voice mispronounces or skips -> spoken words.
_SPOKEN_SYMBOLS = str.maketrans({
    "α": " alpha ", "β": " beta ", "γ": " gamma ", "δ": " delta ", "ε": " epsilon ",
    "ϵ": " epsilon ", "λ": " lambda ", "μ": " mu ", "σ": " sigma ", "θ": " theta ",
    "≈": " approximately ", "≤": " at most ", "≥": " at least ", "×": " times ",
    "÷": " divided by ", "√": " square root of ", "∞": " infinity ", "±": " plus or minus ",
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6",
    "⁷": "7", "⁸": "8", "⁹": "9", "⁻": " minus ",
})


def clean_for_speech(text: str) -> str:
    """Strip markdown and turn symbols into words so TTS reads them naturally."""
    # "10⁻⁹" -> "10 to the power minus 9" (superscripts after a number are exponents)
    text = re.sub(r"(\d)([⁻]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)", r"\1 to the power \2", text)
    text = text.translate(_SPOKEN_SYMBOLS)
    text = re.sub(r"\s=\s", " equals ", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[*_#`>|]+", "", text)
    text = re.sub(r"\[(\d+)\]", "", text)  # citation markers like [1]
    return re.sub(r"\s+", " ", text).strip()


class SentenceBuffer:
    """Accumulate streamed tokens; yield complete sentences as they appear."""

    def __init__(self) -> None:
        self._buf = ""
        self._emitted_any = False

    def push(self, token: str) -> list[str]:
        self._buf += token
        sentences: list[str] = []
        if not self._emitted_any:
            for match in _CLAUSE_END.finditer(self._buf):
                candidate = self._buf[: match.start()].strip()
                if len(candidate) >= _FIRST_CLAUSE_MIN_CHARS:
                    sentences.append(candidate)
                    self._buf = self._buf[match.end() :]
                    break
        start = 0
        for match in _SENTENCE_END.finditer(self._buf):
            candidate = self._buf[start : match.start()].strip()
            if len(candidate) < _MIN_SENTENCE_CHARS or _NO_SPLIT_BEFORE.search(candidate):
                continue  # keep accumulating into the same sentence
            sentences.append(candidate)
            start = match.end()
        self._buf = self._buf[start:]
        self._emitted_any = self._emitted_any or bool(sentences)
        return sentences

    def flush(self) -> str:
        """Return whatever is left at the end of the stream."""
        rest, self._buf = self._buf.strip(), ""
        return rest


async def synthesize_stream(text: str, voice: str | None = None) -> AsyncIterator[bytes]:
    """Yield MP3 byte chunks for `text` as edge-tts produces them."""
    text = clean_for_speech(text)
    if not text:
        return
    communicate = edge_tts.Communicate(text, voice or get_settings().tts_voice)
    async for chunk in communicate.stream():
        if chunk["type"] == "audio" and chunk.get("data"):
            yield chunk["data"]


async def synthesize(text: str, voice: str | None = None) -> bytes:
    """Synthesise one sentence into a single complete MP3 blob."""
    return b"".join([part async for part in synthesize_stream(text, voice)])
