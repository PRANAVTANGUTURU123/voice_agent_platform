"""Speech-to-text with faster-whisper (CTranslate2, CPU, int8)."""

from __future__ import annotations

import logging
import re
import threading
from functools import lru_cache

import numpy as np
from faster_whisper import WhisperModel

from app.config import get_settings

logger = logging.getLogger(__name__)

# Whisper tends to "hear" these on silence/noise. Drop them if they are the
# entire transcript.
_HALLUCINATIONS = {
    "", "you", "thank you", "thanks for watching", "thank you for watching",
    "bye", ".", "uh", "um",
}

_lock = threading.Lock()


@lru_cache(maxsize=1)
def get_whisper() -> WhisperModel:
    """Load the Whisper model once per process."""
    s = get_settings()
    logger.info("Loading faster-whisper %s (%s)", s.whisper_model, s.whisper_compute_type)
    kwargs = dict(device="cpu", compute_type=s.whisper_compute_type, cpu_threads=s.whisper_cpu_threads)
    try:
        # Cached copy first: avoids Hugging Face network checks on every start.
        return WhisperModel(s.whisper_model, local_files_only=True, **kwargs)
    except Exception:  # noqa: BLE001 - not cached yet -> download once
        return WhisperModel(s.whisper_model, **kwargs)


def transcribe(audio: np.ndarray) -> str:
    """Transcribe float32 16 kHz mono audio. Returns "" if nothing was said.

    Settings tuned for latency on short conversational turns:
      - beam_size=1 (greedy): ~2x faster than beam search, tiny accuracy cost
      - no Whisper VAD filter: Silero already trimmed the audio
      - condition_on_previous_text=False: each turn is independent
    """
    if audio.size == 0:
        return ""
    model = get_whisper()
    # CTranslate2 uses all cores for one call; serialising calls avoids
    # oversubscription when several sessions finish speaking at once.
    with _lock:
        segments, _info = model.transcribe(
            audio,
            language="en",
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        texts = [seg.text for seg in segments if seg.no_speech_prob < 0.6]
    text = " ".join(t.strip() for t in texts).strip()
    normalized = re.sub(r"[^a-z ]", "", text.lower()).strip()
    if normalized in _HALLUCINATIONS:
        return ""
    return text
