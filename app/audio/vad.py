"""Streaming voice activity detection with Silero VAD.

Audio arrives in arbitrary-sized chunks (from a mic callback or a WebSocket).
`StreamingVAD.process()` re-frames it into the 512-sample (32 ms @ 16 kHz)
windows Silero expects and emits events:

    SpeechStart          -> user began talking (used for barge-in)
    SpeechEnd(audio)     -> ~500 ms of silence: the user's turn is over;
                            `audio` is the whole utterance as float32 PCM
"""

from __future__ import annotations

import collections
from dataclasses import dataclass

import numpy as np
import torch
from silero_vad import VADIterator, load_silero_vad

from app.config import get_settings

FRAME_SAMPLES = 512  # Silero's required window at 16 kHz


@dataclass
class SpeechStart:
    pass


@dataclass
class SpeechEnd:
    audio: np.ndarray  # float32 mono 16 kHz, values in [-1, 1]

    @property
    def duration_s(self) -> float:
        return len(self.audio) / get_settings().sample_rate


VADEvent = SpeechStart | SpeechEnd


def pcm16_to_float32(data: bytes) -> np.ndarray:
    """Convert little-endian int16 PCM bytes to float32 in [-1, 1]."""
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


class StreamingVAD:
    """Turn a stream of audio chunks into speech-start / speech-end events.

    One instance per audio stream: Silero is a stateful RNN, so sessions must
    not share a model instance (the model is tiny, ~2 MB, and loads in ms).
    """

    def __init__(
        self,
        min_speech_ms: int = 250,
        max_utterance_s: float = 30.0,
        preroll_ms: int = 300,
    ) -> None:
        s = get_settings()
        self.sample_rate = s.sample_rate
        self._model = load_silero_vad()
        self._iterator = VADIterator(
            self._model,
            threshold=s.vad_threshold,
            sampling_rate=s.sample_rate,
            min_silence_duration_ms=s.vad_min_silence_ms,
            speech_pad_ms=s.vad_speech_pad_ms,
        )
        self._min_speech_samples = int(min_speech_ms * s.sample_rate / 1000)
        self._max_utterance_samples = int(max_utterance_s * s.sample_rate)
        # Keep a little audio from *before* speech was detected so the first
        # syllable is not clipped (VAD needs a few frames to become confident).
        self._preroll: collections.deque[np.ndarray] = collections.deque(
            maxlen=max(1, preroll_ms * s.sample_rate // 1000 // FRAME_SAMPLES)
        )
        self._pending = np.zeros(0, dtype=np.float32)
        self._utterance: list[np.ndarray] = []
        self.in_speech = False

    def reset(self) -> None:
        self._iterator.reset_states()
        self._preroll.clear()
        self._pending = np.zeros(0, dtype=np.float32)
        self._utterance = []
        self.in_speech = False

    def process(self, audio: np.ndarray) -> list[VADEvent]:
        """Feed float32 mono audio; return any events it triggered."""
        events: list[VADEvent] = []
        self._pending = np.concatenate([self._pending, audio.astype(np.float32, copy=False)])
        n_frames = len(self._pending) // FRAME_SAMPLES
        for i in range(n_frames):
            frame = self._pending[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]
            result = self._iterator(torch.from_numpy(frame.copy()), return_seconds=False)

            if not self.in_speech:
                self._preroll.append(frame)
                if result and "start" in result:
                    self.in_speech = True
                    self._utterance = list(self._preroll)
                    events.append(SpeechStart())
                continue

            self._utterance.append(frame)
            utterance_len = len(self._utterance) * FRAME_SAMPLES
            ended = bool(result and "end" in result)
            if ended or utterance_len >= self._max_utterance_samples:
                events.extend(self._finish_utterance(force=not ended))
        self._pending = self._pending[n_frames * FRAME_SAMPLES :]
        return events

    def _finish_utterance(self, force: bool) -> list[VADEvent]:
        audio = np.concatenate(self._utterance)
        self.in_speech = False
        self._utterance = []
        self._preroll.clear()
        if force:
            # Max length hit mid-speech: restart the detector cleanly.
            self._iterator.reset_states()
        if len(audio) < self._min_speech_samples:
            return []  # a click or cough, not a real utterance
        return [SpeechEnd(audio)]
