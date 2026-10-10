"""Central configuration, loaded from environment variables / `.env`.

Every tunable (model names, thresholds, paths) lives here so the rest of the
code never hardcodes values or secrets. Swap the LLM by changing
LLM_PROVIDER / LLM_MODEL in `.env` — no code changes needed.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- API keys (never hardcoded; read from .env) ---
    groq_api_key: str = ""
    tavily_api_key: str = ""

    # --- LLM (swappable) ---
    llm_provider: str = "groq"  # "groq" is built in; see agent/graph.py::build_llm
    # llama-3.3-70b-versatile was retired on Groq; gpt-oss-120b is the strongest
    # tool-calling model available there now.
    llm_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = 0.2
    # gpt-oss models "think" before answering; low effort keeps voice latency down.
    # Set to "" for models without reasoning support.
    llm_reasoning_effort: str = "low"

    # --- Speech-to-text ---
    whisper_model: str = "base.en"
    whisper_compute_type: str = "int8"
    # Measured on an i5-1235U (2 P-cores + 8 E-cores): 8 threads was fastest.
    whisper_cpu_threads: int = 8

    # --- Voice activity detection ---
    sample_rate: int = 16_000
    vad_threshold: float = 0.5
    vad_min_silence_ms: int = 500  # silence that ends a user turn
    vad_speech_pad_ms: int = 100

    # --- Text-to-speech ---
    tts_voice: str = "en-IN-NeerjaNeural"

    # --- RAG ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4

    # --- Paths ---
    upload_dir: Path = PROJECT_ROOT / "data" / "uploads"
    sample_docs_dir: Path = PROJECT_ROOT / "data" / "sample_docs"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance (read `.env` once per process)."""
    return Settings()
