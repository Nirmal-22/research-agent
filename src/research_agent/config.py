"""Typed application settings, loaded from environment variables and an optional `.env` file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Models (all served by local Ollama) ---
    ollama_base_url: str = "http://localhost:11434"
    chat_model: str = "qwen3:4b-instruct"
    embed_model: str = "nomic-embed-text"
    temperature: float = 0.2
    num_ctx: int = Field(8192, description="Context window passed to Ollama.")
    request_timeout_s: float = 180.0

    # --- Agent budgets: small models loop and fail, so everything is capped ---
    max_subquestions: int = Field(3, ge=1, le=6)
    researcher_max_model_calls: int = Field(4, ge=1, le=12)
    max_revisions: int = Field(1, ge=0, le=4)
    structured_output_retries: int = Field(2, ge=0, le=5)
    max_concurrency: int = Field(2, ge=1, description="Parallel researchers (Send fan-out).")

    # --- Tools ---
    enable_web: bool = True
    search_max_results: int = 4
    search_retries: int = 1
    search_retry_delay_s: float = 2.0
    fetch_timeout_s: float = 10.0
    fetch_max_chars: int = 3000
    kb_top_k: int = 4
    kb_min_score: float = Field(0.45, description="Drop chunks below this relevance score.")
    kb_grade: bool = Field(True, description="LLM-grade retrieved chunks (corrective RAG).")

    # --- Storage ---
    docs_dir: Path = Path("data/docs")
    chroma_dir: Path = Path("data/chroma")
    chroma_collection: str = "research_kb"
    chunk_size: int = 800
    chunk_overlap: int = 120
    checkpoint_db: Path = Path("data/checkpoints.sqlite")

    # --- Observability ---
    log_level: str = "INFO"
    log_json: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
