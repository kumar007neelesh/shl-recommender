"""Centralized configuration. Everything is overridable via environment variables
so the same code runs locally, in tests, and on free hosting (Render/Railway/Fly)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ---- Data ----------------------------------------------------------------
    # Source catalog (the real SHL hiring dataset, fetched by scripts/ingest.py).
    catalog_source_url: str = (
        "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
    )
    traces_source_url: str = (
        "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/sample_conversations.zip"
    )
    # Normalized catalog produced by ingest.py and consumed at runtime.
    catalog_path: Path = DATA_DIR / "catalog_normalized.json"

    # ---- LLM -----------------------------------------------------------------
    # provider: "gemini" | "groq" | "openrouter" | "none"
    # "none" -> deterministic fallback controller (no network, still valid schema).
    llm_provider: str = "gemini"
    llm_api_key: str = ""
    # gemini-2.5-flash is the free-tier stable alias (2.0 Flash was shut down
    # 2026-06-01). Use gemini-2.5-flash-lite for higher free rate limits.
    llm_model: str = "gemini-2.5-flash"
    llm_timeout_s: float = 18.0          # leave headroom under the 30s eval cap
    llm_max_retries: int = 1

    # ---- Retrieval -----------------------------------------------------------
    top_k: int = 10                      # max recommendations (assignment caps at 10)
    candidate_pool: int = 40             # retrieve wide, then trim
    use_embeddings: bool = False         # optional; BM25+alias is the robust default

    # ---- Agent ---------------------------------------------------------------
    max_turns: int = 8                   # evaluator caps conversations at 8 turns


@lru_cache
def get_settings() -> Settings:
    return Settings()
