"""Central settings. Values come from env / .env."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Groq
    groq_api_key: str = ""
    groq_api_keys: str = ""  # comma-separated, for rotation: GROQ_API_KEYS=key1,key2,...
    llm_model: str = "qwen/qwen3.8-27b"  # TODO: confirm Groq model id
    llm_temperature: float = 0.1

    llm_fallback_model: str = "llama-3.3-70b-versatile"
    vlm_fallback_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"

    # Groq budget (free tier). Limiter keeps a 10% safety margin.
    groq_tpm_limit: int = 8000
    groq_rpm_limit: int = 30
    groq_tpd_limit: int = 200_000
    groq_max_retries: int = 5
    per_request_token_cap: int = 4000
    vlm_max_output_tokens: int = 600
    vlm_max_image_tokens: int = 1536
    vlm_text_tokens: int = 600
    vlm_describe_tables: bool = True
    min_image_px: int = 100

    # Upload limits
    max_file_mb: int = 50
    max_pages: int = 60
    max_text_chars: int = 2_000_000
    max_image_pixels: int = 50_000_000
    max_unzipped_mb: int = 300

    # Chunking
    chunk_max_tokens: int = 512
    chunk_header_reserve: int = 48  # room for the contextual header inside BGE's 512
    render_dpi: int = 144
    catalogue_path: Path = Path("data/papers/catalogue.csv")

    # Embeddings / index
    dense_model: str = "BAAI/bge-base-en-v1.5"  # 768-d
    sparse_model: str = "Qdrant/bm25"
    dense_dim: int = 768
    qdrant_path: Path = Path("data/qdrant")
    qdrant_collection: str = "papers"

    # Retrieval
    dense_k: int = 30
    sparse_k: int = 30
    fused_k: int = 30
    rrf_k: int = 60
    reranker_model: str = "BAAI/bge-reranker-base"
    top_n: int = 6
    max_question_chars: int = 1000

    # Generation
    context_token_budget: int = 6000

    # Cache
    cache_dir: Path = Path("data/cache")

    @property
    def all_groq_keys(self) -> list[str]:
        """GROQ_API_KEY_1..N (numeric order), GROQ_API_KEYS (comma list) and GROQ_API_KEY, deduped."""
        import os
        import re

        from dotenv import dotenv_values

        found = {**(dotenv_values(".env") or {}), **os.environ}
        numbered = sorted(
            ((int(m.group(1)), (v or "").strip()) for k, v in found.items()
             if (m := re.fullmatch(r"GROQ_API_KEY_(\d+)", k, re.I))),
        )
        listed = [k.strip() for k in self.groq_api_keys.replace("\n", ",").split(",")]
        keys = [v for _, v in numbered] + listed + [self.groq_api_key.strip()]
        return list(dict.fromkeys(k for k in keys if k))

    @property
    def figures_cache(self) -> Path:
        return self.cache_dir / "figures.json"


settings = Settings()
