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
    query_rewrite: bool = True
    rewrite_max_tokens: int = 300
    compare_k: int = 15
    synthesis_k: int = 10
    synthesis_per_paper: int = 2

    # Generation
    context_token_budget: int = 6000  # legacy cap; per-route budgets below take precedence
    # Evidence gate: best rerank score (bge-reranker logit) below this -> no LLM call. Tune on the eval set.
    evidence_threshold: float = -4.0
    # Per-route generation budgets (HLD 4.2): (max context chunks, max output tokens)
    gen_request_cap: int = 4700       # whole request (system + question + context + output) stays under this
    gen_system_tokens: int = 450
    gen_max_tokens_general: int = 800
    gen_max_tokens_compare: int = 1200
    gen_chunk_tokens: int = 450       # per-chunk cap inside the prompt
    # Citation check thresholds (tune on the eval set)
    cite_overlap_min: float = 0.3     # content-word overlap, claim vs cited chunk(s)
    cite_semantic_min: float = -2.0   # cross-encoder logit, claim vs cited chunk(s)

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
    def rewrites_cache(self) -> Path:
        return self.cache_dir / "rewrites.json"

    @property
    def figures_cache(self) -> Path:
        return self.cache_dir / "figures.json"


settings = Settings()
