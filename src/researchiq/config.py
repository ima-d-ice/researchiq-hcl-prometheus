"""Central settings. Values come from env / .env."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Groq
    groq_api_key: str = ""
    llm_model: str = "qwen/qwen3.8-27b"  # TODO: confirm Groq model id
    llm_temperature: float = 0.1

    # Upload limits
    max_file_mb: int = 50
    max_pages: int = 60

    # Chunking
    chunk_max_tokens: int = 512

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
    def figures_cache(self) -> Path:
        return self.cache_dir / "figures.json"


settings = Settings()
