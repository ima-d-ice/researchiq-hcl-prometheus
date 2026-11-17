"""B5 · Rerank: cross-encoder -> top 6."""
from functools import lru_cache

from sentence_transformers import CrossEncoder

from researchiq.config import settings
from researchiq.schemas import ScoredChunk


@lru_cache
def _model() -> CrossEncoder:
    return CrossEncoder(settings.reranker_model)


def rerank(question: str, candidates: list[ScoredChunk], top_n: int | None = None) -> list[ScoredChunk]:
    if not candidates:
        return []
    top_n = top_n or settings.top_n
    scores = _model().predict([(question, c.chunk.contextualized or c.chunk.text) for c in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:top_n]
    return [ScoredChunk(chunk=c.chunk, score=float(s)) for c, s in ranked]
