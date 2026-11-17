"""B5 · Rerank: cross-encoder -> top N.

Two backends, chosen by settings.reranker_backend:
  onnx  : small int8-quantized MiniLM cross-encoder through onnxruntime (fastembed). Fast on CPU.
  torch : the original BAAI/bge-reranker-base through sentence-transformers (slow, kept as fallback).

Score scales differ: torch/bge returns 0..1 (sigmoid); onnx/MiniLM returns raw logits (about -10..+8).
The evidence gate threshold (config.evidence_threshold) must be calibrated for the active backend.
"""
import logging
from functools import lru_cache

from researchiq.config import settings
from researchiq.schemas import ScoredChunk

log = logging.getLogger(__name__)


@lru_cache
def _onnx_model():
    from fastembed.common.model_description import ModelSource
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    name = settings.reranker_onnx_model
    if name not in {m["model"] for m in TextCrossEncoder.list_supported_models()}:
        TextCrossEncoder.add_custom_model(
            model=name,
            sources=ModelSource(hf=settings.reranker_onnx_repo),
            model_file=settings.reranker_onnx_file,
            description="int8-quantized ONNX cross-encoder",
            license="apache-2.0",
            size_in_gb=0.03,
        )
    return TextCrossEncoder(model_name=name)


@lru_cache
def _torch_model():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(settings.reranker_model)


def preload() -> None:
    """Load the reranker once at app start so the first user question isn't slow."""
    (_onnx_model if settings.reranker_backend == "onnx" else _torch_model)()


def score_pairs(question: str, texts: list[str]) -> list[float]:
    if settings.reranker_backend == "onnx":
        return [float(s) for s in _onnx_model().rerank(question, texts)]
    return [float(s) for s in _torch_model().predict([(question, t) for t in texts])]


def rerank(question: str, candidates: list[ScoredChunk], top_n: int | None = None) -> list[ScoredChunk]:
    if not candidates:
        return []
    top_n = top_n or settings.top_n
    # candidates arrive best-first from RRF: only score the top slice (cross-encoder cost is linear)
    pool = candidates[: settings.rerank_max_candidates]
    scores = score_pairs(question, [c.chunk.contextualized or c.chunk.text for c in pool])
    ranked = sorted(zip(pool, scores), key=lambda x: x[1], reverse=True)[:top_n]
    return [ScoredChunk(chunk=c.chunk, score=s) for c, s in ranked]
