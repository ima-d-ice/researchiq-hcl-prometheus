"""A7 · Embed ×2: BGE 768-d dense + BM25 sparse."""
from functools import lru_cache

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import models

from researchiq.config import settings


@lru_cache
def dense_model() -> TextEmbedding:
    return TextEmbedding(settings.dense_model)


@lru_cache
def sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(settings.sparse_model)


def embed_dense(texts: list[str]) -> list[list[float]]:
    return [v.tolist() for v in dense_model().embed(texts)]


def embed_sparse(texts: list[str]) -> list[models.SparseVector]:
    return [
        models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())
        for s in sparse_model().embed(texts)
    ]


def embed_query(text: str) -> tuple[list[float], models.SparseVector]:
    # TODO: BGE recommends a query instruction prefix; check fastembed's query_embed
    dense = next(dense_model().query_embed(text)).tolist()
    s = next(sparse_model().query_embed(text))
    return dense, models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())
