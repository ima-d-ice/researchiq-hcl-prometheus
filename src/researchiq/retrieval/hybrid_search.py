"""B3 · Hybrid search: dense top-30 + BM25 top-30 from Qdrant, plus exact figure lookup."""
from qdrant_client import models

from researchiq.config import settings
from researchiq.ingestion.embed import embed_query
from researchiq.ingestion.index import DENSE, SPARSE, qdrant
from researchiq.schemas import Chunk, ScoredChunk


def paper_filter(paper_ids: list[str] | None = None, label: str | None = None) -> models.Filter | None:
    must = []
    if paper_ids:
        must.append(models.FieldCondition(key="paper_id", match=models.MatchAny(any=paper_ids)))
    if label:
        must.append(models.FieldCondition(key="figure_label", match=models.MatchValue(value=label)))
    return models.Filter(must=must) if must else None


def _to_scored(points) -> list[ScoredChunk]:
    return [ScoredChunk(chunk=Chunk(**p.payload), score=getattr(p, "score", 0.0) or 0.0) for p in points]


def hybrid_search(dense_query: str, sparse_query: str | None = None,
                  flt: models.Filter | None = None, k: int | None = None,
                  use_sparse: bool = True) -> tuple[list[ScoredChunk], list[ScoredChunk]]:
    """Returns (dense_hits, sparse_hits), each already rank-ordered.

    The dense side embeds the rewritten question; the BM25 side also gets the keywords.
    """
    dense_vec, _ = embed_query(dense_query)
    client = qdrant()
    dense = client.query_points(settings.qdrant_collection, query=dense_vec, using=DENSE,
                                limit=k or settings.dense_k, query_filter=flt, with_payload=True).points
    sparse: list = []
    if use_sparse:
        _, sparse_vec = embed_query(sparse_query or dense_query)
        sparse = client.query_points(settings.qdrant_collection, query=sparse_vec, using=SPARSE,
                                     limit=k or settings.sparse_k, query_filter=flt,
                                     with_payload=True).points
    return _to_scored(dense), _to_scored(sparse)


def lookup_label(label: str, paper_ids: list[str] | None = None) -> list[ScoredChunk]:
    """Exact payload lookup of 'Figure 3' / 'Table 2' chunks (the figure route's pinned list)."""
    pts, _ = qdrant().scroll(settings.qdrant_collection, scroll_filter=paper_filter(paper_ids, label),
                             limit=10, with_payload=True, with_vectors=False)
    return _to_scored(pts)
