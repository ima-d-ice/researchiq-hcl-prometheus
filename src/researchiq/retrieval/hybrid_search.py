"""B3 · Hybrid search: dense top-30 + BM25 top-30 from Qdrant."""
from qdrant_client import models

from researchiq.config import settings
from researchiq.ingestion.embed import embed_query
from researchiq.ingestion.index import DENSE, SPARSE, qdrant
from researchiq.schemas import Chunk, Route, ScoredChunk


def _filter(r: Route) -> models.Filter | None:
    if r == Route.FIGURE:
        return models.Filter(must=[models.FieldCondition(
            key="kind", match=models.MatchAny(any=["figure", "table"]))])
    return None


def _to_scored(points) -> list[ScoredChunk]:
    return [ScoredChunk(chunk=Chunk(**p.payload), score=p.score) for p in points]


def hybrid_search(question: str, r: Route) -> tuple[list[ScoredChunk], list[ScoredChunk]]:
    """Returns (dense_hits, sparse_hits), each already rank-ordered."""
    dense_vec, sparse_vec = embed_query(question)
    flt = _filter(r)
    client = qdrant()
    dense = client.query_points(settings.qdrant_collection, query=dense_vec, using=DENSE,
                                limit=settings.dense_k, query_filter=flt, with_payload=True).points
    sparse = client.query_points(settings.qdrant_collection, query=sparse_vec, using=SPARSE,
                                 limit=settings.sparse_k, query_filter=flt, with_payload=True).points
    # TODO: for COMPARE / SYNTHESIS, ensure coverage across multiple papers
    return _to_scored(dense), _to_scored(sparse)
