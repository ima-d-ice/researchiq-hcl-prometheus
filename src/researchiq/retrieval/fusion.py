"""B4 · RRF fusion: merge ranked lists by rank -> top 30."""
from researchiq.config import settings
from researchiq.schemas import ScoredChunk


def rrf(*ranked_lists: list[ScoredChunk], k: int | None = None, limit: int | None = None) -> list[ScoredChunk]:
    k = k or settings.rrf_k
    limit = limit or settings.fused_k
    scores: dict[str, float] = {}
    by_id: dict[str, ScoredChunk] = {}
    for lst in ranked_lists:
        for rank, sc in enumerate(lst, start=1):
            cid = sc.chunk.id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            by_id.setdefault(cid, sc)
    ordered = sorted(scores, key=scores.get, reverse=True)[:limit]
    return [ScoredChunk(chunk=by_id[cid].chunk, score=scores[cid]) for cid in ordered]
