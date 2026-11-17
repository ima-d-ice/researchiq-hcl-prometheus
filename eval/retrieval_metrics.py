"""D2 · Retrieval metrics: Recall@k, MRR, ablation configs. Hit = (paper_id, page) match."""
from researchiq.schemas import ScoredChunk

ABLATIONS = {
    "full (dense+bm25+rerank)": dict(use_sparse=True, use_rerank=True),
    "dense+bm25 (no rerank)": dict(use_sparse=True, use_rerank=False),
    "dense only + rerank": dict(use_sparse=False, use_rerank=True),
    "dense only": dict(use_sparse=False, use_rerank=False),
}


def _hits(chunks: list[ScoredChunk], paper_id: str, pages: list[int]) -> list[bool]:
    return [c.chunk.paper_id == paper_id and c.chunk.page in pages for c in chunks]


def recall_at_k(chunks: list[ScoredChunk], paper_id: str, pages: list[int], k: int = 5) -> float:
    """Fraction of expected pages found in the top k."""
    found = {c.chunk.page for c in chunks[:k] if c.chunk.paper_id == paper_id}
    return len(found & set(pages)) / len(pages) if pages else 0.0


def reciprocal_rank(chunks: list[ScoredChunk], paper_id: str, pages: list[int]) -> float:
    for i, hit in enumerate(_hits(chunks, paper_id, pages), start=1):
        if hit:
            return 1.0 / i
    return 0.0
