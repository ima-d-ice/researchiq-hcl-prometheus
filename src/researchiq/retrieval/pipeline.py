"""Lane B orchestration: question -> top-6 chunks."""
from dataclasses import dataclass

from researchiq.retrieval.fusion import rrf
from researchiq.retrieval.guard import clean_question
from researchiq.retrieval.hybrid_search import hybrid_search
from researchiq.retrieval.rerank import rerank
from researchiq.retrieval.router import route
from researchiq.schemas import Route, ScoredChunk


@dataclass
class RetrievalResult:
    question: str
    route: Route
    chunks: list[ScoredChunk]


def retrieve(question: str, use_sparse: bool = True, use_rerank: bool = True) -> RetrievalResult:
    """Flags exist so eval can run ablations (dense-only, no rerank, ...)."""
    q = clean_question(question)
    r = route(q)
    dense, sparse = hybrid_search(q, r)
    fused = rrf(dense, sparse) if use_sparse else rrf(dense)
    top = rerank(q, fused) if use_rerank else fused[:6]
    return RetrievalResult(question=q, route=r, chunks=top)
