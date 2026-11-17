"""Lane B orchestration: question -> top chunks.

guard -> rewrite (Groq) -> router -> hybrid search -> RRF -> cross-encoder, per route.
"""
from dataclasses import dataclass

from researchiq.config import settings
from researchiq.retrieval.fusion import rrf
from researchiq.retrieval.guard import check_question
from researchiq.retrieval.hybrid_search import hybrid_search, lookup_label, paper_filter
from researchiq.retrieval.rerank import rerank
from researchiq.retrieval.rewrite import list_papers, rewrite
from researchiq.retrieval.router import route
from researchiq.schemas import QueryPlan, Route, ScoredChunk


@dataclass
class RetrievalResult:
    question: str
    route: Route
    chunks: list[ScoredChunk]
    plan: QueryPlan | None = None


def _group(plan: QueryPlan, paper_ids: list[str] | None, keep: int, k: int | None,
           use_sparse: bool, use_rerank: bool, pinned: list[ScoredChunk] | None = None) -> list[ScoredChunk]:
    """One search group: dense (+BM25) -> RRF -> rerank -> keep."""
    sparse_q = " ".join([plan.rewritten, *plan.keywords])
    dense, sparse = hybrid_search(plan.rewritten, sparse_q, paper_filter(paper_ids), k, use_sparse)
    lists = [pinned or [], dense] + ([sparse] if use_sparse else [])
    fused = rrf(*[l for l in lists if l])
    return rerank(plan.rewritten, fused, keep) if use_rerank else fused[:keep]


def _interleave(groups: list[list[ScoredChunk]]) -> list[ScoredChunk]:
    out = []
    for i in range(max((len(g) for g in groups), default=0)):
        out += [g[i] for g in groups if i < len(g)]
    return out


def retrieve(question: str, use_sparse: bool = True, use_rerank: bool = True,
             use_rewrite: bool = True) -> RetrievalResult:
    """Flags exist so eval can run ablations (dense-only, no rerank, ...)."""
    q = check_question(question)
    plan = rewrite(q) if use_rewrite else QueryPlan(original=q, rewritten=q)
    r = route(plan)
    n = settings.top_n
    args = dict(use_sparse=use_sparse, use_rerank=use_rerank)

    if r == Route.COMPARE:
        keep = max(3, n // len(plan.paper_ids))
        groups = [_group(plan, [pid], keep, settings.compare_k, **args) for pid in plan.paper_ids]
        chunks = _interleave(groups)
    elif r == Route.SYNTHESIS:
        ids = plan.paper_ids or [pid for pid, _ in list_papers()]
        groups = [_group(plan, [pid], settings.synthesis_per_paper, settings.synthesis_k, **args)
                  for pid in ids]
        chunks = sorted((c for g in groups for c in g), key=lambda c: c.score, reverse=True)
    else:
        pinned = lookup_label(plan.figure_label, plan.paper_ids or None) if r == Route.FIGURE and plan.figure_label else None
        chunks = _group(plan, plan.paper_ids or None, n, None, pinned=pinned, **args)
    return RetrievalResult(question=q, route=r, chunks=chunks, plan=plan)
