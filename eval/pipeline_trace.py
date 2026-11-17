"""Run lanes B + C for one question and keep everything eval needs (sources, timings, flags).

Mirrors researchiq.generation.pipeline.answer() step for step, but also returns the tag -> chunk
map so citations can be scored against (paper_id, page). If generation changes its pipeline,
update this file to match (or have answer() return its sources and call it from here).
"""
import time
from dataclasses import dataclass, field

from researchiq.generation.citations import check_citations
from researchiq.generation.context_builder import build_context
from researchiq.generation.generator import generate
from researchiq.generation.validate import GenerationFailed, validate_with_retry
from researchiq.llm.groq_client import limiter
from researchiq.retrieval.pipeline import retrieve
from researchiq.schemas import Route, ScoredChunk


@dataclass
class Trace:
    question: str
    route: str = ""
    retrieved: list[tuple[str, int]] = field(default_factory=list)   # (paper_id, page) in rank order
    context: str = ""
    answer: str = ""
    claims: list[str] = field(default_factory=list)
    cited: list[tuple[str, int]] = field(default_factory=list)       # (paper_id, page) of each citation
    insufficient_evidence: bool = False
    warnings: list[str] = field(default_factory=list)
    error: str = ""          # what a user would see if the pipeline failed
    seconds: float = 0.0
    groq_tokens: int = 0

    @property
    def full_text(self) -> str:
        """Everything the user would see, for leak / canary checks."""
        return "\n".join([self.answer, *self.claims, *self.warnings, self.error])


def _keys(chunks: list[ScoredChunk]) -> list[tuple[str, int]]:
    return [(c.chunk.paper_id, c.chunk.page) for c in chunks]


def answer_from_chunks(question: str, route: Route, chunks: list[ScoredChunk], t: Trace) -> Trace:
    """Lane C only: used directly by the security suite to inject crafted chunks."""
    t.context, sources = build_context(chunks)
    if not sources:
        t.answer, t.insufficient_evidence = "No relevant sources found in the indexed papers.", True
        return t
    try:
        llm = validate_with_retry(lambda fb: generate(question, route, t.context, fb))
    except GenerationFailed:
        t.error = "The model returned an invalid answer twice."
        return t
    citations, t.warnings = check_citations(llm, sources)
    t.answer, t.insufficient_evidence = llm.answer, llm.insufficient_evidence
    t.claims = [c.text for c in llm.claims]
    by_tag = {c.tag: sources[c.tag] for c in citations}
    t.cited = [(sc.chunk.paper_id, sc.chunk.page) for sc in by_tag.values()]
    return t


def run(question: str, **retrieve_flags) -> Trace:
    t, start, tokens = Trace(question), time.perf_counter(), limiter.run_tokens
    try:
        ret = retrieve(question, **retrieve_flags)
        t.question, t.route, t.retrieved = ret.question, ret.route.value, _keys(ret.chunks)
        answer_from_chunks(ret.question, ret.route, ret.chunks, t)
    except Exception as e:  # record, don't crash the whole eval run
        t.error = f"{type(e).__name__}: {e}"
    t.seconds = time.perf_counter() - start
    t.groq_tokens = limiter.run_tokens - tokens
    return t
