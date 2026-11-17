"""Run lanes B + C for one question and keep everything eval needs (sources, timings, flags).

Calls researchiq.generation.pipeline.generate_answer() (so eval always tests the real pipeline) and
keeps the tag -> chunk map so citations can be scored against (paper_id, page).
"""
import time
from dataclasses import dataclass, field

from researchiq.generation.pipeline import generate_answer
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
    """Lane C only (the real generation pipeline): the security suite injects crafted chunks here."""
    res = generate_answer(question, route, chunks)
    f = res.final
    t.context, t.error = res.context, res.error
    t.answer, t.insufficient_evidence, t.warnings = f.answer, f.insufficient_evidence, f.warnings
    t.claims = [c.text for c in f.claims]
    by_tag = {c.tag: res.sources[c.tag] for c in f.citations if c.tag in res.sources}
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
