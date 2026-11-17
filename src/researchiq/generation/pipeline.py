"""Lanes B + C end to end: question -> cited answer."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from researchiq.config import settings
from researchiq.generation.citations import check_citations
from researchiq.generation.context_builder import build_context
from researchiq.generation.generator import generate
from researchiq.generation.validate import GenerationFailed, validate_with_retry
from researchiq.schemas import FinalAnswer, Route, ScoredChunk

if TYPE_CHECKING:
    from researchiq.retrieval.pipeline import RetrievalResult

log = logging.getLogger(__name__)

NO_SOURCES = "No relevant sources found in the indexed papers."
WEAK_EVIDENCE = "The retrieved passages are not relevant enough to answer this reliably."
FAILED = "I couldn't produce a reliable answer. Please rephrase the question and try again."


@dataclass
class GenResult:
    final: FinalAnswer
    sources: dict[str, ScoredChunk] = field(default_factory=dict)
    context: str = ""
    llm_called: bool = False
    error: str = ""                   # user-safe message when generation failed


def evidence_gate(route: Route, chunks: list[ScoredChunk]) -> str | None:
    """HLD 3.6: best rerank score below the threshold -> don't guess. Returns a reason or None.

    The FIGURE route is exempt: its chunks come from an exact label lookup, not from similarity.
    """
    if not chunks:
        return NO_SOURCES
    if route != Route.FIGURE and max(c.score for c in chunks) < settings.evidence_threshold:
        return WEAK_EVIDENCE
    return None


def generate_answer(question: str, route: Route, chunks: list[ScoredChunk]) -> GenResult:
    """Lane C: gate -> context -> LLM -> validate -> citation check. Never raises."""
    def refuse(reason: str, sources: dict | None = None, context: str = "") -> GenResult:
        final = FinalAnswer(question=question, route=route, answer=reason, citations=[],
                            insufficient_evidence=True, missing_info=reason)
        return GenResult(final, sources or {}, context)

    if (reason := evidence_gate(route, chunks)) is not None:
        return refuse(reason)

    context, sources = build_context(chunks, route=route, question=question)
    if not sources:
        return refuse(NO_SOURCES)

    try:
        llm = validate_with_retry(lambda fb: generate(question, route, context, fb))
    except GenerationFailed as e:
        log.warning("generation failed validation: %s", str(e)[:300])
        return GenResult(FinalAnswer(question=question, route=route, answer=FAILED, citations=[]),
                         sources, context, True, FAILED)
    except Exception as e:  # provider down / rate limits exhausted: details to logs, not the user
        log.exception("generation call failed: %s", type(e).__name__)
        return GenResult(FinalAnswer(question=question, route=route, answer=FAILED, citations=[]),
                         sources, context, True, FAILED)

    rep = check_citations(llm, sources)
    warnings = rep.warnings
    if llm.insufficient_evidence:
        warnings = ["Model reported insufficient evidence in the sources.", *warnings]
    final = FinalAnswer(question=question, route=route, answer=rep.answer, citations=rep.citations,
                        claims=rep.claims, warnings=warnings, insufficient_evidence=llm.insufficient_evidence,
                        missing_info=llm.missing_info, ai_suggestions=llm.ai_suggestions)
    return GenResult(final, sources, context, True)


def answer(question: str, retrieval: RetrievalResult | None = None) -> FinalAnswer:
    """Pass `retrieval` to reuse an existing retrieval (eval) instead of searching again."""
    from researchiq.retrieval.pipeline import retrieve  # lazy: pulls in Qdrant + embedding models

    ret = retrieval or retrieve(question)
    return generate_answer(ret.question, ret.route, ret.chunks).final
