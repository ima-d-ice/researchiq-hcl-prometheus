"""Lanes B + C end to end: question -> cited answer."""
from researchiq.generation.citations import check_citations
from researchiq.generation.context_builder import build_context
from researchiq.generation.generator import generate
from researchiq.generation.validate import validate_with_retry
from researchiq.retrieval.pipeline import retrieve
from researchiq.schemas import FinalAnswer


def answer(question: str) -> FinalAnswer:
    ret = retrieve(question)
    context, sources = build_context(ret.chunks)
    if not sources:
        return FinalAnswer(question=ret.question, route=ret.route,
                           answer="No relevant sources found in the indexed papers.", citations=[])

    llm = validate_with_retry(lambda fb: generate(ret.question, ret.route, context, fb))
    citations, warnings = check_citations(llm, sources)
    if llm.insufficient_evidence:
        warnings.insert(0, "Model reported insufficient evidence in the sources.")
    return FinalAnswer(question=ret.question, route=ret.route, answer=llm.answer,
                       citations=citations, warnings=warnings)
