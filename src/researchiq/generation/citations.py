"""C4 · Citation check: S# exists + supported."""
import re

from researchiq.schemas import Citation, LLMAnswer, ScoredChunk

_TAG = re.compile(r"\[(S\d+)\]")


def is_supported(claim: str, source_text: str) -> bool:
    """Cheap lexical support check. TODO: replace with NLI model or LLM judge."""
    claim_terms = {w for w in re.findall(r"\w+", claim.lower()) if len(w) > 3}
    if not claim_terms:
        return True
    src = source_text.lower()
    return sum(t in src for t in claim_terms) / len(claim_terms) >= 0.3


def check_citations(ans: LLMAnswer, sources: dict[str, ScoredChunk]) -> tuple[list[Citation], list[str]]:
    warnings: list[str] = []
    cited: dict[str, bool] = {}

    for claim in ans.claims:
        for tag in claim.sources:
            if tag not in sources:
                warnings.append(f"Claim cites unknown source {tag}: {claim.text[:80]}")
                continue
            ok = is_supported(claim.text, sources[tag].chunk.text)
            cited[tag] = cited.get(tag, False) or ok
            if not ok:
                warnings.append(f"{tag} may not support: {claim.text[:80]}")

    for tag in _TAG.findall(ans.answer):
        if tag not in sources:
            warnings.append(f"Answer text references unknown source {tag}")
        cited.setdefault(tag, tag in sources)

    citations = [
        Citation(
            tag=tag,
            paper_title=sources[tag].chunk.paper_title,
            page=sources[tag].chunk.page,
            figure_label=sources[tag].chunk.figure_label,
            snippet=sources[tag].chunk.text[:300],
            supported=supported,
        )
        for tag, supported in sorted(cited.items()) if tag in sources
    ]
    return citations, warnings
