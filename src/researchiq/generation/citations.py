"""C4 · Citation check (plain code, zero Groq tokens).

Layers, in order. Each claim ends as supported | weak | unsupported:
  1. ID validity       every cited S# was actually sent; unknown ones are removed
  2. Coverage          each claim has a source; each answer sentence carries a [S#]; answer tags <-> claims
  3. Lexical overlap   content words of the claim found in the cited chunk(s)
  4. Numbers           every number in the claim appears in a cited chunk (else unsupported)
  5. Named entities    acronyms / CamelCase names in the claim appear in a cited chunk (else weak)
  6. Semantic support  local cross-encoder score, claim vs cited chunk
  7. Attribution       a claim naming paper A must not cite only paper B
  8. Consistency       insufficient_evidence vs claims
ai_suggestions are never checked here: they are speculation and the UI labels them as such.
"""
import logging
import re
from dataclasses import dataclass, field
from typing import Callable

from researchiq.config import settings
from researchiq.schemas import Citation, ClaimCheck, LLMAnswer, ScoredChunk

log = logging.getLogger(__name__)
_TAG = re.compile(r"\[(S\d+)\]")
_TAGS_ANY = re.compile(r"\[(S\d+(?:\s*[,;]\s*S\d+)*)\]")
_NUM = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)*%?")
_ENTITY = re.compile(r"\b(?:[A-Z]{2,}[A-Za-z0-9\-]*|[A-Z][a-z]+[A-Z][A-Za-z0-9\-]*|[A-Za-z]+-?\d[A-Za-z0-9\-]*)\b")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])")
_STOP = set("""this that with from were have been which their there these those also than then them they
 into over such more most some only other about under using used uses while where when what does paper
 papers show shows shown study results result method methods approach based between both each across""".split())
_RANK = {"supported": 0, "weak": 1, "unsupported": 2}


@dataclass
class CitationReport:
    answer: str                         # answer text with unknown [S#] markers removed
    citations: list[Citation] = field(default_factory=list)
    claims: list[ClaimCheck] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# --- helpers -------------------------------------------------------------------------------------
def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "ly", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w


def _terms(text: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 3 and w not in _STOP}


def _clean(text: str) -> str:
    return _TAGS_ANY.sub("", text).strip()


def _norm_num(n: str) -> str:
    return n.replace(",", "").rstrip("%")


def _haystack(sc: ScoredChunk) -> str:
    c = sc.chunk
    return " ".join(filter(None, [c.contextualized or c.text, c.text, c.figure_label, c.caption]))


def overlap(claim: str, source: str) -> float:
    """Share of the claim's content words that occur in the source (1.0 if the claim has none)."""
    ct = _terms(claim)
    return 1.0 if not ct else len(ct & _terms(source)) / len(ct)


def missing_numbers(claim: str, source: str) -> list[str]:
    have = {_norm_num(n) for n in _NUM.findall(source)}
    return [n for n in _NUM.findall(claim) if _norm_num(n) not in have]


def missing_entities(claim: str, source: str) -> list[str]:
    low = source.lower()
    return sorted({e for e in _ENTITY.findall(claim) if e.lower() not in low})


def _semantic(pairs: list[tuple[str, str]]) -> list[float]:
    """Cross-encoder relevance (claim, chunk). Reuses the reranker already loaded for retrieval."""
    from researchiq.retrieval.rerank import score_pairs
    # score_pairs takes one question; claims differ per pair, so score each (claim, chunk) pair separately
    return [score_pairs(claim, [chunk])[0] for claim, chunk in pairs]


def _mentioned_papers(claim: str, papers: dict[str, str]) -> set[str]:
    low = claim.lower()
    return {pid for pid, title in papers.items()
            if re.search(rf"(?<![\w]){re.escape(pid.lower())}(?![\w])", low) or (len(title) > 8 and title.lower() in low)}


# --- main ----------------------------------------------------------------------------------------
def check_citations(ans: LLMAnswer, sources: dict[str, ScoredChunk],
                    semantic: Callable[[list[tuple[str, str]]], list[float]] | None = None) -> CitationReport:
    semantic = semantic or _semantic
    rep = CitationReport(answer=ans.answer)
    w = rep.warnings
    papers = {sc.chunk.paper_id: sc.chunk.paper_title for sc in sources.values()}

    # 1. unknown [S#] in the answer text are removed
    def _drop_unknown(m: re.Match) -> str:
        tags = [t.strip() for t in re.split(r"[,;]", m.group(1))]
        known = [t for t in tags if t in sources]
        for t in tags:
            if t not in sources:
                w.append(f"Answer text references unknown source {t} (removed)")
        return f"[{', '.join(known)}]" if known else ""
    rep.answer = re.sub(r"\s{2,}", " ", _TAGS_ANY.sub(_drop_unknown, ans.answer)).strip()

    # 7/8. consistency
    if ans.insufficient_evidence and ans.claims:
        w.append("Model flagged insufficient evidence but still made claims; treat them with caution.")
    if not ans.insufficient_evidence and not ans.claims:
        w.append("Answer has no itemised claims, so it could not be verified.")

    # per-claim layers
    pending: list[tuple[int, str, list[str]]] = []      # claims that still need the semantic score
    checks: list[ClaimCheck] = []
    for claim in ans.claims:
        text = _clean(claim.text)
        ids = list(dict.fromkeys(claim.source_ids))
        for t in ids:
            if t not in sources:
                w.append(f"Claim cites unknown source {t}: {text[:80]}")
        ids = [t for t in ids if t in sources]
        reasons: list[str] = []
        status = "supported"

        def worsen(new: str, why: str) -> None:
            nonlocal status
            reasons.append(why)
            if _RANK[new] > _RANK[status]:
                status = new

        if not ids:
            worsen("unsupported", "no valid source cited")
        else:
            src = " ".join(_haystack(sources[t]) for t in ids)
            ov = overlap(text, src)
            if ov < settings.cite_overlap_min / 2:
                worsen("unsupported", f"very low word overlap ({ov:.0%})")
            elif ov < settings.cite_overlap_min:
                worsen("weak", f"low word overlap ({ov:.0%})")
            if nums := missing_numbers(text, src):
                worsen("unsupported", f"number(s) not in cited source: {', '.join(nums)}")
            if ents := missing_entities(text, src):
                worsen("weak", f"term(s) not in cited source: {', '.join(ents)}")
            cited_papers = {sources[t].chunk.paper_id for t in ids}
            named = _mentioned_papers(text, papers)
            if named and not named & cited_papers:
                worsen("unsupported", "attributed to a paper the cited source is not from")
            pending.append((len(checks), text, ids))
        checks.append(ClaimCheck(text=text, source_ids=ids, status=status, reasons=reasons))

    # 6. semantic support, batched in one local call (best cited chunk per claim)
    if pending:
        pairs = [(text, sources[t].chunk.contextualized or sources[t].chunk.text) for _, text, ids in pending for t in ids]
        try:
            scores = semantic(pairs)
        except Exception as e:  # the check is best-effort: never break answering
            w.append("Semantic support check unavailable.")
            scores = None
            log.warning("semantic check failed: %s", type(e).__name__)
        if scores is not None:
            pos = 0
            for ci, _text, ids in pending:
                best = max(scores[pos:pos + len(ids)])
                pos += len(ids)
                if best < settings.cite_semantic_min:
                    c = checks[ci]
                    c.reasons.append(f"low semantic support ({best:.1f})")
                    c.status = "unsupported" if c.status == "weak" else max(c.status, "weak", key=_RANK.get)

    for c in checks:
        if c.status != "supported":
            tags = "/".join(c.source_ids) or "-"
            w.append(f"{tags} may not support ({c.status}): {c.text[:80]} [{'; '.join(c.reasons)}]")
    rep.claims = checks

    # 2. coverage: answer sentences and answer tags vs claim tags
    if not ans.insufficient_evidence:
        for s in _SENT.split(rep.answer):
            if len(s.split()) >= 5 and not _TAGS_ANY.search(s):
                w.append(f"Uncited sentence: {s[:80]}")
    claim_tags = {t for c in checks for t in c.source_ids}
    answer_tags = set(_TAG.findall(rep.answer)) | set(re.findall(r"S\d+", " ".join(_TAGS_ANY.findall(rep.answer))))
    for t in sorted(answer_tags - claim_tags):
        w.append(f"{t} is cited in the answer but backed by no claim")

    # citations: one per tag in use, status = worst of the claims that cite it
    worst: dict[str, str] = {}
    for c in checks:
        for t in c.source_ids:
            worst[t] = max(worst.get(t, "supported"), c.status, key=_RANK.get)
    for t in answer_tags:
        worst.setdefault(t, "weak")                     # cited in text only: not verified by any claim
    for tag in sorted(worst, key=lambda t: int(t[1:])):
        if tag not in sources:
            continue
        ch = sources[tag].chunk
        rep.citations.append(Citation(tag=tag, paper_id=ch.paper_id, paper_title=ch.paper_title, page=ch.page,
                                      figure_label=ch.figure_label, snippet=ch.text[:300],
                                      status=worst[tag], supported=worst[tag] == "supported"))
    return rep
