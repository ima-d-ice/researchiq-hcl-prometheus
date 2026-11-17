"""Unit tests for the pure, local pieces (no models, no API)."""
import pytest

from researchiq.retrieval.fusion import rrf
from researchiq.retrieval.guard import InvalidQuestion, clean_question
from researchiq.retrieval.router import route
from researchiq.schemas import Chunk, ChunkKind, Route, ScoredChunk


def _sc(cid: str) -> ScoredChunk:
    return ScoredChunk(chunk=Chunk(id=cid, paper_id="p", kind=ChunkKind.TEXT, text=cid, page=1), score=0)


def test_guard_cleans_and_caps():
    assert clean_question("  what   is\tRAG? ") == "what is RAG?"
    with pytest.raises(InvalidQuestion):
        clean_question("   ")
    with pytest.raises(InvalidQuestion):
        clean_question("x" * 5000)


@pytest.mark.parametrize("q,expected", [
    ("What does Figure 3 show?", Route.FIGURE),
    ("Compare BERT vs GPT", Route.COMPARE),
    ("Summarize the trends across all papers", Route.SYNTHESIS),
    ("What dataset was used?", Route.GENERAL),
])
def test_router(q, expected):
    assert route(q) == expected


def test_rrf_rewards_agreement():
    fused = rrf([_sc("a"), _sc("b"), _sc("c")], [_sc("b"), _sc("d")])
    assert fused[0].chunk.id == "b"
    assert {c.chunk.id for c in fused} == {"a", "b", "c", "d"}
