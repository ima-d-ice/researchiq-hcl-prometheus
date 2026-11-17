"""Paper titles from PDFs and code-based paper matching for the compare route. No models, no API."""
import pytest

from researchiq.ingestion.parse import pdf_font_title
from researchiq.retrieval.rewrite import match_papers
from researchiq.retrieval.router import route
from researchiq.schemas import QueryPlan, Route

PAPERS = [
    ("p1", "Multimodal Retrieval-Augmented Generation: Unified Information Processing Across Text, "
           "Image, Table, and Video Modalities"),
    ("p2", "MBA-RAG: a Bandit Approach for Adaptive Retrieval-Augmented Generation through Question Complexity"),
    ("p3", "DAT: Dynamic Alpha Tuning for Hybrid Retrieval in Retrieval-Augmented Generation"),
    ("p4", "GOLFer: Smaller LM-Generated Documents Hallucination Filter & Combiner for Query Expansion"),
    ("p5", "Question Decomposition for Retrieval-Augmented Generation"),
]


def test_title_is_largest_text_on_page_one(tmp_path):
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 60), "Proceedings of Some Workshop 2025", fontsize=8)
    page.insert_text((72, 100), "A Great Title About Retrieval", fontsize=18)
    page.insert_text((72, 140), "Jane Doe, John Roe", fontsize=11)
    page.insert_text((72, 400), "Body text of the paper goes here.", fontsize=10)
    p = tmp_path / "2025.paper-1.pdf"
    doc.save(p)
    assert pdf_font_title(p) == "A Great Title About Retrieval"


@pytest.mark.parametrize("question,expected", [
    ("Compare GOLFer and DAT", ["p4", "p3"]),
    ("compare mba-rag with dat", ["p2", "p3"]),
    (PAPERS[0][1] + " vs " + PAPERS[1][1], ["p1", "p2"]),   # full titles, as typed in the UI
    ("What dataset was used?", []),
    ("What is retrieval augmented generation?", []),           # generic words must not match a title
])
def test_match_papers(question, expected):
    assert sorted(match_papers(question, PAPERS)) == sorted(expected)


def test_short_name_needs_whole_word():
    assert match_papers("what data does it use", PAPERS) == []   # "data" is not "DAT"


def test_compare_route_needs_two_papers():
    two = QueryPlan(original="q", rewritten="q", route=Route.COMPARE, paper_ids=["p1", "p2"])
    one = QueryPlan(original="q", rewritten="q", route=Route.COMPARE, paper_ids=["p2"])
    assert route(two) == Route.COMPARE
    assert route(one) == Route.GENERAL
