"""Ingestion unit tests: fixtures are generated on the fly; no models, no API."""
import zipfile

import pytest

from researchiq.ingestion.contextualize import chunk_id, contextualize, header
from researchiq.ingestion.fallback import fallback_office, fallback_pdf, fallback_text, split_text
from researchiq.ingestion.tables import split_markdown_table
from researchiq.ingestion.upload import UploadError, validate_file
from researchiq.ingestion.visual import find_label
from researchiq.schemas import Chunk, ChunkKind

words = lambda t: len(t.split())  # noqa: E731  (stand-in token counter)


def _pdf(path, pages=1, text="Hello world.\nFigure 1: An architecture overview."):
    import pymupdf as fitz
    doc = fitz.open()
    for _ in range(pages):
        doc.new_page().insert_text((72, 72), text)
    doc.save(path)
    doc.close()
    return path


# ---------- validation ----------
def test_valid_pdf(tmp_path):
    info = validate_file(_pdf(tmp_path / "a.pdf", pages=2))
    assert (info.source_type, info.n_pages) == ("pdf", 2)


def test_pdf_bad_magic(tmp_path):
    p = tmp_path / "a.pdf"
    p.write_bytes(b"not a pdf at all")
    with pytest.raises(UploadError):
        validate_file(p)


def test_pdf_too_many_pages(tmp_path):
    from researchiq.config import settings
    with pytest.raises(UploadError, match="pages"):
        validate_file(_pdf(tmp_path / "a.pdf", pages=settings.max_pages + 1))


def test_encrypted_pdf(tmp_path):
    import pymupdf as fitz
    src = _pdf(tmp_path / "s.pdf")
    doc = fitz.open(src)
    out = tmp_path / "enc.pdf"
    doc.save(out, encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="x", owner_pw="y")
    with pytest.raises(UploadError, match="encrypted"):
        validate_file(out)


def test_wrong_extension_and_legacy(tmp_path):
    (tmp_path / "a.exe").write_bytes(b"MZ")
    (tmp_path / "a.doc").write_bytes(b"x")
    with pytest.raises(UploadError, match="unsupported"):
        validate_file(tmp_path / "a.exe")
    with pytest.raises(UploadError, match=".docx"):
        validate_file(tmp_path / "a.doc")


def test_office_formats(tmp_path):
    import docx, openpyxl
    from pptx import Presentation
    d = docx.Document()
    d.add_paragraph("hi")
    d.save(tmp_path / "a.docx")
    wb = openpyxl.Workbook()
    wb.active.append(["a", "b"])
    wb.save(tmp_path / "a.xlsx")
    Presentation().save(tmp_path / "a.pptx")
    assert validate_file(tmp_path / "a.docx").source_type == "docx"
    assert validate_file(tmp_path / "a.xlsx").source_type == "xlsx"
    assert validate_file(tmp_path / "a.pptx").source_type == "pptx"


def test_zip_without_required_part(tmp_path):
    p = tmp_path / "a.docx"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("hello.txt", "x")
    with pytest.raises(UploadError):
        validate_file(p)


def test_text_formats(tmp_path):
    (tmp_path / "a.md").write_text("# Title\n\nbody")
    (tmp_path / "a.txt").write_text("plain")
    assert validate_file(tmp_path / "a.md").source_type == "md"
    assert validate_file(tmp_path / "a.txt").source_type == "txt"
    (tmp_path / "bin.txt").write_bytes(b"ab\x00cd" * 10)
    with pytest.raises(UploadError, match="binary"):
        validate_file(tmp_path / "bin.txt")


def test_image(tmp_path):
    from PIL import Image
    Image.new("RGB", (50, 50)).save(tmp_path / "a.png")
    assert validate_file(tmp_path / "a.png").source_type == "image"
    (tmp_path / "fake.png").write_bytes(b"nope")
    with pytest.raises(UploadError):
        validate_file(tmp_path / "fake.png")


# ---------- labels, ids, headers ----------
@pytest.mark.parametrize("text,expected", [
    ("Fig. 3: loss curves", "Figure 3"), ("Figure 12a shows", "Figure 12a"),
    ("table 2 results", "Table 2"), ("no label here", None)])
def test_find_label(text, expected):
    assert find_label(text) == expected


def _chunk(**kw):
    base = dict(paper_id="p1", paper_title="T", kind=ChunkKind.TEXT, text="hello", page=3)
    return Chunk(**{**base, **kw})


def test_id_deterministic_and_paper_specific():
    assert chunk_id(_chunk()) == chunk_id(_chunk())
    assert chunk_id(_chunk()) != chunk_id(_chunk(paper_id="p2"))
    assert chunk_id(_chunk()) != chunk_id(_chunk(chunk_index=1))


def test_header_format():
    c = _chunk(kind=ChunkKind.FIGURE, section="3.2 Method", figure_label="Figure 2")
    assert header(c) == "[Paper: T | Section: 3.2 Method | Page 3 | Type: figure | Figure 2]"
    c = contextualize(c)
    assert c.contextualized.endswith("hello") and c.id


# ---------- tables ----------
def test_table_kept_whole_when_small():
    md = "| a | b |\n|---|---|\n| 1 | 2 |"
    assert split_markdown_table(md, 100, words) == [md]


def test_table_split_repeats_header():
    rows = "\n".join(f"| r{i} | v{i} |" for i in range(40))
    md = f"| name | value |\n|---|---|\n{rows}"
    parts = split_markdown_table(md, 30, words)
    assert len(parts) > 1
    for part in parts:
        assert part.startswith("| name | value |\n|---|---|")
        assert words(part) <= 30
    assert sum(p.count("| r") for p in parts) == 40


def test_split_text_respects_limit():
    text = "\n\n".join("word " * 40 for _ in range(10))
    pieces = split_text(text, 100, words)
    assert len(pieces) > 1 and all(words(p) <= 100 for p in pieces)


# ---------- fallbacks ----------
def test_pdf_fallback_has_pages_and_figure(tmp_path, monkeypatch):
    from researchiq.config import settings
    monkeypatch.setattr(settings, "cache_dir", tmp_path / "cache")
    chunks, visuals = fallback_pdf(_pdf(tmp_path / "a.pdf", pages=2), "p1", "T", count=words)
    assert {c.page for c in chunks} == {1, 2}
    assert all(c.parser == "pymupdf" and c.source_type == "pdf" for c in chunks)
    assert any(v.label == "Figure 1" for v in visuals)


def test_xlsx_fallback_tables_with_sheet_location(tmp_path):
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Results"
    ws.append(["model", "acc"])
    ws.append(["a", 0.9])
    wb.save(tmp_path / "a.xlsx")
    chunks, visuals = fallback_office(tmp_path / "a.xlsx", "p1", "T", "xlsx", count=words)
    assert visuals == []
    assert chunks[0].kind == ChunkKind.TABLE and chunks[0].location == "Sheet: Results"
    assert "| model | acc |" in chunks[0].text


def test_docx_and_md_fallback(tmp_path):
    import docx
    d = docx.Document()
    d.add_heading("Intro", 1)
    d.add_paragraph("some body text")
    d.save(tmp_path / "a.docx")
    chunks, _ = fallback_office(tmp_path / "a.docx", "p1", "T", "docx", count=words)
    assert chunks[0].section == "Intro" and chunks[0].source_type == "docx"
    (tmp_path / "a.md").write_text("# One\n\nalpha\n\n# Two\n\nbeta")
    chunks, _ = fallback_text(tmp_path / "a.md", "p1", "T", "md", count=words)
    assert [c.section for c in chunks] == ["One", "Two"]
