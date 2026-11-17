"""A2 fallback: lightweight extraction when Docling fails. Same Chunk / VisualElement schema."""
import re
from pathlib import Path
from typing import Callable

from researchiq.config import settings
from researchiq.ingestion.tables import split_markdown_table
from researchiq.ingestion.visual import find_label
from researchiq.schemas import Chunk, ChunkKind, VisualElement

Count = Callable[[str], int]
_CAPTION = re.compile(r"^\s*(Figure|Fig\.?|Table)\s*\d+[A-Za-z]?\s*[:.]", re.I | re.M)
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def _default_count(text: str) -> int:
    from researchiq.ingestion.tokens import count_tokens
    return count_tokens(text)


def split_text(text: str, max_tokens: int, count: Count | None = None) -> list[str]:
    """Pack paragraphs into pieces of <= max_tokens; oversized paragraphs are split by words."""
    count = count or _default_count
    pieces, cur, used = [], [], 0
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if count(para) <= max_tokens:
            units.append(para)
            continue
        words, buf = para.split(), []
        for w in words:
            buf.append(w)
            if count(" ".join(buf)) > max_tokens and len(buf) > 1:
                units.append(" ".join(buf[:-1]))
                buf = [w]
        if buf:
            units.append(" ".join(buf))
    for u in units:
        n = count(u)
        if cur and used + n > max_tokens:
            pieces.append("\n\n".join(cur))
            cur, used = [], 0
        cur.append(u)
        used += n
    if cur:
        pieces.append("\n\n".join(cur))
    return pieces


def _budget() -> int:
    return settings.chunk_max_tokens - settings.chunk_header_reserve


def _mk(paper_id, title, text, page, source_type, parser, kind=ChunkKind.TEXT,
        section="", location=None, label=None) -> Chunk:
    return Chunk(paper_id=paper_id, paper_title=title, kind=kind, text=text, page=page,
                 section=section, headings=[section] if section else [], source_type=source_type,
                 parser=parser, location=location, figure_label=label)


def fallback_pdf(path: Path, paper_id: str, title: str,
                 count: Count | None = None) -> tuple[list[Chunk], list[VisualElement]]:
    import pymupdf as fitz

    chunks: list[Chunk] = []
    visuals: list[VisualElement] = []
    out_dir = settings.cache_dir / "crops" / paper_id
    with fitz.open(path) as pdf:
        for i, page in enumerate(pdf, start=1):
            text = page.get_text("text")
            for piece in split_text(text, _budget(), count):
                chunks.append(_mk(paper_id, title, piece, i, "pdf", "pymupdf"))
            for m in _CAPTION.finditer(text):
                caption = text[m.start(): m.start() + 300].strip().split("\n\n")[0]
                label = find_label(caption)
                if not label:
                    continue
                out_dir.mkdir(parents=True, exist_ok=True)
                p = out_dir / f"page_{i}_{label.replace(' ', '_')}.png"
                if not p.exists():
                    page.get_pixmap(dpi=settings.render_dpi).save(p)
                visuals.append(VisualElement(
                    paper_id=paper_id, kind="table" if label.startswith("Table") else "figure",
                    label=label, caption=caption, page=i, nearby_text=text[:800], image_path=str(p)))
    return chunks, visuals


def _rows_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    esc = lambda c: str(c).replace("|", "\\|").replace("\n", " ")  # noqa: E731
    lines = ["| " + " | ".join(esc(c) for c in rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def _table_chunks(rows, paper_id, title, page, source_type, location, count) -> list[Chunk]:
    count = count or _default_count
    md = _rows_to_markdown(rows)
    if not md:
        return []
    return [_mk(paper_id, title, part, page, source_type, "office", ChunkKind.TABLE, location=location)
            for part in split_markdown_table(md, _budget(), count)]


def fallback_office(path: Path, paper_id: str, title: str, source_type: str,
                    count: Count | None = None) -> tuple[list[Chunk], list[VisualElement]]:
    chunks: list[Chunk] = []
    if source_type == "docx":
        import docx
        d = docx.Document(str(path))
        section, buf = "", []
        for para in d.paragraphs:
            if para.style.name.lower().startswith("heading"):
                for piece in split_text("\n\n".join(buf), _budget(), count):
                    chunks.append(_mk(paper_id, title, piece, 1, "docx", "office", section=section))
                section, buf = para.text.strip(), []
            elif para.text.strip():
                buf.append(para.text.strip())
        for piece in split_text("\n\n".join(buf), _budget(), count):
            chunks.append(_mk(paper_id, title, piece, 1, "docx", "office", section=section))
        for t in d.tables:
            rows = [[c.text for c in r.cells] for r in t.rows]
            chunks += _table_chunks(rows, paper_id, title, 1, "docx", None, count)
    elif source_type == "pptx":
        from pptx import Presentation
        for i, slide in enumerate(Presentation(str(path)).slides, start=1):
            parts = [s.text_frame.text for s in slide.shapes if s.has_text_frame and s.text_frame.text.strip()]
            for piece in split_text("\n\n".join(parts), _budget(), count):
                chunks.append(_mk(paper_id, title, piece, i, "pptx", "office", location=f"Slide {i}"))
    elif source_type == "xlsx":
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        for i, ws in enumerate(wb.worksheets, start=1):
            rows = [["" if c is None else str(c) for c in r] for r in ws.iter_rows(values_only=True)]
            rows = [r for r in rows if any(r)]
            chunks += _table_chunks(rows, paper_id, title, i, "xlsx", f"Sheet: {ws.title}", count)
    return chunks, []


def fallback_text(path: Path, paper_id: str, title: str, source_type: str,
                  count: Count | None = None) -> tuple[list[Chunk], list[VisualElement]]:
    raw = path.read_text(encoding="utf-8-sig")
    if source_type == "html":
        raw = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
        raw = re.sub(r"(?i)</(p|div|h\d|li|tr)>", "\n\n", raw)
        raw = re.sub(r"<[^>]+>", " ", raw)
    chunks: list[Chunk] = []
    if source_type == "csv":
        import csv, io
        rows = list(csv.reader(io.StringIO(raw)))
        return _table_chunks(rows, paper_id, title, 1, "csv", None, count), []
    section, buf = "", []

    def flush():
        for piece in split_text("\n\n".join(buf), _budget(), count):
            chunks.append(_mk(paper_id, title, piece, 1, source_type, "text", section=section))

    for line in raw.splitlines():
        m = _HEADING.match(line)
        if m:
            flush()
            section, buf = m.group(2).strip(), []
        else:
            buf.append(line)
    flush()
    return chunks, []
