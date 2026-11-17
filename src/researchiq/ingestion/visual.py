"""A4 · Visual path: figure / table crop with caption, label, page, nearby text."""
import base64
import re
from pathlib import Path

from docling_core.types.doc import DoclingDocument

from researchiq.config import settings
from researchiq.schemas import VisualElement

_LABEL = re.compile(r"\b(Figure|Fig\.?|Table)\s*(\d+[A-Za-z]?)", re.I)
_CAPTION_START = re.compile(r"\s*(Figure|Fig\.?)\s*(\d+[A-Za-z]?)\s*[:.]", re.I)


def find_label(text: str) -> str | None:
    """'Fig. 3: ...' -> 'Figure 3', 'table 2' -> 'Table 2'."""
    m = _LABEL.search(text or "")
    if not m:
        return None
    word = "Table" if m.group(1).lower() == "table" else "Figure"
    return f"{word} {m.group(2)}"


def _page(item) -> int:
    return max(item.prov[0].page_no, 1) if getattr(item, "prov", None) else 1


def nearby_text(doc: DoclingDocument, page: int, label: str | None, max_chars: int = 800) -> str:
    """Prefer paragraphs that mention the label, then fill with same-page text."""
    paras = [t.text for t in doc.texts if t.prov and t.prov[0].page_no == page and len(t.text) > 40]
    if label:
        num = label.split()[-1]
        pat = re.compile(rf"\b(Figure|Fig\.?|Table)\s*{re.escape(num)}\b", re.I)
        hits = [p for p in paras if pat.search(p)]
        paras = hits + [p for p in paras if p not in hits]
    return " ".join(paras)[:max_chars]


def _big_enough(img) -> bool:
    return img.width >= settings.min_image_px and img.height >= settings.min_image_px


def extract_visuals(doc: DoclingDocument, paper_id: str, source_type: str = "pdf",
                    path: Path | None = None) -> list[VisualElement]:
    out_dir = settings.cache_dir / "crops" / paper_id
    out_dir.mkdir(parents=True, exist_ok=True)
    elements: list[VisualElement] = []
    seen_figure_labels: set[tuple[int, str]] = set()

    kinds = [("figure", doc.pictures)]
    # Spreadsheet / CSV tables are exact text already: never spend tokens on them.
    if settings.vlm_describe_tables and source_type not in ("xlsx", "csv"):
        kinds.append(("table", doc.tables))

    for kind, items in kinds:
        for i, item in enumerate(items):
            try:
                img = item.get_image(doc)
            except Exception:
                img = None
            if img is None or not _big_enough(img):
                continue
            caption = item.caption_text(doc) or ""
            label = find_label(caption)
            page = _page(item)
            p = out_dir / f"{kind}_{i}.png"
            img.save(p)
            if kind == "figure" and label:
                seen_figure_labels.add((page, label))
            elements.append(VisualElement(
                paper_id=paper_id, kind=kind, label=label, caption=caption, page=page,
                nearby_text=nearby_text(doc, page, label), image_path=str(p)))

    if source_type == "pdf" and path is not None:
        elements += _render_missed_figures(doc, paper_id, Path(path), out_dir, seen_figure_labels)
    return elements


def _render_missed_figures(doc, paper_id, path: Path, out_dir: Path,
                           seen: set[tuple[int, str]]) -> list[VisualElement]:
    """Vector-drawn figures Docling missed: caption on the page, no picture item -> render the page."""
    import pymupdf as fitz

    found: dict[tuple[int, str], str] = {}
    for t in doc.texts:
        if not t.prov:
            continue
        m = _CAPTION_START.match(t.text)
        if m:
            key = (t.prov[0].page_no, f"Figure {m.group(2)}")
            if key not in seen:
                found[key] = t.text
    if not found:
        return []
    out = []
    with fitz.open(path) as pdf:
        for (page, label), caption in found.items():
            pix = pdf[page - 1].get_pixmap(dpi=settings.render_dpi)
            p = out_dir / f"page_{page}_{label.replace(' ', '_')}.png"
            pix.save(p)
            out.append(VisualElement(
                paper_id=paper_id, kind="figure", label=label, caption=caption, page=page,
                nearby_text=nearby_text(doc, page, label), image_path=str(p)))
    return out


def load_b64(path: str | Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()
