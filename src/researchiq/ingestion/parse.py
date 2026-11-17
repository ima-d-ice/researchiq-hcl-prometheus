"""A2 · Docling parse: PDF, DOCX, PPTX, XLSX, MD, CSV, HTML and images -> DoclingDocument."""
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument, DocItemLabel


@lru_cache(maxsize=1)
def build_converter() -> DocumentConverter:
    opts = PdfPipelineOptions()
    opts.generate_picture_images = True  # needed for figure crops
    opts.generate_table_images = True    # needed for table crops
    opts.images_scale = 2.0
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF, InputFormat.DOCX, InputFormat.PPTX, InputFormat.XLSX,
                         InputFormat.MD, InputFormat.CSV, InputFormat.HTML, InputFormat.IMAGE],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)},
    )


def parse_file(path: Path, converter: DocumentConverter | None = None) -> DoclingDocument:
    converter = converter or build_converter()
    path = Path(path)
    if path.suffix.lower() == ".txt":  # Docling has no plain-text backend: read it as Markdown
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d) / (path.stem + ".md")
            shutil.copyfile(path, tmp)
            return converter.convert(tmp).document
    return converter.convert(path).document


parse_pdf = parse_file  # backwards-compatible name


def extract_title(doc: DoclingDocument, path: Path) -> str:
    """Docling TITLE item -> largest text at the top of PDF page 1 -> file metadata -> file name."""
    for item in doc.texts:
        if item.label == DocItemLabel.TITLE and item.text.strip():
            return item.text.strip()[:250]
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        title = pdf_font_title(path)
        if title:
            return title
    meta = _metadata_title(path)
    return meta or path.stem


def pdf_font_title(path: Path) -> str:
    """Paper titles are the largest horizontal text in the top part of page 1."""
    try:
        import pymupdf as fitz
        with fitz.open(path) as d:
            if d.page_count == 0:
                return ""
            page = d[0]
            limit = page.rect.height * 0.4
            lines = []  # (font size, y, text)
            for block in page.get_text("dict").get("blocks", []):
                for line in block.get("lines", []):
                    if tuple(round(x) for x in line.get("dir", (1, 0))) != (1, 0):
                        continue  # skip rotated text such as the arXiv side stamp
                    spans = [s for s in line.get("spans", []) if s["text"].strip()]
                    if not spans or line["bbox"][1] > limit:
                        continue
                    text = " ".join(s["text"].strip() for s in spans)
                    lines.append((max(s["size"] for s in spans), line["bbox"][1], text))
    except Exception:
        return ""
    if not lines:
        return ""
    top = max(size for size, _, text in lines if len(text) >= 3) if any(len(t) >= 3 for _, _, t in lines) else 0
    title = " ".join(t for size, _, t in sorted((l for l in lines if l[0] >= top - 0.5), key=lambda l: l[1]))
    title = " ".join(title.split())
    return title[:250] if 10 <= len(title) and any(c.isalpha() for c in title) else ""


def _metadata_title(path: Path) -> str:
    try:
        ext = path.suffix.lower()
        if ext == ".pdf":
            import pymupdf as fitz
            with fitz.open(path) as d:
                return (d.metadata or {}).get("title", "").strip()[:200]
        if ext == ".docx":
            import docx
            return (docx.Document(str(path)).core_properties.title or "").strip()[:200]
    except Exception:
        pass
    return ""
