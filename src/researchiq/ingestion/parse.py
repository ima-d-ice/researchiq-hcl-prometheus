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
    """Docling TITLE item -> file metadata -> file name."""
    for item in doc.texts:
        if item.label == DocItemLabel.TITLE and item.text.strip():
            return item.text.strip()[:200]
    meta = _metadata_title(Path(path))
    return meta or Path(path).stem


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
