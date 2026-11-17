"""A2 · Docling parse: text, tables, figures, pages."""
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument


def build_converter() -> DocumentConverter:
    opts = PdfPipelineOptions()
    opts.generate_picture_images = True  # needed for figure crops
    opts.generate_table_images = True    # needed for table crops
    opts.images_scale = 2.0
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


def parse_pdf(path: Path, converter: DocumentConverter | None = None) -> DoclingDocument:
    converter = converter or build_converter()
    return converter.convert(path).document
