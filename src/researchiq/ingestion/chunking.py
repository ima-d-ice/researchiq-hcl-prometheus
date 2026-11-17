"""A3 · Text path: HybridChunker (≤512 tokens incl. header, merge small neighbours, tables whole)."""
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc import DocItemLabel, DoclingDocument

from researchiq.config import settings
from researchiq.ingestion.tables import split_markdown_table
from researchiq.ingestion.tokens import bge_tokenizer, count_tokens
from researchiq.schemas import Chunk, ChunkKind


def _budget() -> int:
    return settings.chunk_max_tokens - settings.chunk_header_reserve


def _page(item) -> int:
    return max(item.prov[0].page_no, 1) if getattr(item, "prov", None) else 1


def _location(source_type: str, page: int) -> str | None:
    return {"pptx": f"Slide {page}", "xlsx": f"Sheet {page}"}.get(source_type)


def chunk_text(doc: DoclingDocument, paper_id: str, paper_title: str,
               source_type: str = "pdf") -> list[Chunk]:
    chunker = HybridChunker(
        tokenizer=HuggingFaceTokenizer(tokenizer=bge_tokenizer(), max_tokens=_budget()),
        merge_peers=True,
    )
    chunks: list[Chunk] = []
    for c in chunker.chunk(doc):
        items = c.meta.doc_items or []
        if items and all(i.label == DocItemLabel.TABLE for i in items):
            continue  # tables are handled whole by table_chunks()
        page = _page(items[0]) if items else 1
        headings = c.meta.headings or []
        chunks.append(Chunk(
            paper_id=paper_id, paper_title=paper_title, kind=ChunkKind.TEXT, text=c.text,
            headings=headings, section=headings[-1] if headings else "", page=page,
            source_type=source_type, location=_location(source_type, page),
        ))
    return chunks + table_chunks(doc, paper_id, paper_title, source_type)


def table_chunks(doc: DoclingDocument, paper_id: str, paper_title: str,
                 source_type: str = "pdf") -> list[Chunk]:
    out: list[Chunk] = []
    for t in doc.tables:
        md = t.export_to_markdown(doc=doc)
        if not md.strip():
            continue
        page = _page(t)
        caption = t.caption_text(doc) or None
        label = None
        if caption:
            from researchiq.ingestion.visual import find_label
            label = find_label(caption)
        for part in split_markdown_table(md, _budget(), count_tokens):
            out.append(Chunk(
                paper_id=paper_id, paper_title=paper_title, kind=ChunkKind.TABLE,
                text=(f"{caption}\n{part}" if caption else part), caption=caption,
                figure_label=label, page=page, source_type=source_type,
                location=_location(source_type, page),
            ))
    return out
