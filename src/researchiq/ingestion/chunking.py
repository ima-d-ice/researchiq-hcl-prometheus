"""A3 · Text path: HybridChunker (≤512 tokens, merge small neighbours, tables kept whole)."""
from docling.chunking import HybridChunker
from docling_core.types.doc import DoclingDocument

from researchiq.config import settings
from researchiq.schemas import Chunk, ChunkKind


def chunk_text(doc: DoclingDocument, paper_id: str, paper_title: str) -> list[Chunk]:
    chunker = HybridChunker(
        tokenizer=settings.dense_model,  # align token counts with the embedder
        max_tokens=settings.chunk_max_tokens,
        merge_peers=True,
    )
    chunks: list[Chunk] = []
    for c in chunker.chunk(doc):
        # TODO: detect table chunks (c.meta.doc_items labels) -> ChunkKind.TABLE
        page = c.meta.doc_items[0].prov[0].page_no if c.meta.doc_items else 0
        chunks.append(Chunk(
            paper_id=paper_id,
            paper_title=paper_title,
            kind=ChunkKind.TEXT,
            text=c.text,
            headings=c.meta.headings or [],
            page=page,
        ))
    return chunks
