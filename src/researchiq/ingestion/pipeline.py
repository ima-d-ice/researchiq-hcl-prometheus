"""Lane A orchestration: one PDF in, chunks indexed in Qdrant."""
import hashlib
from pathlib import Path

from researchiq.ingestion.chunking import chunk_text
from researchiq.ingestion.contextualize import contextualize
from researchiq.ingestion.index import upsert_chunks
from researchiq.ingestion.parse import parse_pdf
from researchiq.ingestion.upload import validate_pdf
from researchiq.ingestion.visual import extract_visuals
from researchiq.ingestion.vlm import describe_visuals


def ingest_pdf(path: Path) -> int:
    validate_pdf(path)
    paper_id = hashlib.md5(path.read_bytes()).hexdigest()[:12]
    doc = parse_pdf(path)
    title = doc.name or path.stem  # TODO: better title extraction

    text_chunks = chunk_text(doc, paper_id, title)
    visual_chunks = describe_visuals(extract_visuals(doc, paper_id), title)

    chunks = [contextualize(c) for c in text_chunks + visual_chunks]
    upsert_chunks(chunks)
    return len(chunks)
