"""A6 · Contextualize + ID: header prefix, deterministic MD5 UUID."""
import hashlib
import uuid

from researchiq.schemas import Chunk


def chunk_id(chunk: Chunk) -> str:
    """Stable ID so re-ingesting the same paper upserts instead of duplicating."""
    key = f"{chunk.paper_id}|{chunk.page}|{chunk.kind.value}|{chunk.text}"
    return str(uuid.UUID(hashlib.md5(key.encode()).hexdigest()))


def contextualize(chunk: Chunk) -> Chunk:
    header = " > ".join([chunk.paper_title, *chunk.headings])
    if chunk.figure_label:
        header += f" > {chunk.figure_label}"
    chunk.contextualized = f"{header}\n\n{chunk.text}" if header else chunk.text
    chunk.id = chunk_id(chunk)
    return chunk
