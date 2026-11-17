"""A6 · Contextualize + ID: header prefix, deterministic MD5 UUID."""
import hashlib
import uuid

from researchiq.schemas import Chunk


def chunk_id(chunk: Chunk) -> str:
    """UUID(MD5(paper_id + page + chunk_index + text)): re-ingesting upserts, never duplicates."""
    key = f"{chunk.paper_id}|{chunk.page}|{chunk.chunk_index}|{chunk.text}"
    return str(uuid.UUID(hashlib.md5(key.encode()).hexdigest()))


def header(chunk: Chunk) -> str:
    parts = [f"Paper: {chunk.paper_title}"]
    if chunk.section:
        parts.append(f"Section: {chunk.section}")
    parts.append(f"Page {chunk.page}")
    parts.append(f"Type: {chunk.kind.value}")
    if chunk.figure_label:
        parts.append(chunk.figure_label)
    if chunk.location:
        parts.append(chunk.location)
    return "[" + " | ".join(parts) + "]"


def contextualize(chunk: Chunk) -> Chunk:
    chunk.contextualized = f"{header(chunk)} {chunk.text}"
    chunk.id = chunk_id(chunk)
    return chunk
