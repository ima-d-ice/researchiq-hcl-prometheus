"""Shared data models passed between pipelines."""
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ChunkKind(str, Enum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"


class Chunk(BaseModel):
    """Unit stored in Qdrant (payload) and passed to generation."""
    id: str = ""                      # MD5-derived UUID, set in contextualize
    paper_id: str
    paper_title: str = ""
    kind: ChunkKind
    text: str                         # raw text, or VLM description for figures
    contextualized: str = ""          # header-prefixed text that gets embedded
    headings: list[str] = []
    page: int
    figure_label: str | None = None   # e.g. "Figure 3", "Table 2"
    image_path: str | None = None
    # Added by ingestion (all optional so older payloads still load)
    chunk_index: int = 0
    section: str = ""
    caption: str | None = None
    parser: str = "docling"
    source_type: str = "pdf"
    location: str | None = None       # "Slide 4", "Sheet: Results"
    year: int | None = None
    source_url: str | None = None


class VisualElement(BaseModel):
    """Figure or table crop handed to the VLM."""
    paper_id: str
    kind: Literal["figure", "table"]
    label: str | None
    caption: str
    page: int
    nearby_text: str
    image_path: str


class Route(str, Enum):
    GENERAL = "general"
    COMPARE = "compare"
    FIGURE = "figure"
    SYNTHESIS = "synthesis"


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float


class Claim(BaseModel):
    text: str
    sources: list[str] = Field(description="Source tags like 'S1', 'S3'")


class LLMAnswer(BaseModel):
    """Structure the LLM must return in JSON mode."""
    answer: str
    claims: list[Claim]
    insufficient_evidence: bool = False


class Citation(BaseModel):
    tag: str
    paper_title: str
    page: int
    figure_label: str | None = None
    snippet: str
    supported: bool


class FinalAnswer(BaseModel):
    question: str
    route: Route
    answer: str
    citations: list[Citation]
    warnings: list[str] = []
