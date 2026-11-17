"""Lane A orchestration: one file in, chunks indexed in Qdrant."""
import csv
import hashlib
import logging
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from researchiq.config import settings
from researchiq.ingestion.contextualize import contextualize
from researchiq.ingestion.upload import UploadError, validate_file
from researchiq.ingestion.vlm import VlmStats, describe_visuals
from researchiq.llm.groq_client import limiter
from researchiq.schemas import Chunk, VisualElement

log = logging.getLogger(__name__)


class IngestError(RuntimeError):
    """Message is safe to show to the user; details are in the logs."""


@dataclass
class IngestReport:
    paper_id: str
    title: str
    source_type: str
    parser: str
    counts: dict[str, int] = field(default_factory=dict)
    vlm_calls: int = 0
    vlm_cached: int = 0
    vlm_skipped: int = 0
    groq_tokens: int = 0

    @property
    def n_chunks(self) -> int:
        return sum(self.counts.values())


def load_catalogue(path: Path | None = None) -> dict[str, dict]:
    """catalogue.csv: filename,paper_id,title,year,source_url (all optional except filename)."""
    path = path or settings.catalogue_path
    if not Path(path).exists():
        return {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {r["filename"].strip(): r for r in csv.DictReader(f) if r.get("filename")}


def _image_visual(path: Path, paper_id: str) -> list[VisualElement]:
    out_dir = settings.cache_dir / "crops" / paper_id
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"image{path.suffix.lower()}"
    shutil.copyfile(path, dest)
    return [VisualElement(paper_id=paper_id, kind="figure", label=None, caption=path.stem,
                          page=1, nearby_text="", image_path=str(dest))]


def _extract(path: Path, source_type: str, paper_id: str, title_hint: str | None):
    """Returns (text/table chunks, visual elements, title, parser)."""
    if source_type == "image":
        return [], _image_visual(path, paper_id), title_hint or path.stem, "image"
    try:
        from researchiq.ingestion.chunking import chunk_text
        from researchiq.ingestion.parse import extract_title, parse_file
        from researchiq.ingestion.visual import extract_visuals

        doc = parse_file(path)
        title = title_hint or extract_title(doc, path)
        return (chunk_text(doc, paper_id, title, source_type),
                extract_visuals(doc, paper_id, source_type, path), title, "docling")
    except Exception as e:
        log.warning("Docling failed for %s (%s); using fallback parser", path.name, e, exc_info=True)

    from researchiq.ingestion import fallback
    title = title_hint or path.stem
    if source_type == "pdf":
        chunks, visuals = fallback.fallback_pdf(path, paper_id, title)
        parser = "pymupdf"
    elif source_type in ("docx", "pptx", "xlsx"):
        chunks, visuals = fallback.fallback_office(path, paper_id, title, source_type)
        parser = "office"
    else:
        chunks, visuals = fallback.fallback_text(path, paper_id, title, source_type)
        parser = "text"
    return chunks, visuals, title, parser


def ingest_file(path: Path, describe: bool = True, max_vlm_calls: int | None = None) -> IngestReport:
    path = Path(path)
    info = validate_file(path)  # UploadError propagates with a friendly message
    meta = load_catalogue().get(path.name, {})
    paper_id = (meta.get("paper_id") or "").strip() or hashlib.md5(path.read_bytes()).hexdigest()[:12]
    try:
        text_chunks, visuals, title, parser = _extract(
            path, info.source_type, paper_id, (meta.get("title") or "").strip() or None)
        stats = VlmStats()
        start_tokens = limiter.run_tokens
        visual_chunks = describe_visuals(visuals, title, describe=describe, max_calls=max_vlm_calls,
                                         source_type=info.source_type, stats=stats)
        chunks: list[Chunk] = text_chunks + visual_chunks
        if not chunks:
            raise IngestError(f"{path.name}: no readable content found")

        year = int(meta["year"]) if (meta.get("year") or "").strip().isdigit() else None
        for i, c in enumerate(chunks):
            c.chunk_index, c.parser = i, parser if c.parser == "docling" else c.parser
            c.year, c.source_url = year, (meta.get("source_url") or "").strip() or None
            c.paper_title = title
            contextualize(c)

        from researchiq.ingestion.index import delete_paper, upsert_chunks
        delete_paper(paper_id)  # replace_doc
        upsert_chunks(chunks)
        from researchiq.retrieval.rewrite import list_papers
        list_papers.cache_clear()  # new paper -> refresh the rewrite step's paper list
    except (UploadError, IngestError):
        raise
    except Exception as e:
        log.exception("Ingestion failed for %s", path.name)
        raise IngestError(f"{path.name}: could not be processed ({type(e).__name__})") from None

    return IngestReport(
        paper_id=paper_id, title=title, source_type=info.source_type, parser=parser,
        counts=dict(Counter(c.kind.value for c in chunks)), vlm_calls=stats.calls,
        vlm_cached=stats.cached, vlm_skipped=stats.skipped,
        groq_tokens=limiter.run_tokens - start_tokens)


def ingest_pdf(path: Path) -> int:
    """Backwards-compatible wrapper used by the UI: returns the chunk count."""
    return ingest_file(path).n_chunks
