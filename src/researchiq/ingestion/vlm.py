"""A5 · VLM describe (Groq): figure/table image -> text, cached in figures.json."""
import hashlib
import json

from researchiq.config import settings
from researchiq.ingestion.visual import load_b64
from researchiq.llm.groq_client import describe_image
from researchiq.schemas import Chunk, ChunkKind, VisualElement

PROMPT = """You are describing a {kind} from a research paper for a search index.
Caption: {caption}
Surrounding text: {nearby}

Describe what it shows: axes, variables, trends, key numbers, and the main takeaway.
Be factual and concise. Do not invent values you cannot read."""


def _load_cache() -> dict[str, str]:
    p = settings.figures_cache
    return json.loads(p.read_text()) if p.exists() else {}


def _save_cache(cache: dict[str, str]) -> None:
    settings.figures_cache.parent.mkdir(parents=True, exist_ok=True)
    settings.figures_cache.write_text(json.dumps(cache, indent=2))


def describe_visuals(elements: list[VisualElement], paper_title: str) -> list[Chunk]:
    cache = _load_cache()
    chunks: list[Chunk] = []
    for el in elements:
        img_b64 = load_b64(el.image_path)
        key = hashlib.md5(img_b64.encode()).hexdigest()
        if key not in cache:
            prompt = PROMPT.format(kind=el.kind, caption=el.caption, nearby=el.nearby_text)
            cache[key] = describe_image(prompt, img_b64)
            _save_cache(cache)  # save per item so a crash doesn't lose paid calls
        chunks.append(Chunk(
            paper_id=el.paper_id,
            paper_title=paper_title,
            kind=ChunkKind.FIGURE if el.kind == "figure" else ChunkKind.TABLE,
            text=f"{el.caption}\n{cache[key]}",
            page=el.page,
            figure_label=el.label,
            image_path=el.image_path,
        ))
    return chunks
