"""A5 · VLM describe (Groq): figure/table image -> text, cached in figures.json.

Every request is sized to fit the token budget before it is sent.
"""
import base64
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from researchiq.config import settings
from researchiq.llm import budget
from researchiq.llm.groq_client import describe_image, limiter
from researchiq.schemas import Chunk, ChunkKind, VisualElement

log = logging.getLogger(__name__)

PROMPT = """You are describing a {kind} from a research document for a search index.
First say whether it is an architecture diagram, a chart, a table or something else.
Then describe components and flow, axes, legend, trends and key numbers.
Say "unreadable" if it is unclear. Do not invent values you cannot read.
Caption: {caption}
Surrounding text: {nearby}"""


@dataclass
class VlmStats:
    calls: int = 0
    cached: int = 0
    skipped: int = 0
    tokens: int = 0
    errors: list[str] = field(default_factory=list)


def _load_cache() -> dict[str, dict]:
    p = settings.figures_cache
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    # tolerate the old {key: "text"} layout
    return {k: (v if isinstance(v, dict) else {"description": v}) for k, v in data.items()}


def _save_cache(cache: dict) -> None:
    settings.figures_cache.parent.mkdir(parents=True, exist_ok=True)
    settings.figures_cache.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def build_request(el: VisualElement) -> tuple[str, str, int]:
    """Returns (prompt, image_b64, estimated_total_tokens), guaranteed <= the request cap."""
    img = budget.fit_image(Image.open(el.image_path))
    img_tok = budget.image_tokens(*img.size)
    nearby = budget.truncate_to_tokens(el.nearby_text, settings.vlm_text_tokens)
    caption = budget.truncate_to_tokens(el.caption, settings.vlm_text_tokens)
    cap = min(settings.per_request_token_cap, limiter.tpm_cap)

    def total(n: str) -> int:
        return budget.estimate_request(PROMPT.format(kind=el.kind, caption=caption, nearby=n),
                                       img_tok, settings.vlm_max_output_tokens)

    if total(nearby) > cap:
        nearby = ""                                   # 1) drop surrounding text first
    while total(nearby) > cap and img_tok > 256:      # 2) then shrink the image
        img = budget.fit_image(img, int(img_tok * 0.8))
        img_tok = budget.image_tokens(*img.size)
    if total(nearby) > cap:
        raise ValueError("visual cannot fit the token budget")
    b64 = base64.b64encode(budget.png_bytes(img)).decode()
    return PROMPT.format(kind=el.kind, caption=caption, nearby=nearby), b64, total(nearby)


def describe_visuals(elements: list[VisualElement], paper_title: str, *, describe: bool = True,
                     max_calls: int | None = None, source_type: str = "pdf",
                     stats: VlmStats | None = None) -> list[Chunk]:
    stats = stats if stats is not None else VlmStats()
    cache = _load_cache()
    chunks: list[Chunk] = []
    for el in elements:
        key = hashlib.md5(Path(el.image_path).read_bytes()).hexdigest()
        desc = cache.get(key, {}).get("description")
        if desc:
            stats.cached += 1
        elif describe and (max_calls is None or stats.calls < max_calls):
            try:
                prompt, b64, est = build_request(el)
                before = limiter.run_tokens
                desc = describe_image(prompt, b64, image_tokens=est,
                                      max_tokens=settings.vlm_max_output_tokens)
                stats.calls += 1
                stats.tokens += limiter.run_tokens - before
                cache[key] = {"description": desc, "paper_id": el.paper_id, "page": el.page,
                              "label": el.label, "model": settings.llm_model}
                _save_cache(cache)  # per item: a crash never loses paid calls
            except Exception as e:  # never fail ingestion because of the VLM
                log.warning("VLM failed for %s p.%s: %s", el.label or el.kind, el.page, e)
                stats.errors.append(f"{el.label or el.kind} p.{el.page}")
                desc = None
        else:
            stats.skipped += 1
        # No description: caption + nearby text still makes the visual searchable (not cached).
        body = desc or el.nearby_text
        text = "\n".join(x for x in (el.caption, body) if x).strip() or f"{el.kind} on page {el.page}"
        chunks.append(Chunk(
            paper_id=el.paper_id, paper_title=paper_title,
            kind=ChunkKind.FIGURE if el.kind == "figure" else ChunkKind.TABLE,
            text=text, caption=el.caption or None, page=el.page, figure_label=el.label,
            image_path=el.image_path, source_type=source_type))
    return chunks
