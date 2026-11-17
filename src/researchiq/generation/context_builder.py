"""C1 · Context builder: tagged sources + token budget."""
import tiktoken

from researchiq.config import settings
from researchiq.schemas import ScoredChunk

_enc = tiktoken.get_encoding("cl100k_base")  # approximation of the Qwen tokenizer


def _fmt(tag: str, sc: ScoredChunk) -> str:
    c = sc.chunk
    loc = f"{c.paper_title}, p.{c.page}" + (f", {c.figure_label}" if c.figure_label else "")
    return f"[{tag}] ({loc}) [{c.kind.value}]\n{c.text}"


def build_context(chunks: list[ScoredChunk], budget: int | None = None) -> tuple[str, dict[str, ScoredChunk]]:
    """Returns (context string, tag -> chunk map). Drops lowest-ranked chunks over budget."""
    budget = budget or settings.context_token_budget
    blocks, sources, used = [], {}, 0
    for i, sc in enumerate(chunks, start=1):
        tag = f"S{i}"
        block = _fmt(tag, sc)
        n = len(_enc.encode(block))
        if used + n > budget:
            break
        blocks.append(block)
        sources[tag] = sc
        used += n
    return "\n\n".join(blocks), sources
