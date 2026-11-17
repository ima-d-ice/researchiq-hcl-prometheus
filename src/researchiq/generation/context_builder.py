"""C1 · Context builder: tagged <source> blocks inside a hard token budget (HLD 4.1 / 4.2)."""
import tiktoken

from researchiq.config import settings
from researchiq.generation.prompts import SYSTEM
from researchiq.schemas import Route, ScoredChunk

_enc = tiktoken.get_encoding("cl100k_base")  # approximation of the Qwen tokenizer
_QUESTION_RESERVE = 150                      # question + user-template overhead


def _n(text: str) -> int:
    return len(_enc.encode(text))


def _esc(text: str) -> str:
    """Stop a source from closing its own tag or opening a fake one."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _attr(text: str) -> str:
    return _esc(text).replace('"', "'")


def max_output_tokens(route: Route | str | None) -> int:
    r = route.value if isinstance(route, Route) else route
    return settings.gen_max_tokens_compare if r in ("compare", "synthesis") else settings.gen_max_tokens_general


def context_budget(route: Route | str | None = None, question: str = "") -> int:
    """Tokens left for source text after system prompt, question and the output reserve."""
    cap = min(settings.gen_request_cap, int(settings.groq_tpm_limit * 0.9))
    system = max(settings.gen_system_tokens, _n(SYSTEM))
    return max(cap - system - max(_QUESTION_RESERVE, _n(question) + 50) - max_output_tokens(route), 0)


def _truncate(text: str, limit: int) -> str:
    ids = _enc.encode(text)
    return text if len(ids) <= limit else _enc.decode(ids[:limit]) + " …"


def _fmt(tag: str, sc: ScoredChunk, chunk_tokens: int) -> str:
    c = sc.chunk
    ref = f' ref="{_attr(c.figure_label)}"' if c.figure_label else ""
    return (f'<source id="{tag}" paper="{_attr(c.paper_id)}: {_attr(c.paper_title)}" page="{c.page}" '
            f'type="{c.kind.value}"{ref}>\n{_esc(_truncate(c.text, chunk_tokens))}\n</source>')


def _edge_order(items: list) -> list:
    """Best first, second-best last: models attend most to the start and the end."""
    return items if len(items) < 3 else [items[0], *items[2:], items[1]]


def build_context(chunks: list[ScoredChunk], budget: int | None = None, route: Route | str | None = None,
                  question: str = "") -> tuple[str, dict[str, ScoredChunk]]:
    """Returns (context string, tag -> chunk). `chunks` must be ranked best-first.

    Keeps chunks in rank order while they fit, so over budget the LOWEST-ranked chunks are dropped
    first (never the system prompt). Tags S1.. follow the final display order.
    """
    budget = budget if budget is not None else context_budget(route, question)
    chunk_tokens = settings.gen_chunk_tokens if route not in (Route.COMPARE, "compare") else 350
    kept, used = [], 0
    for sc in chunks:
        n = _n(_fmt("S00", sc, chunk_tokens))
        if used + n > budget:
            break
        kept.append(sc)
        used += n
    sources, blocks = {}, []
    for i, sc in enumerate(_edge_order(kept), start=1):
        tag = f"S{i}"
        sources[tag] = sc
        blocks.append(_fmt(tag, sc, chunk_tokens))
    return "\n".join(blocks), sources
