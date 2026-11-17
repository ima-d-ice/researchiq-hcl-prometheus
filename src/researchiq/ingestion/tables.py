"""Pure helpers: keep tables whole, or split by row groups repeating the header row."""
from typing import Callable


def split_markdown_table(md: str, max_tokens: int, count: Callable[[str], int]) -> list[str]:
    """Return one string if the table fits, else row groups each prefixed with the header."""
    lines = [l for l in md.strip().splitlines() if l.strip()]
    if count(md) <= max_tokens or len(lines) <= 3:
        return [md.strip()]
    # Markdown table: line 0 = header, line 1 = separator, rest = rows
    header = lines[:2]
    rows = lines[2:]
    base = count("\n".join(header))
    parts: list[list[str]] = []
    cur: list[str] = []
    used = base
    for row in rows:
        n = count(row) + 1
        if cur and used + n > max_tokens:
            parts.append(cur)
            cur, used = [], base
        cur.append(row)
        used += n
    if cur:
        parts.append(cur)
    return ["\n".join(header + p) for p in parts]
