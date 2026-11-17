"""B1 · Question guard: length cap, clean input."""
import re
import unicodedata

from researchiq.config import settings


class InvalidQuestion(ValueError):
    pass


def clean_question(q: str) -> str:
    q = unicodedata.normalize("NFKC", q)
    q = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", q)  # control chars
    q = re.sub(r"\s+", " ", q).strip()
    if not q:
        raise InvalidQuestion("Empty question")
    if len(q) > settings.max_question_chars:
        raise InvalidQuestion(f"Question longer than {settings.max_question_chars} chars")
    return q
