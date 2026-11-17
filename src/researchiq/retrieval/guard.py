"""B1 · Question guard: clean input, length cap, prompt-injection rejection."""
import re
import unicodedata

from researchiq.config import settings


class InvalidQuestion(ValueError):
    pass


class PromptInjection(InvalidQuestion):
    pass


_VERBS = r"(ignore|disregard|forget|override|bypass|skip)"
_INJECTION = [re.compile(p, re.I) for p in (
    rf"\b{_VERBS}\b[^.?!]{{0,40}}\b(previous|prior|above|earlier|all|any|your|the|these|those)\b"
    rf"[^.?!]{{0,40}}\b(instructions?|rules?|prompts?|guidelines?|directives?)\b",
    r"\byou\s+are\s+now\b",
    r"\bact\s+as\b[^.?!]{0,30}\b(dan|jailbroken|unrestricted|unfiltered|no\s+restrictions)\b",
    r"\b(reveal|show|print|repeat|output|leak|display)\b[^.?!]{0,30}\b(system\s+prompt|your\s+(instructions|prompt|rules))\b",
    r"\b(developer|god|dan)\s+mode\b",
    r"\bjailbreak(ed|ing)?\b",
    r"</?\s*(system|assistant|source|sources|question|instructions?)\s*>",
    r"^\s*(system|assistant)\s*:",
)]


def clean_question(q: str) -> str:
    q = unicodedata.normalize("NFKC", q)
    q = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", q)  # control chars
    q = re.sub(r"\s+", " ", q).strip()
    if not q:
        raise InvalidQuestion("Empty question")
    if len(q) > settings.max_question_chars:
        raise InvalidQuestion(f"Question longer than {settings.max_question_chars} chars")
    return q


def is_injection(q: str) -> bool:
    return any(p.search(q) for p in _INJECTION)


def check_question(q: str) -> str:
    """Clean, cap and reject prompt-injection attempts. The error never echoes the match."""
    q = clean_question(q)
    if is_injection(q):
        raise PromptInjection("This question looks like an attempt to override the assistant's "
                              "instructions, so it was not processed. Please ask about the papers.")
    return q
