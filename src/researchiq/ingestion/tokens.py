"""BGE token counting (same tokenizer as the dense embedder)."""
from functools import lru_cache

from researchiq.config import settings


@lru_cache(maxsize=1)
def bge_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(settings.dense_model)


def count_tokens(text: str) -> int:
    return len(bge_tokenizer().encode(text, add_special_tokens=False))
