"""Single entry point for every Groq API call (VLM describe, answer, judge).

Every call goes through the shared KeyPool: each API key has its own 8k tokens/minute
limiter, and a rate-limited key is rotated out automatically.
"""
import logging

from groq import Groq, RateLimitError

from researchiq.config import settings
from researchiq.llm.budget import text_tokens
from researchiq.llm.rate_limiter import KeyPool

log = logging.getLogger(__name__)

limiter = KeyPool(settings.all_groq_keys, settings.groq_tpm_limit,
                  settings.groq_rpm_limit, settings.groq_tpd_limit)
_clients: dict[int, Groq] = {}


def client(idx: int = 0) -> Groq:
    if idx not in _clients:
        # max_retries=0: on 429 we rotate to another key instead of retrying the same one
        _clients[idx] = Groq(api_key=limiter.keys[idx], max_retries=0)
    return _clients[idx]


def _retry_after(e: RateLimitError) -> float:
    try:
        return min(float(e.response.headers.get("retry-after", 30)), 120)
    except Exception:
        return 30.0


def _call(models: list[str], messages: list[dict], estimate: int, **kw) -> str:
    """Try each model in order; rotate keys on 429. Every attempt is metered."""
    last: Exception | None = None
    for model in models:
        for _ in range(len(limiter.keys) + settings.groq_max_retries):
            idx, handle = limiter.acquire(estimate)
            try:
                resp = client(idx).chat.completions.create(model=model, messages=messages, **kw)
                usage = getattr(resp, "usage", None)
                limiter.settle(idx, handle, getattr(usage, "total_tokens", None))
                return resp.choices[0].message.content or ""
            except RateLimitError as e:
                limiter.settle(idx, handle, estimate)  # assume the failed call counted
                limiter.cooldown(idx, _retry_after(e))
                last = e
            except Exception as e:
                limiter.settle(idx, handle, 0)
                log.warning("Groq call failed on %s: %s", model, type(e).__name__)
                last = e
                break  # not a rate limit: try the fallback model
    raise last  # type: ignore[misc]


def chat_json(system: str, user: str, max_tokens: int = 1200) -> str:
    """Text chat in JSON mode, thinking off. Returns raw JSON string."""
    # TODO: confirm how to disable thinking for this model on Groq
    est = text_tokens(system) + text_tokens(user) + max_tokens
    return _call(
        [settings.llm_model, settings.llm_fallback_model],
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        est,
        temperature=settings.llm_temperature,
        response_format={"type": "json_object"},
        max_tokens=max_tokens,
    )


def describe_image(prompt: str, image_b64: str, image_tokens: int = 1536,
                   max_tokens: int | None = None) -> str:
    """Vision call: image + prompt -> text description."""
    max_tokens = max_tokens or settings.vlm_max_output_tokens
    est = text_tokens(prompt) + image_tokens + max_tokens
    msg = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        ],
    }]
    return _call([settings.llm_model, settings.vlm_fallback_model], msg, est,
                 temperature=0, max_tokens=max_tokens)
