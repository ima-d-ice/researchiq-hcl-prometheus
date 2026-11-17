"""Single entry point for every Groq API call (VLM describe, answer, judge).

Every call goes through the shared KeyPool: each API key has its own 8k tokens/minute
limiter, and a rate-limited key is rotated out automatically.
"""
import logging
import re

from groq import BadRequestError, Groq, RateLimitError

from researchiq.config import settings
from researchiq.llm.budget import text_tokens
from researchiq.llm.rate_limiter import KeyPool

log = logging.getLogger(__name__)

limiter = KeyPool(settings.all_groq_keys, settings.groq_tpm_limit,
                  settings.groq_rpm_limit, settings.groq_tpd_limit)
_clients: dict[int, Groq] = {}

# Ways to switch Qwen's thinking off on Groq, best first. The first one the API accepts is
# remembered, so one rejected parameter costs a single failed call per process, not per request.
_THINKING_OFF: list[dict] = [{"reasoning_effort": "none"}, {"reasoning_format": "hidden"}, {}]
_thinking_idx = 0
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Safety net: drop any <think>…</think> block (or an unclosed one) before JSON parsing."""
    text = _THINK_BLOCK.sub("", text)
    if "<think>" in text.lower():
        text = re.split(r"<think>", text, flags=re.IGNORECASE)[0]
    return text.strip()


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


def _reasoning_tokens(usage) -> int:
    details = getattr(usage, "completion_tokens_details", None)
    return int(getattr(details, "reasoning_tokens", 0) or 0)


def _call(models: list[str], messages: list[dict], estimate: int, no_thinking: bool = False, **kw) -> str:
    """Try each model in order; rotate keys on 429. Every attempt is metered.

    no_thinking: add the Qwen "thinking off" parameter to the primary (Qwen) model only.
    """
    global _thinking_idx
    last: Exception | None = None
    for mi, model in enumerate(models):
        for _ in range(len(limiter.keys) + settings.groq_max_retries):
            idx, handle = limiter.acquire(estimate)
            extra = _THINKING_OFF[_thinking_idx] if (no_thinking and mi == 0) else {}
            try:
                resp = client(idx).chat.completions.create(model=model, messages=messages, **extra, **kw)
                usage = getattr(resp, "usage", None)
                limiter.settle(idx, handle, getattr(usage, "total_tokens", None))
                if mi == 0 and no_thinking and _reasoning_tokens(usage):
                    log.warning("%s used %d reasoning tokens; thinking is NOT off (params: %s)",
                                model, _reasoning_tokens(usage), extra or "none")
                if mi > 0:
                    log.warning("Answered by fallback model %s, not %s", model, models[0])
                return resp.choices[0].message.content or ""
            except BadRequestError as e:
                limiter.settle(idx, handle, 0)
                last = e
                if (no_thinking and mi == 0 and extra and "reasoning" in str(e).lower()
                        and _thinking_idx < len(_THINKING_OFF) - 1):
                    _thinking_idx += 1  # parameter rejected: try the next way to turn thinking off
                    log.warning("Thinking param %s rejected; trying %s", extra, _THINKING_OFF[_thinking_idx])
                    continue
                log.warning("Groq call failed on %s: BadRequestError", model)
                break
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


def chat_json(system: str, user: str, max_tokens: int = 1200, temperature: float | None = None) -> str:
    """Text chat in JSON mode, thinking off. Returns raw JSON string (any <think> block removed)."""
    est = text_tokens(system) + text_tokens(user) + max_tokens
    raw = _call(
        [settings.llm_model, settings.llm_fallback_model],
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        est,
        no_thinking=True,
        temperature=settings.llm_temperature if temperature is None else temperature,
        response_format={"type": "json_object"},
        max_tokens=max_tokens,
    )
    return strip_thinking(raw)


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
