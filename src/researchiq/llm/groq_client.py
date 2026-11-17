"""Single entry point for every Groq API call (VLM describe, answer, judge)."""
from groq import Groq

from researchiq.config import settings

_client: Groq | None = None


def client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=settings.groq_api_key)
    return _client


def chat_json(system: str, user: str) -> str:
    """Text chat in JSON mode, thinking off. Returns raw JSON string."""
    # TODO: confirm how to disable thinking for this model on Groq
    resp = client().chat.completions.create(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content or ""


def describe_image(prompt: str, image_b64: str) -> str:
    """Vision call: image + prompt -> text description."""
    resp = client().chat.completions.create(
        model=settings.llm_model,
        temperature=0,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ],
        }],
    )
    return resp.choices[0].message.content or ""
