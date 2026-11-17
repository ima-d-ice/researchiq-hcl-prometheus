"""Token arithmetic for Groq's tokens-per-minute budget. No I/O, no API calls."""
import io
import math

from PIL import Image

from researchiq.config import settings

PATCH = 28 * 28  # Qwen-VL style: one token per 28x28 patch


def text_tokens(text: str) -> int:
    """Conservative estimate (chars / 3.5, slightly stricter than chars / 4)."""
    return math.ceil(len(text) / 3.5)


def image_tokens(width: int, height: int) -> int:
    return math.ceil(width * height / PATCH)


def fit_image(img: Image.Image, max_tokens: int | None = None) -> Image.Image:
    """Downscale so the image costs at most max_tokens."""
    max_tokens = max_tokens or settings.vlm_max_image_tokens
    w, h = img.size
    if image_tokens(w, h) <= max_tokens:
        return img
    scale = math.sqrt(max_tokens * PATCH / (w * h)) * 0.98
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    return text[: int(max_tokens * 3.5)]


def estimate_request(prompt_text: str, img_tokens: int, max_output: int) -> int:
    return text_tokens(prompt_text) + img_tokens + max_output
