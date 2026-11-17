"""C3 · Validate: Pydantic parse, retry once."""
from typing import Callable

from pydantic import ValidationError

from researchiq.schemas import LLMAnswer


class GenerationFailed(RuntimeError):
    pass


def validate_with_retry(call: Callable[[str | None], str]) -> LLMAnswer:
    """`call(feedback)` returns raw JSON. Retries once with the validation error as feedback."""
    raw = call(None)
    try:
        return LLMAnswer.model_validate_json(raw)
    except ValidationError as e:
        raw = call(str(e)[:500])
        try:
            return LLMAnswer.model_validate_json(raw)
        except ValidationError as e2:
            raise GenerationFailed(str(e2)) from e2
