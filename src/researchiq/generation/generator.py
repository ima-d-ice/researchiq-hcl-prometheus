"""C2 · Qwen 3.8 27B on Groq: JSON mode, thinking off, low temperature."""
from researchiq.generation.context_builder import max_output_tokens
from researchiq.generation.prompts import ROUTE_HINTS, SYSTEM, USER
from researchiq.llm.groq_client import chat_json
from researchiq.schemas import Route


def generate(question: str, route: Route, context: str, feedback: str | None = None) -> str:
    user = USER.format(route=route.value, hint=ROUTE_HINTS[route.value], context=context, question=question)
    if feedback:
        user += f"\n\nYour previous output was invalid: {feedback}\nReturn valid JSON only."
    return chat_json(SYSTEM, user, max_tokens=max_output_tokens(route), temperature=0.1)
