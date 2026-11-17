"""D3 · RAGAS / LLM judge on Groq: faithfulness, correctness.

Uses a simple LLM-as-judge by default. TODO: optionally wire RAGAS with Groq as the LLM.
"""
import json

from researchiq.llm.groq_client import chat_json

JUDGE_SYSTEM = """You grade answers from a research assistant. Return JSON:
{"faithfulness": 0-1, "correctness": 0-1, "reason": "one sentence"}
faithfulness: every claim in ANSWER is supported by CONTEXT.
correctness: ANSWER agrees with REFERENCE."""


def judge(question: str, answer: str, context: str, reference: str) -> dict:
    user = f"QUESTION: {question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}\n\nREFERENCE:\n{reference}"
    try:
        return json.loads(chat_json(JUDGE_SYSTEM, user))
    except json.JSONDecodeError:
        return {"faithfulness": None, "correctness": None, "reason": "judge returned invalid JSON"}
