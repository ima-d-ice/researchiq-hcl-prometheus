"""B2 · Router: general / compare / figure / synthesis.

Rule-based to start (runs locally, no API call). Swap for a classifier later if needed.
"""
import re

from researchiq.schemas import Route

_FIGURE = re.compile(r"\b(fig(ure)?|table|chart|plot|graph|diagram)\b", re.I)
_COMPARE = re.compile(r"\b(compare|comparison|versus|vs\.?|differ(ence)?|better than)\b", re.I)
_SYNTH = re.compile(r"\b(summari[sz]e|overview|across (all|the) papers|common|trends?|survey)\b", re.I)


def route(question: str) -> Route:
    if _FIGURE.search(question):
        return Route.FIGURE
    if _COMPARE.search(question):
        return Route.COMPARE
    if _SYNTH.search(question):
        return Route.SYNTHESIS
    return Route.GENERAL
