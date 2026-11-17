"""B2 · Router: general / compare / figure / synthesis.

Uses the query plan from the Groq rewrite, validated by regex rules. With no plan hint
(rewrite off or failed) it is purely rule-based.
"""
import re

from researchiq.schemas import QueryPlan, Route

_FIGURE = re.compile(r"\b(fig(ure)?|table|chart|plot|graph|diagram)\b", re.I)
_COMPARE = re.compile(r"\b(compare|comparison|versus|vs\.?|differ(ence)?|better than)\b", re.I)
_SYNTH = re.compile(r"\b(summari[sz]e|overview|across (all|the) papers|common|trends?|survey)\b", re.I)


def route_by_rules(question: str) -> Route:
    if _FIGURE.search(question):
        return Route.FIGURE
    if _COMPARE.search(question):
        return Route.COMPARE
    if _SYNTH.search(question):
        return Route.SYNTHESIS
    return Route.GENERAL


def route(plan: QueryPlan) -> Route:
    if plan.figure_label:
        return Route.FIGURE
    hint = plan.route or route_by_rules(plan.original)
    if hint == Route.FIGURE:
        return Route.FIGURE
    if hint == Route.COMPARE:
        # compare needs two papers to split on; otherwise the paper filter would hide everything
        return Route.COMPARE if len(plan.paper_ids) >= 2 else Route.GENERAL
    return hint
