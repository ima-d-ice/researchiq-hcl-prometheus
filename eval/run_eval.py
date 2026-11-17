"""D · Offline evaluation: retrieval (lane B), answers (lane C) and security. Writes eval/report.md.

  python eval/run_eval.py                      # everything
  python eval/run_eval.py --no-llm             # retrieval + local security checks only, 0 Groq tokens
  python eval/run_eval.py --only retrieval     # or answers / security
  python eval/run_eval.py --limit 5 --judge-limit 3

Token cost (free tier, ~8k tokens/min): about 4k per answered question and 3k per judged one.
Don't run this at the same time as ingestion on the same API key.
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import checks  # noqa: E402

ABLATIONS = {
    "dense only": dict(use_sparse=False, use_rerank=False),
    "dense + rerank": dict(use_sparse=False, use_rerank=True),
    "dense + BM25 (RRF)": dict(use_sparse=True, use_rerank=False),
    "dense + BM25 (RRF) + rerank": dict(use_sparse=True, use_rerank=True),   # the shipped config
}
FULL = "dense + BM25 (RRF) + rerank"


# ---------------------------------------------------------------- gold set
def load_gold(path: Path = HERE / "gold_set.jsonl") -> tuple[list[dict], int]:
    rows, placeholders = [], 0
    for i, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = checks.validate_row(json.loads(line), i)
        if checks.is_placeholder(row):
            placeholders += 1
        else:
            rows.append(row)
    return rows, placeholders


# ---------------------------------------------------------------- retrieval (local, 0 tokens)
def eval_retrieval(gold: list[dict]) -> dict:
    from researchiq.retrieval.pipeline import retrieve
    from researchiq.retrieval.router import route

    scored = [g for g in gold if g["expected_sources"]]
    ablation, per_type = {}, defaultdict(list)
    for name, flags in ABLATIONS.items():
        r5, mrr, hit = [], [], []
        for g in scored:
            found = [(c.chunk.paper_id, c.chunk.page) for c in retrieve(g["question"], **flags).chunks]
            exp = checks.expected_set(g)
            r, rr = checks.recall_at_k(found, exp, 5), checks.reciprocal_rank(found, exp)
            r5.append(r), mrr.append(rr), hit.append(float(rr > 0 and rr >= 1 / 5))
            if name == FULL:
                per_type[g["type"]].append(r)
        ablation[name] = {"Recall@5": _avg(r5), "MRR": _avg(mrr), "Hit@5": _avg(hit)}

    routed = [g for g in gold if g.get("route")]
    misrouted = [{"id": g["id"], "expected": g["route"], "got": route(g["question"]).value}
                 for g in routed if route(g["question"]).value != g["route"]]
    return {"n": len(scored), "ablation": ablation,
            "recall_by_type": {k: _avg(v) for k, v in per_type.items()},
            "routing": {"n": len(routed), "accuracy": _avg([1.0] * (len(routed) - len(misrouted))
                                                            + [0.0] * len(misrouted)),
                        "misrouted": misrouted}}


# ---------------------------------------------------------------- answers (Groq)
def eval_answers(gold: list[dict], judge_limit: int) -> list[dict]:
    import pipeline_trace
    from judge import judge

    rows, judged = [], 0
    for g in gold:
        t = pipeline_trace.run(g["question"])
        exp = checks.expected_set(g)
        refused = checks.refused(t.insufficient_evidence, t.answer)
        row = {"id": g["id"], "type": g["type"], "question": g["question"], "answer": t.answer,
               "error": t.error, "route": t.route, "refused": refused,
               "citation_accuracy": checks.citation_accuracy(t.cited, exp),
               "n_citations": len(t.cited), "warnings": t.warnings,
               "seconds": round(t.seconds, 2), "groq_tokens": t.groq_tokens}
        for bad in g.get("must_not_contain", []):
            if bad.lower() in t.full_text.lower():
                row.setdefault("violations", []).append(bad)
        # judge answerable questions only, up to the budget
        if g["type"] != "unanswerable" and not t.error and judged < judge_limit:
            row.update(judge(g["question"], t.answer, t.context, g.get("reference_answer", "")))
            judged += 1
        rows.append(row)
    return rows


def summarize_answers(rows: list[dict]) -> dict:
    from judge import METRICS
    unans = [r for r in rows if r["type"] == "unanswerable"]
    ans = [r for r in rows if r["type"] != "unanswerable"]
    out = {m: _avg([r[m] for r in rows if r.get(m) is not None]) for m in (*METRICS, "correctness")}
    out["judged"] = sum(r.get("faithfulness") is not None for r in rows)
    out["citation_accuracy"] = _avg([r["citation_accuracy"] for r in ans if r["citation_accuracy"] is not None])
    out["uncited_answers"] = sum(r["n_citations"] == 0 and not r["refused"] for r in ans)
    out["refusal_correctness"] = _avg([float(r["refused"]) for r in unans])
    out["false_refusals"] = sum(r["refused"] for r in ans)
    out["errors"] = sum(bool(r["error"]) for r in rows)
    out["avg_seconds"] = _avg([r["seconds"] for r in rows])
    out["groq_tokens"] = sum(r["groq_tokens"] for r in rows)
    out["n_answerable"], out["n_unanswerable"] = len(ans), len(unans)
    return out


# ---------------------------------------------------------------- report
def _avg(xs):
    xs = [x for x in xs if x is not None]
    return round(mean(xs), 3) if xs else None


def _fmt(x) -> str:
    return "n/a" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def _cell(s: str, n: int = 160) -> str:
    s = " ".join(str(s).split()).replace("|", "\\|")
    return s if len(s) <= n else s[:n - 1] + "…"


def write_report(meta: dict, ret: dict | None, ans: list[dict] | None, sec: list | None) -> Path:
    L = ["# ResearchIQ evaluation report", "",
         f"Run: {meta['date']} · model `{meta['model']}` · {meta['n_gold']} gold questions"
         f" ({meta['types']}) · {meta['placeholders']} placeholder rows skipped", ""]

    if ret is not None:
        L += ["## Retrieval", "", f"{ret['n']} questions with expected sources. A hit is a matching "
              "(paper_id, page) in the retrieved chunks. Local models only, 0 Groq tokens.", "",
              "| Config | Recall@5 | MRR | Hit@5 |", "|---|---|---|---|"]
        L += [f"| {k}{' (shipped)' if k == FULL else ''} | {_fmt(v['Recall@5'])} | {_fmt(v['MRR'])} "
              f"| {_fmt(v['Hit@5'])} |" for k, v in ret["ablation"].items()]
        if ret["recall_by_type"]:
            L += ["", "Recall@5 of the shipped config by question type:", "",
                  "| Type | Recall@5 |", "|---|---|"]
            L += [f"| {k} | {_fmt(v)} |" for k, v in sorted(ret["recall_by_type"].items())]
        r = ret["routing"]
        if r["n"]:
            L += ["", f"Router accuracy: {_fmt(r['accuracy'])} on {r['n']} labelled questions."]
            L += [f"- {m['id']}: expected `{m['expected']}`, got `{m['got']}`" for m in r["misrouted"]]
        L += [""]

    if ans is not None:
        s = summarize_answers(ans)
        L += ["## Answers", "",
              f"{s['n_answerable']} answerable and {s['n_unanswerable']} unanswerable questions; "
              f"{s['judged']} judged by the LLM judge. {s['groq_tokens']:,} Groq tokens for answers.", "",
              "| Metric | Score | What it measures |", "|---|---|---|",
              f"| Faithfulness | {_fmt(s['faithfulness'])} | claims supported by the retrieved context (judge) |",
              f"| Answer relevancy | {_fmt(s['answer_relevancy'])} | answer addresses the question (judge) |",
              f"| Context precision | {_fmt(s['context_precision'])} | retrieved sources are relevant (judge) |",
              f"| Context recall | {_fmt(s['context_recall'])} | reference facts present in context (judge) |",
              f"| Correctness | {_fmt(s['correctness'])} | agrees with the reference answer (judge) |",
              f"| Citation accuracy | {_fmt(s['citation_accuracy'])} | cited (paper, page) is an expected source |",
              f"| Refusal correctness | {_fmt(s['refusal_correctness'])} | unanswerable questions are declined |",
              f"| False refusals | {s['false_refusals']} | answerable questions that were declined |",
              f"| Uncited answers | {s['uncited_answers']} | answered with no valid citation |",
              f"| Pipeline errors | {s['errors']} | questions that failed outright |",
              f"| Avg latency | {_fmt(s['avg_seconds'])} s | question to validated answer |", "",
              "Judge scores are not ground truth: they come from the same model family that wrote the "
              "answers. Hand-check a sample before quoting them.", "",
              "### Per question", "",
              "| ID | Type | Route | Faith. | Correct. | Cit. acc. | Refused | Answer |",
              "|---|---|---|---|---|---|---|---|"]
        L += [f"| {r['id']} | {r['type']} | {r['route']} | {_fmt(r.get('faithfulness'))} | "
              f"{_fmt(r.get('correctness'))} | {_fmt(r['citation_accuracy'])} | {'yes' if r['refused'] else 'no'} | "
              f"{_cell(r['error'] or r['answer'], 120)} |" for r in ans]
        L += [""]

    if sec is not None:
        groups = defaultdict(lambda: [0, 0, 0])
        for r in sec:
            groups[r.group][["pass", "fail", "skipped"].index(r.status)] += 1
        L += ["## Security and responsible AI", "",
              "| Group | Passed | Failed | Skipped |", "|---|---|---|---|"]
        L += [f"| {g} | {p} | {f} | {k} |" for g, (p, f, k) in groups.items()]
        L += ["", "| ID | Group | Test | Result | Detail |", "|---|---|---|---|---|"]
        L += [f"| {r.id} | {r.group} | {r.name} | **{r.status.upper()}** | "
              f"{_cell('; '.join(r.failed_checks) or r.note or 'all checks passed')} |" for r in sec]
        L += ["", "Injection tests use synthetic papers and a canary string (eval/attacks/cases.py). "
              "A test fails if the canary, an injected false fact or an injected link reaches the "
              "answer, if the model cites a source tag it was never given, or if the real answer is lost.", ""]

    L += ["## Failure cases", ""]
    fails = []
    if ans:
        judged = [r for r in ans if r.get("faithfulness") is not None and r.get("correctness") is not None]
        fails += [(f"{r['id']} ({r['type']}): {r['question']}",
                   f"**Answer:** {_cell(r['answer'], 400)}\n\n**Judge:** {r.get('reason', '')}")
                  for r in sorted(judged, key=lambda r: r["faithfulness"] + r["correctness"])[:2]
                  if r["faithfulness"] + r["correctness"] < 1.6]
        fails += [(f"{r['id']} (unanswerable): {r['question']}", f"**Answered instead of declining:** "
                   f"{_cell(r['answer'], 400)}") for r in ans if r["type"] == "unanswerable" and not r["refused"]]
    if sec:
        fails += [(f"{r.id} ({r.group}): {r.name}", f"**Failed checks:** {'; '.join(r.failed_checks)}"
                   + (f"\n\n**Answer:** {_cell(r.answer, 400)}" if r.answer else ""))
                  for r in sec if r.status == "fail"]
    if not fails:
        L += ["No failures in this run.", ""]
    for title, body in fails:
        L += [f"### {title}", "", body, "", "**Root cause:** _fill in by hand_", "",
              "**Fix:** _fill in by hand_", ""]

    out = HERE / "report.md"
    out.write_text("\n".join(L))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["retrieval", "answers", "security"])
    ap.add_argument("--no-llm", action="store_true", help="skip everything that calls Groq")
    ap.add_argument("--limit", type=int, help="use only the first N gold questions")
    ap.add_argument("--judge-limit", type=int, default=8, help="max questions sent to the judge (default 8)")
    args = ap.parse_args()

    from researchiq.config import settings
    gold, placeholders = load_gold()
    gold = gold[:args.limit] if args.limit else gold
    if not gold and args.only != "security":
        print(f"gold_set.jsonl has no real questions yet ({placeholders} TODO rows). "
              "Retrieval and answer evals need them; running security only.")
    want = lambda part: args.only in (None, part)  # noqa: E731
    types = defaultdict(int)
    for g in gold:
        types[g["type"]] += 1

    meta = {"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "model": settings.llm_model,
            "n_gold": len(gold), "placeholders": placeholders,
            "types": ", ".join(f"{v} {k}" for k, v in sorted(types.items())) or "none"}
    start = time.perf_counter()

    ret = eval_retrieval(gold) if gold and want("retrieval") else None
    ans = eval_answers(gold, args.judge_limit) if gold and want("answers") and not args.no_llm else None
    sec = None
    if want("security"):
        import security
        sec = security.run_all(include_llm=not args.no_llm)

    path = write_report(meta, ret, ans, sec)
    results = HERE / "results"
    results.mkdir(exist_ok=True)
    (results / f"{datetime.now():%Y%m%d-%H%M%S}.json").write_text(json.dumps(
        {"meta": meta, "retrieval": ret, "answers": ans,
         "security": [{**r.__dict__, "findings": [f.__dict__ for f in r.findings]} for r in sec or []]},
        indent=2, default=str))
    print(f"Wrote {path} in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
