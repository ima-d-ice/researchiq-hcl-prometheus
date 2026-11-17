"""D · Offline eval: scores retrieval (B) and answers (C), writes eval/report.md.

Run: python eval/run_eval.py
"""
import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).parent))

from judge import judge  # noqa: E402
from retrieval_metrics import ABLATIONS, recall_at_k, reciprocal_rank  # noqa: E402

from researchiq.generation.context_builder import build_context  # noqa: E402
from researchiq.generation.pipeline import answer  # noqa: E402
from researchiq.retrieval.pipeline import retrieve  # noqa: E402

HERE = Path(__file__).parent


def load_gold() -> list[dict]:
    return [json.loads(l) for l in (HERE / "gold_set.jsonl").read_text().splitlines() if l.strip()]


def eval_retrieval(gold: list[dict]) -> dict[str, dict[str, float]]:
    out = {}
    for name, flags in ABLATIONS.items():
        r5, mrr = [], []
        for g in gold:
            chunks = retrieve(g["question"], **flags).chunks
            r5.append(recall_at_k(chunks, g["paper_id"], g["expected_pages"], k=5))
            mrr.append(reciprocal_rank(chunks, g["paper_id"], g["expected_pages"]))
        out[name] = {"Recall@5": mean(r5), "MRR": mean(mrr)}
    return out


def eval_answers(gold: list[dict]) -> list[dict]:
    rows = []
    for g in gold:
        res = answer(g["question"])
        context, _ = build_context(retrieve(g["question"]).chunks)
        rows.append({"id": g["id"], "question": g["question"], "answer": res.answer,
                     **judge(g["question"], res.answer, context, g["reference_answer"])})
    return rows


def write_report(ret: dict, ans: list[dict]) -> None:
    lines = ["# ResearchIQ eval report", "", "## Retrieval", "", "| Config | Recall@5 | MRR |", "|---|---|---|"]
    lines += [f"| {k} | {v['Recall@5']:.3f} | {v['MRR']:.3f} |" for k, v in ret.items()]

    scored = [r for r in ans if r["faithfulness"] is not None]
    lines += ["", "## Answers", ""]
    if scored:
        lines += [f"- Faithfulness: {mean(r['faithfulness'] for r in scored):.3f}",
                  f"- Correctness: {mean(r['correctness'] for r in scored):.3f}"]

    worst = sorted(scored, key=lambda r: r["faithfulness"] + r["correctness"])[:2]
    lines += ["", "## Failure cases", ""]
    for r in worst:
        lines += [f"### {r['id']}: {r['question']}", "", f"**Answer:** {r['answer']}", "",
                  f"**Judge:** {r['reason']}", ""]
    # TODO: add manual root-cause notes for each failure case

    (HERE / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    gold = load_gold()
    write_report(eval_retrieval(gold), eval_answers(gold))
    print(f"Wrote {HERE / 'report.md'}")
