"""Inspect what ingestion put into Qdrant (read-only, no Groq calls).

Usage:
  python scripts/check_index.py                     # summary per paper + sanity checks
  python scripts/check_index.py --show figure       # print sample figure chunks
  python scripts/check_index.py --search "Figure 1" # try a real retrieval query
"""
import argparse
from collections import Counter

from researchiq.config import settings
from researchiq.ingestion.index import qdrant


def all_points():
    client, pts, offset = qdrant(), [], None
    while True:
        batch, offset = client.scroll(settings.qdrant_collection, limit=256, offset=offset,
                                      with_payload=True, with_vectors=False)
        pts += batch
        if offset is None:
            return pts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", choices=["text", "table", "figure"], help="print sample chunks of a kind")
    ap.add_argument("--search", help="run a hybrid search + rerank and print the top hits")
    args = ap.parse_args()

    client = qdrant()
    if not client.collection_exists(settings.qdrant_collection):
        print("No collection yet: run scripts/ingest.py first")
        return
    pts = all_points()
    print(f"Collection '{settings.qdrant_collection}': {len(pts)} points\n")

    by_paper: dict[str, list[dict]] = {}
    for p in pts:
        by_paper.setdefault(p.payload["paper_id"], []).append(p.payload)

    problems = 0
    for pid, rows in by_paper.items():
        kinds = Counter(r["kind"] for r in rows)
        pages = sorted({r["page"] for r in rows})
        labels = sorted({r["figure_label"] for r in rows if r.get("figure_label")})
        print(f"{rows[0]['paper_title'][:70]!r}  id={pid}  type={rows[0]['source_type']}  "
              f"parser={rows[0]['parser']}")
        print(f"  chunks: {dict(kinds)}   pages: {pages[0]}-{pages[-1]}   labels: {labels or 'none'}")

        empty = [r for r in rows if not r["text"].strip()]
        no_page = [r for r in rows if r["page"] < 1]
        no_ctx = [r for r in rows if not r.get("contextualized")]
        fig_no_img = [r for r in rows if r["kind"] == "figure" and not r.get("image_path")]
        for name, bad in (("empty text", empty), ("page < 1", no_page),
                          ("missing contextualized header", no_ctx), ("figure without image", fig_no_img)):
            if bad:
                problems += len(bad)
                print(f"  WARNING: {len(bad)} chunk(s) with {name}")
        print()
    print("Sanity checks:", "all passed" if not problems else f"{problems} problem(s) found")

    if args.show:
        rows = [p.payload for p in pts if p.payload["kind"] == args.show][:3]
        for r in rows:
            print(f"\n--- {r['kind']} | p.{r['page']} | {r.get('figure_label')} | section={r.get('section')!r}")
            print(r["contextualized"][:600])

    if args.search:
        from researchiq.retrieval.pipeline import retrieve
        res = retrieve(args.search)
        print(f"\nQuery: {args.search!r}  route={res.route.value}")
        for i, sc in enumerate(res.chunks, 1):
            c = sc.chunk
            print(f"{i}. score={sc.score:.3f} [{c.kind.value}] p.{c.page} {c.figure_label or ''} "
                  f"{c.text[:100]!r}")

    client.close()


if __name__ == "__main__":
    main()
