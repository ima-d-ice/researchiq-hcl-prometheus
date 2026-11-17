"""CLI: ingest documents into Qdrant.

Usage: python scripts/ingest.py data/papers [--no-vlm] [--max-vlm-calls N]
Accepts files, directories or globs of PDF / DOCX / PPTX / XLSX / CSV / MD / TXT / HTML / PNG / JPG.
"""
import argparse
import glob
import logging
from pathlib import Path

from researchiq.ingestion.pipeline import ingest_file
from researchiq.ingestion.upload import ALLOWED_EXTENSIONS


def expand(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        for m in glob.glob(item) or [item]:
            p = Path(m)
            if p.is_dir():
                files += sorted(f for f in p.rglob("*") if f.suffix.lower() in ALLOWED_EXTENSIONS)
            else:
                files.append(p)
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--no-vlm", action="store_true", help="skip Groq figure descriptions (0 tokens)")
    ap.add_argument("--max-vlm-calls", type=int, default=None, help="cap on Groq image calls this run")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    for f in expand(args.inputs):
        try:
            r = ingest_file(f, describe=not args.no_vlm, max_vlm_calls=args.max_vlm_calls)
            kinds = ", ".join(f"{v} {k}" for k, v in sorted(r.counts.items()))
            print(f"{f.name}: {r.n_chunks} chunks ({kinds}) [{r.parser}] id={r.paper_id} "
                  f"vlm: {r.vlm_calls} calls, {r.vlm_cached} cached, {r.vlm_skipped} skipped, "
                  f"{r.groq_tokens} Groq tokens")
        except Exception as e:  # keep going on bad files
            print(f"{f.name}: FAILED ({e})")

    # Close local Qdrant explicitly so it doesn't error during interpreter shutdown
    from researchiq.ingestion.index import qdrant
    qdrant().close()


if __name__ == "__main__":
    main()
