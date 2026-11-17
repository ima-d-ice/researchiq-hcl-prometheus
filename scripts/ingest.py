"""CLI: ingest PDFs into Qdrant.  Usage: python scripts/ingest.py data/papers/*.pdf"""
import argparse
from pathlib import Path

from researchiq.ingestion.pipeline import ingest_pdf


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("pdfs", nargs="+", type=Path)
    args = p.parse_args()
    for pdf in args.pdfs:
        try:
            print(f"{pdf.name}: {ingest_pdf(pdf)} chunks")
        except Exception as e:  # keep going on bad files
            print(f"{pdf.name}: FAILED ({e})")


if __name__ == "__main__":
    main()
