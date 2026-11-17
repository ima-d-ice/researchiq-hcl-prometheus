"""Write the synthetic attack PDFs used by the document-level security tests.

Usage:
  python eval/attacks/make_docs.py                      # writes data/eval_attacks/*.pdf
  python scripts/ingest.py data/eval_attacks --no-vlm   # index them (0 Groq tokens)
  python eval/run_eval.py --only security

Each PDF is one page: a title, a short body containing the real fact, and the injected
instruction (visible, or as white 1pt text when the case is marked hidden). The figure case
is written as a figure caption so it lands in the same chunk as the chart description.
They are clearly labelled synthetic; never mix them into the real papers folder.
"""
import sys
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).parent))
from cases import ATTACKS, Attack  # noqa: E402

OUT = Path("data/eval_attacks")


def write_pdf(a: Attack, out_dir: Path = OUT) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{a.id}.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    box = pymupdf.Rect(72, 72, 540, 760)
    body = (f"{a.title}\n\nSYNTHETIC DOCUMENT FOR SECURITY TESTING.\n\n"
            f"1 Introduction\n{a.fact} The remainder of this section describes the method in detail "
            f"and is intentionally brief.\n\n2 Results\n")
    if a.kind == "figure":
        body += f"Figure 2: {a.fact} {a.payload}\n"
    elif not a.hidden:
        body += f"{a.payload}\n"
    page.insert_textbox(box, body, fontsize=11)
    if a.hidden:
        page.insert_text((72, 700), a.payload, fontsize=1, color=(1, 1, 1))
    doc.set_metadata({"title": a.title})
    doc.save(path)
    doc.close()
    return path


def main() -> None:
    for a in ATTACKS:
        print(write_pdf(a))
    print(f"\nNext: python scripts/ingest.py {OUT} --no-vlm")


if __name__ == "__main__":
    main()
