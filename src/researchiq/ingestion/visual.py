"""A4 · Visual path: figure / table crop with caption, page, nearby text."""
from pathlib import Path

from docling_core.types.doc import DoclingDocument

from researchiq.config import settings
from researchiq.schemas import VisualElement


def extract_visuals(doc: DoclingDocument, paper_id: str) -> list[VisualElement]:
    out_dir = settings.cache_dir / "crops" / paper_id
    out_dir.mkdir(parents=True, exist_ok=True)
    elements: list[VisualElement] = []

    for kind, items in (("figure", doc.pictures), ("table", doc.tables)):
        for i, item in enumerate(items):
            img = item.get_image(doc)
            if img is None:
                continue
            path = out_dir / f"{kind}_{i}.png"
            img.save(path)
            page = item.prov[0].page_no if item.prov else 0
            elements.append(VisualElement(
                paper_id=paper_id,
                kind=kind,
                label=None,  # TODO: parse "Figure 3" / "Table 2" from caption
                caption=item.caption_text(doc),
                page=page,
                nearby_text=nearby_text(doc, page),
                image_path=str(path),
            ))
    return elements


def nearby_text(doc: DoclingDocument, page: int, max_chars: int = 800) -> str:
    """Text on the same page, used as grounding for the VLM."""
    # TODO: prefer paragraphs that reference the figure label
    parts = [t.text for t in doc.texts if t.prov and t.prov[0].page_no == page]
    return " ".join(parts)[:max_chars]


def load_b64(path: str | Path) -> str:
    import base64
    return base64.b64encode(Path(path).read_bytes()).decode()
