"""A1 · PDF upload: validate type, size, pages."""
from pathlib import Path

from researchiq.config import settings


class UploadError(ValueError):
    pass


def validate_pdf(path: Path) -> None:
    if path.suffix.lower() != ".pdf":
        raise UploadError(f"{path.name}: not a PDF")
    with path.open("rb") as f:
        if f.read(5) != b"%PDF-":
            raise UploadError(f"{path.name}: bad PDF header")
    size_mb = path.stat().st_size / 1e6
    if size_mb > settings.max_file_mb:
        raise UploadError(f"{path.name}: {size_mb:.1f} MB > {settings.max_file_mb} MB")
    # TODO: page count check against settings.max_pages (e.g. via pypdfium2)
