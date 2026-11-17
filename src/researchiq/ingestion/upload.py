"""A1 · Upload: validate type, size, pages for PDF / Office / text / image files."""
import zipfile
from dataclasses import dataclass
from pathlib import Path

from researchiq.config import settings


class UploadError(ValueError):
    """Message is safe to show to the user."""


OFFICE = {".docx": "word/", ".pptx": "ppt/", ".xlsx": "xl/"}
TEXT = {".md", ".txt", ".csv", ".html", ".htm"}
IMAGES = {".png": b"\x89PNG\r\n\x1a\n", ".jpg": b"\xff\xd8\xff", ".jpeg": b"\xff\xd8\xff"}
LEGACY = {".doc": ".docx", ".xls": ".xlsx", ".ppt": ".pptx"}

ALLOWED_EXTENSIONS = sorted({".pdf", *OFFICE, *TEXT, *IMAGES})


@dataclass
class FileInfo:
    source_type: str  # pdf | docx | pptx | xlsx | md | txt | csv | html | image
    n_pages: int = 1


def _validate_pdf(path: Path) -> int:
    import fitz  # PyMuPDF

    with path.open("rb") as f:
        if f.read(5) != b"%PDF-":
            raise UploadError(f"{path.name}: not a valid PDF (bad header)")
    try:
        doc = fitz.open(path)
    except Exception:
        raise UploadError(f"{path.name}: the PDF is corrupt or unreadable") from None
    with doc:
        if doc.needs_pass or doc.is_encrypted:
            raise UploadError(f"{path.name}: encrypted PDFs are not supported")
        n = doc.page_count
    if n == 0:
        raise UploadError(f"{path.name}: the PDF has no pages")
    if n > settings.max_pages:
        raise UploadError(f"{path.name}: {n} pages > limit of {settings.max_pages}")
    return n


def _validate_office(path: Path, part: str) -> None:
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            if not any(n.startswith(part) for n in names):
                raise UploadError(f"{path.name}: not a valid {path.suffix} file")
            total = sum(i.file_size for i in z.infolist())
    except zipfile.BadZipFile:
        raise UploadError(f"{path.name}: corrupt or not a {path.suffix} file") from None
    if total > settings.max_unzipped_mb * 1e6:
        raise UploadError(f"{path.name}: file expands to more than {settings.max_unzipped_mb} MB")


def _validate_text(path: Path) -> None:
    raw = path.read_bytes()
    if b"\x00" in raw[:8192]:
        raise UploadError(f"{path.name}: looks like a binary file, not text")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise UploadError(f"{path.name}: text files must be UTF-8 encoded") from None
    if not text.strip():
        raise UploadError(f"{path.name}: the file is empty")
    if len(text) > settings.max_text_chars:
        raise UploadError(f"{path.name}: more than {settings.max_text_chars:,} characters")


def _validate_image(path: Path) -> None:
    from PIL import Image

    ext = path.suffix.lower()
    with path.open("rb") as f:
        if not f.read(8).startswith(IMAGES[ext]):
            raise UploadError(f"{path.name}: not a valid {ext} image")
    try:
        with Image.open(path) as im:
            w, h = im.size
    except Exception:
        raise UploadError(f"{path.name}: corrupt image") from None
    if w * h > settings.max_image_pixels:
        raise UploadError(f"{path.name}: image is too large")


def validate_file(path: Path) -> FileInfo:
    path = Path(path)
    ext = path.suffix.lower()
    if ext in LEGACY:
        raise UploadError(f"{path.name}: old {ext} format is not supported, save it as {LEGACY[ext]}")
    if ext not in ALLOWED_EXTENSIONS:
        raise UploadError(f"{path.name}: unsupported type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
    if not path.is_file():
        raise UploadError(f"{path.name}: file not found")
    size_mb = path.stat().st_size / 1e6
    if size_mb > settings.max_file_mb:
        raise UploadError(f"{path.name}: {size_mb:.1f} MB > {settings.max_file_mb} MB")
    if size_mb == 0:
        raise UploadError(f"{path.name}: the file is empty")

    if ext == ".pdf":
        return FileInfo("pdf", _validate_pdf(path))
    if ext in OFFICE:
        _validate_office(path, OFFICE[ext])
        return FileInfo(ext[1:])
    if ext in IMAGES:
        _validate_image(path)
        return FileInfo("image")
    _validate_text(path)
    return FileInfo("html" if ext == ".htm" else ext[1:])


def validate_pdf(path: Path) -> None:  # backwards-compatible alias
    validate_file(path)
