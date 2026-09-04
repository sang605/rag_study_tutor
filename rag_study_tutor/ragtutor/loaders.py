"""Read study material out of a folder. Supports .txt, .md and .pdf."""

from dataclasses import dataclass
from pathlib import Path
from typing import List

SUPPORTED = {".txt", ".md", ".markdown", ".pdf"}


@dataclass
class Document:
    text: str
    source: str  # file name, used for citations


def read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PDF support needs pypdf:  pip install pypdf") from exc

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(f"[page {i}]\n{text}")
    return "\n\n".join(pages)


def read_file(path: Path) -> Document:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text = read_pdf(path)
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")
    return Document(text=text.strip(), source=path.name)


def load_documents(folder: Path) -> List[Document]:
    """Load every supported file in *folder* (recursively). Empty files are skipped."""
    folder = Path(folder)
    docs: List[Document] = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        try:
            doc = read_file(path)
        except Exception as exc:  # a broken file should not kill the whole run
            print(f"  ! skipped {path.name}: {exc}")
            continue
        if doc.text:
            docs.append(doc)
    return docs
