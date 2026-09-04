"""Split long text into overlapping chunks.

Why chunks? An LLM prompt has a size limit, and retrieval works better on small,
focused pieces of text than on a whole 40-page PDF. The overlap keeps a sentence
that straddles a boundary from being lost to both chunks.
"""

import re
from typing import List

_PARAGRAPH = re.compile(r"\n\s*\n")


def _split_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in _PARAGRAPH.split(text) if p.strip()]


def _hard_split(text: str, size: int, overlap: int) -> List[str]:
    """Last resort for a single paragraph that is longer than one chunk."""
    step = max(1, size - overlap)
    return [text[i : i + size] for i in range(0, len(text), step)]


def split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> List[str]:
    """Group paragraphs into chunks of about *chunk_size* characters."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = (text or "").strip()
    if not text:
        return []

    chunks: List[str] = []
    current = ""

    for para in _split_paragraphs(text):
        if len(para) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_hard_split(para, chunk_size, chunk_overlap))
            continue

        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            chunks.append(current)
            # carry the tail of the previous chunk over as context
            tail = current[-chunk_overlap:] if chunk_overlap else ""
            if " " in tail:  # do not start the overlap in the middle of a word
                tail = tail.split(" ", 1)[1]
            current = f"{tail}\n\n{para}".strip() if tail else para

    if current:
        chunks.append(current)

    return [c.strip() for c in chunks if c.strip()]
