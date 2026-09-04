import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragtutor.chunking import split_text


def test_empty_text_gives_no_chunks():
    assert split_text("") == []
    assert split_text("   \n  ") == []


def test_short_text_is_one_chunk():
    assert split_text("A primary key is unique.", 100, 20) == [
        "A primary key is unique."
    ]


def test_long_text_is_split_and_nothing_is_empty():
    text = "\n\n".join(f"Paragraph number {i}. " * 10 for i in range(20))
    chunks = split_text(text, chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)


def test_single_huge_paragraph_is_hard_split():
    chunks = split_text("x" * 2500, chunk_size=500, chunk_overlap=100)
    assert len(chunks) > 1
    assert all(len(chunk) <= 500 for chunk in chunks)


def test_overlap_must_be_smaller_than_chunk_size():
    try:
        split_text("hello", chunk_size=100, chunk_overlap=100)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
