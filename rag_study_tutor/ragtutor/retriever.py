"""Build the index from data/, and load it back for searching."""

from __future__ import annotations

from pathlib import Path
from typing import List

from .chunking import split_text
from .config import Settings, settings as default_settings
from .embeddings import get_embedder
from .loaders import load_documents
from .vectorstore import Hit, VectorStore


def build_index(settings: Settings = default_settings, verbose: bool = True) -> VectorStore:
    docs = load_documents(settings.data_dir)
    if not docs:
        raise SystemExit(
            f"No notes found in {settings.data_dir}.\n"
            "Put some .pdf, .txt or .md files there and run this again."
        )

    texts: List[str] = []
    sources: List[str] = []
    for doc in docs:
        chunks = split_text(doc.text, settings.chunk_size, settings.chunk_overlap)
        texts.extend(chunks)
        sources.extend([doc.source] * len(chunks))
        if verbose:
            print(f"  {doc.source}: {len(chunks)} chunks")

    embedder = get_embedder(settings.embedder)
    if verbose:
        print(f"Embedding {len(texts)} chunks with '{embedder.kind}' ...")
    embedder.fit(texts)
    vectors = embedder.embed(texts)

    store = VectorStore(embedder)
    store.add(vectors, texts, sources)
    store.save(settings.storage_dir)
    if verbose:
        print(f"Index saved to {settings.storage_dir}")
    return store


def load_index(settings: Settings = default_settings) -> VectorStore:
    return VectorStore.load(settings.storage_dir)


def retrieve(store: VectorStore, question: str, k: int = 4) -> List[Hit]:
    return store.search(question, k=k)
