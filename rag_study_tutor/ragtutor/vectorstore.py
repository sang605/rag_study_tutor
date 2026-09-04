"""A tiny vector database: a NumPy matrix + metadata, saved to disk.

Real projects use FAISS / Chroma / pgvector. For a few thousand chunks a plain
matrix multiply is fast, and you can see exactly how the search works.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .embeddings import embedder_from_state


@dataclass
class Hit:
    text: str
    source: str
    score: float


class VectorStore:
    def __init__(self, embedder):
        self.embedder = embedder
        self.vectors: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self.texts: List[str] = []
        self.sources: List[str] = []

    def __len__(self) -> int:
        return len(self.texts)

    def add(self, vectors: np.ndarray, texts: Sequence[str], sources: Sequence[str]) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if len(vectors) != len(texts) or len(texts) != len(sources):
            raise ValueError("vectors, texts and sources must be the same length")
        if len(self.texts) == 0:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        self.texts.extend(texts)
        self.sources.extend(sources)

    def search(self, query: str, k: int = 4) -> List[Hit]:
        if len(self) == 0:
            return []
        query_vec = self.embedder.embed([query])[0]
        # vectors are normalised, so this dot product is the cosine similarity
        scores = self.vectors @ query_vec
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [
            Hit(text=self.texts[i], source=self.sources[i], score=float(scores[i]))
            for i in top
        ]

    # ---------- persistence ----------

    def save(self, folder: Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "vectors.npy", self.vectors)
        meta: Dict = {
            "texts": self.texts,
            "sources": self.sources,
            "embedder": self.embedder.state_dict(),
        }
        (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, folder: Path) -> "VectorStore":
        folder = Path(folder)
        meta_path = folder / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                "No index found. Build one first with:  python main.py index"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        store = cls(embedder_from_state(meta["embedder"]))
        store.vectors = np.load(folder / "vectors.npy")
        store.texts = meta["texts"]
        store.sources = meta["sources"]
        return store
