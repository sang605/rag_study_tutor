"""Turn text into vectors.

Two embedders, same interface:

* HashingEmbedder          - pure NumPy TF-IDF over hashed word/bigram features.
                             No downloads, no API key, works offline. Good enough
                             for keyword-ish study questions.
* SentenceTransformerEmbedder - real semantic embeddings. Needs
                             `pip install sentence-transformers` (~90 MB model on
                             first run) and understands paraphrases much better.

Both return L2-normalised vectors, so a dot product IS the cosine similarity.
"""

from __future__ import annotations

import re
import zlib
from typing import Dict, List, Optional, Sequence

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


def stable_hash(token: str) -> int:
    """Python's built-in hash() is randomised per process, which would make a saved
    index unreadable on the next run. CRC32 is stable across runs and machines."""
    return zlib.crc32(token.encode("utf-8"))


STOPWORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
    "do", "does", "for", "from", "has", "have", "how", "i", "if", "in", "is", "it",
    "its", "me", "my", "not", "of", "on", "or", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "to", "was", "we", "were", "what",
    "when", "which", "who", "why", "will", "with", "you", "your",
}


def tokenize(text: str) -> List[str]:
    """Lowercase content words plus bigrams ('primary key' becomes one feature).

    Stopwords are dropped: without this, a question like "what is X and why do we
    do it" matches every chunk that happens to contain 'do' and 'we'.
    """
    words = _TOKEN.findall(text.lower())
    bigrams = [
        f"{a}_{b}"
        for a, b in zip(words, words[1:])
        if a not in STOPWORDS and b not in STOPWORDS
    ]
    content = [w for w in words if w not in STOPWORDS and len(w) > 1]
    return content + bigrams


class HashingEmbedder:
    kind = "hashing"

    # dim is the number of hash buckets. Too small and unrelated words collide into
    # the same bucket, which shows up as odd search results; too large wastes memory
    # because every chunk stores a dense vector of this length.
    def __init__(self, dim: int = 16384, idf: Optional[np.ndarray] = None):
        self.dim = dim
        self.idf = idf

    def _counts(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in tokenize(text):
            vec[stable_hash(token) % self.dim] += 1.0
        return vec

    def fit(self, texts: Sequence[str]) -> "HashingEmbedder":
        """Learn how rare each feature is, so common words count for less."""
        n = max(1, len(texts))
        df = np.zeros(self.dim, dtype=np.float32)
        for text in texts:
            df += (self._counts(text) > 0).astype(np.float32)
        self.idf = np.log((1.0 + n) / (1.0 + df)).astype(np.float32) + 1.0
        return self

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if self.idf is None:
            raise RuntimeError("call fit() before embed(), or load a saved index")
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            counts = self._counts(text)
            tf = np.where(counts > 0, 1.0 + np.log(np.maximum(counts, 1.0)), 0.0)
            out[i] = tf * self.idf
        return normalize(out)

    def state_dict(self) -> Dict:
        return {"kind": self.kind, "dim": self.dim, "idf": self.idf.tolist()}

    @classmethod
    def from_state(cls, state: Dict) -> "HashingEmbedder":
        return cls(dim=state["dim"], idf=np.asarray(state["idf"], dtype=np.float32))


class SentenceTransformerEmbedder:
    kind = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit(self, texts: Sequence[str]) -> "SentenceTransformerEmbedder":
        return self  # nothing to learn, the model is pre-trained

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self.model.encode(list(texts), convert_to_numpy=True)
        return normalize(np.asarray(vectors, dtype=np.float32))

    def state_dict(self) -> Dict:
        return {"kind": self.kind, "model_name": self.model_name}

    @classmethod
    def from_state(cls, state: Dict) -> "SentenceTransformerEmbedder":
        return cls(model_name=state.get("model_name", "all-MiniLM-L6-v2"))


def normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-10)


def _sentence_transformers_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401

        return True
    except Exception:
        return False


def get_embedder(name: str = "auto"):
    """Build a fresh embedder by name: auto | hashing | sentence-transformers."""
    if name == "sentence-transformers":
        return SentenceTransformerEmbedder()
    if name == "hashing":
        return HashingEmbedder()
    # auto
    if _sentence_transformers_available():
        return SentenceTransformerEmbedder()
    return HashingEmbedder()


def embedder_from_state(state: Dict):
    if state.get("kind") == SentenceTransformerEmbedder.kind:
        return SentenceTransformerEmbedder.from_state(state)
    return HashingEmbedder.from_state(state)
