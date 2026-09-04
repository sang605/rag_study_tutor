import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragtutor.embeddings import HashingEmbedder, stable_hash
from ragtutor.vectorstore import VectorStore

CHUNKS = [
    "A foreign key refers to the primary key of another table.",
    "Normalization reduces data redundancy and avoids update anomalies.",
    "An index makes SELECT faster but slows down INSERT and UPDATE.",
    "Atomicity means all operations succeed or none of them do.",
]
SOURCES = ["dbms.md"] * len(CHUNKS)


def build_store() -> VectorStore:
    embedder = HashingEmbedder().fit(CHUNKS)
    store = VectorStore(embedder)
    store.add(embedder.embed(CHUNKS), CHUNKS, SOURCES)
    return store


def test_stable_hash_is_deterministic():
    assert stable_hash("primary_key") == stable_hash("primary_key")


def test_vectors_are_normalised():
    store = build_store()
    norms = (store.vectors**2).sum(axis=1) ** 0.5
    assert all(abs(n - 1.0) < 1e-4 for n in norms)


def test_search_finds_the_right_chunk():
    store = build_store()
    hits = store.search("what does normalization do?", k=2)
    assert hits
    assert "Normalization" in hits[0].text


def test_search_returns_k_results():
    store = build_store()
    assert len(store.search("index", k=3)) == 3


def test_save_and_load_round_trip(tmp_path):
    store = build_store()
    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path)
    assert len(loaded) == len(CHUNKS)
    assert loaded.search("atomicity", k=1)[0].text == CHUNKS[3]
