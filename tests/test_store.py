import numpy as np
import pytest

from askdocs.embeddings import HashingEmbedder
from askdocs.store import VectorStore


def make_store():
    e = HashingEmbedder(dim=128)
    texts = [
        "python is a programming language",
        "docker packages applications into containers",
        "bananas are a yellow fruit",
    ]
    store = VectorStore(dim=128)
    store.add("notes.md", texts, e.embed(texts))
    return e, store


def test_search_ranks_most_relevant_first():
    e, store = make_store()
    hits = store.search(e.embed(["what is a docker container"])[0], k=2)
    assert hits[0].record.text.startswith("docker")
    assert hits[0].score >= hits[1].score


def test_k_larger_than_store_is_clamped():
    e, store = make_store()
    assert len(store.search(e.embed(["anything"])[0], k=50)) == 3


def test_empty_store_returns_nothing():
    assert VectorStore(dim=8).search(np.ones(8, dtype=np.float32)) == []


def test_ids_are_sequential_across_adds():
    e, store = make_store()
    store.add("more.md", ["extra"], e.embed(["extra"]))
    assert [r.id for r in store.records] == [0, 1, 2, 3]


def test_dimension_mismatch_rejected():
    store = VectorStore(dim=4)
    with pytest.raises(ValueError):
        store.add("x", ["t"], np.ones((1, 5), dtype=np.float32))

def test_save_and_load_round_trip(tmp_path):
    e, store = make_store()
    store.save(tmp_path / "idx")
    loaded = VectorStore.load(tmp_path / "idx")
    assert loaded.dim == store.dim
    assert loaded.records == store.records
    q = e.embed(["python language"])[0]
    assert [h.record.id for h in loaded.search(q)] == [h.record.id for h in store.search(q)]


def test_load_detects_corruption(tmp_path):
    _, store = make_store()
    store.save(tmp_path)
    np.save(tmp_path / "vectors.npy", np.zeros((1, 128), dtype=np.float32))
    with pytest.raises(ValueError):
        VectorStore.load(tmp_path)