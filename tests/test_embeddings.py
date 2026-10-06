import numpy as np

from askdocs.embeddings import HashingEmbedder, tokenize


def test_tokenize_drops_stopwords_and_plurals():
    assert tokenize("The servers are running") == ["server", "running"]


def test_shape_and_unit_norm():
    vecs = HashingEmbedder(dim=64).embed(["hello world", "another sentence here"])
    assert vecs.shape == (2, 64)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)


def test_deterministic():
    e = HashingEmbedder()
    assert np.array_equal(e.embed(["same text"]), e.embed(["same text"]))


def test_empty_text_is_zero_vector_not_nan():
    vec = HashingEmbedder().embed([""])
    assert not np.isnan(vec).any()
    assert np.count_nonzero(vec) == 0


def test_related_text_scores_higher_than_unrelated():
    e = HashingEmbedder()
    q, related, unrelated = e.embed(
        [
            "how do I deploy the service with docker",
            "deploy the service using a docker container",
            "the cat sat quietly on the warm windowsill",
        ]
    )
    assert q @ related > q @ unrelated