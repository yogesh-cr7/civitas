import numpy as np
import pytest

from civitas.embeddings import TfidfEmbedder
from civitas.index import VectorIndex


def test_tfidf_embedder_ranks_matching_document_first():
    texts = [
        "the quick brown fox jumps over the lazy dog",
        "congress shall make no law abridging freedom of speech",
        "the judicial power shall extend to all cases in law and equity",
    ]
    embedder = TfidfEmbedder().fit(texts)
    vectors = embedder.encode(texts)
    query_vec = embedder.encode(["freedom of speech and of the press"])[0]

    scores = vectors @ query_vec
    assert np.argmax(scores) == 1  # the free-speech sentence should win


def test_tfidf_vectors_are_unit_normalized():
    embedder = TfidfEmbedder().fit(["alpha beta gamma", "delta epsilon"])
    vectors = embedder.encode(["alpha beta gamma", "delta epsilon"])
    norms = np.linalg.norm(vectors, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


@pytest.fixture(scope="module")
def index():
    return VectorIndex.build(embedder_name="tfidf")


def test_index_search_returns_k_results_sorted_by_score(index):
    results = index.search("freedom of the press", k=5)
    assert len(results) == 5
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize(
    "query,expected_title_substring",
    [
        ("cruel and unusual punishments excessive bail", "Amendment VIII"),
        ("right to keep and bear arms militia", "Amendment II"),
        ("checks and balances ambition must counteract ambition", "Federalist No. 51"),
        ("courts may declare an act of the legislature void", "Federalist No. 78"),
    ],
)
def test_known_queries_retrieve_the_right_document(index, query, expected_title_substring):
    """Regression test: these are the easy cases retrieval should never miss.

    They're intentionally close paraphrases of the source text (TF-IDF is a
    lexical method), which is the honest scope for this backend. The eval
    suite is where harder, more paraphrased queries get measured properly
    across backends.
    """
    results = index.search(query, k=3)
    titles = [chunk.title for chunk, _ in results]
    assert any(expected_title_substring in t for t in titles), titles


def test_save_and_load_round_trip(tmp_path):
    idx = VectorIndex.build(embedder_name="tfidf")
    path = tmp_path / "index.json"
    idx.save(path)

    loaded = VectorIndex.load(path)
    assert len(loaded.chunks) == len(idx.chunks)
    np.testing.assert_allclose(loaded.vectors, idx.vectors)

    original_results = idx.search("due process of law", k=3)
    loaded_results = loaded.search("due process of law", k=3)
    assert [c.id for c, _ in original_results] == [c.id for c, _ in loaded_results]
