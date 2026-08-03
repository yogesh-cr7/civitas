from civitas.ingest import Chunk
from eval.metrics import (
    chunk_matches,
    mean_reciprocal_rank,
    mean_recall_at_k,
    rank_of_first_match,
    recall_at_k,
    reciprocal_rank,
)


def make_chunk(doc, **metadata):
    return Chunk(id=f"{doc}-{metadata}", doc=doc, title="x", text="x", metadata=metadata)


def test_chunk_matches_checks_doc_and_metadata():
    c = make_chunk("amendments", amendment_number=4)
    assert chunk_matches(c, {"doc": "amendments", "amendment_number": 4})
    assert not chunk_matches(c, {"doc": "amendments", "amendment_number": 5})
    assert not chunk_matches(c, {"doc": "federalist", "amendment_number": 4})


def test_chunk_matches_ignores_extra_metadata_not_in_expect():
    c = make_chunk("federalist", essay_number=10, author="MADISON", part=2)
    assert chunk_matches(c, {"doc": "federalist", "essay_number": 10})


def test_rank_of_first_match_is_one_indexed():
    results = [make_chunk("amendments", amendment_number=1), make_chunk("amendments", amendment_number=4)]
    assert rank_of_first_match(results, {"doc": "amendments", "amendment_number": 4}) == 2
    assert rank_of_first_match(results, {"doc": "amendments", "amendment_number": 9}) is None


def test_recall_at_k_respects_the_cutoff():
    results = [make_chunk("amendments", amendment_number=n) for n in [1, 2, 3, 4]]
    expect = {"doc": "amendments", "amendment_number": 4}
    assert recall_at_k(results, expect, k=4) is True
    assert recall_at_k(results, expect, k=2) is False


def test_reciprocal_rank():
    results = [make_chunk("amendments", amendment_number=n) for n in [9, 1, 4]]
    assert reciprocal_rank(results, {"doc": "amendments", "amendment_number": 1}) == 1 / 2
    assert reciprocal_rank(results, {"doc": "amendments", "amendment_number": 99}) == 0.0


def test_mean_helpers_average_across_questions():
    results_a = [make_chunk("amendments", amendment_number=1)]  # hit at rank 1
    results_b = [make_chunk("amendments", amendment_number=9)]  # miss
    expects = [{"doc": "amendments", "amendment_number": 1}, {"doc": "amendments", "amendment_number": 1}]

    assert mean_recall_at_k([results_a, results_b], expects, k=1) == 0.5
    assert mean_reciprocal_rank([results_a, results_b], expects) == 0.5
