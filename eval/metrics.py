"""Retrieval metrics: recall@k and MRR, scored against the Question.expect predicate."""
from __future__ import annotations

from civitas.ingest import Chunk


def chunk_matches(chunk: Chunk, expect: dict) -> bool:
    for key, value in expect.items():
        if key == "doc":
            if chunk.doc != value:
                return False
        elif chunk.metadata.get(key) != value:
            return False
    return True


def rank_of_first_match(results: list[Chunk], expect: dict) -> int | None:
    """1-based rank of the first matching chunk in results, or None if not found."""
    for i, chunk in enumerate(results, start=1):
        if chunk_matches(chunk, expect):
            return i
    return None


def recall_at_k(results: list[Chunk], expect: dict, k: int) -> bool:
    rank = rank_of_first_match(results[:k], expect)
    return rank is not None


def reciprocal_rank(results: list[Chunk], expect: dict) -> float:
    rank = rank_of_first_match(results, expect)
    return 1.0 / rank if rank else 0.0


def mean_recall_at_k(all_results: list[list[Chunk]], all_expect: list[dict], k: int) -> float:
    hits = [recall_at_k(r, e, k) for r, e in zip(all_results, all_expect)]
    return sum(hits) / len(hits) if hits else 0.0


def mean_reciprocal_rank(all_results: list[list[Chunk]], all_expect: list[dict]) -> float:
    scores = [reciprocal_rank(r, e) for r, e in zip(all_results, all_expect)]
    return sum(scores) / len(scores) if scores else 0.0
