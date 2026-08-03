"""Run the question set against a fresh index and print a retrieval report."""
from __future__ import annotations

from civitas.index import VectorIndex
from eval.metrics import mean_reciprocal_rank, mean_recall_at_k, rank_of_first_match
from eval.questions import QUESTIONS


def run(embedder_name: str = "tfidf", k_values: tuple[int, ...] = (1, 3, 5)) -> None:
    print(f"Building a fresh index with embedder={embedder_name!r} ...")
    index = VectorIndex.build(embedder_name=embedder_name)

    max_k = max(k_values)
    all_results = [[c for c, _ in index.search(q.query, k=max_k)] for q in QUESTIONS]
    all_expect = [q.expect for q in QUESTIONS]

    print(f"\n{len(QUESTIONS)} questions, embedder={embedder_name!r}\n")
    misses = []
    for q, results in zip(QUESTIONS, all_results):
        rank = rank_of_first_match(results, q.expect)
        tag = "hard" if q.hard else "    "
        status = f"rank {rank}" if rank else "MISS"
        print(f"[{tag}] {status:>7}  {q.query}")
        if rank is None:
            misses.append(q)

    print("\n--- summary ---")
    for k in k_values:
        print(f"recall@{k}: {mean_recall_at_k(all_results, all_expect, k):.2f}")
    print(f"MRR:      {mean_reciprocal_rank(all_results, all_expect):.2f}")

    if misses:
        print(f"\n{len(misses)} complete misses (not in top {max_k}):")
        for q in misses:
            print(f"  - {q.query!r} (hard={q.hard})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedder", choices=["tfidf", "sentence-transformers"], default="tfidf")
    args = parser.parse_args()
    run(embedder_name=args.embedder)
