"""Command-line entry point: `python -m civitas <command>`.

    python -m civitas index                 build the local index (fast, free)
    python -m civitas query "..."            retrieve the top-k passages
    python -m civitas query "..." --llm      also synthesize an answer (needs API key)
"""
from __future__ import annotations

import argparse
import sys

from .index import DEFAULT_INDEX_PATH, VectorIndex


def cmd_index(args: argparse.Namespace) -> None:
    print(f"Building index with embedder={args.embedder!r} ...")
    idx = VectorIndex.build(embedder_name=args.embedder)
    idx.save()
    print(f"Indexed {len(idx.chunks)} chunks -> {DEFAULT_INDEX_PATH}")


def cmd_query(args: argparse.Namespace) -> None:
    if not DEFAULT_INDEX_PATH.exists():
        print("No index found. Run `python -m civitas index` first.", file=sys.stderr)
        sys.exit(1)

    idx = VectorIndex.load()
    results = idx.search(args.query, k=args.k)

    print(f"\nTop {len(results)} passages for: {args.query!r}\n")
    for rank, (chunk, score) in enumerate(results, start=1):
        print(f"{rank}. [{score:.3f}] {chunk.title}")
        preview = chunk.text.strip().replace("\n", " ")
        print(f"   {preview[:220]}{'...' if len(preview) > 220 else ''}\n")

    if args.llm:
        from .llm import synthesize_answer
        print("Synthesizing answer with Claude...\n")
        answer = synthesize_answer(args.query, [c for c, _ in results])
        print(answer)


def main() -> None:
    parser = argparse.ArgumentParser(prog="civitas", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Build and save the local vector index")
    p_index.add_argument(
        "--embedder", choices=["tfidf", "sentence-transformers"], default="tfidf",
        help="Embedding backend (default: tfidf, no extra downloads required)",
    )
    p_index.set_defaults(func=cmd_index)

    p_query = sub.add_parser("query", help="Retrieve passages for a question")
    p_query.add_argument("query", help="Your question, in quotes")
    p_query.add_argument("-k", type=int, default=5, help="Number of passages to retrieve (default: 5)")
    p_query.add_argument(
        "--llm", action="store_true",
        help="Also synthesize a cited answer via the Anthropic API (requires ANTHROPIC_API_KEY)",
    )
    p_query.set_defaults(func=cmd_query)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
