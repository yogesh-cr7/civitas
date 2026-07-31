"""Optional answer synthesis on top of retrieval, via the Anthropic API.

Retrieval (civitas.index) never needs this module or an API key -- it's pure
local computation. This module is only imported when the CLI is run with
--llm, mirroring the pattern of keeping the free, deterministic path fully
separate from the part that costs money and depends on a network call.
"""
from __future__ import annotations

import os

from .ingest import Chunk

SYSTEM_PROMPT = (
    "You are a careful research assistant answering questions about U.S. founding "
    "legal documents (the Constitution, its amendments, and the Federalist Papers). "
    "Answer ONLY using the excerpts provided in the context below. If the context "
    "does not contain enough information to answer, say so explicitly instead of "
    "guessing. Cite which document each part of your answer comes from, using the "
    "bracketed labels shown before each excerpt (e.g. [Amendment IV], "
    "[Federalist No. 51 (...)])."
)


def synthesize_answer(query: str, chunks: list[Chunk], model: str = "claude-sonnet-4-5-20250929") -> str:
    try:
        import anthropic
    except ImportError as e:
        raise ImportError(
            "The `anthropic` package is required for --llm. Install it with "
            "`pip install anthropic`."
        ) from e

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it in your shell "
            "(or add it to a .env file you load yourself) before using --llm."
        )

    context = "\n\n---\n\n".join(c.as_context() for c in chunks)
    user_message = f"Context:\n\n{context}\n\n---\n\nQuestion: {query}"

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return "".join(block.text for block in response.content if block.type == "text")
