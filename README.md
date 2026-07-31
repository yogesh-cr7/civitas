# civitas

A small retrieval-augmented generation (RAG) system over the U.S. founding legal
documents — the Constitution, its 27 amendments, and nine landmark Federalist
Papers — built as a portfolio project with a real evaluation suite, not just a
retrieval demo.

Most RAG side-projects stop at "it retrieves something and an LLM answers."
This one is about the other half: how do you know if it's actually any good,
and how do you compare design choices (chunk size, embedding backend, prompt)
with numbers instead of vibes? That eval layer lands in session 2. This
session is the foundation it needs to stand on: a clean corpus, a correct
ingestion pipeline, and retrieval that works without needing an API key.

## Why this corpus

Public-domain, dense, and factual. Every claim in the Constitution and its
amendments has exactly one correct reading, which means eval questions like
*"what does the Fourth Amendment protect against?"* have unambiguous ground
truth — no fabricated data, no ambiguity about what "correct" means.

- **Constitution**: Preamble + all 7 Articles, split by section ([Project Gutenberg](https://www.gutenberg.org/ebooks/5))
- **Amendments**: all 27, 1 through 27 ([National Archives](https://www.archives.gov/founding-docs/bill-of-rights-transcript))
- **Federalist Papers**: No. 1, 10, 15, 39, 47, 51, 70, 78, 84 — a curated set
  covering union, faction, separation of powers, checks and balances, the
  executive, judicial review, and the bill-of-rights debate ([Avalon Project, Yale Law School](https://avalon.law.yale.edu/subject_menus/fed.asp))

The Federalist Papers are lightly excerpted in a couple of places (repetitive
enumerations, tangential asides) to keep chunk sizes reasonable — everything
kept is verbatim, nothing is paraphrased. The Constitution and amendments are
complete and unabridged.

## Architecture

```
civitas/
  ingest.py       parse raw corpus text -> normalized Chunk objects
  embeddings.py   TF-IDF (default, no deps) or sentence-transformers (optional)
  index.py        build/save/load a brute-force cosine-similarity vector index
  llm.py          optional answer synthesis via the Anthropic API
  cli.py          `python -m civitas index|query`
data/corpus/      the three source text files
tests/            ingestion + retrieval regression tests
eval/             placeholder -- the eval suite lands in session 2
```

Retrieval defaults to **TF-IDF**, not a downloaded embedding model. For a
139-document corpus that's not a compromise — it's the honest choice: no
multi-hundred-megabyte download, no GPU, instant indexing, and a lexical
method is easy to reason about when something goes wrong. `sentence-transformers`
is wired in as a drop-in alternative specifically so the eval suite can
measure whether semantic embeddings actually help here, instead of assuming
they do.

## Setup

```bash
pip install -r requirements.txt
python -m civitas index
python -m civitas query "what does the fourth amendment protect against"
```

Add `--llm` to get a synthesized, cited answer instead of raw passages
(requires `pip install anthropic` and `ANTHROPIC_API_KEY` set):

```bash
python -m civitas query "why did Hamilton think the Constitution didn't need a bill of rights" --llm
```

To index with real semantic embeddings instead of TF-IDF:

```bash
pip install sentence-transformers
python -m civitas index --embedder sentence-transformers
```

## Tests

```bash
pytest
```

Covers: ingestion (all 27 amendments parse in order, chunk IDs are unique,
long-essay splitting preserves every paragraph) and retrieval (known easy
queries hit the right document, save/load round-trips exactly).

## Known limitation

TF-IDF is lexical, not semantic. Queries phrased as *"the fourth amendment"*
can lose to a document that happens to share more surface words (e.g.
"protect," "against") even when a differently-numbered amendment is the
actually-correct answer, because the amendment text itself never spells out
its own number. This is a real, reproducible failure mode — not a bug to
quietly patch, but exactly the kind of thing the eval suite in session 2 is
meant to catch and quantify, including whether semantic embeddings fix it.

## Roadmap

- **Session 2**: eval suite — a hand-written question set with ground truth,
  retrieval metrics (recall@k, MRR), answer-quality scoring, and a
  side-by-side comparison of TF-IDF vs. sentence-transformers.
