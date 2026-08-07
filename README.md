# civitas

A small retrieval-augmented generation (RAG) system over the U.S. founding legal
documents — the Constitution, its 27 amendments, and nine landmark Federalist
Papers — built as a portfolio project with a real evaluation suite, not just a
retrieval demo.

Most RAG side-projects stop at "it retrieves something and an LLM answers."
This one is about the other half: how do you know if it's actually any good,
and how do you compare design choices (chunk size, embedding backend, prompt)
with numbers instead of vibes. There's a hand-written eval suite (32 questions
with ground truth, recall@k and MRR) sitting on top of a clean corpus, a
correct ingestion pipeline, and retrieval that works without needing an API key.

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
eval/             hand-written question set, recall@k / MRR metrics, eval runner
```

Retrieval defaults to **TF-IDF**, not a downloaded embedding model. For a
139-document corpus that's not a compromise — it's the honest choice: no
multi-hundred-megabyte download, no GPU, instant indexing, and a lexical
method is easy to reason about when something goes wrong. `sentence-transformers`
is wired in as a drop-in alternative, and the eval suite below actually
measures whether it's worth the download rather than assuming it is.

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
quietly patch, but exactly the kind of thing the eval suite below is built
to catch and quantify.

## Retrieval eval

```bash
python -m civitas eval
python -m civitas eval --embedder sentence-transformers
```

32 hand-written questions across the Constitution, amendments, and Federalist
Papers, each checked against ground-truth metadata (document + article/amendment/
essay number) rather than an exact chunk id, so re-chunking doesn't silently
break the eval.

| metric    | TF-IDF | sentence-transformers |
|-----------|--------|------------------------|
| recall@1  | 0.50   | 0.62                   |
| recall@3  | 0.69   | 0.84                   |
| recall@5  | 0.72   | 0.91                   |
| MRR       | 0.58   | 0.73                   |

Semantic embeddings win clearly on aggregate — recall@5 goes from 0.72 to
0.91, and the numeral-reference problem above mostly resolves ("the fourth
amendment" moves from a complete miss to rank 4). But it's not a strict
improvement: two queries TF-IDF got exactly right at rank 1 — "equal
protection under the law" and "ambition must be made to counteract ambition"
— become complete misses for sentence-transformers, likely because a
general-purpose sentence embedding doesn't weight exact legal phrasing as
heavily as an overlapping-vocabulary method does. Semantic search is the
better default here, but an aggregate MRR alone would hide the fact that it
trades some failures for others rather than strictly fixing them.

## Roadmap

- Score synthesized answers, not just retrieved passages — including a
  citation-fidelity check after finding a real case where the model cited a
  source outside the retrieved context
