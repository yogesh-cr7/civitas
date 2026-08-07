"""Citation-fidelity check for synthesized answers.

The LLM system prompt only allows citing the bracketed labels shown before
each retrieved excerpt (e.g. "[Amendment IV]", "[Federalist No. 84 (...)]").
This doesn't try to fully parse prose - it does two cheap, honest checks: no
bare URLs (the model has no business citing an external link it wasn't
given), and any bracketed "Federalist No. N" / "Amendment N" / "Article N"
reference has to point at a document that was actually retrieved. The model
often paraphrases the bracket text rather than reproducing a chunk's title
verbatim, so this matches on the document number rather than exact string
equality.
"""
from __future__ import annotations

import re

from civitas.ingest import Chunk

_BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")
_URL_RE = re.compile(r"https?://\S+")

_FEDERALIST_NUM_RE = re.compile(r"Federalist No\.?\s*(\d+)", re.IGNORECASE)
_AMENDMENT_NUM_RE = re.compile(r"Amendment\s+([IVXLCDM]+|\d+)", re.IGNORECASE)
_ARTICLE_NUM_RE = re.compile(r"Article\s+(\d+)", re.IGNORECASE)

_ROMAN_TO_INT = {
    roman: n
    for n, roman in enumerate(
        [
            "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
            "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX",
            "XXI", "XXII", "XXIII", "XXIV", "XXV", "XXVI", "XXVII",
        ],
        start=1,
    )
}


def _bracket_matches_a_retrieved_chunk(bracketed: str, chunks: list[Chunk]) -> bool:
    m = _FEDERALIST_NUM_RE.search(bracketed)
    if m:
        n = int(m.group(1))
        return any(c.doc == "federalist" and c.metadata.get("essay_number") == n for c in chunks)

    m = _AMENDMENT_NUM_RE.search(bracketed)
    if m:
        raw = m.group(1).upper()
        n = _ROMAN_TO_INT.get(raw) if not raw.isdigit() else int(raw)
        return any(c.doc == "amendments" and c.metadata.get("amendment_number") == n for c in chunks)

    m = _ARTICLE_NUM_RE.search(bracketed)
    if m:
        n = int(m.group(1))
        return any(c.doc == "constitution" and c.metadata.get("article") == n for c in chunks)

    # doesn't look like one of the known citation shapes at all - not our call to make
    return True


def check_citation_fidelity(answer: str, chunks: list[Chunk]) -> list[str]:
    """Return a list of problems found in `answer`; empty means it only cited what it was given."""
    problems = []

    for url in _URL_RE.findall(answer):
        problems.append(f"cites an external url not in the retrieved context: {url}")

    for bracketed in _BRACKET_RE.findall(answer):
        if not _bracket_matches_a_retrieved_chunk(bracketed, chunks):
            problems.append(f"cites '[{bracketed}]', which doesn't match a retrieved passage")

    return problems
