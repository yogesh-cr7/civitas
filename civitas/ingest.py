"""Parse the raw corpus text files into normalized Chunk objects.

Each source document (the Constitution's articles, the 27 amendments, and a
curated set of Federalist Papers) has its own light markup, because that's
what real ingestion pipelines look like -- different sources need different
parsers even when they feed the same downstream index. All three converge on
the same Chunk shape so retrieval doesn't need to know where a passage came
from.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "corpus"


@dataclass
class Chunk:
    id: str
    doc: str          # "constitution" | "amendments" | "federalist"
    title: str         # human-readable label, e.g. "Amendment IV" or "Federalist No. 10"
    text: str
    metadata: dict = field(default_factory=dict)

    def as_context(self) -> str:
        """How this chunk is rendered when stuffed into an LLM prompt."""
        return f"[{self.title}]\n{self.text.strip()}"


_ROMAN = [
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
    "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX",
    "XXI", "XXII", "XXIII", "XXIV", "XXV", "XXVI", "XXVII",
]


def _to_roman(n: int) -> str:
    return _ROMAN[n - 1] if 1 <= n <= len(_ROMAN) else str(n)


def _split_records(raw: str) -> list[str]:
    """Split on the ===END=== delimiter and drop empty records."""
    records = [r.strip() for r in raw.split("===END===")]
    return [r for r in records if r]


def parse_constitution(path: Path = CORPUS_DIR / "constitution_articles.txt") -> list[Chunk]:
    chunks = []
    for record in _split_records(path.read_text(encoding="utf-8")):
        lines = record.split("\n", 1)
        header = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""

        if header == "PREAMBLE":
            title = "Preamble"
            metadata = {"section": "preamble"}
        else:
            m = re.match(r"ARTICLE (\d+)(?: SECTION (\d+))?", header)
            if not m:
                raise ValueError(f"Unrecognized constitution header: {header!r}")
            article, section = m.group(1), m.group(2)
            title = f"Article {article}" + (f", Section {section}" if section else "")
            metadata = {"article": int(article), "section": int(section) if section else None}

        chunks.append(Chunk(
            id=f"const-{header.lower().replace(' ', '-')}",
            doc="constitution",
            title=title,
            text=body,
            metadata=metadata,
        ))
    return chunks


def parse_amendments(path: Path = CORPUS_DIR / "amendments.txt") -> list[Chunk]:
    chunks = []
    for record in _split_records(path.read_text(encoding="utf-8")):
        lines = record.split("\n")
        header = lines[0].strip()
        m = re.match(r"AMENDMENT (\d+)", header)
        if not m:
            raise ValueError(f"Unrecognized amendment header: {header!r}")
        number = int(m.group(1))
        ratified_line = lines[1].strip() if len(lines) > 1 else ""
        body = "\n".join(lines[2:]).strip()

        title = f"Amendment {_to_roman(number)}"
        chunks.append(Chunk(
            id=f"amend-{number}",
            doc="amendments",
            title=title,
            text=body,
            metadata={"amendment_number": number, "ratification": ratified_line},
        ))
    return chunks


_FEDERALIST_RE = re.compile(
    r"===ESSAY (\d+)===\s*\n"
    r"TITLE: (.*?)\n"
    r"AUTHOR: (.*?)\n"
    r"DATE: (.*?)\n"
    r"---\n"
    r"(.*?)\n===END===",
    re.DOTALL,
)


def parse_federalist(path: Path = CORPUS_DIR / "federalist_papers.txt") -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    chunks = []
    for number, title, author, date, body in _FEDERALIST_RE.findall(raw):
        chunks.append(Chunk(
            id=f"fed-{number}",
            doc="federalist",
            title=f"Federalist No. {number} ({title.strip()})",
            text=body.strip(),
            metadata={"essay_number": int(number), "author": author.strip(), "date": date.strip()},
        ))
    return chunks


def split_long_chunk(chunk: Chunk, max_words: int = 180, overlap_paragraphs: int = 1) -> list[Chunk]:
    """Break a long chunk (a full Federalist essay) into paragraph-aligned sub-chunks.

    Constitution articles and amendments are already short and atomic -- splitting
    the Fourth Amendment in half would just hurt retrieval. Federalist essays run
    800-1500+ words on one continuous argument, so we window them by paragraph
    with a one-paragraph overlap, which keeps each sub-chunk topically coherent
    without cutting a sentence in half.
    """
    paragraphs = [p.strip() for p in chunk.text.split("\n\n") if p.strip()]
    if sum(len(p.split()) for p in paragraphs) <= max_words or len(paragraphs) <= 1:
        return [chunk]

    windows: list[list[str]] = []
    current: list[str] = []
    current_words = 0
    for para in paragraphs:
        para_words = len(para.split())
        if current and current_words + para_words > max_words:
            windows.append(current)
            # start next window with the overlap tail of the previous one
            current = current[-overlap_paragraphs:] if overlap_paragraphs else []
            current_words = sum(len(p.split()) for p in current)
        current.append(para)
        current_words += para_words
    if current:
        windows.append(current)

    sub_chunks = []
    for i, window in enumerate(windows, start=1):
        sub_chunks.append(Chunk(
            id=f"{chunk.id}-p{i}",
            doc=chunk.doc,
            title=f"{chunk.title}, part {i}/{len(windows)}",
            text="\n\n".join(window),
            metadata={**chunk.metadata, "part": i, "parts_total": len(windows)},
        ))
    return sub_chunks


def load_corpus(split_long: bool = True) -> list[Chunk]:
    """Load and combine all three source documents into one flat chunk list.

    Federalist essays are windowed into smaller sub-chunks by default (see
    split_long_chunk); pass split_long=False to keep one chunk per essay.
    """
    chunks = parse_constitution() + parse_amendments() + parse_federalist()
    if not split_long:
        return chunks
    result = []
    for c in chunks:
        result.extend(split_long_chunk(c) if c.doc == "federalist" else [c])
    return result


if __name__ == "__main__":
    corpus = load_corpus()
    print(f"Loaded {len(corpus)} chunks:")
    by_doc: dict[str, int] = {}
    for c in corpus:
        by_doc[c.doc] = by_doc.get(c.doc, 0) + 1
    for doc, count in by_doc.items():
        print(f"  {doc}: {count}")
    words = [len(c.text.split()) for c in corpus]
    print(f"  chunk word count: min={min(words)} max={max(words)} avg={sum(words)/len(words):.0f}")
