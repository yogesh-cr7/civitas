from civitas.ingest import (
    Chunk,
    load_corpus,
    parse_amendments,
    parse_constitution,
    parse_federalist,
    split_long_chunk,
)


def test_parse_constitution_covers_preamble_through_article_seven():
    chunks = parse_constitution()
    titles = {c.title for c in chunks}
    assert "Preamble" in titles
    assert "Article 1, Section 8" in titles  # the enumerated-powers clause
    assert "Article 7" in titles
    assert all(c.doc == "constitution" for c in chunks)
    assert all(c.text for c in chunks), "every chunk should have non-empty body text"


def test_parse_amendments_returns_all_27_in_order():
    chunks = parse_amendments()
    assert len(chunks) == 27
    numbers = [c.metadata["amendment_number"] for c in chunks]
    assert numbers == list(range(1, 28))
    assert chunks[3].title == "Amendment IV"
    assert "unreasonable searches" in chunks[3].text


def test_parse_federalist_extracts_metadata():
    chunks = parse_federalist()
    numbers = {c.metadata["essay_number"] for c in chunks}
    assert numbers == {1, 10, 15, 39, 47, 51, 70, 78, 84}
    fed10 = next(c for c in chunks if c.metadata["essay_number"] == 10)
    assert fed10.metadata["author"] == "MADISON"
    assert "faction" in fed10.text.lower()


def test_split_long_chunk_preserves_all_text_and_keeps_short_chunks_whole():
    short = Chunk(id="x", doc="amendments", title="Amendment I", text="A short amendment.")
    assert split_long_chunk(short) == [short]

    long_text = "\n\n".join(f"Paragraph {i} " + ("word " * 60) for i in range(6))
    long_chunk = Chunk(id="fed-99", doc="federalist", title="Federalist No. 99", text=long_text)
    parts = split_long_chunk(long_chunk, max_words=100, overlap_paragraphs=1)

    assert len(parts) > 1
    assert all(p.id.startswith("fed-99-p") for p in parts)
    # every paragraph from the original should show up in at least one sub-chunk
    for i in range(6):
        assert any(f"Paragraph {i} " in p.text for p in parts)


def test_load_corpus_has_no_duplicate_ids():
    chunks = load_corpus()
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk ids must be unique for the index to work"
    assert len(chunks) > 100
