from civitas.ingest import Chunk
from eval.citation_check import check_citation_fidelity


def make_chunk(doc, title, **metadata):
    return Chunk(id=title.lower(), doc=doc, title=title, text="x", metadata=metadata)


def test_no_problems_when_answer_only_cites_retrieved_chunks():
    chunks = [make_chunk("amendments", "Amendment IV", amendment_number=4)]
    answer = "The Fourth Amendment protects against unreasonable searches. [Amendment IV]"
    assert check_citation_fidelity(answer, chunks) == []


def test_paraphrased_bracket_still_matches_by_document_number():
    # the model often shortens "Federalist No. 84 (Certain General ...), part 8/12"
    # down to something like this instead of reproducing the title verbatim
    chunks = [make_chunk("federalist", "Federalist No. 84 (...), part 8/12", essay_number=84)]
    answer = "Hamilton argued this. [Federalist No. 84, part 8/12]"
    assert check_citation_fidelity(answer, chunks) == []


def test_flags_a_bare_url_not_in_the_context():
    chunks = [make_chunk("federalist", "Federalist No. 84 (...)", essay_number=84)]
    answer = "See [Federalist No. 84]. Cite: https://www.consource.org/document/x"
    problems = check_citation_fidelity(answer, chunks)
    assert len(problems) == 1
    assert "consource.org" in problems[0]


def test_flags_a_citation_to_a_document_that_was_never_retrieved():
    chunks = [make_chunk("federalist", "Federalist No. 10", essay_number=10)]
    answer = "This echoes [Federalist No. 51], which was never actually retrieved."
    problems = check_citation_fidelity(answer, chunks)
    assert len(problems) == 1
    assert "Federalist No. 51" in problems[0]


def test_amendment_roman_numerals_resolve_to_the_right_number():
    chunks = [make_chunk("amendments", "Amendment XIV", amendment_number=14)]
    answer = "Equal protection comes from [Amendment XIV]."
    assert check_citation_fidelity(answer, chunks) == []


def test_article_citations_check_against_retrieved_articles():
    chunks = [make_chunk("constitution", "Article 1, Section 8", article=1, section=8)]
    assert check_citation_fidelity("war powers, see [Article 1]", chunks) == []
    problems = check_citation_fidelity("see [Article 2]", chunks)
    assert len(problems) == 1
