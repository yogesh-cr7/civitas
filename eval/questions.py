"""Hand-written eval set: questions with ground truth, not exact chunk ids.

`expect` matches on doc type + metadata (e.g. amendment_number, essay_number)
instead of a literal chunk id, since Federalist essays get split into several
sub-chunks and a chunk id like "fed-51-p3" is an indexing detail, not the
actual right answer. Any sub-chunk belonging to the right document counts.

`hard=True` marks queries we expect a lexical method like TF-IDF to struggle
with (paraphrases, numeral-only references) -- kept in the set on purpose so
the eval reports an honest number instead of only easy wins.
"""
from dataclasses import dataclass


@dataclass
class Question:
    query: str
    expect: dict
    hard: bool = False


QUESTIONS = [
    # Constitution
    Question("what is the minimum age to be a senator", {"doc": "constitution", "article": 1, "section": 3}),
    Question("how long is a president's term in years", {"doc": "constitution", "article": 2, "section": 1}),
    Question("what power does congress have to declare war", {"doc": "constitution", "article": 1, "section": 8}),
    Question("how are amendments to the constitution proposed", {"doc": "constitution", "article": 5, "section": None}),
    Question("what happens to an officer convicted on impeachment", {"doc": "constitution", "article": 2, "section": 4}),
    Question("how many votes are needed to override a presidential veto", {"doc": "constitution", "article": 1, "section": 7}),
    Question("how many states had to ratify the constitution for it to take effect", {"doc": "constitution", "article": 7, "section": None}),
    Question("what is the minimum age to be president", {"doc": "constitution", "article": 2, "section": 1}, hard=True),

    # Amendments
    Question("freedom of speech and of the press", {"doc": "amendments", "amendment_number": 1}),
    Question("the fourth amendment", {"doc": "amendments", "amendment_number": 4}, hard=True),
    Question("protection against unreasonable searches and seizures", {"doc": "amendments", "amendment_number": 4}),
    Question("right to a speedy and public trial by jury", {"doc": "amendments", "amendment_number": 6}),
    Question("protection against self-incrimination", {"doc": "amendments", "amendment_number": 5}),
    Question("cruel and unusual punishment", {"doc": "amendments", "amendment_number": 8}),
    Question("abolition of slavery", {"doc": "amendments", "amendment_number": 13}),
    Question("equal protection under the law", {"doc": "amendments", "amendment_number": 14}),
    Question("women's right to vote", {"doc": "amendments", "amendment_number": 19}, hard=True),
    Question("voting age lowered to eighteen", {"doc": "amendments", "amendment_number": 26}),
    Question("prohibition of alcohol", {"doc": "amendments", "amendment_number": 18}),
    Question("repeal of prohibition", {"doc": "amendments", "amendment_number": 21}),
    Question("federal income tax", {"doc": "amendments", "amendment_number": 16}),
    Question("direct election of senators by the people", {"doc": "amendments", "amendment_number": 17}),
    Question("a president can only serve two terms", {"doc": "amendments", "amendment_number": 22}, hard=True),

    # Federalist Papers
    Question("why a large republic controls factions better than a small one", {"doc": "federalist", "essay_number": 10}),
    Question("why the constitution doesn't need a bill of rights", {"doc": "federalist", "essay_number": 84}),
    Question("ambition must be made to counteract ambition", {"doc": "federalist", "essay_number": 51}),
    Question("why judges should hold office during good behavior", {"doc": "federalist", "essay_number": 78}),
    Question("why a single executive is better than a plural executive", {"doc": "federalist", "essay_number": 70}),
    Question("separation of powers among departments of government", {"doc": "federalist", "essay_number": 47}),
    Question("weaknesses of the articles of confederation", {"doc": "federalist", "essay_number": 15}, hard=True),
    Question("is the proposed government national or federal", {"doc": "federalist", "essay_number": 39}),
    Question("general introduction to the federalist papers", {"doc": "federalist", "essay_number": 1}),
]
